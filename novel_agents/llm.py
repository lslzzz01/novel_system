from __future__ import annotations

import http.client
import json
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


class LanguageModel(Protocol):
    def generate_json(self, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]: ...

    def generate_text(self, system_prompt: str, payload: dict[str, Any]) -> str: ...


@dataclass(slots=True)
class OpenAICompatibleClient:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 180
    temperature: float | None = 0.4
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    reasoning_effort: str | None = None
    use_json_response_format: bool = True
    supported_parameters: tuple[str, ...] = (
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "max_tokens",
    )
    extra_parameters: dict[str, Any] = field(default_factory=dict)

    def generate_json(
        self, system_prompt: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        last_error: ValueError | None = None
        for attempt in range(2):
            request_payload = dict(payload)
            if last_error is not None:
                request_payload["output_repair"] = (
                    "上一次输出不是有效 JSON。请重新生成完整 JSON 对象，不要省略字段，"
                    f"不要添加说明。错误：{last_error}"
                )
            content = self._complete(
                system_prompt,
                request_payload,
                json_mode=True,
                temperature_override=0.2 if attempt else None,
            )
            try:
                return self._parse_json_object(content)
            except ValueError as exc:
                last_error = exc
        raise RuntimeError(f"模型连续两次没有返回有效 JSON：{last_error}")

    def generate_text(self, system_prompt: str, payload: dict[str, Any]) -> str:
        return self._complete(system_prompt, payload, json_mode=False).strip()

    def _complete(
        self,
        system_prompt: str,
        payload: dict[str, Any],
        json_mode: bool,
        temperature_override: float | None = None,
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                },
            ],
        }
        parameter_values = {
            "temperature": (
                temperature_override
                if temperature_override is not None
                else self.temperature
            ),
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "reasoning_effort": self.reasoning_effort,
        }
        allowed = set(self.supported_parameters)
        for name, value in parameter_values.items():
            if name not in allowed or value is None:
                continue
            # Some gateways reject top_p=0 / seed=0 / max_tokens=0 as invalid.
            if name in {"top_p", "frequency_penalty", "presence_penalty"} and float(value) == 0.0:
                continue
            if name in {"max_tokens", "seed"} and int(value) <= 0:
                continue
            if name == "max_tokens" and int(value) > 32000:
                # Guard against misconfigured huge ceilings that some providers reject.
                body[name] = 16000
                continue
            body[name] = value
        for name, value in self.extra_parameters.items():
            if name not in {"model", "messages", "stream"} and value is not None:
                body[name] = value
        if json_mode and self.use_json_response_format:
            body["response_format"] = {"type": "json_object"}
        errors: list[str] = []
        # Start with ordinary responses because they are easier to validate. Use a
        # streaming attempt for gateways that drop long non-stream responses, then
        # return to ordinary transport once in case the gateway only rejects SSE.
        # This changes transport only; it never alters the configured output limit.
        transports = (False, False, True, False)
        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                if transports[attempt]:
                    return self._request_stream(body)
                result = self._request_json(body)
                content = self._response_content(result)
                if content.strip():
                    return content
                # A 200 response with an empty content field is a known gateway
                # failure mode. Treat it as transient so another transport can run.
                errors.append("Model returned an empty response")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code not in {408, 409, 425, 429, 500, 502, 503, 504}:
                    raise RuntimeError(
                        f"模型接口返回 HTTP {exc.code}：{detail}"
                    ) from exc
                errors.append(f"HTTP {exc.code}: {detail[:300]}")
            except urllib.error.URLError as exc:
                reason = exc.reason
                if self._is_local_permission_error(reason):
                    raise RuntimeError(self._local_permission_message(reason)) from exc
                if not self._is_transient_connection_error(reason) and not self._is_transient_message(
                    reason
                ):
                    raise RuntimeError(f"无法连接模型接口：{reason}") from exc
                errors.append(str(reason))
            except (
                http.client.RemoteDisconnected,
                http.client.IncompleteRead,
                ConnectionResetError,
                BrokenPipeError,
                TimeoutError,
                socket.timeout,
                json.JSONDecodeError,
            ) as exc:
                errors.append(str(exc) or exc.__class__.__name__)
            except RuntimeError as exc:
                message = str(exc)
                # Retry transient upstream/stream failures; keep hard errors immediate.
                if not self._is_transient_message(message):
                    raise
                errors.append(message)
            if attempt < max_attempts - 1:
                time.sleep(0.8 * (2**attempt))
        detail = errors[-1] if errors else "未知连接错误"
        raise RuntimeError(f"模型请求重试 {max_attempts} 次后仍失败：{detail}")

    def _request_json(self, body: dict[str, Any]) -> dict[str, Any]:
        request = self._request(body)
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected model response: {result}")
        return result

    def _request_stream(self, body: dict[str, Any]) -> str:
        streaming_body = dict(body)
        streaming_body["stream"] = True
        request = self._request(streaming_body)
        content_parts: list[str] = []
        raw_lines: list[str] = []
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                raw_lines.append(line)
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                if isinstance(event, dict) and event.get("error"):
                    raise RuntimeError(f"Model streaming error: {event['error']}")
                try:
                    delta = event["choices"][0].get("delta", {})
                    part = delta.get("content", "")
                except (KeyError, IndexError, TypeError, AttributeError):
                    part = ""
                content_parts.append(self._content_text(part))
        content = "".join(content_parts)
        if content:
            return content

        # Some compatible gateways ignore stream=true and return one JSON response.
        raw = "\n".join(raw_lines)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Streaming response did not contain model output") from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected streaming response: {result}")
        return self._response_content(result)

    def _request(self, body: dict[str, Any]) -> urllib.request.Request:
        url = self._chat_completions_url()
        request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        return request

    def _chat_completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        # Accept both https://host and https://host/v1 style base URLs.
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    @classmethod
    def _response_content(cls, result: dict[str, Any]) -> str:
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected model response: {result}") from exc
        return cls._content_text(content)

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, list):
            return "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content or "")

    @staticmethod
    def _is_transient_connection_error(error: Any) -> bool:
        if OpenAICompatibleClient._is_local_permission_error(error):
            return False
        return isinstance(
            error,
            (
                http.client.RemoteDisconnected,
                http.client.IncompleteRead,
                ConnectionResetError,
                BrokenPipeError,
                TimeoutError,
                socket.timeout,
                OSError,
            ),
        )

    @staticmethod
    def _is_local_permission_error(error: Any) -> bool:
        if not isinstance(error, OSError):
            return False
        code = getattr(error, "winerror", None) or getattr(error, "errno", None)
        return code == 10013

    @staticmethod
    def _local_permission_message(error: OSError) -> str:
        code = getattr(error, "winerror", None) or getattr(error, "errno", None)
        return (
            "本机网络权限拒绝了模型服务连接（WinError "
            f"{code}）。请检查系统防火墙、安全软件或代理设置后再试。原始错误：{error}"
        )

    @staticmethod
    def _is_transient_message(error: Any) -> bool:
        text = str(error or "").lower()
        markers = (
            "remote end closed connection",
            "connection reset",
            "connection aborted",
            "broken pipe",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "upstream",
            "http/2 stream",
            "incomplete read",
            "empty response",
            "did not contain model output",
            "10054",
            "10053",
            "eof occurred",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any]:
        cleaned = content.strip()
        fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            decoder = json.JSONDecoder()
            parsed = None
            for index, character in enumerate(cleaned):
                if character != "{":
                    continue
                try:
                    candidate, _ = decoder.raw_decode(cleaned[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    parsed = candidate
                    break
            if parsed is None:
                raise ValueError(
                    f"模型没有返回有效 JSON：{content[:500]}"
                ) from exc
        if not isinstance(parsed, dict):
            raise ValueError("Model JSON response must be an object")
        return parsed


@dataclass
class MockLanguageModel:
    """Deterministic local model used to verify workflow wiring."""

    calls: list[str] = field(default_factory=list)

    def generate_json(
        self, system_prompt: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        task = str(payload["task"])
        self.calls.append(task)
        project = payload.get("project", {})
        if task == "genre_analysis":
            return {
                "positioning": f"A focused {project['genre']} novel",
                "target_readers": project["target_readers"],
                "core_promises": ["clear progression", "meaningful choices"],
                "tone": project["style"],
                "taboos": project.get("constraints", []),
            }
        if task == "world_bible":
            return {
                "premise": project["premise"],
                "rules": ["Every power has a visible cost"],
                "factions": [{"name": "Central faction", "goal": "Preserve order"}],
                "locations": [{"name": "Opening location", "purpose": "Start the conflict"}],
                "history": ["A past event created the present conflict"],
                "glossary": {},
            }
        if task == "character_bible":
            return {
                "protagonists": [
                    {
                        "name": "Protagonist",
                        "goal": "Resolve the central conflict",
                        "flaw": "Acts before trusting others",
                        "arc": "Learns responsibility without losing agency",
                    }
                ],
                "supporting_cast": [],
                "relationships": [],
                "arcs": ["Protagonist: isolation to earned trust"],
            }
        if task == "story_outline":
            count = int(project["chapter_count"])
            return {
                "logline": project["premise"],
                "theme": "Choices create identity",
                "ending": "The central conflict is resolved through the protagonist's choice",
                "acts": [
                    {"name": "Setup", "purpose": "Establish desire and disruption"},
                    {"name": "Escalation", "purpose": "Raise cost and reveal truth"},
                    {"name": "Resolution", "purpose": "Force the final choice"},
                ],
                "volumes": [{"number": 1, "goal": "Complete the primary arc"}],
                "chapter_map": [
                    {
                        "chapter_number": number,
                        "purpose": f"Advance the central conflict at step {number}",
                    }
                    for number in range(1, count + 1)
                ],
            }
        if task == "story_outline_chapter_map":
            return {
                "chapter_map": [
                    {
                        "chapter_number": int(number),
                        "purpose": (
                            "Advance the central conflict at step "
                            f"{int(number)}"
                        ),
                    }
                    for number in payload.get("requested_chapter_numbers", [])
                ]
            }
        if task == "relationship_arcs":
            character_bible = payload.get("inputs", {}).get("character_bible", {})
            cast = list(character_bible.get("protagonists", [])) + list(
                character_bible.get("supporting_cast", [])
            )
            names = [
                str(item.get("name", "")).strip()
                for item in cast
                if isinstance(item, dict) and str(item.get("name", "")).strip()
            ]
            relationships = []
            if len(names) >= 2:
                relationships.append(
                    {
                        "id": "mock-primary-pair",
                        "character_a": names[0],
                        "character_b": names[1],
                        "relationship_type": "romance",
                        "initial_stage": "stranger",
                        "target_stage": "dating",
                        "public_status": "互不熟悉",
                        "private_status": "尚无私人关系",
                        "a_to_b": {
                            "trust": 0,
                            "attraction": 0,
                            "dependence": 0,
                            "guard": 3,
                            "perception": "陌生人",
                        },
                        "b_to_a": {
                            "trust": 0,
                            "attraction": 0,
                            "dependence": 0,
                            "guard": 3,
                            "perception": "陌生人",
                        },
                        "address_terms": {"a_to_b": "", "b_to_a": ""},
                        "touch_boundary": "保持普通社交距离",
                        "arc_summary": "从谨慎接触发展为经过验证的信任。",
                        "milestones": [
                            {
                                "from_chapter": 1,
                                "stage": "stranger",
                                "trigger": "开局保持距离",
                            }
                        ],
                        "stage_schedule": [
                            {
                                "from_chapter": 1,
                                "stage": "stranger",
                                "mode": "confirm",
                                "summary": "开局保持距离",
                                "required_evidence": [],
                            }
                        ],
                        "next_milestone": "完成一次有代价的合作",
                        "forbidden_leaps": ["一次帮助后直接告白"],
                        "genre_signals": ["称呼和合作方式发生细微变化"],
                    }
                )
            return {
                "global_rules": ["关系变化必须由定稿中的具体事件支持"],
                "relationships": relationships,
            }
        if task == "subplot_register":
            return {
                "subplots": [
                    {
                        "name": "Trust subplot",
                        "entry": 1,
                        "development": "Pressure tests cooperation",
                        "payoff": project["chapter_count"],
                    }
                ],
                "foreshadowing": [
                    {
                        "seed": "An unexplained object",
                        "plant_chapter": 1,
                        "payoff_chapter": project["chapter_count"],
                    }
                ],
                "payoff_matrix": [],
            }
        if task == "chapter_plan":
            number = int(payload["chapter_number"])
            minimum_chars = int(project["chapter_word_count"])
            target_chars = (minimum_chars * 110 + 99) // 100
            scene_count = 5 if minimum_chars >= 4000 else 2
            base_scene_chars, extra_scene_chars = divmod(
                target_chars, scene_count
            )
            return {
                "chapter_number": number,
                "title": f"Chapter {number}",
                "objective": f"Advance the main conflict in chapter {number}",
                "conflict": "The protagonist must choose between speed and trust",
                "characters": ["Protagonist"],
                "required_reveals": [f"Reveal one consequence for chapter {number}"],
                "withhold": ["The complete truth behind the central conflict"],
                "continuity_inputs": [],
                "ending_hook": "A new cost becomes visible",
                "target_word_count": minimum_chars,
                "mainline": {
                    "trigger": "A pending consequence forces the protagonist to act",
                    "action": "The protagonist pursues the chapter objective",
                    "turn": "Resistance makes the original approach unusable",
                    "result": "A deliberate choice changes the situation",
                },
                "subplot": {
                    "trigger": "An existing trust problem complicates the main action",
                    "action": "The protagonist tests a limited cooperation",
                    "cost": "The cooperation exposes a personal vulnerability",
                    "payoff": "The response changes how the main choice can be made",
                },
                "length_budget": {
                    "target_chars": target_chars,
                    "minimum_chars": minimum_chars,
                    "scene_count": scene_count,
                },
                "emotional_state_before": {"Protagonist": "guarded"},
                "emotional_state_after": {"Protagonist": "more committed"},
                "scene_cards": [
                    {
                        "scene_number": index,
                        "time": f"Chapter {number}, phase {index}",
                        "location": "Opening location",
                        "pov": "Protagonist",
                        "surface_goal": "Advance the chapter objective",
                        "real_purpose": "Force a meaningful choice",
                        "trigger": "The previous result creates an immediate task",
                        "action": "The protagonist takes a concrete step",
                        "obstacle": "Limited time and incomplete trust block the easy route",
                        "turn": "A response changes the available choice",
                        "result": "The scene leaves a concrete consequence for the next step",
                        "target_chars": base_scene_chars
                        + (1 if index <= extra_scene_chars else 0),
                    }
                    for index in range(1, scene_count + 1)
                ],
                "subtext": ["The protagonist does not state the full concern"],
                "lived_in_details": ["A practical task reveals current pressure"],
                "dialogue_intents": {"Protagonist": "Obtain a clear answer"},
                "anti_repetition": ["Avoid repeating the previous chapter structure"],
                "ending_aftertaste": "The choice leaves a visible emotional cost",
                "forbidden_major_additions": [
                    "No new major enemy, power breakthrough, or world revelation"
                ],
                "dialogue_information_limits": [
                    "Characters do not explain facts already known to each other"
                ],
                "emotion_actions": {
                    "pressure": "The protagonist adjusts a practical object twice"
                },
                "rhythm_variation": [
                    "Use one short exchange and one slowed physical action"
                ],
                "incidental_details": [
                    "A physical inconvenience forces a small adjustment"
                ],
                "ending_concrete_residue": "An unanswered practical question remains",
            }
        if task == "relationship_beat":
            number = int(payload["chapter_number"])
            context = payload.get("inputs", {}).get("relationship_context", {})
            pairs = context.get("pairs", []) if isinstance(context, dict) else []
            focus_pairs = []
            if pairs:
                pair = pairs[0]
                focus_pairs.append(
                    {
                        "pair_id": pair["id"],
                        "direction": "双向反应不同步",
                        "stage_before": pair.get("stage", "stranger"),
                        "stage_after_allowed": pair.get("stage", "stranger"),
                        "scene_signals": ["通过一次具体协作表现当前边界"],
                        "callback": "",
                        "boundary": "不跨越当前阶段",
                        "reason": "模拟关系节拍",
                    }
                )
            return {
                "chapter_number": number,
                "intensity": "subtle" if focus_pairs else "hold",
                "focus_pairs": focus_pairs,
                "behavior_signals": ["一次具体协作"] if focus_pairs else [],
                "callbacks": [],
                "forbidden_leaps": ["突然改变称呼或肢体边界"] if focus_pairs else [],
                "progression_summary": "关系保持当前阶段，只出现细微可见变化。"
                if focus_pairs
                else "本章没有需要单独安排的人物关系变化。",
                "stage_change_allowed": False,
            }
        if task == "memory_extract":
            chunks = payload.get("chunks", [])
            final_text = str(payload.get("final_text", ""))
            return {
                "summary": final_text[:240] or "本章完成一次状态变化。",
                "chunk_annotations": [
                    {
                        "chunk_id": item["id"],
                        "importance": 0.6,
                        "characters": ["Protagonist"],
                        "location": "",
                        "timeline": "",
                        "keywords": ["章节事件"],
                    }
                    for item in chunks
                ],
                "facts": [
                    {
                        "fact_type": "event",
                        "subject": "Protagonist",
                        "predicate": "完成",
                        "object": "本章目标",
                        "timeline": "",
                        "importance": 0.7,
                        "confidence": 1.0,
                        "locked": False,
                        "source_chunk_id": chunks[0]["id"] if chunks else "",
                        "source_quote": chunks[0]["text"][:80] if chunks else "",
                        "payload": {},
                    }
                ],
                "style_observations": [],
            }
        if task == "memory_compress":
            return {
                "stage_summary": "阶段中的主要事件形成了连续因果关系。",
                "character_states": [],
                "unresolved_threads": [],
                "world_state_changes": [],
                "timeline_digest": [],
            }
        if task == "memory_rerank":
            return {
                "ordered_ids": [item["id"] for item in payload.get("candidates", [])],
                "reason": "按当前候选顺序保留。",
            }
        if task == "style_audit":
            findings = [str(item) for item in payload.get("rule_findings", [])]
            issues = [
                {
                    "id": f"mock-{index:03d}",
                    "category": "rule_check",
                    "severity": "medium",
                    "paragraph_id": "",
                    "quote": "",
                    "reason": finding,
                    "suggestion": "删除解释性表达，保留具体动作和剧情事实。",
                    "continuity_risk": "low",
                }
                for index, finding in enumerate(findings, start=1)
            ]
            return {
                "status": "review" if issues else "pass",
                "score": max(0, 100 - len(issues) * 10),
                "summary": "模拟文风检测完成。",
                "strengths": ["章节结构保持完整"],
                "rewrite_recommended": bool(issues),
                "issues": issues,
            }
        if task == "relationship_audit":
            return {
                "status": "pass",
                "score": 100,
                "summary": "模拟关系一致性检查完成。",
                "strengths": ["人物关系未越过当前阶段"],
                "rewrite_recommended": False,
                "issues": [],
            }
        if task == "relationship_memory":
            return {"events": []}
        raise ValueError(f"Unsupported mock JSON task: {task}")

    def generate_text(self, system_prompt: str, payload: dict[str, Any]) -> str:
        task = str(payload["task"])
        self.calls.append(task)
        if task == "chapter_draft_humanize":
            return str(payload.get("draft", ""))
        if task == "style_edit":
            text = str(payload.get("source_text", ""))
            replacements = {
                "显然": "",
                "这说明": "",
                "这意味着": "",
                "命运的齿轮": "后续变化",
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
            return text
        number = int(payload["chapter_number"])
        plan = payload["inputs"]["chapter_plan"]
        title = plan["title"]
        budget = plan.get("length_budget", {})
        target_chars = int(
            budget.get("target_chars", plan.get("target_word_count", 300))
        )
        passages = (
            "主角把眼前的任务拆成可以立刻完成的步骤，先核对手边材料，再沿着已经确定的目标行动。阻力没有消失，他只能放慢速度，重新判断哪一项代价能够承担。",
            "对方没有给出完整答案，只把能确认的部分摆到桌面。主角追问一次，得到的回应仍留着缺口，于是改用更小的承诺测试合作边界。",
            "原先的办法在执行中遇到限制，时间和条件都不允许继续照搬。主角收回已经做出的安排，保留真正有用的一步，并承担由此产生的麻烦。",
            "一件普通事务拖慢了进度，人物需要亲手整理、移动和确认。这个过程暴露出双方处理压力的差异，也让支线对主线形成实际影响。",
            "新的回应改变了可选路径。主角没有立刻得到理想结果，却完成了本章必须作出的选择，并看见这项选择留下的具体代价。",
        )
        paragraphs: list[str] = []
        current_chars = 0
        passage_index = number - 1
        while current_chars < target_chars:
            passage = passages[passage_index % len(passages)]
            paragraphs.append(passage)
            current_chars += sum(
                1 for character in passage if not character.isspace()
            )
            passage_index += 1
        return f"# {title}\n\n" + "\n\n".join(paragraphs)
