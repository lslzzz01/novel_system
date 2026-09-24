from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .llm import OpenAICompatibleClient


REGISTRY_VERSION = 2
DEFAULT_SUPPORTED_PARAMETERS = [
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "max_tokens",
]
CLIENT_FIELDS = {
    "base_url",
    "model",
    "temperature",
    "timeout_seconds",
    "api_key",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "max_tokens",
    "seed",
    "reasoning_effort",
    "use_json_response_format",
    "supported_parameters",
    "extra_parameters",
}
CONNECTION_FIELDS = CLIENT_FIELDS | {"enabled"}


@dataclass(slots=True)
class ModelSettings:
    id: str = "default"
    display_name: str = "默认模型"
    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    temperature: float = 0.4
    timeout_seconds: int = 180
    api_key: str = ""
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    reasoning_effort: str | None = None
    use_json_response_format: bool = True
    supported_parameters: list[str] = field(
        default_factory=lambda: list(DEFAULT_SUPPORTED_PARAMETERS)
    )
    extra_parameters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_tested_at: str = ""
    last_test_ok: bool | None = None
    last_test_message: str = ""


class ModelSettingsStore:
    """Versioned local model registry with a legacy single-model facade."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.local_dir = self.workspace / ".novel_agents"
        self.path = self.local_dir / "model.json"
        self._lock = threading.RLock()

    def load(self) -> ModelSettings:
        """Return the active model for callers that use the old single-model API."""
        with self._lock:
            registry = self._load_registry()
            return self._active_profile(registry)

    def public(self) -> dict[str, Any]:
        """Return the active model using the original `/api/config` response shape."""
        with self._lock:
            registry = self._load_registry()
            return self._public_profile(
                self._active_profile(registry), str(registry["active_model_id"])
            )

    def registry_public(self) -> dict[str, Any]:
        with self._lock:
            registry = self._load_registry()
            active_id = str(registry["active_model_id"])
            return {
                "version": REGISTRY_VERSION,
                "active_model_id": active_id,
                "models": [
                    self._public_profile(profile, active_id)
                    for profile in registry["models"]
                ],
            }

    def update(self, values: dict[str, Any]) -> dict[str, Any]:
        """Update the active profile for backwards compatibility."""
        with self._lock:
            active_id = str(self._load_registry()["active_model_id"])
            self.update_model(active_id, values)
            return self.public()

    def create_model(self, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            registry = self._load_registry()
            model_id = self._new_id(registry)
            profile = self._profile_from_values(
                None,
                {**values, "id": model_id},
            )
            self._assert_unique_display_name(registry, profile.display_name)

            models: list[ModelSettings] = registry["models"]
            virtual_default = (
                not self.path.exists()
                and len(models) == 1
                and models[0].id == "default"
                and not models[0].model
                and not models[0].api_key
            )
            if virtual_default:
                registry["models"] = [profile]
                registry["active_model_id"] = profile.id
            else:
                models.append(profile)
            self._write_registry(registry)
            return self._public_profile(profile, str(registry["active_model_id"]))

    def update_model(self, model_id: str, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            registry = self._load_registry()
            current = self._profile_by_id(registry, model_id)
            updated = self._profile_from_values(current, values)
            self._assert_unique_display_name(
                registry, updated.display_name, exclude_id=current.id
            )
            if current.id == registry["active_model_id"] and not updated.enabled:
                raise ValueError("默认模型不能停用，请先选择其他默认模型")

            changed_connection = any(
                field_name in values
                for field_name in CONNECTION_FIELDS
                if field_name != "enabled"
            ) or values.get("clear_api_key") is True
            if changed_connection:
                updated.last_tested_at = ""
                updated.last_test_ok = None
                updated.last_test_message = ""

            registry["models"] = [
                updated if profile.id == current.id else profile
                for profile in registry["models"]
            ]
            self._write_registry(registry)
            return self._public_profile(updated, str(registry["active_model_id"]))

    def delete_model(self, model_id: str) -> dict[str, Any]:
        with self._lock:
            registry = self._load_registry()
            profile = self._profile_by_id(registry, model_id)
            if profile.id == registry["active_model_id"]:
                raise ValueError("默认模型不能删除，请先选择其他默认模型")
            registry["models"] = [
                item for item in registry["models"] if item.id != profile.id
            ]
            self._write_registry(registry)
            return {"deleted": True, "id": profile.id, "display_name": profile.display_name}

    def set_active(self, model_id: str) -> dict[str, Any]:
        with self._lock:
            registry = self._load_registry()
            profile = self._profile_by_id(registry, model_id)
            if not profile.enabled:
                raise ValueError("已停用的模型不能设为默认模型")
            if not profile.model:
                raise ValueError("模型名称不能为空")
            if not profile.api_key:
                raise ValueError("该模型尚未配置 API 密钥")
            registry["active_model_id"] = profile.id
            self._write_registry(registry)
            return self._public_profile(profile, profile.id)

    def record_test(
        self, model_id: str, ok: bool, message: str = ""
    ) -> dict[str, Any]:
        with self._lock:
            registry = self._load_registry()
            profile = self._profile_by_id(registry, model_id)
            profile.last_tested_at = datetime.now(timezone.utc).isoformat()
            profile.last_test_ok = bool(ok)
            profile.last_test_message = str(message).strip()[:300]
            self._write_registry(registry)
            return self._public_profile(profile, str(registry["active_model_id"]))

    def model_name_map(self) -> dict[str, list[str]]:
        with self._lock:
            result: dict[str, list[str]] = {}
            for profile in self._load_registry()["models"]:
                if profile.model:
                    result.setdefault(profile.model, []).append(profile.id)
            return result

    def resolved_public(
        self,
        model_profile_id: str | None = None,
        legacy_model: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            registry = self._load_registry()
            active = self._active_profile(registry)
            requested = str(model_profile_id or "").strip()
            profile = active
            fallback = False
            if requested:
                try:
                    candidate = self._profile_by_id(registry, requested)
                    if candidate.enabled:
                        profile = candidate
                    else:
                        fallback = True
                except ValueError:
                    fallback = True
            public = self._public_profile(profile, str(registry["active_model_id"]))
            public["requested_model_id"] = requested or None
            public["fallback"] = fallback
            if legacy_model and not requested:
                public["model"] = str(legacy_model)
                public["legacy_override"] = True
            return public

    def create_client(
        self,
        overrides: dict[str, Any] | None = None,
        model_profile_id: str | None = None,
    ) -> OpenAICompatibleClient:
        with self._lock:
            override_values = dict(overrides or {})
            requested_id = str(
                override_values.pop("model_profile_id", None)
                or model_profile_id
                or ""
            ).strip()
            registry = self._load_registry()
            settings = (
                self._profile_by_id(registry, requested_id)
                if requested_id
                else self._active_profile(registry)
            )
            if not settings.enabled:
                raise ValueError(f"模型配置“{settings.display_name}”已停用")

            values = {
                key: value
                for key, value in asdict(settings).items()
                if key in CLIENT_FIELDS
            }
            for key, value in override_values.items():
                if key in values and value is not None:
                    values[key] = value
            if not values["api_key"]:
                raise ValueError("尚未配置 API 密钥")
            if not values["model"]:
                raise ValueError("尚未配置模型名称")
            return OpenAICompatibleClient(
                api_key=str(values["api_key"]),
                model=str(values["model"]),
                base_url=str(values["base_url"]),
                timeout_seconds=int(values["timeout_seconds"]),
                temperature=values["temperature"],
                top_p=values["top_p"],
                frequency_penalty=values["frequency_penalty"],
                presence_penalty=values["presence_penalty"],
                max_tokens=values["max_tokens"],
                seed=values["seed"],
                reasoning_effort=values["reasoning_effort"],
                use_json_response_format=bool(values["use_json_response_format"]),
                supported_parameters=tuple(values["supported_parameters"]),
                extra_parameters=dict(values["extra_parameters"]),
            )

    def _load_registry(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._environment_registry()
        with self.path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        data = loaded if isinstance(loaded, dict) else {}
        if data.get("version") == REGISTRY_VERSION and isinstance(
            data.get("models"), list
        ):
            profiles = [
                self._profile_from_data(item)
                for item in data["models"]
                if isinstance(item, dict)
            ]
            if not profiles:
                return self._environment_registry()
            active_id = str(data.get("active_model_id", ""))
            if not any(profile.id == active_id for profile in profiles):
                active_id = profiles[0].id
            return {
                "version": REGISTRY_VERSION,
                "active_model_id": active_id,
                "models": profiles,
            }

        profile = self._legacy_profile(data)
        registry = {
            "version": REGISTRY_VERSION,
            "active_model_id": profile.id,
            "models": [profile],
        }
        self._write_registry(registry)
        return registry

    def _environment_registry(self) -> dict[str, Any]:
        model = str(os.getenv("NOVEL_MODEL") or "")
        profile = ModelSettings(
            id="default",
            display_name=model or "默认模型",
            base_url=str(
                os.getenv("NOVEL_BASE_URL") or "https://api.openai.com/v1"
            ).rstrip("/"),
            model=model,
            temperature=float(os.getenv("NOVEL_TEMPERATURE", "0.4")),
            api_key=str(
                os.getenv("NOVEL_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
            ),
        )
        return {
            "version": REGISTRY_VERSION,
            "active_model_id": profile.id,
            "models": [profile],
        }

    def _legacy_profile(self, data: dict[str, Any]) -> ModelSettings:
        model = str(data.get("model") or os.getenv("NOVEL_MODEL") or "")
        legacy = dict(data)
        legacy.update(
            {
                "id": "default",
                "display_name": str(data.get("display_name") or model or "默认模型"),
                "base_url": str(
                    data.get("base_url")
                    or os.getenv("NOVEL_BASE_URL")
                    or "https://api.openai.com/v1"
                ),
                "model": model,
                "temperature": data.get(
                    "temperature", os.getenv("NOVEL_TEMPERATURE", "0.4")
                ),
                "api_key": str(
                    data.get("api_key")
                    or os.getenv("NOVEL_API_KEY")
                    or os.getenv("OPENAI_API_KEY")
                    or ""
                ),
            }
        )
        return self._profile_from_data(legacy)

    def _profile_from_data(self, data: dict[str, Any]) -> ModelSettings:
        return ModelSettings(
            id=str(data.get("id") or "default"),
            display_name=str(
                data.get("display_name") or data.get("model") or "默认模型"
            ),
            base_url=_normalize_base_url(
                str(data.get("base_url") or "https://api.openai.com/v1")
            ),
            model=str(data.get("model") or ""),
            temperature=float(data.get("temperature", 0.4)),
            timeout_seconds=int(data.get("timeout_seconds", 180)),
            api_key=str(data.get("api_key") or ""),
            top_p=_optional_float(data.get("top_p")),
            frequency_penalty=_optional_float(data.get("frequency_penalty")),
            presence_penalty=_optional_float(data.get("presence_penalty")),
            max_tokens=_optional_int(data.get("max_tokens")),
            seed=_optional_int(data.get("seed")),
            reasoning_effort=_optional_string(data.get("reasoning_effort")),
            use_json_response_format=_bool_value(
                data.get("use_json_response_format", True)
            ),
            supported_parameters=[
                str(item)
                for item in data.get(
                    "supported_parameters", DEFAULT_SUPPORTED_PARAMETERS
                )
            ],
            extra_parameters=(
                dict(data.get("extra_parameters", {}))
                if isinstance(data.get("extra_parameters", {}), dict)
                else {}
            ),
            enabled=_bool_value(data.get("enabled", True)),
            last_tested_at=str(data.get("last_tested_at") or ""),
            last_test_ok=(
                bool(data["last_test_ok"])
                if isinstance(data.get("last_test_ok"), bool)
                else None
            ),
            last_test_message=str(data.get("last_test_message") or "")[:300],
        )

    def _profile_from_values(
        self,
        current: ModelSettings | None,
        values: dict[str, Any],
    ) -> ModelSettings:
        base = asdict(current) if current else asdict(ModelSettings())
        for key, value in values.items():
            if key in base and value is not None:
                base[key] = value

        supplied_key = str(values.get("api_key", "")).strip()
        if current is not None and not supplied_key:
            base["api_key"] = current.api_key
        else:
            base["api_key"] = supplied_key
        if values.get("clear_api_key") is True:
            base["api_key"] = ""

        base["id"] = current.id if current else str(values.get("id") or "")
        base["display_name"] = str(base.get("display_name") or base.get("model") or "").strip()
        base["base_url"] = _normalize_base_url(str(base.get("base_url") or ""))
        base["model"] = str(base.get("model") or "").strip()
        base["temperature"] = float(base.get("temperature", 0.4))
        base["timeout_seconds"] = int(base.get("timeout_seconds", 180))
        base["top_p"] = _optional_float(base.get("top_p"))
        base["frequency_penalty"] = _optional_float(base.get("frequency_penalty"))
        base["presence_penalty"] = _optional_float(base.get("presence_penalty"))
        base["max_tokens"] = _optional_int(base.get("max_tokens"))
        base["seed"] = _optional_int(base.get("seed"))
        # Treat 0 as unset for optional sampling controls. Some providers reject
        # top_p=0 / seed=0 / max_tokens=0 with HTTP 400.
        if base["top_p"] is not None and float(base["top_p"]) <= 0:
            base["top_p"] = None
        if base["frequency_penalty"] is not None and float(base["frequency_penalty"]) == 0:
            base["frequency_penalty"] = None
        if base["presence_penalty"] is not None and float(base["presence_penalty"]) == 0:
            base["presence_penalty"] = None
        if base["max_tokens"] is not None and int(base["max_tokens"]) <= 0:
            base["max_tokens"] = None
        if base["seed"] is not None and int(base["seed"]) == 0:
            base["seed"] = None
        base["reasoning_effort"] = _optional_string(base.get("reasoning_effort"))
        base["use_json_response_format"] = _bool_value(
            base.get("use_json_response_format", True)
        )
        base["enabled"] = _bool_value(base.get("enabled", True))
        base["supported_parameters"] = [
            str(item) for item in base.get("supported_parameters", [])
        ]
        base["extra_parameters"] = (
            dict(base.get("extra_parameters", {}))
            if isinstance(base.get("extra_parameters", {}), dict)
            else {}
        )

        profile = ModelSettings(**base)
        self._validate(profile)
        return profile

    @staticmethod
    def _validate(settings: ModelSettings) -> None:
        if not settings.id:
            raise ValueError("模型配置 ID 不能为空")
        if not settings.display_name:
            raise ValueError("显示名称不能为空")
        if len(settings.display_name) > 80:
            raise ValueError("显示名称不能超过 80 个字符")
        parsed = urlparse(settings.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("API 地址必须是有效的 http 或 https 地址")
        if not settings.model:
            raise ValueError("模型名称不能为空")
        if not 0 <= settings.temperature <= 2:
            raise ValueError("温度必须在 0 到 2 之间")
        if not 10 <= settings.timeout_seconds <= 900:
            raise ValueError("超时时间必须在 10 到 900 秒之间")
        if settings.top_p is not None and not 0 <= settings.top_p <= 1:
            raise ValueError("top_p 必须在 0 到 1 之间")
        for name, penalty in (
            ("frequency_penalty", settings.frequency_penalty),
            ("presence_penalty", settings.presence_penalty),
        ):
            if penalty is not None and not -2 <= penalty <= 2:
                raise ValueError(f"{name} 必须在 -2 到 2 之间")
        if settings.max_tokens is not None and settings.max_tokens < 1:
            raise ValueError("最大输出 token 必须大于 0")

    @staticmethod
    def _active_profile(registry: dict[str, Any]) -> ModelSettings:
        active_id = str(registry["active_model_id"])
        for profile in registry["models"]:
            if profile.id == active_id:
                return profile
        return registry["models"][0]

    @staticmethod
    def _profile_by_id(registry: dict[str, Any], model_id: str) -> ModelSettings:
        for profile in registry["models"]:
            if profile.id == model_id:
                return profile
        raise ValueError("模型配置不存在")

    @staticmethod
    def _public_profile(profile: ModelSettings, active_id: str) -> dict[str, Any]:
        hint = ""
        if profile.api_key:
            suffix = profile.api_key[-4:] if len(profile.api_key) >= 4 else "****"
            hint = f"****{suffix}"
        return {
            "id": profile.id,
            "display_name": profile.display_name,
            "base_url": profile.base_url,
            "model": profile.model,
            "temperature": profile.temperature,
            "timeout_seconds": profile.timeout_seconds,
            "has_api_key": bool(profile.api_key),
            "api_key_hint": hint,
            "top_p": profile.top_p,
            "frequency_penalty": profile.frequency_penalty,
            "presence_penalty": profile.presence_penalty,
            "max_tokens": profile.max_tokens,
            "seed": profile.seed,
            "reasoning_effort": profile.reasoning_effort,
            "use_json_response_format": profile.use_json_response_format,
            "supported_parameters": list(profile.supported_parameters),
            "extra_parameters": dict(profile.extra_parameters),
            "enabled": profile.enabled,
            "last_tested_at": profile.last_tested_at,
            "last_test_ok": profile.last_test_ok,
            "last_test_message": profile.last_test_message,
            "is_default": profile.id == active_id,
        }

    @staticmethod
    def _assert_unique_display_name(
        registry: dict[str, Any], display_name: str, exclude_id: str = ""
    ) -> None:
        normalized = display_name.casefold()
        if any(
            profile.id != exclude_id and profile.display_name.casefold() == normalized
            for profile in registry["models"]
        ):
            raise ValueError("模型显示名称已存在")

    @staticmethod
    def _new_id(registry: dict[str, Any]) -> str:
        existing = {profile.id for profile in registry["models"]}
        while True:
            candidate = f"model-{uuid.uuid4().hex[:10]}"
            if candidate not in existing:
                return candidate

    def _write_registry(self, registry: dict[str, Any]) -> None:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "version": REGISTRY_VERSION,
            "active_model_id": str(registry["active_model_id"]),
            "models": [asdict(profile) for profile in registry["models"]],
        }
        fd, temp_name = tempfile.mkstemp(
            prefix=".model.", suffix=".tmp", dir=self.local_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_base_url(value: str) -> str:
    base = str(value or "").strip().rstrip("/")
    if not base:
        return "https://api.openai.com/v1"
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")].rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    return base


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)
