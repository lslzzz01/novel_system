export async function api(path, options = {}) {
  const request = { ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}) } };
  if (request.body && typeof request.body !== "string") request.body = JSON.stringify(request.body);
  const response = await fetch(path, request);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `请求失败：${response.status}`);
  return payload;
}

export function projectPath(projectId) {
  return encodeURIComponent(projectId || "");
}

export function lines(value) {
  return String(value || "").split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
}

export function joinLines(value) {
  return Array.isArray(value) ? value.join("\n") : String(value || "");
}

export function asNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}
