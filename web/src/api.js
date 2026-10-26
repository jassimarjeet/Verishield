const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function readBody(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

function detailMessage(payload, fallback) {
  const detail = payload?.detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail) && detail[0]?.msg) return String(detail[0].msg);
  return fallback;
}

export async function fetchHealth() {
  const response = await fetch(`${API_BASE}/health`);
  const payload = await readBody(response);
  if (!response.ok) {
    throw new Error(detailMessage(payload, "The API is not responding."));
  }
  return payload;
}

export async function fetchThresholds() {
  const response = await fetch(`${API_BASE}/api/thresholds`);
  const payload = await readBody(response);
  if (!response.ok) {
    throw new Error(detailMessage(payload, "The validation cuts could not be loaded."));
  }
  return payload;
}

export async function detectFile(file) {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch(`${API_BASE}/api/detect`, { method: "POST", body });
  const payload = await readBody(response);
  if (!response.ok) {
    throw new Error(detailMessage(payload, "Detection did not return a score."));
  }
  if (typeof payload.score !== "number" || typeof payload.decision !== "string") {
    throw new Error("The API response did not include a score.");
  }
  return payload;
}
