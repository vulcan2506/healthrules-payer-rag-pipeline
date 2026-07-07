import { API_BASE, ApiError } from "@/lib/api/client";

export async function uploadPdf(file: File): Promise<{ filename: string; size_bytes: number }> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body.detail ?? `Upload failed (${res.status})`);
  }
  return res.json();
}
