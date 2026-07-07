import { apiFetch } from "@/lib/api/client";
import type { SettingsStatus } from "@/lib/types";

export function getSettingsStatus(): Promise<SettingsStatus> {
  return apiFetch("/api/settings/status");
}

export function setApiKey(apiKey: string): Promise<{ valid: boolean; error?: string }> {
  return apiFetch("/api/settings/key", {
    method: "POST",
    body: JSON.stringify({ api_key: apiKey }),
  });
}
