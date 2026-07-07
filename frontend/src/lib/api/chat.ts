import { apiFetch } from "@/lib/api/client";
import type { ChatRequestParams, ChatResponse } from "@/lib/types";

// Wraps POST /api/chat (retrieval_layer/api_server.py), which itself wraps
// redis_cache.answer_query() / retriever.retrieve() — no retrieval, gating,
// or citation logic lives on the frontend.
export function sendChatMessage(params: ChatRequestParams): Promise<ChatResponse> {
  return apiFetch<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify(params),
  });
}
