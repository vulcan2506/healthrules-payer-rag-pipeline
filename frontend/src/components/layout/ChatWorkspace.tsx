"use client";

import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import { Conversation } from "@/components/chat/Conversation";
import { MessageInput } from "@/components/chat/MessageInput";
import { sendChatMessage } from "@/lib/api/chat";
import { useChatStore, buildChatParams } from "@/lib/chat-store";
import { ApiError } from "@/lib/api/client";

export function ChatWorkspace() {
  const {
    addUserMessage,
    addPendingAssistant,
    resolveAssistant,
    failAssistant,
    retryAssistant,
    mode,
    intent,
    bestOf,
    sessionId,
  } = useChatStore();

  const mutation = useMutation({
    mutationFn: sendChatMessage,
  });

  const runQuery = (assistantId: string, query: string) => {
    mutation.mutate(
      { ...buildChatParams(mode, intent, bestOf), query, session_id: sessionId },
      {
        onSuccess: (response) => resolveAssistant(assistantId, response),
        onError: (err) => {
          const message =
            err instanceof ApiError
              ? err.message
              : "Couldn't reach the backend — is api_server.py running on port 8000?";
          failAssistant(assistantId, message);
          toast.error("Chat request failed", { description: message });
        },
      }
    );
  };

  const handleSend = (query: string) => {
    addUserMessage(query);
    const assistantId = addPendingAssistant(query);
    runQuery(assistantId, query);
  };

  const handleRetry = (id: string, query: string) => {
    retryAssistant(id);
    runQuery(id, query);
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <Conversation onRetry={handleRetry} />
      <MessageInput onSend={handleSend} disabled={mutation.isPending} />
    </div>
  );
}
