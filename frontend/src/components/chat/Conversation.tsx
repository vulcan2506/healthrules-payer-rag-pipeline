"use client";

import { useEffect, useRef } from "react";
import { Sparkles, User, AlertCircle, RotateCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ResponseCard } from "@/components/chat/ResponseCard";
import { PipelineStepper } from "@/components/chat/PipelineStepper";
import { useChatStore } from "@/lib/chat-store";
import { usePreferencesStore, bubbleColorClassName } from "@/lib/preferences-store";
import { cn } from "@/lib/utils";

interface ConversationProps {
  onRetry: (id: string, query: string) => void;
}

export function Conversation({ onRetry }: ConversationProps) {
  const messages = useChatStore((s) => s.messages);
  const bubbleColor = usePreferencesStore((s) => s.bubbleColor);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, messages[messages.length - 1]?.pending]);

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 p-8 text-center">
        <Sparkles className="size-8 text-muted-foreground/50" />
        <div>
          <p className="font-medium">Ask about the HealthRules Payer documentation</p>
          <p className="max-w-sm text-sm text-muted-foreground">
            Every answer shows its confidence, retrieval strategy, and cited sources — not just a response.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col gap-4 overflow-y-auto px-4 pt-2 pb-4 sm:px-6">
      {messages.map((m) =>
        m.role === "user" ? (
          <div key={m.id} className="flex animate-in items-start gap-2 fade-in slide-in-from-bottom-1 duration-300">
            <User className="mt-1 size-5 shrink-0 text-muted-foreground" />
            <div
              className={cn(
                "max-w-[85%] rounded-2xl rounded-tl-sm px-4 py-2 text-sm shadow-sm",
                bubbleColorClassName(bubbleColor)
              )}
            >
              {m.query}
            </div>
          </div>
        ) : (
          <div key={m.id} className="flex animate-in items-start justify-end gap-2 fade-in slide-in-from-bottom-1 duration-300">
            <div className="min-w-0 max-w-[85%]">
              {m.pending && <PipelineStepper />}
              {m.error && (
                <div className="flex items-center gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive shadow-sm">
                  <AlertCircle className="size-4 shrink-0" />
                  <span className="flex-1">{m.error}</span>
                  {m.query && (
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-7 gap-1 border-destructive/30 text-destructive hover:bg-destructive/10"
                      onClick={() => onRetry(m.id, m.query as string)}
                    >
                      <RotateCw className="size-3" /> Retry
                    </Button>
                  )}
                </div>
              )}
              {m.response && <ResponseCard response={m.response} />}
            </div>
            <Sparkles className="mt-1 size-5 shrink-0 text-muted-foreground" />
          </div>
        )
      )}
      <div ref={bottomRef} />
    </div>
  );
}
