"use client";

import { useState, useRef, KeyboardEvent } from "react";
import { ArrowUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ChatControls } from "@/components/chat/ChatControls";

interface MessageInputProps {
  onSend: (query: string) => void;
  disabled?: boolean;
}

export function MessageInput({ onSend, disabled }: MessageInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
    textareaRef.current?.focus();
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="flex flex-col gap-2 border-t bg-background p-3">
      <ChatControls />
      <div className="flex items-end gap-2">
        <Textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about the HealthRules Payer documentation…"
          rows={2}
          className="min-h-[52px] resize-none"
          disabled={disabled}
        />
        <Button
          size="icon"
          className="size-9 shrink-0 rounded-full"
          onClick={submit}
          disabled={disabled || !value.trim()}
          aria-label="Send message"
        >
          <ArrowUp className="size-4" />
        </Button>
      </div>
    </div>
  );
}
