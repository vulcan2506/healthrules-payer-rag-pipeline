"use client";

import { History, Trash2, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useUIStore } from "@/lib/store";
import { useChatStore } from "@/lib/chat-store";
import { useHistoryStore } from "@/lib/history-store";

function relativeTime(ts: number): string {
  const diffMs = Date.now() - ts;
  const minutes = Math.round(diffMs / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return `${days}d ago`;
}

export function HistoryDialog() {
  const { historyOpen, setHistoryOpen } = useUIStore();
  const { threads, deleteThread, saveThread } = useHistoryStore();
  const { loadMessages, messages } = useChatStore();

  const handleRestore = (threadMessages: typeof messages) => {
    if (messages.length > 0) saveThread(messages);
    loadMessages(threadMessages);
    setHistoryOpen(false);
  };

  return (
    <Dialog open={historyOpen} onOpenChange={setHistoryOpen}>
      <DialogContent className="flex h-[70vh] flex-col overflow-hidden sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-1.5">
            <History className="size-4" /> History
          </DialogTitle>
        </DialogHeader>

        {threads.length === 0 ? (
          <div className="flex flex-1 items-center justify-center px-4 text-center text-sm text-muted-foreground">
            No past conversations yet — they&apos;re saved here automatically when you start a
            New Chat.
          </div>
        ) : (
          <ScrollArea className="min-h-0 flex-1">
            <div className="flex flex-col gap-1 pr-3">
              {threads.map((thread) => (
                <div
                  key={thread.id}
                  className="group flex items-start gap-2 rounded-md p-2 hover:bg-muted"
                >
                  <button
                    type="button"
                    onClick={() => handleRestore(thread.messages)}
                    className="flex min-w-0 flex-1 items-start gap-2 text-left"
                  >
                    <MessageSquare className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm">{thread.title}</span>
                      <span className="text-xs text-muted-foreground">{relativeTime(thread.savedAt)}</span>
                    </span>
                  </button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="size-6 shrink-0 opacity-0 group-hover:opacity-100"
                    onClick={() => deleteThread(thread.id)}
                    aria-label="Delete conversation"
                  >
                    <Trash2 className="size-3.5" />
                  </Button>
                </div>
              ))}
            </div>
          </ScrollArea>
        )}
      </DialogContent>
    </Dialog>
  );
}

// Exposed for Sidebar's "New Chat" action — archives the current thread (if
// any) into history, then clears the active conversation.
export function useNewChat() {
  const { messages, clear } = useChatStore();
  const { saveThread } = useHistoryStore();
  return () => {
    if (messages.length > 0) saveThread(messages);
    clear();
  };
}
