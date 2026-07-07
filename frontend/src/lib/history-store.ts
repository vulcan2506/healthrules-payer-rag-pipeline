import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { ChatMessage } from "@/lib/chat-store";

export interface ChatThread {
  id: string;
  title: string;
  messages: ChatMessage[];
  savedAt: number;
}

interface HistoryState {
  threads: ChatThread[];
  saveThread: (messages: ChatMessage[]) => void;
  deleteThread: (id: string) => void;
}

function deriveTitle(messages: ChatMessage[]): string {
  const firstUser = messages.find((m) => m.role === "user");
  const text = firstUser?.query?.trim() || "Untitled conversation";
  return text.length > 60 ? `${text.slice(0, 60)}…` : text;
}

// Client-side only for now — the backend has no chat-history storage today.
// Persisted to localStorage so it survives a page refresh.
export const useHistoryStore = create<HistoryState>()(
  persist(
    (set) => ({
      threads: [],

      saveThread: (messages) => {
        if (messages.length === 0) return;
        const thread: ChatThread = {
          id: `thread_${Date.now()}`,
          title: deriveTitle(messages),
          messages,
          savedAt: Date.now(),
        };
        set((s) => ({ threads: [thread, ...s.threads].slice(0, 50) }));
      },

      deleteThread: (id) => set((s) => ({ threads: s.threads.filter((t) => t.id !== id) })),
    }),
    { name: "chat-history" }
  )
);
