import { create } from "zustand";
import type { ChatResponse, ChatMode, ChatIntent, BestOf, ChatRequestParams } from "@/lib/types";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  query?: string;
  response?: ChatResponse;
  pending?: boolean;
  error?: string;
}

interface ChatStoreState {
  messages: ChatMessage[];
  sessionId: string;
  addUserMessage: (query: string) => string;
  addPendingAssistant: (query: string) => string;
  resolveAssistant: (id: string, response: ChatResponse) => void;
  failAssistant: (id: string, error: string) => void;
  retryAssistant: (id: string) => void;
  clear: () => void;
  loadMessages: (messages: ChatMessage[]) => void;

  mode: ChatMode;
  setMode: (mode: ChatMode) => void;

  intent: ChatIntent;
  setIntent: (intent: ChatIntent) => void;

  bestOf: BestOf;
  setBestOf: (bestOf: BestOf) => void;
}

let counter = 0;
function nextId() {
  counter += 1;
  return `msg_${Date.now()}_${counter}`;
}

// Ties every /api/chat call in a conversation to the same backend
// ConversationSession (see api_server.py) so follow-ups like "explain that in
// more detail" get resolved against real history instead of retrieved on
// their own literal (near-contentless) text. A new id per New Chat/loaded
// thread starts a fresh session — the backend has no memory of it anyway.
function nextSessionId() {
  return `session_${Date.now()}_${Math.random().toString(36).slice(2)}`;
}

export const useChatStore = create<ChatStoreState>((set) => ({
  messages: [],
  sessionId: nextSessionId(),

  addUserMessage: (query) => {
    const id = nextId();
    set((s) => ({ messages: [...s.messages, { id, role: "user", query }] }));
    return id;
  },

  addPendingAssistant: (query) => {
    const id = nextId();
    set((s) => ({ messages: [...s.messages, { id, role: "assistant", query, pending: true }] }));
    return id;
  },

  resolveAssistant: (id, response) =>
    set((s) => ({
      messages: s.messages.map((m) => (m.id === id ? { ...m, pending: false, response } : m)),
    })),

  failAssistant: (id, error) =>
    set((s) => ({
      messages: s.messages.map((m) => (m.id === id ? { ...m, pending: false, error } : m)),
    })),

  retryAssistant: (id) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, pending: true, error: undefined, response: undefined } : m
      ),
    })),

  clear: () => set({ messages: [], sessionId: nextSessionId() }),
  loadMessages: (messages) => set({ messages, sessionId: nextSessionId() }),

  mode: "concise",
  setMode: (mode) => set({ mode }),

  intent: "auto",
  setIntent: (intent) => set({ intent }),

  bestOf: 3,
  setBestOf: (bestOf) => set({ bestOf }),
}));

// Mirrors cli.py exactly: forcing a real intent only has effect on the raw
// path (see api_server.py's /api/chat) — so choosing anything but "auto"
// here switches the request into diagnostic mode automatically, same as
// cli.py's --intent implicitly needing --raw to do anything.
export function buildChatParams(mode: ChatMode, intent: ChatIntent, bestOf: BestOf): Omit<ChatRequestParams, "query"> {
  if (intent !== "auto") {
    return { best_of: bestOf, intent, raw: true };
  }
  return { best_of: bestOf, mode };
}
