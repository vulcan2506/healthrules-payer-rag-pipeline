import { create } from "zustand";
import { persist } from "zustand/middleware";

export const BUBBLE_COLORS = [
  { value: "default", label: "Default", swatch: "bg-primary", className: "bg-primary text-primary-foreground" },
  { value: "blue", label: "Blue", swatch: "bg-blue-600", className: "bg-blue-600 text-white dark:bg-blue-500" },
  { value: "emerald", label: "Emerald", swatch: "bg-emerald-600", className: "bg-emerald-600 text-white dark:bg-emerald-500" },
  { value: "violet", label: "Violet", swatch: "bg-violet-600", className: "bg-violet-600 text-white dark:bg-violet-500" },
  { value: "rose", label: "Rose", swatch: "bg-rose-600", className: "bg-rose-600 text-white dark:bg-rose-500" },
] as const;

export type BubbleColor = (typeof BUBBLE_COLORS)[number]["value"];

export function bubbleColorClassName(color: BubbleColor): string {
  return BUBBLE_COLORS.find((c) => c.value === color)?.className ?? BUBBLE_COLORS[0].className;
}

interface PreferencesState {
  bubbleColor: BubbleColor;
  setBubbleColor: (color: BubbleColor) => void;
}

// Persisted client-side only — purely cosmetic, no backend concept of this.
export const usePreferencesStore = create<PreferencesState>()(
  persist(
    (set) => ({
      bubbleColor: "default",
      setBubbleColor: (color) => set({ bubbleColor: color }),
    }),
    { name: "chat-preferences" }
  )
);
