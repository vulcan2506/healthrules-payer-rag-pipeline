"use client";

import { Check } from "lucide-react";
import { usePreferencesStore, BUBBLE_COLORS } from "@/lib/preferences-store";
import { cn } from "@/lib/utils";

export function BubbleColorSettings() {
  const { bubbleColor, setBubbleColor } = usePreferencesStore();

  return (
    <div className="flex flex-col gap-2">
      <h4 className="text-sm font-medium">Your message color</h4>
      <div className="flex gap-1.5">
        {BUBBLE_COLORS.map(({ value, label, swatch }) => (
          <button
            key={value}
            type="button"
            aria-label={label}
            onClick={() => setBubbleColor(value)}
            className={cn(
              "flex size-7 items-center justify-center rounded-full shadow-sm ring-1 ring-black/10 transition-transform hover:scale-105 dark:ring-white/10",
              swatch
            )}
          >
            {bubbleColor === value && <Check className="size-3.5 text-white" />}
          </button>
        ))}
      </div>
    </div>
  );
}
