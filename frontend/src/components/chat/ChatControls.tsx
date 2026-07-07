"use client";

import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useChatStore } from "@/lib/chat-store";
import type { ChatIntent, ChatMode, BestOf } from "@/lib/types";

const MODE_OPTIONS: { value: ChatMode; label: string; description: string }[] = [
  { value: "concise", label: "Concise", description: "Best-of-3 focused pass, escalates only if confidence is low." },
  { value: "detailed", label: "Detailed", description: "Always pools 3 retrieval strategies together — no confidence check." },
];

// "auto" plus the 4 real retriever.classify() intents. Forcing any of the
// 4 switches this turn into the raw/diagnostic path — no generated answer,
// routing + retrieved chunks only. See chat-store.ts's buildChatParams().
const INTENT_OPTIONS: { value: ChatIntent; label: string; description: string }[] = [
  { value: "auto", label: "Auto", description: "Let the backend classify normally — generates a real answer." },
  { value: "specific", label: "Specific", description: "Force a targeted topic lookup. Diagnostic view — no generated answer." },
  { value: "factual", label: "Factual", description: "Force a raw corpus search. Diagnostic view — no generated answer." },
  { value: "intelligence", label: "Summarize", description: "Force an overview/summary search. Diagnostic view — no generated answer." },
  { value: "delta", label: "Delta", description: "Force a version-comparison search. Diagnostic view — no generated answer." },
];

const BEST_OF_OPTIONS: BestOf[] = [1, 3, 5, 7];

export function ChatControls() {
  const { mode, setMode, intent, setIntent, bestOf, setBestOf } = useChatStore();

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs">
      <div className="flex items-center gap-1.5">
        <span className="text-muted-foreground">Mode</span>
        <ToggleGroup
          size="sm"
          variant="outline"
          value={[mode]}
          onValueChange={(v) => v[0] && setMode(v[0] as ChatMode)}
        >
          {MODE_OPTIONS.map((opt) => (
            <Tooltip key={opt.value}>
              <TooltipTrigger render={<ToggleGroupItem value={opt.value}>{opt.label}</ToggleGroupItem>} />
              <TooltipContent side="top" className="max-w-56 text-balance">
                {opt.description}
              </TooltipContent>
            </Tooltip>
          ))}
        </ToggleGroup>
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-muted-foreground">Intent</span>
        <ToggleGroup
          size="sm"
          variant="outline"
          className="flex-wrap"
          value={[intent]}
          onValueChange={(v) => v[0] && setIntent(v[0] as ChatIntent)}
        >
          {INTENT_OPTIONS.map((opt) => (
            <Tooltip key={opt.value}>
              <TooltipTrigger render={<ToggleGroupItem value={opt.value}>{opt.label}</ToggleGroupItem>} />
              <TooltipContent side="top" className="max-w-56 text-balance">
                {opt.description}
              </TooltipContent>
            </Tooltip>
          ))}
        </ToggleGroup>
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <Tooltip>
          <TooltipTrigger render={<span className="text-muted-foreground">Best-of</span>} />
          <TooltipContent side="top" className="max-w-56 text-balance">
            Number of query reformulations tried in parallel — the highest-scoring one wins. Ignored
            in Detailed mode (it never reformulates).
          </TooltipContent>
        </Tooltip>
        <ToggleGroup
          size="sm"
          variant="outline"
          className="flex-wrap"
          value={[String(bestOf)]}
          onValueChange={(v) => v[0] && setBestOf(Number(v[0]) as BestOf)}
        >
          {BEST_OF_OPTIONS.map((n) => (
            <ToggleGroupItem key={n} value={String(n)} disabled={mode === "detailed" && intent === "auto"}>
              {n}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>

      {intent !== "auto" && (
        <span className="rounded-full bg-amber-500/10 px-2 py-0.5 text-amber-700 dark:text-amber-400">
          Diagnostic view — no generated answer
        </span>
      )}
    </div>
  );
}
