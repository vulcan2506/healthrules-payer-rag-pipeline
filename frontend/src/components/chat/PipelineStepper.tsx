"use client";

import { useEffect, useState } from "react";
import { Check, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

// Purely presentational — the backend doesn't emit per-stage progress events
// today (see KT doc / Frontend plan), so this advances on a timer rather than
// real signals. It still communicates the real shape of the pipeline
// (routing -> retrieval -> reranking -> generation) rather than a generic spinner.
const STAGES = ["Planning", "Retrieving", "Reranking", "Generating"] as const;
const STAGE_DURATION_MS = 1400;

export function PipelineStepper() {
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveIndex((i) => Math.min(i + 1, STAGES.length - 1));
    }, STAGE_DURATION_MS);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex items-center gap-1.5">
      {STAGES.map((stage, i) => {
        const done = i < activeIndex;
        const active = i === activeIndex;
        return (
          <div key={stage} className="flex items-center gap-1.5">
            <div
              className={cn(
                "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs transition-colors",
                done && "text-muted-foreground",
                active && "bg-muted font-medium text-foreground",
                !done && !active && "text-muted-foreground/40"
              )}
            >
              {done && <Check className="size-3 text-emerald-600 dark:text-emerald-400" />}
              {active && <Loader2 className="size-3 animate-spin" />}
              {stage}
            </div>
            {i < STAGES.length - 1 && (
              <div className={cn("h-px w-3", done ? "bg-muted-foreground/40" : "bg-muted-foreground/15")} />
            )}
          </div>
        );
      })}
    </div>
  );
}
