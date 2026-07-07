"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Loader2, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { generateVisualization, suggestVizType } from "@/lib/api/visualize";
import { VIZ_TYPE_OPTIONS } from "@/lib/viz-types";
import { MermaidDiagram } from "@/components/chat/MermaidDiagram";
import type { ChatChunk, VizType } from "@/lib/types";
import { cn } from "@/lib/utils";

interface VisualizeDialogProps {
  open: boolean;
  onClose: () => void;
  question: string;
  answer: string;
  chunks: ChatChunk[];
}

export function VisualizeDialog({ open, onClose, question, answer, chunks }: VisualizeDialogProps) {
  const [selected, setSelected] = useState<VizType>("flowchart");

  const suggestQuery = useQuery({
    queryKey: ["visualize-suggest", question],
    queryFn: () => suggestVizType(question),
    enabled: open,
  });

  useEffect(() => {
    if (suggestQuery.data?.suggested) setSelected(suggestQuery.data.suggested);
  }, [suggestQuery.data]);

  const generateMutation = useMutation({
    mutationFn: generateVisualization,
  });

  const handleGenerate = () => {
    generateMutation.mutate({ question, answer, chunks, viz_type: selected });
  };

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="flex h-[85vh] flex-col overflow-hidden sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-1.5">
            <Sparkles className="size-4" /> Visualize Answer
          </DialogTitle>
        </DialogHeader>

        <p className="text-xs text-muted-foreground">
          Generated only from this answer and its cited sources — never new facts.
        </p>

        <div className="flex flex-wrap gap-1.5">
          {VIZ_TYPE_OPTIONS.map((opt) => {
            const chip = (
              <Button
                key={opt.value}
                type="button"
                size="sm"
                variant={selected === opt.value ? "default" : "outline"}
                aria-disabled={!opt.enabled}
                // Not a native `disabled` button for the disabled options —
                // that sets pointer-events:none, which silently kills hover
                // too, so the "coming soon" tooltip below would never fire.
                onClick={() => opt.enabled && setSelected(opt.value as VizType)}
                className={cn(
                  "h-7 gap-1.5 rounded-full text-xs",
                  !opt.enabled && "cursor-not-allowed opacity-50 hover:bg-transparent"
                )}
              >
                <opt.icon className="size-3.5" />
                {opt.label}
                {!opt.enabled && <span className="text-[10px] opacity-70">(soon)</span>}
              </Button>
            );
            if (!opt.enabled) {
              return (
                <Tooltip key={opt.value}>
                  <TooltipTrigger render={chip} />
                  <TooltipContent side="top" className="max-w-56 text-balance">
                    {opt.disabledReason}
                  </TooltipContent>
                </Tooltip>
              );
            }
            return chip;
          })}
        </div>

        <Button onClick={handleGenerate} disabled={generateMutation.isPending} className="gap-1.5">
          {generateMutation.isPending ? <Loader2 className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
          Generate
        </Button>

        <div className="min-h-0 flex-1 overflow-y-auto rounded-md border bg-muted/20 p-4">
          {generateMutation.isPending && (
            <div className="flex h-40 items-center justify-center text-sm text-muted-foreground">
              <Loader2 className="mr-2 size-4 animate-spin" /> Generating diagram…
            </div>
          )}
          {generateMutation.isError && (
            <p className="text-sm text-destructive">
              {generateMutation.error instanceof Error ? generateMutation.error.message : "Generation failed"}
            </p>
          )}
          {generateMutation.data && <MermaidDiagram code={generateMutation.data.mermaid} />}
          {!generateMutation.data && !generateMutation.isPending && !generateMutation.isError && (
            <p className={cn("text-center text-sm text-muted-foreground")}>
              Choose a type and click Generate.
            </p>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
