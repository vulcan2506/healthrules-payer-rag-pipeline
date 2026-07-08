"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ChevronDown, Clock, Gauge, Route, FileText, Sparkles, MessageCircleQuestion } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { methodInfo, explainConfidence, formatLatency } from "@/lib/chat-explain";
import type { ChatResponse } from "@/lib/types";
import { cn } from "@/lib/utils";
import { VisualizeDialog } from "@/components/chat/VisualizeDialog";

export function ResponseCard({ response }: { response: ChatResponse }) {
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [visualizeOpen, setVisualizeOpen] = useState(false);
  const method = methodInfo(response.method);
  const confidence = explainConfidence(response);

  return (
    <Card className="gap-3 py-4 shadow-sm">
      <CardContent className="flex flex-col gap-3">
        {response.answer && (
          <div className="prose prose-sm dark:prose-invert max-w-none text-sm leading-relaxed">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{response.answer}</ReactMarkdown>
          </div>
        )}

        {/* Why this answer looks the way it does — surfaced, not hidden */}
        <div className="flex flex-wrap items-center gap-2 border-t pt-3">
          <Tooltip>
            <TooltipTrigger
              render={
                <Badge variant="secondary" className="gap-1.5 py-1">
                  <Gauge className="size-3" />
                  {confidence.headline}
                </Badge>
              }
            />
            <TooltipContent side="bottom" className="max-w-xs text-balance">
              {confidence.detail}
            </TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger
              render={
                <Badge variant="outline" className="gap-1.5 py-1">
                  <Route className="size-3" />
                  {method.label}
                </Badge>
              }
            />
            <TooltipContent side="bottom" className="max-w-xs text-balance">
              {method.description}
            </TooltipContent>
          </Tooltip>

          <Badge variant="outline" className="gap-1.5 py-1">
            <Clock className="size-3" />
            {formatLatency(response.latency_s, response.from_cache)}
          </Badge>

          {response.was_rewritten && response.standalone_query && (
            <Tooltip>
              <TooltipTrigger
                render={
                  <Badge variant="outline" className="gap-1.5 py-1">
                    <MessageCircleQuestion className="size-3" />
                    Follow-up resolved
                  </Badge>
                }
              />
              <TooltipContent side="bottom" className="max-w-xs text-balance">
                Understood this as a follow-up and searched for: &ldquo;{response.standalone_query}&rdquo;
              </TooltipContent>
            </Tooltip>
          )}
        </div>

        {response.chunks.length > 0 && (
          <Collapsible open={sourcesOpen} onOpenChange={setSourcesOpen}>
            <CollapsibleTrigger className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground">
              <FileText className="size-3.5" />
              {sourcesOpen ? "Hide" : "Expand"} {response.chunks.length} source
              {response.chunks.length === 1 ? "" : "s"}
              <ChevronDown className={cn("size-3.5 transition-transform", sourcesOpen && "rotate-180")} />
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-2 flex flex-col gap-2">
              {response.chunks.map((chunk, i) => (
                <div key={chunk.chunk_id || i} className="rounded-md border bg-muted/30 p-2.5 text-xs shadow-sm">
                  <div className="mb-1 flex flex-wrap items-center gap-1.5 font-medium">
                    <span className="text-muted-foreground">[{i + 1}]</span>
                    <span>{chunk.section_header || "Untitled section"}</span>
                    {chunk.version && (
                      <Badge variant="outline" className="h-4 px-1.5 text-[10px]">
                        v{chunk.version}
                      </Badge>
                    )}
                  </div>
                  <p className="mb-1 text-muted-foreground">{chunk.source_doc}</p>
                  <p className="line-clamp-3 text-foreground/80">{chunk.text}</p>
                </div>
              ))}
            </CollapsibleContent>
          </Collapsible>
        )}

        {response.answer && (
          <>
            <Button
              size="sm"
              variant="outline"
              className="w-fit gap-1.5 rounded-full"
              onClick={() => setVisualizeOpen(true)}
            >
              <Sparkles className="size-3.5" /> Visualize
            </Button>
            <VisualizeDialog
              open={visualizeOpen}
              onClose={() => setVisualizeOpen(false)}
              question={response.query}
              answer={response.answer}
              chunks={response.chunks}
            />
          </>
        )}
      </CardContent>
    </Card>
  );
}
