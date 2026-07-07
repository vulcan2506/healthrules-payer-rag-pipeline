"use client";

import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { useUIStore } from "@/lib/store";

export function AboutDialog() {
  const { aboutOpen, setAboutOpen } = useUIStore();

  return (
    <Dialog open={aboutOpen} onOpenChange={setAboutOpen}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Adaptive Enterprise Knowledge Platform</DialogTitle>
          <DialogDescription>
            A knowledge assistant over HealthRules Payer release documentation — built on
            taxonomy-routed retrieval, cross-encoder reranking, and confidence-gated escalation.
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-wrap gap-2 text-sm text-muted-foreground">
          <Badge variant="secondary">Confidence-gated retrieval</Badge>
          <Badge variant="secondary">Version-aware answers</Badge>
          <Badge variant="secondary">Cited sources</Badge>
          <Badge variant="secondary">Claude-powered</Badge>
        </div>
        <p className="text-sm text-muted-foreground">
          Every answer is produced by the same dynamic gate the CLI backend uses — a cheap
          pipeline pass first, escalating to a wider pooled search only when the retrieved
          evidence isn&apos;t confident enough on its own.
        </p>
      </DialogContent>
    </Dialog>
  );
}
