import { apiFetch } from "@/lib/api/client";
import type { VisualizeRequest, VisualizeResponse, VizType } from "@/lib/types";

export function suggestVizType(question: string): Promise<{ suggested: VizType }> {
  return apiFetch(`/api/visualize/suggest?question=${encodeURIComponent(question)}`);
}

export function generateVisualization(req: VisualizeRequest): Promise<VisualizeResponse> {
  return apiFetch("/api/visualize", {
    method: "POST",
    body: JSON.stringify(req),
  });
}
