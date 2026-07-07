// Plain-language explanations of the backend's dynamic gating decisions —
// surfaced so first-time users see WHY an answer looks the way it does,
// not just a raw method string. See retrieval_layer/redis_cache.py's
// gate_and_retrieve()/retrieve_detailed() docstrings for the ground truth
// this is translating.
import type { ChatResponse } from "@/lib/types";

export const CONFIDENCE_THRESHOLD = 0.3;

interface MethodInfo {
  label: string;
  description: string;
}

// "confidence" is a raw cross-encoder rerank score (roughly -5..+5 in
// practice), NOT a 0-1 probability — never render it as a percentage.
const METHOD_INFO: Record<string, MethodInfo> = {
  pipeline: {
    label: "Focused search",
    description: "A single taxonomy-routed search found confident matches on the first pass.",
  },
  merged: {
    label: "Combined search (2-way)",
    description: "Pooled two retrieval strategies together after the focused pass wasn't confident enough alone.",
  },
  merged_all: {
    label: "Comprehensive search (3-way)",
    description: "Pooled three retrieval strategies together for the most thorough coverage available.",
  },
  naive: {
    label: "Flat vector search",
    description: "A plain similarity search with no taxonomy routing — a baseline comparison method.",
  },
  traditional: {
    label: "Traditional RAG",
    description: "A sliding-window search over raw text with no taxonomy — the closest analogue to off-the-shelf RAG.",
  },
  specific: {
    label: "Targeted lookup",
    description: "Routed directly to the most relevant topic in the taxonomy.",
  },
  factual: {
    label: "Factual lookup",
    description: "Searched the raw document corpus directly.",
  },
  intelligence: {
    label: "Overview search",
    description: "Searched summarized topic and hierarchy knowledge — best for breadth questions.",
  },
  delta: {
    label: "Version comparison",
    description: "Searched version-change analysis specifically — best for \"what changed\" questions.",
  },
};

export function methodInfo(method: string): MethodInfo {
  return (
    METHOD_INFO[method] ?? {
      label: method,
      description: "Retrieval method reported directly by the backend.",
    }
  );
}

export function explainConfidence(response: ChatResponse): { headline: string; detail: string } {
  if (response.mode === "raw") {
    const scoreText = response.confidence != null ? response.confidence.toFixed(2) : "n/a";
    return {
      headline: `Diagnostic view (score ${scoreText})`,
      detail:
        "A forced intent bypasses the confidence gate entirely — this is the plain routing/retrieval result, not a gated decision.",
    };
  }

  if (response.mode === "detailed") {
    return {
      headline: "Comprehensive mode — no confidence gate",
      detail:
        "Detailed mode always pools three retrieval strategies together for maximum thoroughness, regardless of confidence.",
    };
  }

  if (response.confidence == null) {
    return { headline: "Confidence unavailable", detail: "" };
  }

  const score = response.confidence;
  const scoreText = score.toFixed(2);
  const escalated = !["pipeline", "specific"].includes(response.method);

  if (!escalated) {
    return {
      headline: `Confident on the first pass (score ${scoreText})`,
      detail: `Above the ${CONFIDENCE_THRESHOLD.toFixed(2)} threshold the system uses to decide whether to search further — no escalation was needed.`,
    };
  }

  const poolSize = response.method === "merged" ? "two" : "three";
  return {
    headline: `Automatically broadened the search (initial score ${scoreText})`,
    detail: `The first focused pass scored below the ${CONFIDENCE_THRESHOLD.toFixed(2)} confidence threshold, so the system pooled ${poolSize} retrieval strategies together for more thorough coverage.`,
  };
}

export function formatLatency(latencySeconds: number, fromCache: boolean): string {
  if (fromCache) return "Instant (cached)";
  if (latencySeconds < 1) return `${Math.round(latencySeconds * 1000)}ms`;
  return `${latencySeconds.toFixed(1)}s`;
}
