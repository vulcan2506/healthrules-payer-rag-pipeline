import { Workflow, GitBranch, Waypoints, Network, Boxes, Image, Users } from "lucide-react";
import type { VizType } from "@/lib/types";

export interface VizTypeOption {
  value: VizType | "infographic" | "comic-strip";
  label: string;
  icon: typeof Workflow;
  enabled: boolean;
  disabledReason?: string;
}

// Infographic/Comic Strip are intentionally disabled — Anthropic has no
// image-generation API, and faking them with the diagram-as-code approach
// used for the other 5 types wouldn't produce anything resembling either.
export const VIZ_TYPE_OPTIONS: VizTypeOption[] = [
  { value: "flowchart", label: "Flowchart", icon: Workflow, enabled: true },
  { value: "timeline", label: "Timeline", icon: GitBranch, enabled: true },
  { value: "relationship", label: "Relationship Diagram", icon: Network, enabled: true },
  { value: "mindmap", label: "Mind Map", icon: Waypoints, enabled: true },
  { value: "architecture", label: "Architecture Diagram", icon: Boxes, enabled: true },
  {
    value: "infographic",
    label: "Infographic",
    icon: Image,
    enabled: false,
    disabledReason: "Needs an image-generation model — not available on Anthropic's API today.",
  },
  {
    value: "comic-strip",
    label: "Comic Strip (Fun Mode)",
    icon: Users,
    enabled: false,
    disabledReason: "Needs an image-generation model — not available on Anthropic's API today.",
  },
];
