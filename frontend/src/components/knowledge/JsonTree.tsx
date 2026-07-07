"use client";

import { useState } from "react";
import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

function valuePreview(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return `Array(${value.length})`;
  if (typeof value === "object") return `Object(${Object.keys(value as object).length})`;
  if (typeof value === "string") return `"${value}"`;
  return String(value);
}

function JsonNode({ label, value, depth }: { label: string; value: unknown; depth: number }) {
  const [open, setOpen] = useState(depth < 1);
  const isExpandable = value !== null && typeof value === "object";

  if (!isExpandable) {
    return (
      <div className="flex gap-1.5 py-0.5 pl-5 font-mono text-xs">
        <span className="text-muted-foreground">{label}:</span>
        <span
          className={cn(
            typeof value === "string" && "text-emerald-600 dark:text-emerald-400",
            typeof value === "number" && "text-blue-600 dark:text-blue-400"
          )}
        >
          {valuePreview(value)}
        </span>
      </div>
    );
  }

  const entries = Array.isArray(value)
    ? value.map((v, i) => [String(i), v] as const)
    : Object.entries(value as Record<string, unknown>);

  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-1 py-0.5 font-mono text-xs hover:bg-muted/50"
      >
        <ChevronRight className={cn("size-3 shrink-0 transition-transform", open && "rotate-90")} />
        <span className="text-muted-foreground">{label}:</span>
        <span className="text-muted-foreground/70">{valuePreview(value)}</span>
      </button>
      {open && (
        <div className="border-l ml-1.5 pl-2">
          {entries.map(([k, v]) => (
            <JsonNode key={k} label={k} value={v} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

export function JsonTree({ data }: { data: unknown }) {
  return (
    <div className="overflow-x-auto">
      <JsonNode label="root" value={data} depth={0} />
    </div>
  );
}
