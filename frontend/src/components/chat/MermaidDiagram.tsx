"use client";

import { useEffect, useRef, useState, useId } from "react";
import { useTheme } from "next-themes";
import { AlertTriangle } from "lucide-react";

interface MermaidDiagramProps {
  code: string;
}

export function MermaidDiagram({ code }: MermaidDiagramProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const rawId = useId();
  const diagramId = `mermaid-${rawId.replace(/:/g, "")}`;
  const { resolvedTheme } = useTheme();

  useEffect(() => {
    let cancelled = false;

    async function render() {
      setError(null);
      try {
        const mermaid = (await import("mermaid")).default;
        mermaid.initialize({
          startOnLoad: false,
          theme: resolvedTheme === "dark" ? "dark" : "default",
          securityLevel: "strict",
          fontFamily: "var(--font-sans)",
        });
        const { svg } = await mermaid.render(diagramId, code);
        if (!cancelled && containerRef.current) {
          containerRef.current.innerHTML = svg;
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Failed to render diagram");
      }
    }

    render();
    return () => {
      cancelled = true;
    };
  }, [code, diagramId, resolvedTheme]);

  if (error) {
    return (
      <div className="flex flex-col gap-2 rounded-md border border-destructive/30 bg-destructive/10 p-3 text-xs text-destructive">
        <div className="flex items-center gap-1.5 font-medium">
          <AlertTriangle className="size-3.5 shrink-0" />
          Couldn&apos;t render this diagram
        </div>
        <pre className="overflow-x-auto whitespace-pre-wrap text-[11px] opacity-80">{code}</pre>
      </div>
    );
  }

  return <div ref={containerRef} className="flex justify-center overflow-x-auto [&_svg]:max-w-full" />;
}
