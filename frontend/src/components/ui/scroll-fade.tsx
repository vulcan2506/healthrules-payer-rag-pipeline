"use client";

import { useEffect, useRef, useState, UIEvent } from "react";
import { cn } from "@/lib/utils";

interface ScrollFadeProps {
  children: React.ReactNode;
  className?: string;
}

/**
 * Native-scrollbar scroll container with top/bottom fade indicators that
 * appear only when there's actually more content in that direction — makes
 * it visually obvious the box scrolls instead of looking like content is
 * cut off.
 */
export function ScrollFade({ children, className }: ScrollFadeProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [atTop, setAtTop] = useState(true);
  const [atBottom, setAtBottom] = useState(true);

  const updateFades = () => {
    const el = ref.current;
    if (!el) return;
    setAtTop(el.scrollTop <= 1);
    setAtBottom(el.scrollTop + el.clientHeight >= el.scrollHeight - 1);
  };

  useEffect(() => {
    updateFades();
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver(updateFades);
    observer.observe(el);
    return () => observer.disconnect();
  }, [children]);

  const handleScroll = (_e: UIEvent<HTMLDivElement>) => updateFades();

  return (
    <div className="relative min-h-0 flex-1">
      {/* absolute+inset-0 (not h-full) — sizes to the positioned ancestor's
          box regardless of content, sidestepping percentage-height
          resolution inside nested flex containers. */}
      <div
        ref={ref}
        onScroll={handleScroll}
        className={cn("absolute inset-0 overflow-y-auto overscroll-contain", className)}
      >
        {children}
      </div>
      <div
        className={cn(
          "pointer-events-none absolute inset-x-0 top-0 h-6 bg-gradient-to-b from-background to-transparent transition-opacity",
          atTop ? "opacity-0" : "opacity-100"
        )}
      />
      <div
        className={cn(
          "pointer-events-none absolute inset-x-0 bottom-0 h-8 bg-gradient-to-t from-background to-transparent transition-opacity",
          atBottom ? "opacity-0" : "opacity-100"
        )}
      />
    </div>
  );
}
