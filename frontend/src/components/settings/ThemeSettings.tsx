"use client";

import { useEffect, useState } from "react";
import { useTheme } from "next-themes";
import { Sun, Moon, Monitor } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const OPTIONS = [
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
  { value: "system", label: "System", icon: Monitor },
] as const;

export function ThemeSettings() {
  const { theme, setTheme } = useTheme();
  // Avoid hydration mismatch — next-themes resolves the real value client-side only.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  return (
    <div className="flex flex-col gap-2">
      <h4 className="text-sm font-medium">Theme</h4>
      <div className="flex gap-1.5">
        {OPTIONS.map(({ value, label, icon: Icon }) => (
          <Button
            key={value}
            size="sm"
            variant={mounted && theme === value ? "default" : "outline"}
            onClick={() => setTheme(value)}
            className={cn("flex-1 gap-1.5")}
          >
            <Icon className="size-3.5" />
            {label}
          </Button>
        ))}
      </div>
    </div>
  );
}
