"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Eye, EyeOff, CheckCircle2, XCircle, Loader2, KeyRound } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { getSettingsStatus, setApiKey } from "@/lib/api/settings";
import { cn } from "@/lib/utils";

export function ApiKeySettings() {
  const [key, setKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const queryClient = useQueryClient();

  const statusQuery = useQuery({
    queryKey: ["settings-status"],
    queryFn: getSettingsStatus,
  });

  const mutation = useMutation({
    mutationFn: setApiKey,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["settings-status"] });
      setKey("");
    },
  });

  const status = statusQuery.data;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h4 className="flex items-center gap-1.5 text-sm font-medium">
          <KeyRound className="size-4" /> Anthropic API Key
        </h4>
        {status && (
          <Badge
            variant={status.has_key ? "secondary" : "outline"}
            className={cn(
              "gap-1.5",
              status.has_key && "text-emerald-700 dark:text-emerald-400"
            )}
          >
            {status.has_key ? <CheckCircle2 className="size-3" /> : <XCircle className="size-3" />}
            {status.has_key ? `Connected — ${status.model}` : "No key configured"}
          </Badge>
        )}
      </div>

      <p className="text-xs text-muted-foreground">
        Bring your own key (BYOK) — stored only in the backend&apos;s local <code>.env</code> file,
        never logged or displayed once saved. Used for document processing, retrieval, and chat
        generation; falls back to a local model automatically if it stops working.
      </p>

      <div className="flex gap-2">
        <div className="relative flex-1">
          <Input
            type={showKey ? "text" : "password"}
            placeholder="sk-ant-api03-…"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            className="pr-9"
          />
          <button
            type="button"
            onClick={() => setShowKey((s) => !s)}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            aria-label={showKey ? "Hide key" : "Show key"}
          >
            {showKey ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
          </button>
        </div>
        <Button onClick={() => mutation.mutate(key)} disabled={!key.trim() || mutation.isPending}>
          {mutation.isPending ? <Loader2 className="size-4 animate-spin" /> : "Validate"}
        </Button>
      </div>

      {mutation.data && (
        <p
          className={cn(
            "flex items-center gap-1.5 text-xs",
            mutation.data.valid ? "text-emerald-600 dark:text-emerald-400" : "text-destructive"
          )}
        >
          {mutation.data.valid ? <CheckCircle2 className="size-3.5" /> : <XCircle className="size-3.5" />}
          {mutation.data.valid ? "Key validated and saved." : mutation.data.error || "Key rejected."}
        </p>
      )}
    </div>
  );
}
