"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Folder, FileText, FileJson, ChevronRight, RefreshCw, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { getKnowledgeFiles } from "@/lib/api/knowledge";
import type { KnowledgeNode } from "@/lib/types";
import { FileViewerDialog } from "@/components/knowledge/FileViewerDialog";
import { cn } from "@/lib/utils";

function TreeNode({
  node,
  depth,
  onOpenFile,
}: {
  node: KnowledgeNode;
  depth: number;
  onOpenFile: (path: string, displayName: string) => void;
}) {
  const [open, setOpen] = useState(depth === 0);

  if (node.type === "file") {
    const Icon = node.extension === ".json" ? FileJson : FileText;
    return (
      <button
        type="button"
        onClick={() => onOpenFile(node.path, node.display_name)}
        className="flex w-full min-w-0 items-center gap-1.5 rounded-md px-2 py-1.5 text-left text-sm hover:bg-muted"
        style={{ paddingLeft: `${depth * 14 + 8}px` }}
      >
        <Icon className="size-3.5 shrink-0 text-muted-foreground" />
        <span className="min-w-0 flex-1 truncate">{node.display_name}</span>
      </button>
    );
  }

  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full min-w-0 items-center gap-1.5 rounded-md px-2 py-1.5 text-left text-sm font-medium hover:bg-muted"
        style={{ paddingLeft: `${depth * 14 + 8}px` }}
      >
        <ChevronRight className={cn("size-3.5 shrink-0 transition-transform", open && "rotate-90")} />
        <Folder className="size-3.5 shrink-0 text-muted-foreground" />
        <span className="min-w-0 flex-1 truncate">{node.display_name}</span>
        <span className="ml-auto shrink-0 text-xs font-normal text-muted-foreground">
          {node.children.length}
        </span>
      </button>
      {open &&
        node.children.map((child) => (
          <TreeNode key={child.path} node={child} depth={depth + 1} onOpenFile={onOpenFile} />
        ))}
    </div>
  );
}

export function KnowledgeExplorer() {
  const [openFile, setOpenFile] = useState<{ path: string; displayName: string } | null>(null);
  const { data, isLoading, isError, refetch, isFetching } = useQuery({
    queryKey: ["knowledge-files"],
    queryFn: getKnowledgeFiles,
  });

  return (
    <div className="flex h-full min-w-0 flex-col">
      <div className="flex items-center justify-between border-b px-3 py-2">
        <h3 className="text-sm font-semibold">Knowledge Explorer</h3>
        <Button
          variant="ghost"
          size="icon"
          className="size-6"
          onClick={() => refetch()}
          aria-label="Refresh"
        >
          <RefreshCw className={cn("size-3.5", isFetching && "animate-spin")} />
        </Button>
      </div>

      <div className="min-w-0 flex-1 overflow-x-hidden overflow-y-auto p-1.5">
        {isLoading && (
          <div className="flex flex-col gap-2 p-2">
            <Skeleton className="h-5 w-full" />
            <Skeleton className="h-5 w-full" />
            <Skeleton className="h-5 w-3/4" />
          </div>
        )}
        {isError && (
          <div className="flex flex-col items-start gap-2 p-3 text-xs text-destructive">
            <div className="flex items-center gap-2">
              <AlertCircle className="size-3.5 shrink-0" />
              Couldn&apos;t reach the backend — is api_server.py running?
            </div>
            <Button size="sm" variant="outline" className="h-7 gap-1" onClick={() => refetch()}>
              <RefreshCw className="size-3" /> Retry
            </Button>
          </div>
        )}
        {data?.length === 0 && (
          <p className="p-3 text-xs text-muted-foreground">
            No knowledge artifacts yet — run the pipeline first (Document Processing above).
          </p>
        )}
        {data?.map((node) => (
          <TreeNode
            key={node.path}
            node={node}
            depth={0}
            onOpenFile={(path, displayName) => setOpenFile({ path, displayName })}
          />
        ))}
      </div>

      <FileViewerDialog
        path={openFile?.path ?? null}
        displayName={openFile?.displayName ?? ""}
        onClose={() => setOpenFile(null)}
      />
    </div>
  );
}
