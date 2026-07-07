"use client";

import { useQuery } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ScrollFade } from "@/components/ui/scroll-fade";
import { Skeleton } from "@/components/ui/skeleton";
import { getKnowledgeFile } from "@/lib/api/knowledge";
import { JsonTree } from "@/components/knowledge/JsonTree";

interface FileViewerDialogProps {
  path: string | null;
  displayName: string;
  onClose: () => void;
}

export function FileViewerDialog({ path, displayName, onClose }: FileViewerDialogProps) {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["knowledge-file", path],
    queryFn: () => getKnowledgeFile(path as string),
    enabled: !!path,
  });

  return (
    <Dialog open={!!path} onOpenChange={(open) => !open && onClose()}>
      {/* h-[80vh], not max-h — a flex child needs the container's height to
          be definite (not just capped) for flex-grow to actually give it
          space; max-height alone lets the dialog shrink-to-fit and the
          scroll area collapses instead of scrolling. */}
      <DialogContent className="flex h-[80vh] flex-col overflow-hidden sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{displayName}</DialogTitle>
        </DialogHeader>
        <ScrollFade className="pr-4">
          {isLoading && (
            <div className="flex flex-col gap-2">
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-2/3" />
            </div>
          )}
          {isError && (
            <p className="text-sm text-destructive">
              Couldn&apos;t load this file: {error instanceof Error ? error.message : "unknown error"}
            </p>
          )}
          {data?.type === "markdown" && (
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{data.content as string}</ReactMarkdown>
            </div>
          )}
          {data?.type === "json" && <JsonTree data={data.content} />}
        </ScrollFade>
      </DialogContent>
    </Dialog>
  );
}
