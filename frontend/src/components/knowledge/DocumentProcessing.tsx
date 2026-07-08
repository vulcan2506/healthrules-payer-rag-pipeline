"use client";

import { useState, useRef, useEffect, DragEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { UploadCloud, FileCheck2, Play, Loader2, CircleCheck, CircleX, Info, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter, DialogClose } from "@/components/ui/dialog";
import { uploadPdf } from "@/lib/api/upload";
import { startProcessing, getProcessStatus, resetCorpus } from "@/lib/api/process";
import { cn } from "@/lib/utils";

export function DocumentProcessing() {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [resetDialogOpen, setResetDialogOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();

  const uploadMutation = useMutation({
    mutationFn: uploadPdf,
    onSuccess: (data) => setUploadedFiles((prev) => [...prev, data.filename]),
  });

  const processMutation = useMutation({
    mutationFn: startProcessing,
    onSuccess: (data) => setJobId(data.job_id),
  });

  const statusQuery = useQuery({
    queryKey: ["process-status", jobId],
    queryFn: () => getProcessStatus(jobId as string),
    enabled: !!jobId,
    refetchInterval: (query) => (query.state.data?.status === "running" ? 2000 : false),
  });

  const resetMutation = useMutation({
    mutationFn: resetCorpus,
    onSuccess: () => {
      setUploadedFiles([]);
      setJobId(null);
      setResetDialogOpen(false);
      queryClient.invalidateQueries({ queryKey: ["knowledge-files"] });
    },
  });

  const job = statusQuery.data;
  const isRunning = job?.status === "running";

  // Refresh the Knowledge Explorer exactly once when processing finishes.
  useEffect(() => {
    if (job?.status === "done") {
      queryClient.invalidateQueries({ queryKey: ["knowledge-files"] });
    }
  }, [job?.status, queryClient]);

  const handleFiles = (files: FileList | null) => {
    if (!files) return;
    Array.from(files).forEach((file) => {
      if (!file.name.toLowerCase().endsWith(".pdf")) return;
      uploadMutation.mutate(file);
    });
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  return (
    <div className="flex flex-col gap-3 p-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Document Processing</h3>
        <Tooltip>
          <TooltipTrigger render={<Info className="size-3.5 text-muted-foreground" />} />
          <TooltipContent side="left" className="max-w-64 text-balance">
            Processing reruns the full pipeline over every PDF in the corpus (not just the new
            one) — the backend doesn&apos;t support incremental single-document processing today.
          </TooltipContent>
        </Tooltip>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={cn(
          "flex cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed p-6 text-center transition-colors",
          isDragging ? "border-primary bg-primary/5" : "border-muted-foreground/25 hover:border-muted-foreground/40"
        )}
      >
        <UploadCloud className="size-6 text-muted-foreground" />
        <p className="text-xs text-muted-foreground">
          Drag & drop a PDF here, or click to browse
        </p>
        <input
          ref={fileInputRef}
          type="file"
          accept="application/pdf"
          multiple
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      {uploadMutation.isPending && (
        <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Loader2 className="size-3 animate-spin" /> Uploading…
        </p>
      )}
      {uploadMutation.isError && (
        <p className="text-xs text-destructive">
          {uploadMutation.error instanceof Error ? uploadMutation.error.message : "Upload failed"}
        </p>
      )}
      {uploadedFiles.map((name) => (
        <div key={name} className="flex min-w-0 items-center gap-1.5 text-xs">
          <FileCheck2 className="size-3.5 shrink-0 text-emerald-600 dark:text-emerald-400" />
          <span className="min-w-0 flex-1 truncate">{name}</span>
        </div>
      ))}

      <div className="flex gap-1.5">
        <Button
          size="sm"
          onClick={() => processMutation.mutate()}
          disabled={uploadedFiles.length === 0 || isRunning || processMutation.isPending}
          className="flex-1 gap-1.5"
        >
          {isRunning || processMutation.isPending ? (
            <Loader2 className="size-3.5 animate-spin" />
          ) : (
            <Play className="size-3.5" />
          )}
          Process
        </Button>

        <Tooltip>
          <TooltipTrigger
            render={
              <Button
                size="sm"
                variant="destructive"
                onClick={() => setResetDialogOpen(true)}
                disabled={isRunning || resetMutation.isPending}
              >
                {resetMutation.isPending ? (
                  <Loader2 className="size-3.5 animate-spin" />
                ) : (
                  <Trash2 className="size-3.5" />
                )}
              </Button>
            }
          />
          <TooltipContent side="left" className="max-w-64 text-balance">
            Clear the current corpus (source PDFs, generated output, vector store) so a new set
            of PDFs can be processed from a clean slate, instead of merging with what&apos;s here now.
          </TooltipContent>
        </Tooltip>
      </div>

      {job && (
        <div className="flex flex-col gap-1.5 rounded-md border bg-muted/30 p-2.5 text-xs">
          <div className="flex min-w-0 items-center gap-1.5 font-medium">
            {job.status === "running" && <Loader2 className="size-3.5 shrink-0 animate-spin" />}
            {job.status === "done" && <CircleCheck className="size-3.5 shrink-0 text-emerald-600 dark:text-emerald-400" />}
            {job.status === "error" && <CircleX className="size-3.5 shrink-0 text-destructive" />}
            <span className="min-w-0 flex-1 truncate">{job.stage}</span>
          </div>
        </div>
      )}

      {resetMutation.isError && (
        <p className="text-xs text-destructive">
          {resetMutation.error instanceof Error ? resetMutation.error.message : "Reset failed"}
        </p>
      )}

      <Dialog open={resetDialogOpen} onOpenChange={setResetDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Clear the current corpus?</DialogTitle>
            <DialogDescription>
              This permanently deletes all processed PDFs, generated output, and the vector store
              on this deployment. Use this before processing a client&apos;s own PDF set so it
              doesn&apos;t merge with what&apos;s here now. This can&apos;t be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <DialogClose render={<Button variant="outline" size="sm" />}>Cancel</DialogClose>
            <Button
              size="sm"
              variant="destructive"
              onClick={() => resetMutation.mutate()}
              disabled={resetMutation.isPending}
              className="gap-1.5"
            >
              {resetMutation.isPending && <Loader2 className="size-3.5 animate-spin" />}
              Clear corpus
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
