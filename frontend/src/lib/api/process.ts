import { apiFetch } from "@/lib/api/client";
import type { ProcessJobStatus } from "@/lib/types";

// Triggers Stage 1's main.py -> run_tail.py --skip-eval as a background
// subprocess chain — reprocesses the ENTIRE data/pdfs/ corpus (main.py has
// no incremental/single-file mode), not just the file just uploaded.
export function startProcessing(): Promise<{ job_id: string }> {
  return apiFetch("/api/process", { method: "POST" });
}

export function getProcessStatus(jobId: string): Promise<ProcessJobStatus> {
  return apiFetch(`/api/process/${jobId}`);
}

// Clears the current corpus (source PDFs, generated output, vector store) so
// a fresh set of PDFs can be processed without merging with what was there
// before — /api/process reprocesses everything currently in data/pdfs/.
export function resetCorpus(): Promise<{ status: string; message: string }> {
  return apiFetch("/api/reset", {
    method: "POST",
    body: JSON.stringify({ confirm: true }),
  });
}
