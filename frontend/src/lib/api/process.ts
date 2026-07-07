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
