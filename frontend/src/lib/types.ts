// Mirrors retrieval_layer/api_server.py's response shapes exactly — see that
// file (and redis_cache.py / retriever.py, which it wraps) for ground truth.

export interface ChatChunk {
  chunk_id: string;
  text: string;
  section_header: string;
  source_doc: string;
  version: string;
  score?: number;
  rerank_score?: number;
}

export interface ChatResponse {
  query: string;
  standalone_query?: string; // present when session_id was sent — query after follow-up rewriting
  was_rewritten?: boolean;   // true if the backend resolved a follow-up ("explain that more") using conversation history
  method: string; // "pipeline" | "merged" | "merged_all" | "naive" | "traditional" | "specific" | "factual" | "intelligence" | "delta"
  mode: string; // "concise" | "detailed" | "raw"
  answer: string | null; // null only in raw diagnostic mode
  confidence: number | null; // raw cross-encoder rerank score, NOT a 0-1 probability
  chunks: ChatChunk[];
  from_cache: boolean;
  latency_s: number;
}

export interface ChatRequestParams {
  query: string;
  mode?: "concise" | "detailed";
  best_of?: number;
  intent?: string;
  version?: string;
  raw?: boolean;
  session_id?: string;
}

export type ChatMode = "concise" | "detailed";
export type BestOf = 1 | 3 | 5 | 7;

// Real retriever.classify() intents, plus "auto" (don't force one — the
// backend classifies automatically). Forcing a real intent only has effect
// on the raw/diagnostic retrieval path — redis_cache.answer_query() never
// receives it (confirmed by reading cli.py's gated-mode branch). There is
// no forceable "cross-relations" intent — relationship-shaped queries are
// auto-detected by the confidence ladder, not a selectable dial.
export type ChatIntent = "auto" | "specific" | "factual" | "intelligence" | "delta";

export interface KnowledgeFileEntry {
  type: "file";
  name: string;
  display_name: string;
  path: string;
  extension: string;
  size_bytes: number;
}

export interface KnowledgeDirEntry {
  type: "directory";
  name: string;
  display_name: string;
  path: string;
  children: KnowledgeNode[];
}

export type KnowledgeNode = KnowledgeFileEntry | KnowledgeDirEntry;

export interface KnowledgeFileContent {
  type: "markdown" | "json";
  content: string | unknown;
}

export interface SettingsStatus {
  backend: "anthropic" | "local";
  model: string;
  has_key: boolean;
}

export interface ProcessJobStatus {
  job_id: string;
  status: "running" | "done" | "error";
  stage: string;
  log_tail: string;
}

// Only these 5 render as real diagrams (Claude generates Mermaid diagram-as-
// code, client-rendered). Infographic/Comic Strip are NOT implemented —
// Anthropic has no image-generation API, so faking them would contradict
// this app's Anthropic-only design.
export type VizType = "flowchart" | "timeline" | "relationship" | "mindmap" | "architecture";

export interface VisualizeRequest {
  question: string;
  answer: string;
  chunks: ChatChunk[];
  viz_type: VizType;
}

export interface VisualizeResponse {
  mermaid: string;
  viz_type: VizType;
}
