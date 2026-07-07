import { apiFetch } from "@/lib/api/client";
import type { KnowledgeNode, KnowledgeFileContent } from "@/lib/types";

export async function getKnowledgeFiles(): Promise<KnowledgeNode[]> {
  const { tree } = await apiFetch<{ tree: KnowledgeNode[] }>("/api/knowledge/files");
  return tree;
}

export function getKnowledgeFile(path: string): Promise<KnowledgeFileContent> {
  return apiFetch(`/api/knowledge/file?path=${encodeURIComponent(path)}`);
}
