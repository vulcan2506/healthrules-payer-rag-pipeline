import { DocumentProcessing } from "@/components/knowledge/DocumentProcessing";
import { KnowledgeExplorer } from "@/components/knowledge/KnowledgeExplorer";

export function KnowledgeWorkspace() {
  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col divide-y">
      <div className="max-h-[45%] min-w-0 overflow-x-hidden overflow-y-auto">
        <DocumentProcessing />
      </div>
      <div className="min-h-0 min-w-0 flex-1">
        <KnowledgeExplorer />
      </div>
    </div>
  );
}
