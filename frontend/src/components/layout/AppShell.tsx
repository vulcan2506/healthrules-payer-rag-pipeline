"use client";

import { BookOpen, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Sidebar } from "@/components/layout/Sidebar";
import { ChatWorkspace } from "@/components/layout/ChatWorkspace";
import { KnowledgeWorkspace } from "@/components/layout/KnowledgeWorkspace";
import { AboutDialog } from "@/components/layout/AboutDialog";
import { HistoryDialog } from "@/components/layout/HistoryDialog";
import { SettingsDialog } from "@/components/settings/SettingsDialog";
import { useUIStore } from "@/lib/store";

export function AppShell() {
  const { knowledgePanelOpen, setKnowledgePanelOpen, mobileSidebarOpen, setMobileSidebarOpen } = useUIStore();

  return (
    <div className="flex h-dvh w-full overflow-hidden bg-background">
      {/* Desktop: persistent sidebar */}
      <div className="hidden md:flex">
        <Sidebar />
      </div>

      <div className="relative flex min-w-0 flex-1 flex-col">
        {/* Mobile: top bar with hamburger trigger for the sidebar drawer */}
        <div className="flex items-center gap-2 border-b p-2 md:hidden">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setMobileSidebarOpen(true)}
            aria-label="Open navigation menu"
          >
            <Menu className="size-5" />
          </Button>
          <span className="text-sm font-semibold tracking-tight">Knowledge Platform</span>
        </div>

        <div className="relative flex min-h-0 flex-1">
          {/* Chat Workspace — 70% on desktop, full width on mobile */}
          <main className="min-w-0 flex-1 md:flex-[7]">
            <ChatWorkspace />
          </main>

          {/* Knowledge Workspace — 30% on desktop, hidden (drawer) on mobile */}
          <aside className="hidden min-w-0 border-l md:flex md:flex-[3]">
            <KnowledgeWorkspace />
          </aside>

          {/* Mobile: floating trigger for the Knowledge Workspace drawer */}
          <Button
            size="icon"
            className="absolute bottom-4 right-4 size-12 rounded-full shadow-lg md:hidden"
            onClick={() => setKnowledgePanelOpen(true)}
            aria-label="Open knowledge workspace"
          >
            <BookOpen className="size-5" />
          </Button>
        </div>
      </div>

      {/* Mobile: sidebar drawer */}
      <Sheet open={mobileSidebarOpen} onOpenChange={setMobileSidebarOpen}>
        <SheetContent side="left" className="w-64 p-0">
          <SheetHeader className="sr-only">
            <SheetTitle>Navigation</SheetTitle>
          </SheetHeader>
          <Sidebar forceExpanded />
        </SheetContent>
      </Sheet>

      {/* Mobile: knowledge workspace drawer */}
      <Sheet open={knowledgePanelOpen} onOpenChange={setKnowledgePanelOpen}>
        <SheetContent side="right" className="w-[85vw] max-w-sm p-0 sm:max-w-md">
          <SheetHeader className="border-b px-4 py-3">
            <SheetTitle>Knowledge Workspace</SheetTitle>
          </SheetHeader>
          <div className="min-h-0 flex-1">
            <KnowledgeWorkspace />
          </div>
        </SheetContent>
      </Sheet>

      <SettingsDialog />
      <AboutDialog />
      <HistoryDialog />
    </div>
  );
}
