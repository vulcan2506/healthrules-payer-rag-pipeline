import { create } from "zustand";

interface UIState {
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;

  settingsOpen: boolean;
  setSettingsOpen: (open: boolean) => void;

  aboutOpen: boolean;
  setAboutOpen: (open: boolean) => void;

  historyOpen: boolean;
  setHistoryOpen: (open: boolean) => void;

  // Mobile: the 30% Knowledge Workspace collapses into a Sheet drawer.
  knowledgePanelOpen: boolean;
  setKnowledgePanelOpen: (open: boolean) => void;

  // Mobile: the left Sidebar collapses into a Sheet drawer too.
  mobileSidebarOpen: boolean;
  setMobileSidebarOpen: (open: boolean) => void;

  pinnedVersion: string | null;
  setPinnedVersion: (version: string | null) => void;
}

export const useUIStore = create<UIState>((set) => ({
  sidebarCollapsed: false,
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),

  settingsOpen: false,
  setSettingsOpen: (open) => set({ settingsOpen: open }),

  aboutOpen: false,
  setAboutOpen: (open) => set({ aboutOpen: open }),

  historyOpen: false,
  setHistoryOpen: (open) => set({ historyOpen: open }),

  knowledgePanelOpen: false,
  setKnowledgePanelOpen: (open) => set({ knowledgePanelOpen: open }),

  mobileSidebarOpen: false,
  setMobileSidebarOpen: (open) => set({ mobileSidebarOpen: open }),

  pinnedVersion: null,
  setPinnedVersion: (version) => set({ pinnedVersion: version }),
}));
