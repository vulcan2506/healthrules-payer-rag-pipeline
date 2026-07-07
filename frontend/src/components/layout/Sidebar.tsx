"use client";

import { MessageSquarePlus, History, Database, Settings, Info, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useUIStore } from "@/lib/store";
import { useNewChat } from "@/components/layout/HistoryDialog";
import { cn } from "@/lib/utils";

interface NavItemProps {
  icon: React.ReactNode;
  label: string;
  collapsed: boolean;
  onClick?: () => void;
  active?: boolean;
}

function NavItem({ icon, label, collapsed, onClick, active }: NavItemProps) {
  const button = (
    <Button
      variant={active ? "secondary" : "ghost"}
      onClick={onClick}
      className={cn("w-full gap-2", collapsed ? "justify-center px-0" : "justify-start")}
    >
      {icon}
      {!collapsed && <span className="truncate">{label}</span>}
    </Button>
  );

  if (!collapsed) return button;

  return (
    <Tooltip>
      <TooltipTrigger render={button} />
      <TooltipContent side="right">{label}</TooltipContent>
    </Tooltip>
  );
}

interface SidebarProps {
  /** Ignore the collapsed-width toggle — used when rendering inside the mobile drawer. */
  forceExpanded?: boolean;
}

export function Sidebar({ forceExpanded = false }: SidebarProps) {
  const {
    sidebarCollapsed: storeCollapsed,
    toggleSidebar,
    setSettingsOpen,
    setAboutOpen,
    setHistoryOpen,
    setKnowledgePanelOpen,
  } = useUIStore();
  const sidebarCollapsed = forceExpanded ? false : storeCollapsed;
  const newChat = useNewChat();

  return (
    <aside
      className={cn(
        "flex h-full flex-col border-r bg-sidebar text-sidebar-foreground transition-[width] duration-200",
        sidebarCollapsed ? "w-14" : "w-56"
      )}
    >
      <div className={cn("flex items-center gap-2 p-3", sidebarCollapsed && "justify-center")}>
        {!sidebarCollapsed && (
          <span className="text-sm font-semibold tracking-tight">Knowledge Platform</span>
        )}
        {!forceExpanded && (
          <Button
            variant="ghost"
            size="icon"
            className="ml-auto size-7"
            onClick={toggleSidebar}
            aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {sidebarCollapsed ? <PanelLeftOpen className="size-4" /> : <PanelLeftClose className="size-4" />}
          </Button>
        )}
      </div>

      <Separator />

      <nav className="flex flex-1 flex-col gap-1 p-2">
        <NavItem
          icon={<MessageSquarePlus className="size-4" />}
          label="New Chat"
          collapsed={sidebarCollapsed}
          onClick={newChat}
        />
        <NavItem
          icon={<History className="size-4" />}
          label="History"
          collapsed={sidebarCollapsed}
          onClick={() => setHistoryOpen(true)}
        />
        <NavItem
          icon={<Database className="size-4" />}
          label="Knowledge Bases"
          collapsed={sidebarCollapsed}
          onClick={() => setKnowledgePanelOpen(true)}
        />
      </nav>

      <Separator />

      <div className="flex flex-col gap-1 p-2">
        <NavItem
          icon={<Settings className="size-4" />}
          label="Settings"
          collapsed={sidebarCollapsed}
          onClick={() => setSettingsOpen(true)}
        />
        <NavItem
          icon={<Info className="size-4" />}
          label="About"
          collapsed={sidebarCollapsed}
          onClick={() => setAboutOpen(true)}
        />
      </div>
    </aside>
  );
}
