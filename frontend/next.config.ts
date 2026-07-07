import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  turbopack: {
    root: path.join(__dirname),
  },
  // Default bottom-left position collides with the sidebar's own bottom nav
  // (Settings/About) — move the dev-only indicator out of the way.
  devIndicators: {
    position: "top-right",
  },
};

export default nextConfig;
