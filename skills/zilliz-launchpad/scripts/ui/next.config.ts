import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_SIDECAR_URL:
      process.env.NEXT_PUBLIC_SIDECAR_URL ?? "http://127.0.0.1:8000",
  },
};

export default nextConfig;
