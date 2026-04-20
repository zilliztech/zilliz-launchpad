/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_SIDECAR_URL: process.env.NEXT_PUBLIC_SIDECAR_URL || "http://127.0.0.1:8000",
  },
};

export default nextConfig;
