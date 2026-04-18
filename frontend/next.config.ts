import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Proxy API calls to the FastAPI backend so the frontend can just
  // fetch('/api/...') without worrying about CORS or ports.
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8000/api/:path*",
      },
    ];
  },
};

export default nextConfig;
