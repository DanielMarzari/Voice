import type { NextConfig } from "next";

// We used to proxy /api/* → :8000 here, but Next's rewrite proxy has a
// ~30s idle timeout that kills long-running TTS synthesis requests with
// a socket hang-up. Now src/lib/api.ts fetches http://127.0.0.1:8000
// directly; CORS is set up backend-side. See api.ts for details.
const nextConfig: NextConfig = {};

export default nextConfig;
