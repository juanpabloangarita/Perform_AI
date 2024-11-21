import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  trailingSlash: true,
  rewrites:
    process.env.NODE_ENV === "development"
      ? async () => {
          return [
            {
              source: "/api/:path*",
              destination: "http://localhost:5000/:path*/",
            },
          ];
        }
      : undefined,
};

export default nextConfig;
