import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  reactStrictMode: true,
  rewrites: async () => {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.API_URL}/:path*/`,
      },
    ];
  }
};

export default nextConfig;
