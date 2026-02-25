/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ['@neo4j-nvl/react', '@neo4j-nvl/base'],
  experimental: {
    optimizePackageImports: ['@chakra-ui/react'],
  },
};

module.exports = nextConfig;
