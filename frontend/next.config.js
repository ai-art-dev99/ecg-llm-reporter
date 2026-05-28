/** @type {import('next').NextConfig} */
const nextConfig = {
    output: 'standalone',
    env: {
        NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    },
    async rewrites() {
        // In production, proxy /api/* to backend to avoid CORS issues
        const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
        return [
            {
                source: '/api/:path*',
                destination: `${backendUrl}/:path*`,
            },
        ]
    },
}

module.exports = nextConfig