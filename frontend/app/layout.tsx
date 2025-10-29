import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'SkyGuard Tactical - Drone Detection System',
  description: 'Real-time acoustic drone detection and classification',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-slate-900">{children}</body>
    </html>
  )
}
