'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

/** Internal navigation links (Next.js client-side routing). */
const NAV_ITEMS = [
  { label: 'Dashboard', href: '/', icon: '▦' },
  { label: 'Experiments', href: '/experiments', icon: '⚗' },
  { label: 'Models', href: '/models', icon: '⬡' },
  { label: 'Inference', href: '/inference', icon: '◎' },
];

/**
 * External service links — ports default to common dev values and can be
 * overridden by setting NEXT_PUBLIC_* env vars if needed.
 * These open in a new tab.
 */
const EXTERNAL_LINKS = [
  { label: 'MLflow UI', icon: '⎇', port: 5000, path: '' },
  { label: 'API Docs', icon: '📄', port: 8000, path: '/docs' },
  { label: 'Grafana', icon: '📊', port: 3001, path: '' },
  { label: 'MinIO', icon: '🗄', port: 9001, path: '' },
];

/** Collapsible sidebar navigation with internal routes and external service links. */
export default function Sidebar() {
  const pathname = usePathname();

  // Build external URLs using the current browser hostname so the links work
  // both locally and on a remote dev server.
  const hostname = typeof window !== 'undefined' ? window.location.hostname : 'localhost';

  return (
    <aside className="w-52 shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col">
      {/* Brand */}
      <div className="px-4 py-5 border-b border-gray-800">
        <h1 className="text-base font-bold text-blue-400 tracking-wide">AI-Forge</h1>
        <p className="text-xs text-gray-600 mt-0.5">VLM MLOps Platform</p>
      </div>

      {/* Primary navigation */}
      <nav className="py-3">
        {NAV_ITEMS.map((item) => {
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-4 py-2.5 text-sm transition-colors ${
                active
                  ? 'bg-blue-950/60 text-blue-300 border-r-2 border-blue-500'
                  : 'text-gray-400 hover:text-gray-100 hover:bg-gray-800/60'
              }`}
            >
              <span className="text-base leading-none">{item.icon}</span>
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* External service links */}
      <div className="border-t border-gray-800 py-3 flex-1">
        <p className="px-4 mb-1 text-xs text-gray-700 uppercase tracking-wider">Services</p>
        {EXTERNAL_LINKS.map((link) => (
          <a
            key={link.label}
            href={`http://${hostname}:${link.port}${link.path}`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 px-4 py-2 text-sm text-gray-500 hover:text-gray-200 hover:bg-gray-800/60 transition-colors"
          >
            <span className="text-sm leading-none">{link.icon}</span>
            <span>{link.label}</span>
            <span className="ml-auto text-xs text-gray-700">:{link.port}</span>
          </a>
        ))}
      </div>

      {/* Footer version */}
      <div className="px-4 py-3 border-t border-gray-800 text-xs text-gray-700">v1.0.0</div>
    </aside>
  );
}
