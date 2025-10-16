import Link from 'next/link';

export const metadata = {
  title: 'Offline',
  description: 'You are currently offline'
};

export default function OfflinePage() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4">
      <div className="text-center">
        <div className="mb-8">
          <svg
            className="mx-auto h-24 w-24 text-gray-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M18.364 5.636a9 9 0 010 12.728m0 0l-2.829-2.829m2.829 2.829L21 21M15.536 8.464a5 5 0 010 7.072m0 0l-2.829-2.829m-4.243 2.829a4.978 4.978 0 01-1.414-2.83m-1.414 5.658a9 9 0 01-2.167-9.238m7.824 2.167a1 1 0 111.414 1.414m-1.414-1.414L3 3m8.293 8.293l1.414 1.414"
            />
          </svg>
        </div>

        <h1 className="mb-4 text-4xl font-bold tracking-tight">
          You're Offline
        </h1>

        <p className="mb-8 text-lg text-gray-600 dark:text-gray-400">
          Check your internet connection and try again
        </p>

        <div className="space-y-4">
          <Link
            href="/"
            className="inline-block rounded-lg bg-black px-8 py-3 text-white transition-colors hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200"
          >
            Return Home
          </Link>

          <p className="text-sm text-gray-500">
            Some cached content may still be available
          </p>
        </div>
      </div>
    </div>
  );
}
