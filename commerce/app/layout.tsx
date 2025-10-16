import { CartProvider } from 'components/cart/cart-context';
// import { Navbar } from 'components/layout/navbar';
import GlobalNav from 'components/layout/global-nav';
import { WelcomeToast } from 'components/welcome-toast';
import { ToastContainer } from 'components/ui/toast';
import { ServiceWorkerRegister } from 'components/service-worker-register';
import { GeistSans } from 'geist/font/sans';
import { getCart } from 'lib/shopify';
import { ReactNode } from 'react';
import { Toaster } from 'sonner';
import './globals.css';
import { baseUrl } from 'lib/utils';

const { SITE_NAME } = process.env;

export const metadata = {
  metadataBase: new URL(baseUrl),
  title: {
    default: SITE_NAME!,
    template: `%s | ${SITE_NAME}`
  },
  description: 'AI 기반 맞춤형 향수 제작 및 추천 플랫폼 - Fragrance AI',
  keywords: ['향수', 'AI', '조향', '맞춤형', 'fragrance', 'perfume'],
  authors: [{ name: 'Fragrance AI Team' }],
  manifest: '/manifest.json',
  robots: {
    follow: true,
    index: true
  },
  openGraph: {
    type: 'website',
    locale: 'ko_KR',
    url: baseUrl,
    siteName: SITE_NAME!,
    title: SITE_NAME!,
    description: 'AI 기반 맞춤형 향수 제작 및 추천 플랫폼'
  },
  twitter: {
    card: 'summary_large_image',
    title: SITE_NAME!,
    description: 'AI 기반 맞춤형 향수 제작 및 추천 플랫폼'
  },
  viewport: {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 5,
    userScalable: true
  },
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#000000' }
  ]
};

export default async function RootLayout({
  children
}: {
  children: ReactNode;
}) {
  // Don't await the fetch, pass the Promise to the context provider
  const cart = getCart();

  return (
    <html lang="en" className={GeistSans.variable}>
      <body className="bg-[var(--luxury-cream)] text-[var(--luxury-midnight)] selection:bg-[var(--luxury-rose-gold)] selection:text-[var(--luxury-cream)] dark:bg-[var(--luxury-obsidian)] dark:text-[var(--luxury-pearl)] dark:selection:bg-[var(--luxury-gold)] dark:selection:text-[var(--luxury-midnight)]">
        <ServiceWorkerRegister />
        <CartProvider cartPromise={cart}>
          {/* <Navbar /> */}
          <GlobalNav />
          <main>
            {children}
            <Toaster closeButton />
            <WelcomeToast />
            <ToastContainer />
          </main>
        </CartProvider>
      </body>
    </html>
  );
}
