import { Capacitor } from "@capacitor/core";

/**
 * CRITICAL for Capacitor apps: your web deployment (Vercel) can use relative
 * paths like fetch("/api/chat") because the browser resolves them against
 * the page's own domain. Inside the native app, there IS no "page domain" —
 * the WebView loads bundled local files from a capacitor://localhost origin,
 * so a relative fetch("/api/chat") would try to hit
 * capacitor://localhost/api/chat, which does not exist, and silently fail.
 *
 * Anything that calls a relative path (like ChatWidget.jsx → /api/chat) must
 * be routed through this helper instead, so it uses your real deployed
 * domain when running as a native app.
 *
 * IMPORTANT: replace the placeholder below with your actual deployed URL.
 */
const DEPLOYED_ORIGIN = "https://rare-disease-identification-system.vercel.app";

export function apiUrl(path) {
    if (Capacitor.isNativePlatform()) {
        return `${DEPLOYED_ORIGIN}${path}`;
    }
    return path; // relative path is fine in the browser/web deployment
}

export const isNativeApp = () => Capacitor.isNativePlatform();