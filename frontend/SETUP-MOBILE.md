# Turning AI DOC into a native mobile app (Capacitor)

This wraps your existing Vite + React web app into real installable iOS/Android
apps. Your React components, routing, and styling all stay exactly as they are —
Capacitor just adds a native shell around the same web build.

---

## 0. File placement (from earlier in this session)

Before starting, make sure these are already in place:
- `capacitor.config.json` → repo root
- `lib-platform.js` → save as `src/lib/platform.js`
- `ChatWidget.jsx` (updated version) → `src/components/ChatWidget.jsx`

---

## 1. Install Capacitor

Run these in your project root:

```bash
npm install @capacitor/core @capacitor/cli
npx cap init
```

When `cap init` asks for App name / App ID, just confirm — they're already set in
`capacitor.config.json` (`AI DOC` / `com.aidoc.rarediseaseassistant`). If you want
a different app ID (must be unique, reverse-domain style, e.g.
`com.yourname.aidoc`), edit `capacitor.config.json` before continuing.

---

## 2. Build your web app

Capacitor wraps your **built** output (the `dist` folder), not your source code directly:

```bash
npm run build
```

You'll re-run this (then `npx cap sync`, step 5) every time you want the app to
reflect new changes — Capacitor doesn't watch your source files automatically.

---

## 3. Add the native platforms

You need a Mac (with Xcode) to build for iOS. Android works on any OS with
Android Studio installed.

```bash
npm install @capacitor/android @capacitor/ios
npx cap add android
npx cap add ios
```

This creates `android/` and `ios/` folders in your repo — real native Xcode/Gradle
projects. Commit these to your repo (they're meant to be checked in, unlike
`node_modules`).

---

## 4. Add your app icon and splash screen

Two source images were generated earlier this session:
- `icon-1024-source.png` (1024×1024 app icon)
- `splash-source.png` (2732×2732 splash screen)

Put both in a new folder `resources/` at your repo root, named exactly:
```
resources/icon.png
resources/splash.png
```

Then generate every platform-specific size automatically:

```bash
npm install @capacitor/assets --save-dev
npx capacitor-assets generate
```

This writes all the actual iOS/Android icon and splash files into `ios/` and
`android/` for you — no manual resizing needed.

---

## 5. Sync your web build into the native projects

```bash
npx cap sync
```

Run this after every `npm run build` — it copies your `dist` folder into both
native projects and updates any native dependencies.

---

## 6. Open and run

**Android** (needs Android Studio installed):
```bash
npx cap open android
```
This opens Android Studio. Press the green ▶ Run button, pick an emulator or a
plugged-in device.

**iOS** (needs a Mac + Xcode):
```bash
npx cap open ios
```
This opens Xcode. Select a simulator or device, press ▶ Run.

---

## 7. Two real bugs your specific app WILL hit — fixed / flagged below

### ✅ Already fixed: relative API paths break in the native shell
Inside the native app, there's no "current domain" the way a browser page has
one — the WebView loads local bundled files from an internal address, so a
relative call like `fetch("/api/chat")` silently fails (it's not a CORS error,
it just resolves to a URL that doesn't exist). This is why `ChatWidget.jsx` was
updated to route through `apiUrl()` from `src/lib/platform.js`, which forces an
absolute URL to your real deployed backend when running natively. Your other
API calls in `Api.js` were already safe since `BASE_URL` there is already
absolute (points at your Hugging Face Space).

**Action needed:** open `src/lib/platform.js` and make sure `DEPLOYED_ORIGIN`
is set to your actual live Vercel URL.

### ⚠️ Not yet fixed — Google Sign-In will very likely break natively
`signInWithPopup` (what `signInWithGoogle()` in your `firebase.js` almost
certainly uses) generally fails inside embedded native WebViews — Google
actively blocks OAuth logins from "disallowed user agents" like this, often
surfacing as `Error 403: disallowed_useragent`. This is a known, common issue,
not a bug in your code — it'll likely still work fine on the web deployment,
but break in the native app specifically.

**The real fix** requires a native-aware auth plugin — typically
`@capacitor-firebase/authentication`, which performs sign-in through the OS's
native Google account picker instead of an embedded webview. That requires
changes inside your `firebase.js`, which wasn't part of the files shared with
me this session. Share that file with me and I'll wire in the native-aware
version directly — I didn't want to guess at your Firebase initialization code
and hand back something that silently doesn't compile.

---

## 8. Iterating after this initial setup

Every time you make changes to your React app:
```bash
npm run build
npx cap sync
npx cap open android   # or ios
```
Then re-run from Xcode/Android Studio.

---

## 9. Publishing to app stores (when you're ready)

- **Android**: Android Studio → Build → Generate Signed Bundle/APK. You'll need
  a Google Play Developer account ($25 one-time).
- **iOS**: Xcode → Product → Archive, then upload via Xcode's Organizer. You'll
  need an Apple Developer account ($99/year) — this is Apple's requirement, not
  something Capacitor or I can work around.