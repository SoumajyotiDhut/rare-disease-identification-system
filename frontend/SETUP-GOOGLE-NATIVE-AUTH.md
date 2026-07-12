# Fixing Google Sign-In for the native app

This is the one piece of the mobile setup that isn't just running commands —
it needs a few manual steps in the Firebase console, because Google requires
each platform (web, Android, iOS) to be individually registered before native
sign-in will work. Do this once, carefully, in order.

---

## 1. Install the plugin

```powershell
cd D:\rare-disease-identification-system\frontend
npm install @capacitor-firebase/authentication
npx cap sync
```

`firebase.js` and `capacitor.config.json` (already updated above) expect this
package to exist — the build will fail without it.

---

## 2. Register an Android app in Firebase (if not already done)

1. [Firebase Console](https://console.firebase.google.com) → your project
2. ⚙️ **Project settings** → scroll to **"Your apps"**
3. Click **Add app** → Android icon
4. **Android package name** must exactly match your `capacitor.config.json`
   `appId`: `com.aidoc.rarediseaseassistant`
5. Nickname: anything (e.g. "AI DOC Android")
6. **SHA-1 signing certificate**: for now, during development, get this by running:
   ```powershell
   cd android
   ./gradlew signingReport
   ```
   Look for the `SHA1` value under `Variant: debug`. Paste that into Firebase.
   (You'll need to add your **release** SHA-1 here too later, before publishing
   to the Play Store — debug and release use different signing keys.)
7. Click **Register app**, then **Download `google-services.json`**

**Place the downloaded file at:**
```
frontend/android/app/google-services.json
```

---

## 3. Add the Google Services Gradle plugin (Android build config)

Open `frontend/android/build.gradle` (the **project-level** one, not
`app/build.gradle`) and add to the `dependencies` block:

```gradle
buildscript {
    dependencies {
        // ...existing lines...
        classpath 'com.google.gms:google-services:4.4.2'
    }
}
```

Then open `frontend/android/app/build.gradle` and add this line at the very
bottom of the file:

```gradle
apply plugin: 'com.google.gms.google-services'
```

---

## 4. Register an iOS app in Firebase (Mac + Xcode only)

1. Same Firebase console page → **Add app** → iOS icon
2. **Bundle ID** must exactly match `com.aidoc.rarediseaseassistant`
3. Download **`GoogleService-Info.plist`**
4. Open `frontend/ios/App/App.xcworkspace` in Xcode
5. Drag `GoogleService-Info.plist` into the `App` folder in Xcode's left
   sidebar — **make sure "Copy items if needed" is checked** and it's added
   to the `App` target
6. Open `Info.plist` (in the same Xcode project) and add a URL scheme:
   - Find the `REVERSED_CLIENT_ID` value inside the `GoogleService-Info.plist`
     you just added (it looks like `com.googleusercontent.apps.XXXXXXXXX`)
   - In Xcode: select the `App` target → **Info** tab → **URL Types** →
     click **+** → paste that value into **URL Schemes**

---

## 5. Confirm Google Sign-In is enabled in Firebase Auth itself

This part you likely already have, since Google Sign-In already works on
web — just confirm: **Authentication** → **Sign-in method** → **Google** →
should show "Enabled".

---

## 6. Rebuild and test

```powershell
npm run build
npx cap sync
npx cap open android
```

Run it on an emulator or device, try Google Sign-In. It should now open the
native Google account picker instead of an embedded browser popup.

---

## Why this couldn't be "just code"

Every native mobile OAuth integration — not just this one, for any app —
requires the platform (Google, in this case) to know in advance which
specific app package/bundle ID and signing certificate is allowed to
request sign-in on its behalf. That's a security boundary Google enforces
at their end, not something any code change on our side can bypass. The
`google-services.json` / `GoogleService-Info.plist` files above are exactly
that registration, downloaded per-platform from Firebase.