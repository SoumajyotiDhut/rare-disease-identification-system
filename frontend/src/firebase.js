import { initializeApp } from "firebase/app";
import {
    getAuth,
    GoogleAuthProvider,
    signInWithPopup,
    signInWithCredential,
    createUserWithEmailAndPassword,
    signInWithEmailAndPassword,
    signOut,
    onAuthStateChanged,
    updateProfile,
    sendPasswordResetEmail,
} from "firebase/auth";
import { Capacitor } from "@capacitor/core";
import { FirebaseAuthentication } from "@capacitor-firebase/authentication";

const firebaseConfig = {
    apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
    authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
    projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
    storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
    messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
    appId: import.meta.env.VITE_FIREBASE_APP_ID,
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);

const googleProvider = new GoogleAuthProvider();

export const signUpWithEmail = (email, password, displayName) =>
    createUserWithEmailAndPassword(auth, email, password).then(async (cred) => {
        if (displayName) {
            await updateProfile(cred.user, { displayName });
        }
        return cred;
    });

export const signInWithEmail = (email, password) =>
    signInWithEmailAndPassword(auth, email, password);

/**
 * Google Sign-In, platform-aware.
 *
 * WEB: unchanged — signInWithPopup works fine in a real browser.
 *
 * NATIVE (Capacitor/iOS/Android): signInWithPopup fails inside embedded
 * WebViews — Google actively blocks OAuth from what it detects as an
 * embedded user agent (commonly surfaces as "Error 403:
 * disallowed_useragent"). Instead we use @capacitor-firebase/authentication,
 * which opens the OS's native Google account picker, then hands back a
 * credential that we sign into the SAME firebase/auth `auth` object with.
 * This keeps onAuthStateChanged/AuthContext working identically regardless
 * of platform — nothing else in the app needs to know or care which path
 * was used.
 */
export const signInWithGoogle = async () => {
    if (Capacitor.isNativePlatform()) {
        const result = await FirebaseAuthentication.signInWithGoogle();
        const idToken = result?.credential?.idToken;
        const accessToken = result?.credential?.accessToken;
        if (!idToken) {
            throw new Error("Native Google sign-in did not return a credential.");
        }
        const credential = GoogleAuthProvider.credential(idToken, accessToken);
        return signInWithCredential(auth, credential);
    }
    return signInWithPopup(auth, googleProvider);
};

export const logOut = async () => {
    if (Capacitor.isNativePlatform()) {
        // Keep native Google session and JS SDK session in sync on sign-out too
        await FirebaseAuthentication.signOut().catch(() => { });
    }
    return signOut(auth);
};

export const resetPassword = (email) => sendPasswordResetEmail(auth, email);

export const watchAuthState = (callback) => onAuthStateChanged(auth, callback);

/** Friendly error messages — Firebase's raw codes are not user-facing */
export const getAuthErrorMessage = (error) => {
    const code = error?.code || "";
    const map = {
        "auth/email-already-in-use": "An account with this email already exists.",
        "auth/invalid-email": "That doesn't look like a valid email address.",
        "auth/weak-password": "Password should be at least 6 characters.",
        "auth/user-not-found": "No account found with this email.",
        "auth/wrong-password": "Incorrect password. Please try again.",
        "auth/invalid-credential": "Incorrect email or password.",
        "auth/too-many-requests": "Too many attempts. Please wait a moment and try again.",
        "auth/popup-closed-by-user": "Sign-in popup was closed before completing.",
        "auth/network-request-failed": "Network error. Check your connection and try again.",
    };
    return map[code] || "Something went wrong. Please try again.";
};