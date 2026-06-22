import { initializeApp } from "firebase/app";
import {
    getAuth,
    GoogleAuthProvider,
    signInWithPopup,
    createUserWithEmailAndPassword,
    signInWithEmailAndPassword,
    signOut,
    onAuthStateChanged,
    updateProfile,
    sendPasswordResetEmail,
} from "firebase/auth";

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

export const signInWithGoogle = () => signInWithPopup(auth, googleProvider);

export const logOut = () => signOut(auth);

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