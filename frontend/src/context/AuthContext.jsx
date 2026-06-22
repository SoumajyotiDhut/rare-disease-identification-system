import { createContext, useContext, useEffect, useState } from "react";
import { auth, watchAuthState, logOut as fbLogOut } from "../firebase";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [authLoading, setAuthLoading] = useState(true);

    useEffect(() => {
        const unsubscribe = watchAuthState((firebaseUser) => {
            setUser(firebaseUser);
            setAuthLoading(false);
        });
        return unsubscribe;
    }, []);

    const logout = async () => {
        await fbLogOut();
        setUser(null);
    };

    return (
        <AuthContext.Provider value={{ user, authLoading, logout, currentAuth: auth }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const ctx = useContext(AuthContext);
    if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
    return ctx;
}