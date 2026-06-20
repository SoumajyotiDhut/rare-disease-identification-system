import { createContext, useContext, useState, useEffect } from "react";
import { LIGHT, DARK } from "../theme";

const ThemeContext = createContext(null);

export function ThemeProvider({ children }) {
    const [dark, setDark] = useState(() => {
        try {
            const saved = window.localStorage?.getItem("aidoc-theme");
            if (saved) return saved === "dark";
        } catch (_) { }
        return false;
    });

    useEffect(() => {
        try { window.localStorage?.setItem("aidoc-theme", dark ? "dark" : "light"); } catch (_) { }
        document.body.style.background = dark ? DARK.bg : LIGHT.bg;
    }, [dark]);

    const c = dark ? DARK : LIGHT;

    return (
        <ThemeContext.Provider value={{ dark, setDark, c }}>
            {children}
        </ThemeContext.Provider>
    );
}

export function useTheme() {
    const ctx = useContext(ThemeContext);
    if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
    return ctx;
}