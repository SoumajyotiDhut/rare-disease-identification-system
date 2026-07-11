import { createContext, useContext, useState, useCallback, useRef } from "react";
import { useTheme } from "./ThemeContext";

const ToastContext = createContext(null);
let idCounter = 0;

export function ToastProvider({ children }) {
    const [toasts, setToasts] = useState([]);
    const timers = useRef({});

    const remove = useCallback((id) => {
        setToasts(t => t.filter(x => x.id !== id));
        clearTimeout(timers.current[id]);
        delete timers.current[id];
    }, []);

    const push = useCallback((message, type = "info", duration = 4200) => {
        const id = ++idCounter;
        setToasts(t => [...t, { id, message, type }]);
        timers.current[id] = setTimeout(() => remove(id), duration);
        return id;
    }, [remove]);

    const toast = {
        success: (msg, d) => push(msg, "success", d),
        error: (msg, d) => push(msg, "error", d),
        info: (msg, d) => push(msg, "info", d),
        warn: (msg, d) => push(msg, "warn", d),
    };

    return (
        <ToastContext.Provider value={toast}>
            {children}
            <ToastViewport toasts={toasts} remove={remove} />
        </ToastContext.Provider>
    );
}

function ToastViewport({ toasts, remove }) {
    const { c } = useTheme();
    const ICONS = { success: "✓", error: "✕", warn: "⚠", info: "ℹ" };
    const COLORS = {
        success: { color: c.teal, border: c.tealB },
        error: { color: c.red, border: c.redB },
        warn: { color: c.amber, border: c.ambB },
        info: { color: c.blue, border: c.blueB },
    };

    return (
        <div className="toast-viewport" style={{
            position: "fixed", top: 20, right: 20, zIndex: 9999,
            display: "flex", flexDirection: "column", gap: 10,
            maxWidth: "calc(100vw - 40px)", width: 360,
        }}>
            <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600&display=swap');
        @keyframes toastIn  { from{opacity:0;transform:translateX(30px) scale(.97)} to{opacity:1;transform:translateX(0) scale(1)} }
        @media(max-width:600px){
          .toast-viewport { left:16px!important; right:16px!important; width:auto!important; top:auto!important; bottom:16px!important }
        }
      `}</style>
            {toasts.map(t => {
                const s = COLORS[t.type] || COLORS.info;
                return (
                    <div key={t.id} onClick={() => remove(t.id)} style={{
                        background: c.card, border: `1px solid ${c.border}`, borderLeft: `2.5px solid ${s.color}`,
                        padding: "14px 16px", display: "flex", alignItems: "flex-start", gap: 11,
                        boxShadow: c.shadowLg, cursor: "pointer",
                        animation: "toastIn .25s ease",
                    }}>
                        <span style={{
                            width: 20, height: 20, borderRadius: "50%", color: s.color,
                            display: "flex", alignItems: "center", justifyContent: "center",
                            fontSize: 12, fontWeight: 800, flexShrink: 0, marginTop: 1,
                        }}>{ICONS[t.type]}</span>
                        <p style={{ fontSize: 13.5, color: c.text, margin: 0, lineHeight: 1.55, fontWeight: 500, flex: 1 }}>{t.message}</p>
                        <span style={{ fontSize: 13, color: c.muted, flexShrink: 0 }}>✕</span>
                    </div>
                );
            })}
        </div>
    );
}

export function useToast() {
    const ctx = useContext(ToastContext);
    if (!ctx) throw new Error("useToast must be used within ToastProvider");
    return ctx;
}