import { useTheme } from "../context/ThemeContext";

function Loader({ message = "Loading…" }) {
    const { c } = useTheme();
    return (
        <div style={{ minHeight: "100vh", background: c.bg, display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 24, fontFamily: "'Inter',sans-serif" }}>
            <div style={{ position: "relative", width: 64, height: 64 }}>
                <div style={{ position: "absolute", inset: 0, border: `3px solid ${c.border}`, borderTop: `3px solid ${c.teal}`, borderRadius: "50%", animation: "spin .9s linear infinite" }} />
                <div style={{ position: "absolute", inset: 10, border: `2px solid ${c.border}`, borderTop: `2px solid ${c.blue}`, borderRadius: "50%", animation: "spin 1.5s linear infinite reverse" }} />
                <div style={{ position: "absolute", inset: "50%", transform: "translate(-50%,-50%)", width: 12, height: 12, borderRadius: "50%", background: `linear-gradient(135deg,${c.teal},${c.blue})` }} />
            </div>
            <div style={{ textAlign: "center" }}>
                <p style={{ fontSize: 15, color: c.sub, fontWeight: 600, margin: "0 0 5px" }}>{message}</p>
                <p style={{ fontSize: 12, color: c.muted, margin: 0 }}>AI DOC · Rare Disease Assistant</p>
            </div>
            <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
        </div>
    );
}
export default Loader;