import { useTheme } from "../context/ThemeContext";

export default function Loader({ message = "Loading…" }) {
    const { c } = useTheme();
    return (
        <div style={{
            minHeight: "100vh", background: c.bg,
            display: "flex", alignItems: "center", justifyContent: "center",
            flexDirection: "column", gap: 26, fontFamily: "'Inter',sans-serif",
        }}>
            <style>{`
        @keyframes spin     { to{transform:rotate(360deg)} }
        @keyframes spinRev  { to{transform:rotate(-360deg)} }
        @keyframes pulse    { 0%,100%{opacity:.5;transform:scale(.95)} 50%{opacity:1;transform:scale(1.05)} }
        @keyframes fadeUp   { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
        @keyframes drawLine { from{stroke-dashoffset:600} to{stroke-dashoffset:0} }
      `}</style>

            {/* Layered spinner */}
            <div style={{ position: "relative", width: 68, height: 68 }}>
                <div style={{
                    position: "absolute", inset: 0,
                    border: `2px solid ${c.border}`,
                    borderTop: `2px solid ${c.teal}`,
                    borderRadius: "50%",
                    animation: "spin .9s linear infinite",
                }} />
                <div style={{
                    position: "absolute", inset: 10,
                    border: `1.5px solid ${c.border}`,
                    borderTop: `1.5px solid ${c.gold}`,
                    borderRadius: "50%",
                    animation: "spinRev 1.4s linear infinite",
                }} />
                <div style={{
                    position: "absolute", inset: 20,
                    border: `1.5px solid ${c.border}`,
                    borderTop: `1.5px solid ${c.blue}`,
                    borderRadius: "50%",
                    animation: "spin 1.8s linear infinite",
                }} />
                <div style={{
                    position: "absolute",
                    top: "50%", left: "50%",
                    transform: "translate(-50%,-50%)",
                    width: 12, height: 12, borderRadius: "50%",
                    background: c.gradPrimary,
                    animation: "pulse 2s ease-in-out infinite",
                }} />
            </div>

            {/* Text */}
            <div style={{ textAlign: "center", animation: "fadeUp .5s ease both .2s" }}>
                <p style={{ fontFamily: "'Fraunces',serif", fontSize: 16, color: c.text, fontWeight: 600, margin: "0 0 8px", letterSpacing: "-0.01em" }}>
                    {message}
                </p>
                <p style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 11, color: c.muted, margin: 0, display: "flex", alignItems: "center", justifyContent: "center", gap: 8, letterSpacing: "0.04em" }}>
                    <span style={{
                        display: "inline-block", width: 5, height: 5, borderRadius: "50%",
                        background: c.teal, animation: "pulse 1.4s ease-in-out infinite",
                    }} />
                    AI DOC · RARE DISEASE ASSISTANT
                </p>
            </div>
        </div>
    );
}