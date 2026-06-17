const PALETTE = [
    { bar: "#0B7B6F", light: "#EBF8F6", text: "#0B7B6F", border: "#B2E8E2" },
    { bar: "#1D6FA4", light: "#EBF4F9", text: "#1D6FA4", border: "#B3D8EE" },
    { bar: "#5B3DB8", light: "#F2EEF9", text: "#5B3DB8", border: "#C8B8EC" },
    { bar: "#C05B1A", light: "#FFF4EC", text: "#C05B1A", border: "#F5D8B8" },
    { bar: "#8FA5B5", light: "#F0F5F8", text: "#5A7184", border: "#C8D8E4" },
];

const PredictionCard = ({ item }) => {
    const c = PALETTE[(item.rank - 1) % PALETTE.length];
    return (
        <div
            style={{ background: "#fff", border: `1px solid ${c.border}`, borderRadius: 18, padding: "22px 24px", transition: "box-shadow 0.2s, transform 0.15s", cursor: "default" }}
            onMouseEnter={e => { e.currentTarget.style.boxShadow = "0 8px 28px rgba(15,28,46,0.08)"; e.currentTarget.style.transform = "translateY(-3px)"; }}
            onMouseLeave={e => { e.currentTarget.style.boxShadow = "none"; e.currentTarget.style.transform = "none"; }}
        >
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
                <span style={{ width: 32, height: 32, borderRadius: 9, background: c.light, border: `1px solid ${c.border}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 800, color: c.text, flexShrink: 0 }}>#{item.rank}</span>
                <h2 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 15, fontWeight: 700, color: "#0F1C2E", margin: 0, lineHeight: 1.3 }}>{item.disease}</h2>
            </div>

            <div style={{ marginBottom: 13 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 7 }}>
                    <span style={{ color: "#8FA5B5", fontWeight: 500 }}>Probability</span>
                    <span style={{ color: c.text, fontWeight: 800 }}>{item.probability}%</span>
                </div>
                <div style={{ height: 6, background: "#F0F5F8", borderRadius: 100 }}>
                    <div style={{ height: 6, width: `${item.probability}%`, background: c.bar, borderRadius: 100, transition: "width 0.9s ease" }} />
                </div>
            </div>

            <span style={{ fontSize: 10, fontWeight: 800, padding: "4px 12px", borderRadius: 100, background: c.light, color: c.text, border: `1px solid ${c.border}`, textTransform: "uppercase", letterSpacing: 0.7 }}>
                {item.confidence}
            </span>
        </div>
    );
};

export default PredictionCard;
ENDOFFILE

cat > /home/claude / Loader.jsx << 'ENDOFFILE'
function Loader({ message = "Loading…" }) {
    return (
        <div style={{ minHeight: "100vh", background: "#F4F8FB", display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 22, fontFamily: "'Inter',sans-serif" }}>
            <div style={{ position: "relative", width: 60, height: 60 }}>
                <div style={{ position: "absolute", inset: 0, border: "3px solid #E8EFF5", borderTop: "3px solid #0B7B6F", borderRadius: "50%", animation: "spin 0.9s linear infinite" }} />
                <div style={{ position: "absolute", inset: 9, border: "2px solid #EDF2F6", borderTop: "2px solid #1D6FA4", borderRadius: "50%", animation: "spin 1.5s linear infinite reverse" }} />
                <div style={{ position: "absolute", inset: "50%", transform: "translate(-50%,-50%)", width: 10, height: 10, borderRadius: "50%", background: "#EBF8F6", border: "2px solid #B2E8E2" }} />
            </div>
            <div style={{ textAlign: "center" }}>
                <p style={{ fontSize: 15, color: "#4A6275", fontWeight: 600, margin: "0 0 5px" }}>{message}</p>
                <p style={{ fontSize: 12, color: "#9BB8CC", margin: 0 }}>AI DOC · Rare Disease Assistant</p>
            </div>
            <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
        </div>
    );
}
export default Loader;