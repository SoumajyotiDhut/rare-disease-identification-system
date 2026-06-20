import { useTheme } from "../context/ThemeContext";

const CONF_KEY = (c) => ({
    High: { bg: c.tealL, color: c.teal, border: c.tealB },
    Medium: { bg: c.ambL, color: c.amber, border: c.ambB },
    Low: { bg: c.redL, color: c.red, border: c.redB },
});

export default function HistoryDrawer({ item, onClose }) {
    const { c } = useTheme();
    if (!item) return null;

    const CONF = CONF_KEY(c);
    const predictions = item.predictions?.length ? item.predictions : (item.disease ? [item] : []);
    const symptoms = Array.isArray(item.symptoms) ? item.symptoms : (item.symptoms ? [item.symptoms] : []);

    return (
        <>
            <style>{`
        @keyframes drawerIn  { from{opacity:0} to{opacity:1} }
        @keyframes drawerSlide { from{transform:translateX(100%)} to{transform:translateX(0)} }
        @media(max-width:640px){ .hd-panel{ width:100%!important } }
      `}</style>

            {/* Backdrop */}
            <div onClick={onClose} style={{
                position: "fixed", inset: 0, background: "rgba(10,20,32,0.45)",
                backdropFilter: "blur(2px)", zIndex: 400, animation: "drawerIn .2s ease",
            }} />

            {/* Panel */}
            <div className="hd-panel" style={{
                position: "fixed", top: 0, right: 0, height: "100%", width: 460,
                background: c.card, zIndex: 401, animation: "drawerSlide .3s cubic-bezier(.2,.7,.3,1)",
                boxShadow: "-12px 0 40px rgba(0,0,0,0.18)", overflowY: "auto",
                borderLeft: `1px solid ${c.border}`,
            }}>
                {/* Header */}
                <div style={{
                    padding: "24px 28px", borderBottom: `1px solid ${c.border}`,
                    display: "flex", justifyContent: "space-between", alignItems: "flex-start",
                    position: "sticky", top: 0, background: c.card, zIndex: 2,
                }}>
                    <div>
                        <span style={{
                            fontSize: 10, fontWeight: 800, color: c.teal, background: c.tealL,
                            border: `1px solid ${c.tealB}`, padding: "4px 12px", borderRadius: 100,
                            letterSpacing: 1, textTransform: "uppercase", display: "inline-block", marginBottom: 10
                        }}>
                            Session Detail
                        </span>
                        <p style={{ fontSize: 13, color: c.sub, margin: 0, fontWeight: 500 }}>
                            {item.timestamp ? new Date(item.timestamp).toLocaleString("en-IN") : "Timestamp unavailable"}
                        </p>
                    </div>
                    <button onClick={onClose} style={{
                        width: 32, height: 32, borderRadius: 9, border: `1px solid ${c.borderI}`,
                        background: c.bgAlt, color: c.sub, cursor: "pointer", fontSize: 15,
                        display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                    }}>✕</button>
                </div>

                <div style={{ padding: "24px 28px" }}>
                    {/* Symptoms */}
                    <div style={{ marginBottom: 28 }}>
                        <p style={{
                            fontSize: 10, fontWeight: 800, color: c.muted, textTransform: "uppercase",
                            letterSpacing: 1, margin: "0 0 12px"
                        }}>Reported Symptoms</p>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
                            {symptoms.length ? symptoms.map((s, i) => (
                                <span key={i} style={{
                                    fontSize: 12.5, color: c.sub, background: c.cardAlt,
                                    border: `1px solid ${c.border}`, padding: "6px 13px", borderRadius: 100, fontWeight: 500
                                }}>
                                    {s}
                                </span>
                            )) : <span style={{ fontSize: 13, color: c.muted }}>No symptom data recorded</span>}
                        </div>
                    </div>

                    {/* Predictions list */}
                    <div>
                        <p style={{
                            fontSize: 10, fontWeight: 800, color: c.muted, textTransform: "uppercase",
                            letterSpacing: 1, margin: "0 0 14px"
                        }}>
                            Differential Diagnosis ({predictions.length})
                        </p>
                        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                            {predictions.map((p, i) => {
                                const conf = CONF[p.confidence] || CONF.Low;
                                return (
                                    <div key={i} style={{
                                        border: `1px solid ${i === 0 ? c.tealB : c.border}`, borderRadius: 16,
                                        padding: "16px 18px", background: i === 0 ? c.tealL : c.bgAlt,
                                    }}>
                                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
                                            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                                <span style={{
                                                    fontSize: 10, fontWeight: 800, color: c.teal, background: c.card,
                                                    border: `1px solid ${c.tealB}`, padding: "3px 9px", borderRadius: 7
                                                }}>#{p.rank || i + 1}</span>
                                                <span style={{ fontWeight: 700, fontSize: 14.5, color: c.text }}>{p.disease}</span>
                                            </div>
                                            <span style={{
                                                padding: "3px 11px", borderRadius: 100, fontSize: 10, fontWeight: 800,
                                                background: conf.bg, color: conf.color, border: `1px solid ${conf.border}`,
                                                textTransform: "uppercase", letterSpacing: .6
                                            }}>{p.confidence || "—"}</span>
                                        </div>
                                        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                                            <span style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 18, fontWeight: 800, color: c.teal }}>
                                                {p.probability ?? "—"}%
                                            </span>
                                            <div style={{ flex: 1, height: 5, background: c.border, borderRadius: 100 }}>
                                                <div style={{ height: 5, width: `${p.probability || 0}%`, background: c.teal, borderRadius: 100 }} />
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* Footer note */}
                    <div style={{
                        marginTop: 28, padding: "14px 16px", background: c.ambL,
                        border: `1px solid ${c.ambB}`, borderRadius: 12
                    }}>
                        <p style={{ fontSize: 12, color: c.amber, margin: 0, lineHeight: 1.6 }}>
                            <strong>⚠ Research use only.</strong> This record reflects an AI-generated prediction and should not substitute professional clinical judgment.
                        </p>
                    </div>
                </div>
            </div>
        </>
    );
}