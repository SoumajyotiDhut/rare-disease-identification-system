import { useTheme } from "../context/ThemeContext";

const CONF = (c) => ({
    High: { bg: c.tealL, color: c.teal, border: c.tealB },
    Medium: { bg: c.ambL, color: c.amber, border: c.ambB },
    Low: { bg: c.redL, color: c.red, border: c.redB },
});

export default function HistoryDrawer({ item, onClose }) {
    const { c } = useTheme();
    if (!item) return null;

    const CMAP = CONF(c);
    const predictions = item.predictions?.length ? item.predictions : (item.disease ? [item] : []);
    const symptoms = Array.isArray(item.symptoms)
        ? item.symptoms
        : item.symptoms ? item.symptoms.split(",").map(s => s.trim()) : [];
    const topDx = predictions[0];
    const topConf = CMAP[topDx?.confidence] || CMAP.Low;

    return (
        <>
            <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Mono:wght@500;600&display=swap');
        @keyframes drawerIn    { from{opacity:0} to{opacity:1} }
        @keyframes drawerSlide { from{transform:translateX(100%)} to{transform:translateX(0)} }
        @media(max-width:600px){
          .hd-panel{ width:100%!important; border-radius:0!important }
          .hd-header{ padding:18px 20px!important }
          .hd-body{ padding:20px 20px!important }
        }
      `}</style>

            {/* Backdrop */}
            <div onClick={onClose} style={{
                position: "fixed", inset: 0,
                background: "rgba(8,14,26,0.55)",
                backdropFilter: "blur(4px)",
                zIndex: 400,
                animation: "drawerIn .2s ease",
            }} />

            {/* Panel */}
            <div className="hd-panel" style={{
                position: "fixed", top: 0, right: 0, height: "100%", width: 480,
                background: c.card,
                zIndex: 401,
                animation: "drawerSlide .3s cubic-bezier(.2,.7,.3,1)",
                boxShadow: `-16px 0 60px rgba(0,0,0,0.25)`,
                overflowY: "auto",
                borderLeft: `1px solid ${c.border}`,
            }}>

                {/* Header */}
                <div className="hd-header" style={{
                    padding: "22px 28px",
                    borderBottom: `1px solid ${c.border}`,
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    position: "sticky", top: 0, background: c.card, zIndex: 2,
                    backdropFilter: "blur(10px)",
                }}>
                    <div>
                        <span style={{
                            fontFamily: "'IBM Plex Mono',monospace",
                            fontSize: 10, fontWeight: 600, color: c.gold,
                            border: `1px solid ${c.goldB}`, padding: "4px 12px",
                            letterSpacing: "0.1em", textTransform: "uppercase",
                            display: "inline-block", marginBottom: 8,
                        }}>Session Detail</span>
                        <p style={{ fontSize: 13, color: c.sub, margin: 0, fontWeight: 500 }}>
                            {item.timestamp
                                ? new Date(item.timestamp).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" })
                                : "Timestamp unavailable"}
                        </p>
                    </div>
                    <button onClick={onClose} style={{
                        width: 34, height: 34, borderRadius: 4,
                        border: `1px solid ${c.border}`, background: c.cardAlt,
                        color: c.sub, cursor: "pointer", fontSize: 14,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        transition: "all .15s", flexShrink: 0,
                    }}
                        onMouseEnter={e => { e.currentTarget.style.background = c.redL; e.currentTarget.style.color = c.red; e.currentTarget.style.borderColor = c.redB; }}
                        onMouseLeave={e => { e.currentTarget.style.background = c.cardAlt; e.currentTarget.style.color = c.sub; e.currentTarget.style.borderColor = c.border; }}
                    >✕</button>
                </div>

                <div className="hd-body" style={{ padding: "24px 28px" }}>

                    {/* Top result hero */}
                    {topDx && (
                        <div style={{
                            background: `linear-gradient(135deg, ${c.tealL}, ${c.blueL})`,
                            border: `1px solid ${c.tealB}`, borderTop: `2px solid ${c.teal}`,
                            padding: "22px 24px", marginBottom: 24,
                        }}>
                            <div style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 10, fontWeight: 600, color: c.teal, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 12 }}>
                                Top Diagnosis
                            </div>
                            <h3 style={{
                                fontFamily: "'Fraunces',serif",
                                fontSize: 19, fontWeight: 600, color: c.text,
                                margin: "0 0 14px", lineHeight: 1.25,
                            }}>{topDx.disease}</h3>
                            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                                <span style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 28, fontWeight: 600, color: c.teal }}>{topDx.probability}%</span>
                                <div style={{ flex: 1 }}>
                                    <div style={{ height: 4, background: `${c.teal}22`, borderRadius: 100, overflow: "hidden" }}>
                                        <div style={{ height: 4, width: `${topDx.probability}%`, background: c.teal, borderRadius: 100 }} />
                                    </div>
                                    <span style={{ fontSize: 10.5, fontWeight: 700, color: topConf.color, background: topConf.bg, border: `1px solid ${topConf.border}`, padding: "3px 10px", borderRadius: 100, textTransform: "uppercase", letterSpacing: "0.06em", display: "inline-block", marginTop: 8 }}>
                                        {topDx.confidence}
                                    </span>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Symptoms */}
                    <div style={{ marginBottom: 24 }}>
                        <p style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 10, fontWeight: 600, color: c.muted, textTransform: "uppercase", letterSpacing: "0.1em", margin: "0 0 12px" }}>
                            Reported Symptoms
                        </p>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
                            {symptoms.length ? symptoms.map((s, i) => (
                                <span key={i} style={{
                                    fontSize: 12.5, color: c.sub, background: c.bgDeep,
                                    border: `1px solid ${c.border}`,
                                    padding: "6px 13px", borderRadius: 100, fontWeight: 500,
                                }}>{s}</span>
                            )) : <span style={{ fontSize: 13, color: c.muted, fontStyle: "italic" }}>No symptom data</span>}
                        </div>
                    </div>

                    {/* All predictions */}
                    {predictions.length > 1 && (
                        <div>
                            <p style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 10, fontWeight: 600, color: c.muted, textTransform: "uppercase", letterSpacing: "0.1em", margin: "0 0 14px" }}>
                                Differential Diagnosis ({predictions.length})
                            </p>
                            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                                {predictions.map((p, i) => {
                                    const conf = CMAP[p.confidence] || CMAP.Low;
                                    return (
                                        <div key={i} style={{
                                            border: `1px solid ${i === 0 ? c.tealB : c.border}`,
                                            borderLeft: `2px solid ${i === 0 ? c.teal : c.border}`,
                                            padding: "14px 16px",
                                            background: i === 0 ? c.tealL : c.cardAlt,
                                            display: "flex", alignItems: "center", gap: 14,
                                        }}>
                                            <span style={{
                                                fontFamily: "'Fraunces',serif", fontStyle: "italic",
                                                width: 22, fontSize: 15, fontWeight: 600, flexShrink: 0,
                                                color: i === 0 ? c.teal : c.muted, textAlign: "center",
                                            }}>{p.rank || i + 1}</span>
                                            <div style={{ flex: 1, minWidth: 0 }}>
                                                <p style={{ fontSize: 13.5, fontWeight: 700, color: c.text, margin: "0 0 2px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{p.disease}</p>
                                                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                                    <span style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 12, color: c.teal, fontWeight: 600 }}>{p.probability}%</span>
                                                    <span style={{ fontSize: 9.5, fontWeight: 700, color: conf.color, background: conf.bg, border: `1px solid ${conf.border}`, padding: "2px 8px", borderRadius: 100, textTransform: "uppercase" }}>{p.confidence}</span>
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {/* Disclaimer */}
                    <div style={{
                        marginTop: 24, padding: "14px 16px",
                        background: c.ambL, border: `1px solid ${c.ambB}`,
                    }}>
                        <p style={{ fontSize: 12, color: c.amber, margin: 0, lineHeight: 1.6 }}>
                            <strong>⚠ Research use only.</strong> AI-generated predictions should not replace professional clinical judgment.
                        </p>
                    </div>
                </div>
            </div>
        </>
    );
}