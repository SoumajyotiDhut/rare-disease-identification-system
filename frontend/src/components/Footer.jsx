import { Link } from "react-router-dom";
import { useTheme } from "../context/ThemeContext";

const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&display=swap');
  .fl:hover { opacity:0.8 }
  .fnl:hover { opacity:0.85 }
  @media(max-width:768px){
    .fg { grid-template-columns:1fr 1fr!important }
    .fp { padding:48px 20px 24px!important }
    .fb { flex-direction:column!important; gap:12px!important; text-align:center }
  }
  @media(max-width:480px){ .fg { grid-template-columns:1fr!important } }
`;

function Footer() {
    const { c, dark } = useTheme();
    // Footer always stays on the dark navy regardless of theme, for a consistent
    // anchor — but in dark mode we shift it slightly to match the page tone.
    const bg = dark ? c.footerBg : "#0F1C2E";

    return (
        <footer style={{ background: bg, fontFamily: "'Inter',sans-serif" }}>
            <style>{CSS}</style>
            <div style={{ height: 3, background: `linear-gradient(90deg,${c.teal},${c.blue},${c.purple})` }} />
            <div className="fp" style={{ maxWidth: 1200, margin: "0 auto", padding: "56px 32px 32px" }}>
                <div className="fg" style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr 1fr 1fr", gap: 48, marginBottom: 52, paddingBottom: 48, borderBottom: "1px solid rgba(255,255,255,0.07)" }}>
                    <div>
                        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
                            <div style={{ width: 36, height: 36, borderRadius: 10, background: `linear-gradient(135deg,${c.teal},${c.blue})`, display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 800, fontSize: 13, color: "#fff" }}>AI</div>
                            <span style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontWeight: 800, fontSize: 17, color: "#F8FAFC" }}>AI DOC</span>
                        </div>
                        <p style={{ fontSize: 14, color: "#7A94A8", lineHeight: 1.8, margin: "0 0 24px", maxWidth: 220 }}>AI-powered rare disease detection for clinical and research use worldwide.</p>
                        <div style={{ display: "flex", gap: 8 }}>
                            {["🔬", "🧬", "🩺"].map((ic, i) => (
                                <div key={i} style={{ width: 36, height: 36, borderRadius: 9, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, cursor: "pointer", transition: "background .15s" }}
                                    onMouseEnter={e => e.currentTarget.style.background = `${c.teal}40`}
                                    onMouseLeave={e => e.currentTarget.style.background = "rgba(255,255,255,0.06)"}>{ic}</div>
                            ))}
                        </div>
                    </div>
                    <div>
                        <p style={{ fontSize: 10, fontWeight: 800, color: "#5A7184", textTransform: "uppercase", letterSpacing: 1.2, margin: "0 0 18px" }}>Platform</p>
                        {[{ to: "/predict", label: "Predict Disease" }, { to: "/dashboard", label: "Analytics" }, { to: "/history", label: "History" }, { to: "/", label: "Home" }].map(({ to, label }) => (
                            <Link key={to} to={to} className="fnl" style={{ display: "block", fontSize: 14, color: "#7A94A8", textDecoration: "none", marginBottom: 11, fontWeight: 500, transition: "opacity .15s" }}>{label}</Link>
                        ))}
                    </div>
                    <div>
                        <p style={{ fontSize: 10, fontWeight: 800, color: "#5A7184", textTransform: "uppercase", letterSpacing: 1.2, margin: "0 0 18px" }}>Models</p>
                        {[["Symptom Classifier", c.teal, "Active"], ["Image Detector", c.amber, "Active"], ["Fusion Model", c.purple, "Active"]].map(([n, col, s]) => (
                            <div key={n} style={{ marginBottom: 13 }}>
                                <p style={{ fontSize: 13, color: "#94A8BA", margin: "0 0 4px", fontWeight: 500 }}>{n}</p>
                                <span style={{ fontSize: 10, fontWeight: 800, color: col, background: `${col}22`, border: `1px solid ${col}44`, padding: "2px 10px", borderRadius: 100, letterSpacing: .6, textTransform: "uppercase" }}>{s}</span>
                            </div>
                        ))}
                    </div>
                    <div>
                        <p style={{ fontSize: 10, fontWeight: 800, color: "#5A7184", textTransform: "uppercase", letterSpacing: 1.2, margin: "0 0 18px" }}>Dataset</p>
                        <p style={{ fontSize: 13, color: "#94A8BA", margin: "0 0 8px", fontWeight: 500 }}>ZebraMap Dataset</p>
                        <span style={{ fontSize: 10, fontWeight: 800, color: c.teal, background: `${c.teal}22`, border: `1px solid ${c.teal}44`, padding: "3px 11px", borderRadius: 100, letterSpacing: .6, display: "inline-block", marginBottom: 18 }}>CC BY 4.0</span>
                        <div style={{ background: `${c.teal}15`, border: `1px solid ${c.teal}28`, borderRadius: 12, padding: "14px 16px" }}>
                            <p style={{ fontSize: 11, color: c.teal, fontWeight: 700, margin: "0 0 3px" }}>36,374 patient cases</p>
                            <p style={{ fontSize: 11, color: "#7A94A8", margin: 0 }}>1,374 rare diseases indexed</p>
                        </div>
                    </div>
                </div>
                <div className="fb" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <p style={{ fontSize: 12, color: "#46627A", margin: 0 }}>© 2026 AI DOC · Rare Disease Detection System</p>
                    <div style={{ display: "flex", gap: 22 }}>
                        {["Privacy Policy", "Terms of Use", "Contact"].map(l => (
                            <a key={l} href="#" className="fl" style={{ fontSize: 12, color: "#46627A", textDecoration: "none", transition: "opacity .15s" }}>{l}</a>
                        ))}
                    </div>
                </div>
            </div>
        </footer>
    );
}

export default Footer;