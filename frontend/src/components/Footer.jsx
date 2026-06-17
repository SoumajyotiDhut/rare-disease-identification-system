import { Link } from "react-router-dom";

function Footer() {
    return (
        <footer style={{ background: "#0F1C2E", fontFamily: "'Inter',sans-serif" }}>
            <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&display=swap" rel="stylesheet" />
            <style>{`
        .footer-link:hover{color:#4ABFB5!important}
        .footer-nav-link:hover{color:#F8FAFB!important}
        @media(max-width:768px){
          .footer-grid{grid-template-columns:1fr 1fr!important}
          .footer-pad{padding:48px 20px 24px!important}
        }
        @media(max-width:480px){
          .footer-grid{grid-template-columns:1fr!important}
          .footer-bottom{flex-direction:column!important;gap:12px!important;text-align:center}
        }
      `}</style>

            {/* Top accent bar */}
            <div style={{ height: 3, background: "linear-gradient(90deg,#0B7B6F,#1D6FA4,#5B3DB8)" }} />

            <div className="footer-pad" style={{ maxWidth: 1200, margin: "0 auto", padding: "56px 32px 32px" }}>
                {/* Top row */}
                <div className="footer-grid" style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr 1fr 1fr", gap: 48, marginBottom: 52, paddingBottom: 48, borderBottom: "1px solid rgba(255,255,255,0.07)" }}>

                    {/* Brand */}
                    <div>
                        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
                            <div style={{ width: 36, height: 36, borderRadius: 10, background: "#0B7B6F", display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 800, fontSize: 13, color: "#fff" }}>AI</div>
                            <span style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontWeight: 800, fontSize: 17, color: "#F8FAFC" }}>AI DOC</span>
                        </div>
                        <p style={{ fontSize: 14, color: "#5A7184", lineHeight: 1.75, margin: "0 0 24px", maxWidth: 220 }}>AI-powered rare disease detection for clinical and research use worldwide.</p>
                        <div style={{ display: "flex", gap: 10 }}>
                            {["🔬", "🧬", "🩺"].map((icon, i) => (
                                <div key={i} style={{ width: 36, height: 36, borderRadius: 9, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, cursor: "pointer", transition: "background 0.15s" }}
                                    onMouseEnter={e => e.currentTarget.style.background = "rgba(11,123,111,0.2)"}
                                    onMouseLeave={e => e.currentTarget.style.background = "rgba(255,255,255,0.06)"}
                                >{icon}</div>
                            ))}
                        </div>
                    </div>

                    {/* Platform */}
                    <div>
                        <p style={{ fontSize: 10, fontWeight: 800, color: "#4A6275", textTransform: "uppercase", letterSpacing: 1.2, margin: "0 0 18px" }}>Platform</p>
                        {[{ to: "/predict", label: "Predict Disease" }, { to: "/dashboard", label: "Analytics" }, { to: "/history", label: "History" }, { to: "/", label: "Home" }].map(({ to, label }) => (
                            <Link key={to} to={to} className="footer-nav-link" style={{ display: "block", fontSize: 14, color: "#5A7184", textDecoration: "none", marginBottom: 11, fontWeight: 500, transition: "color 0.15s" }}>{label}</Link>
                        ))}
                    </div>

                    {/* Models */}
                    <div>
                        <p style={{ fontSize: 10, fontWeight: 800, color: "#4A6275", textTransform: "uppercase", letterSpacing: 1.2, margin: "0 0 18px" }}>Models</p>
                        {[["Symptom Classifier", "Active", "#0B7B6F", "#EBF8F6"], ["Image Detector", "Training", "#C05B1A", "#FFF4EC"], ["Fusion Model", "Coming Soon", "#5B3DB8", "#F2EEF9"]].map(([name, status, color, bg]) => (
                            <div key={name} style={{ marginBottom: 12 }}>
                                <p style={{ fontSize: 13, color: "#7A94A8", margin: "0 0 4px", fontWeight: 500 }}>{name}</p>
                                <span style={{ fontSize: 10, fontWeight: 800, color, background: "rgba(255,255,255,0.04)", border: `1px solid ${color}44`, padding: "2px 10px", borderRadius: 100, letterSpacing: 0.6, textTransform: "uppercase" }}>{status}</span>
                            </div>
                        ))}
                    </div>

                    {/* Dataset */}
                    <div>
                        <p style={{ fontSize: 10, fontWeight: 800, color: "#4A6275", textTransform: "uppercase", letterSpacing: 1.2, margin: "0 0 18px" }}>Dataset</p>
                        <p style={{ fontSize: 13, color: "#7A94A8", margin: "0 0 8px", fontWeight: 500 }}>ZebraMap Dataset</p>
                        <span style={{ fontSize: 10, fontWeight: 800, color: "#0B7B6F", background: "rgba(11,123,111,0.12)", border: "1px solid rgba(11,123,111,0.25)", padding: "3px 11px", borderRadius: 100, letterSpacing: 0.6, display: "inline-block", marginBottom: 18 }}>CC BY 4.0</span>
                        <div style={{ background: "rgba(11,123,111,0.08)", border: "1px solid rgba(11,123,111,0.15)", borderRadius: 12, padding: "14px 16px" }}>
                            <p style={{ fontSize: 11, color: "#4ABFB5", fontWeight: 700, margin: "0 0 4px" }}>36,374 patient cases</p>
                            <p style={{ fontSize: 11, color: "#4A6275", margin: 0 }}>1,374 rare diseases indexed</p>
                        </div>
                    </div>
                </div>

                {/* Bottom */}
                <div className="footer-bottom" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <p style={{ fontSize: 13, color: "#334155", margin: 0 }}>© 2026 AI DOC · Rare Disease Detection System</p>
                    <div style={{ display: "flex", gap: 24 }}>
                        {["Privacy Policy", "Terms of Use", "Contact"].map(l => (
                            <a key={l} href="#" className="footer-link" style={{ fontSize: 12, color: "#334155", textDecoration: "none", transition: "color 0.15s" }}>{l}</a>
                        ))}
                    </div>
                </div>
            </div>
        </footer>
    );
}

export default Footer;