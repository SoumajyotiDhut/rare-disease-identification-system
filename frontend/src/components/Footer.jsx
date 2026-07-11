import { Link } from "react-router-dom";
import { useTheme } from "../context/ThemeContext";

const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Mono:wght@500;600&display=swap');
  .fl:hover  { opacity:.75!important }
  .fnl:hover { color:#94A3B8!important }
  @media(max-width:768px){
    .fg  { grid-template-columns:1fr 1fr!important }
    .fp  { padding:48px 24px 28px!important }
    .fbot{ flex-direction:column!important; gap:14px!important; text-align:center }
  }
  @media(max-width:480px){
    .fg { grid-template-columns:1fr!important; gap:36px!important }
    .fp { padding:40px 20px 24px!important }
    .flinks { justify-content:center!important; flex-wrap:wrap!important; gap:14px 20px!important }
  }
`;

export default function Footer() {
    const { c } = useTheme();

    return (
        <footer style={{ background: "#060D1A", fontFamily: "'Inter',sans-serif" }}>
            <style>{CSS}</style>

            {/* Top hairline */}
            <div style={{ height: 1, background: "linear-gradient(90deg, transparent, rgba(203,175,116,0.5), transparent)" }} />

            <div className="fp" style={{ maxWidth: 1200, margin: "0 auto", padding: "64px 32px 36px" }}>
                <div className="fg" style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr 1fr 1fr", gap: 52, marginBottom: 56, paddingBottom: 48, borderBottom: "1px solid rgba(255,255,255,0.06)" }}>

                    {/* Brand */}
                    <div>
                        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 18 }}>
                            <div style={{ width: 36, height: 36, borderRadius: 4, background: c.gradPrimary, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'Fraunces',serif", fontStyle: "italic", fontWeight: 600, fontSize: 14, color: "#fff" }}>AI</div>
                            <span style={{ fontFamily: "'Fraunces',serif", fontWeight: 600, fontSize: 18, color: "#F1F5F9", letterSpacing: "-0.02em" }}>AI DOC</span>
                        </div>
                        <p style={{ fontSize: 13.5, color: "#64748B", lineHeight: 1.8, margin: "0 0 24px", maxWidth: 230 }}>
                            AI-powered multimodal rare disease identification built on 36,487 real patient cases from ZebraMap.
                        </p>
                        <div style={{ display: "flex", gap: 8 }}>
                            {[{ ic: "R", label: "Research" }, { ic: "AI", label: "Machine Learning" }, { ic: "C", label: "Clinical" }].map(({ ic, label }) => (
                                <div key={label} title={label} style={{
                                    height: 30, borderRadius: 4, padding: "0 12px",
                                    background: "rgba(255,255,255,0.04)",
                                    border: "1px solid rgba(255,255,255,0.08)",
                                    display: "flex", alignItems: "center", justifyContent: "center",
                                    fontFamily: "'IBM Plex Mono',monospace", fontSize: 10.5, fontWeight: 600,
                                    color: "#8A93A3", letterSpacing: "0.04em",
                                }}>{ic}</div>
                            ))}
                        </div>
                    </div>

                    {/* Platform links */}
                    <div>
                        <p style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 10, fontWeight: 600, color: "#8A7B5C", textTransform: "uppercase", letterSpacing: "0.12em", margin: "0 0 18px" }}>Platform</p>
                        {[{ to: "/predict", label: "Predict Disease" }, { to: "/dashboard", label: "Dashboard" }, { to: "/history", label: "History" }, { to: "/", label: "Home" }].map(({ to, label }) => (
                            <Link key={to} to={to} className="fnl" style={{ display: "block", fontSize: 13.5, color: "#475569", textDecoration: "none", marginBottom: 12, fontWeight: 500, transition: "color .15s" }}>{label}</Link>
                        ))}
                    </div>

                    {/* Models */}
                    <div>
                        <p style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 10, fontWeight: 600, color: "#8A7B5C", textTransform: "uppercase", letterSpacing: "0.12em", margin: "0 0 18px" }}>Models</p>
                        {[
                            { name: "TF-IDF + LR", status: "Active", color: c.teal },
                            { name: "EfficientNet-B4", status: "Active", color: c.blue },
                            { name: "Late Fusion", status: "Active", color: c.purple },
                            { name: "FastGAN", status: "Research", color: c.amber },
                        ].map(({ name, status, color }) => (
                            <div key={name} style={{ marginBottom: 13 }}>
                                <p style={{ fontSize: 13, color: "#475569", margin: "0 0 4px", fontWeight: 500 }}>{name}</p>
                                <span style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 9.5, fontWeight: 600, color, background: `${color}1A`, border: `1px solid ${color}44`, padding: "2px 9px", borderRadius: 100, letterSpacing: "0.06em", textTransform: "uppercase" }}>{status}</span>
                            </div>
                        ))}
                    </div>

                    {/* Dataset */}
                    <div>
                        <p style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 10, fontWeight: 600, color: "#8A7B5C", textTransform: "uppercase", letterSpacing: "0.12em", margin: "0 0 18px" }}>Dataset</p>
                        <p style={{ fontSize: 13.5, color: "#475569", margin: "0 0 8px", fontWeight: 600 }}>ZebraMap</p>
                        <span style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 9.5, fontWeight: 600, color: c.teal, background: `${c.teal}1A`, border: `1px solid ${c.teal}44`, padding: "3px 10px", borderRadius: 100, display: "inline-block", marginBottom: 18, letterSpacing: "0.06em" }}>CC BY 4.0</span>
                        <div style={{ background: "rgba(51,179,159,0.07)", border: "1px solid rgba(51,179,159,0.18)", borderRadius: 4, padding: "16px 18px" }}>
                            <p style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 12, color: c.teal, fontWeight: 600, margin: "0 0 6px" }}>36,487 patient cases</p>
                            <p style={{ fontSize: 11.5, color: "#475569", margin: "0 0 4px" }}>1,374 rare diseases</p>
                            <p style={{ fontSize: 11.5, color: "#475569", margin: 0 }}>94,384 biomedical images</p>
                        </div>
                    </div>
                </div>

                {/* Bottom */}
                <div className="fbot" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <p style={{ fontSize: 12, color: "#1E293B", margin: 0 }}>
                        © 2026 AI DOC · Built for PROJ-IT781 · Techno Main Salt Lake, Kolkata
                    </p>
                    <div className="flinks" style={{ display: "flex", gap: 20 }}>
                        {/* TODO: Privacy/Terms currently point nowhere (#) — either build
                            those pages/routes or remove them if not needed for an academic project */}
                        <a href="#" className="fl" style={{ fontSize: 12, color: "#1E293B", textDecoration: "none", transition: "opacity .15s" }}>Privacy</a>
                        <a href="#" className="fl" style={{ fontSize: 12, color: "#1E293B", textDecoration: "none", transition: "opacity .15s" }}>Terms</a>
                        {/* TODO: replace with your real contact email */}
                        <a href="mailto:contact@aidoc.example" className="fl" style={{ fontSize: 12, color: "#1E293B", textDecoration: "none", transition: "opacity .15s" }}>Contact</a>
                    </div>
                </div>
            </div>
        </footer>
    );
}