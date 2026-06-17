import { Link } from "react-router-dom";

function Footer() {
    return (
        <footer style={{
            background: "#fff",
            borderTop: "1px solid #EDF2F6",
            fontFamily: "'Inter', sans-serif",
        }}>
            <div style={{
                maxWidth: 1200,
                margin: "0 auto",
                padding: "48px 40px 32px",
            }}>
                {/* Top row */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 40, marginBottom: 48 }}>
                    {/* Brand */}
                    <div>
                        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
                            <div style={{
                                width: 34, height: 34, borderRadius: 9,
                                background: "#0B7B6F",
                                display: "flex", alignItems: "center", justifyContent: "center",
                                fontWeight: 800, fontSize: 12, color: "#fff", letterSpacing: 0.3,
                            }}>AI</div>
                            <span style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif", fontWeight: 800, fontSize: 16, color: "#0F1C2E" }}>AI DOC</span>
                        </div>
                        <p style={{ fontSize: 13, color: "#8FA5B5", lineHeight: 1.7, margin: 0, maxWidth: 200 }}>
                            AI-powered rare disease detection for clinical and research use.
                        </p>
                    </div>

                    {/* Platform */}
                    <div>
                        <p style={{ fontSize: 11, fontWeight: 700, color: "#8FA5B5", textTransform: "uppercase", letterSpacing: 1, margin: "0 0 16px" }}>Platform</p>
                        {[
                            { to: "/predict", label: "Predict Disease" },
                            { to: "/dashboard", label: "Analytics Dashboard" },
                            { to: "/history", label: "Prediction History" },
                        ].map(({ to, label }) => (
                            <Link key={to} to={to} style={{
                                display: "block", fontSize: 14, color: "#5A7184", textDecoration: "none",
                                marginBottom: 10, fontWeight: 500, transition: "color 0.15s",
                            }}
                                onMouseEnter={e => e.currentTarget.style.color = "#0B7B6F"}
                                onMouseLeave={e => e.currentTarget.style.color = "#5A7184"}
                            >
                                {label}
                            </Link>
                        ))}
                    </div>

                    {/* Models */}
                    <div>
                        <p style={{ fontSize: 11, fontWeight: 700, color: "#8FA5B5", textTransform: "uppercase", letterSpacing: 1, margin: "0 0 16px" }}>Models</p>
                        {["Symptom Classifier", "Image Detection", "Fusion Model"].map(m => (
                            <p key={m} style={{ fontSize: 14, color: "#8FA5B5", margin: "0 0 10px" }}>{m}</p>
                        ))}
                    </div>

                    {/* Dataset */}
                    <div>
                        <p style={{ fontSize: 11, fontWeight: 700, color: "#8FA5B5", textTransform: "uppercase", letterSpacing: 1, margin: "0 0 16px" }}>Dataset</p>
                        <p style={{ fontSize: 14, color: "#8FA5B5", margin: "0 0 6px" }}>ZebraMap Dataset</p>
                        <span style={{
                            fontSize: 11, fontWeight: 700, color: "#0B7B6F",
                            background: "#EBF8F6", border: "1px solid #B2E8E2",
                            padding: "3px 10px", borderRadius: 100, letterSpacing: 0.6,
                        }}>CC BY 4.0</span>
                    </div>
                </div>

                {/* Bottom row */}
                <div style={{
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    paddingTop: 24, borderTop: "1px solid #EDF2F6",
                }}>
                    <p style={{ fontSize: 12, color: "#9BB8CC", margin: 0 }}>
                        © 2026 AI DOC · Rare Disease Detection System
                    </p>
                    <div style={{ display: "flex", gap: 20 }}>
                        {["Privacy Policy", "Terms of Use", "Contact"].map(l => (
                            <a key={l} href="#" style={{
                                fontSize: 12, color: "#9BB8CC", textDecoration: "none",
                                transition: "color 0.15s",
                            }}
                                onMouseEnter={e => e.currentTarget.style.color = "#0B7B6F"}
                                onMouseLeave={e => e.currentTarget.style.color = "#9BB8CC"}
                            >{l}</a>
                        ))}
                    </div>
                </div>
            </div>
        </footer>
    );
}

export default Footer;