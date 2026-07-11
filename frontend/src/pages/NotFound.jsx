import { Link } from "react-router-dom";
import { useTheme } from "../context/ThemeContext";

const CSS = (c) => `
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
  @keyframes fadeUp   { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
  @keyframes drawLine { from{stroke-dashoffset:600} to{stroke-dashoffset:0} }
  @keyframes flatline { 0%,100%{opacity:1} 50%{opacity:.35} }

  .eyebrow {
    display:inline-flex; align-items:center; gap:10px;
    font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:600;
    color:${c.gold}; letter-spacing:0.14em; text-transform:uppercase;
  }
  .eyebrow::before { content:''; width:20px; height:1px; background:${c.gold}; display:inline-block; }

  .nf-btn {
    display:inline-flex; align-items:center; gap:10px;
    padding:15px 28px; border-radius:4px; border:none;
    background:${c.text}; color:${c.bg}; font-size:14.5px;
    font-weight:600; cursor:pointer; text-decoration:none;
    font-family:'Inter',sans-serif; transition:all .25s;
  }
  .nf-btn:hover { background:${c.teal}; color:#fff; transform:translateY(-2px); box-shadow:${c.shadowTeal}; }

  .nf-btn-outline {
    display:inline-flex; align-items:center; gap:10px;
    padding:14px 26px; border-radius:4px;
    border:1px solid ${c.borderI}; background:transparent; color:${c.text};
    font-size:14px; font-weight:600; cursor:pointer; text-decoration:none;
    font-family:'Inter',sans-serif; transition:all .25s;
  }
  .nf-btn-outline:hover { border-color:${c.teal}; color:${c.teal}; background:${c.tealL}; }

  @media(max-width:560px){
    .nf-pad   { padding:60px 20px!important }
    .nf-code  { font-size:64px!important }
    .nf-row   { flex-direction:column!important; align-items:stretch!important }
    .nf-btn, .nf-btn-outline { width:100%!important; justify-content:center!important }
  }
`;

export default function NotFound() {
    const { c } = useTheme();

    return (
        <div style={{ minHeight: "100vh", background: c.bg, fontFamily: "'Inter',sans-serif", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <style>{CSS(c)}</style>

            <div className="nf-pad" style={{ maxWidth: 560, margin: "0 auto", padding: "80px 32px", textAlign: "center", animation: "fadeUp .5s cubic-bezier(.2,.7,.3,1) both" }}>

                <span className="eyebrow" style={{ justifyContent: "center", marginBottom: 24 }}>Signal Lost</span>

                {/* Flatline monitor illustration */}
                <svg viewBox="0 0 320 90" width="100%" style={{ maxWidth: 300, margin: "0 auto 8px", display: "block", height: "auto" }}>
                    <path
                        d="M0 45H90L104 15L128 78L146 45H320"
                        fill="none" stroke={c.red} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
                        strokeDasharray="600" style={{ animation: "drawLine 1.2s ease forwards, flatline 2.4s ease-in-out 1.2s infinite" }}
                    />
                </svg>

                <p className="nf-code" style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 88, fontWeight: 600, color: c.text, margin: "8px 0 4px", letterSpacing: "-0.02em", lineHeight: 1 }}>
                    404
                </p>

                <h1 style={{ fontFamily: "'Fraunces',serif", fontSize: 26, fontWeight: 600, color: c.text, margin: "10px 0 14px", letterSpacing: "-0.02em" }}>
                    This page didn't make it to differential diagnosis
                </h1>

                <p style={{ fontSize: 14.5, color: c.sub, margin: "0 0 36px", lineHeight: 1.7, maxWidth: 420, marginLeft: "auto", marginRight: "auto" }}>
                    The page you're looking for doesn't exist, was moved, or the link is out of date.
                    Let's get you back to somewhere useful.
                </p>

                <div className="nf-row" style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
                    <Link to="/" className="nf-btn">Back to Home</Link>
                    <Link to="/predict" className="nf-btn-outline">Go to Predict</Link>
                </div>
            </div>
        </div>
    );
}