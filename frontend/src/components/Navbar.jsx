import { Link, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";

const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&display=swap');
  .nav-link:hover { color:#0B7B6F!important; background:#F0FAF9!important }
  .clock          { display:flex }
  .desktop-nav    { display:flex }
  .mobile-toggle  { display:none!important }
  .mobile-menu    { display:none }
  @media(max-width:768px){
    .clock         { display:none!important }
    .desktop-nav   { display:none!important }
    .mobile-toggle { display:flex!important }
    .mobile-menu   { display:block }
  }
`;

function Navbar({ dark, setDark }) {
  const [time, setTime] = useState("");
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const t = setInterval(() => setTime(new Date().toLocaleTimeString("en-IN")), 1000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", fn);
    return () => window.removeEventListener("scroll", fn);
  }, []);
  useEffect(() => setOpen(false), [location]);

  const isActive = p => location.pathname === p;
  const links = [{ path: "/", label: "Home" }, { path: "/predict", label: "Predict" }, { path: "/dashboard", label: "Dashboard" }, { path: "/history", label: "History" }];

  return (
    <>
      <style>{CSS}</style>
      <header style={{
        position: "sticky", top: 0, zIndex: 300,
        background: scrolled ? "rgba(255,255,255,0.97)" : "#fff",
        backdropFilter: "blur(20px)",
        borderBottom: "1px solid #EDF2F6",
        boxShadow: scrolled ? "0 1px 24px rgba(15,28,46,0.07)" : "none",
        transition: "box-shadow .3s",
      }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", height: 68, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 24px" }}>

          {/* Logo */}
          <Link to="/" style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 36, height: 36, borderRadius: 10, background: "linear-gradient(135deg,#0B7B6F,#1D6FA4)", display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 800, fontSize: 13, color: "#fff", boxShadow: "0 4px 12px rgba(11,123,111,0.25)" }}>AI</div>
            <div>
              <div style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontWeight: 800, fontSize: 16, color: "#0F1C2E", letterSpacing: -.3 }}>AI DOC</div>
              <div style={{ fontSize: 9, color: "#0B7B6F", letterSpacing: 1.5, textTransform: "uppercase", fontWeight: 700 }}>Rare Disease Assistant</div>
            </div>
          </Link>

          {/* Clock */}
          <div className="clock" style={{ fontFamily: "monospace", fontSize: 12, color: "#7A94A8", background: "#F4F8FB", border: "1px solid #E0EBF2", padding: "5px 12px", borderRadius: 8, letterSpacing: 1, fontWeight: 500 }}>{time}</div>

          {/* Desktop nav */}
          <nav className="desktop-nav" style={{ alignItems: "center", gap: 2 }}>
            {links.map(({ path, label }) => (
              <Link key={path} to={path} className="nav-link" style={{
                textDecoration: "none", padding: "8px 15px", borderRadius: 9, fontSize: 14,
                fontWeight: isActive(path) ? 700 : 500,
                color: isActive(path) ? "#0B7B6F" : "#5A7184",
                background: isActive(path) ? "#EBF8F6" : "transparent",
                border: isActive(path) ? "1px solid #B2E8E2" : "1px solid transparent",
                transition: "all .15s",
              }}>{label}</Link>
            ))}
            <button onClick={() => setDark(!dark)} style={{
              marginLeft: 8, padding: "7px 13px", borderRadius: 9,
              border: "1px solid #E0EBF2", background: "#F4F8FB",
              color: "#5A7184", cursor: "pointer", fontSize: 15, transition: "all .15s",
            }}
              onMouseEnter={e => { e.currentTarget.style.background = "#EDF2F6"; e.currentTarget.style.borderColor = "#C8D8E4"; }}
              onMouseLeave={e => { e.currentTarget.style.background = "#F4F8FB"; e.currentTarget.style.borderColor = "#E0EBF2"; }}>
              {dark ? "☀️" : "🌙"}
            </button>
          </nav>

          {/* Mobile hamburger */}
          <button className="mobile-toggle" onClick={() => setOpen(!open)} style={{
            flexDirection: "column", gap: 5, padding: "8px", border: "1px solid #E0EBF2",
            background: "#F4F8FB", borderRadius: 9, cursor: "pointer",
          }}>
            {[0, 1, 2].map(i => (
              <span key={i} style={{
                width: 20, height: 2, background: "#5A7184", borderRadius: 2, display: "block",
                transform: open && i === 0 ? "rotate(45deg) translate(5px,5px)" : open && i === 2 ? "rotate(-45deg) translate(5px,-5px)" : "none",
                opacity: open && i === 1 ? 0 : 1, transition: "all .2s",
              }} />
            ))}
          </button>
        </div>

        {/* Mobile dropdown */}
        {open && (
          <div className="mobile-menu" style={{ background: "#fff", borderTop: "1px solid #EDF2F6", padding: "12px 20px 20px" }}>
            {links.map(({ path, label }) => (
              <Link key={path} to={path} style={{
                display: "block", padding: "12px 16px", borderRadius: 10, fontSize: 15,
                fontWeight: isActive(path) ? 700 : 500, color: isActive(path) ? "#0B7B6F" : "#0F1C2E",
                background: isActive(path) ? "#EBF8F6" : "transparent", textDecoration: "none",
                marginBottom: 4, transition: "all .15s",
              }}>{label}</Link>
            ))}
            <div style={{ marginTop: 10, paddingTop: 14, borderTop: "1px solid #EDF2F6", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 12, color: "#9BB8CC", fontFamily: "monospace" }}>{time}</span>
              <button onClick={() => setDark(!dark)} style={{ padding: "7px 14px", borderRadius: 9, border: "1px solid #E0EBF2", background: "#F4F8FB", color: "#5A7184", cursor: "pointer", fontSize: 15 }}>{dark ? "☀️" : "🌙"}</button>
            </div>
          </div>
        )}
      </header>
    </>
  );
}

export default Navbar;
ENDOFFILE

cat > /home/claude / Footer.jsx << 'ENDOFFILE'
import { Link } from "react-router-dom";

const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&display=swap');
  .fl:hover { color:#4ABFB5!important }
  .fnl:hover { color:#F8FAFC!important }
  @media(max-width:768px){
    .fg { grid-template-columns:1fr 1fr!important }
    .fp { padding:48px 20px 24px!important }
    .fb { flex-direction:column!important; gap:12px!important; text-align:center }
  }
  @media(max-width:480px){ .fg { grid-template-columns:1fr!important } }
`;

function Footer() {
  return (
    <footer style={{ background: "#0F1C2E", fontFamily: "'Inter',sans-serif" }}>
      <style>{CSS}</style>
      <div style={{ height: 3, background: "linear-gradient(90deg,#0B7B6F,#1D6FA4,#5B3DB8)" }} />
      <div className="fp" style={{ maxWidth: 1200, margin: "0 auto", padding: "56px 32px 32px" }}>
        <div className="fg" style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr 1fr 1fr", gap: 48, marginBottom: 52, paddingBottom: 48, borderBottom: "1px solid rgba(255,255,255,0.07)" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
              <div style={{ width: 36, height: 36, borderRadius: 10, background: "linear-gradient(135deg,#0B7B6F,#1D6FA4)", display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 800, fontSize: 13, color: "#fff" }}>AI</div>
              <span style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontWeight: 800, fontSize: 17, color: "#F8FAFC" }}>AI DOC</span>
            </div>
            <p style={{ fontSize: 14, color: "#4A6275", lineHeight: 1.8, margin: "0 0 24px", maxWidth: 220 }}>AI-powered rare disease detection for clinical and research use worldwide.</p>
            <div style={{ display: "flex", gap: 8 }}>
              {["🔬", "🧬", "🩺"].map((ic, i) => (
                <div key={i} style={{ width: 36, height: 36, borderRadius: 9, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, cursor: "pointer", transition: "background .15s" }}
                  onMouseEnter={e => e.currentTarget.style.background = "rgba(11,123,111,0.25)"}
                  onMouseLeave={e => e.currentTarget.style.background = "rgba(255,255,255,0.06)"}>{ic}</div>
              ))}
            </div>
          </div>
          <div>
            <p style={{ fontSize: 10, fontWeight: 800, color: "#4A6275", textTransform: "uppercase", letterSpacing: 1.2, margin: "0 0 18px" }}>Platform</p>
            {[{ to: "/predict", label: "Predict Disease" }, { to: "/dashboard", label: "Analytics" }, { to: "/history", label: "History" }, { to: "/", label: "Home" }].map(({ to, label }) => (
              <Link key={to} to={to} className="fnl" style={{ display: "block", fontSize: 14, color: "#4A6275", textDecoration: "none", marginBottom: 11, fontWeight: 500, transition: "color .15s" }}>{label}</Link>
            ))}
          </div>
          <div>
            <p style={{ fontSize: 10, fontWeight: 800, color: "#4A6275", textTransform: "uppercase", letterSpacing: 1.2, margin: "0 0 18px" }}>Models</p>
            {[["Symptom Classifier", "#0B7B6F", "Active"], ["Image Detector", "#C05B1A", "Training"], ["Fusion Model", "#5B3DB8", "Soon"]].map(([n, c, s]) => (
              <div key={n} style={{ marginBottom: 13 }}>
                <p style={{ fontSize: 13, color: "#7A94A8", margin: "0 0 4px", fontWeight: 500 }}>{n}</p>
                <span style={{ fontSize: 10, fontWeight: 800, color: c, background: `${c}22`, border: `1px solid ${c}44`, padding: "2px 10px", borderRadius: 100, letterSpacing: .6, textTransform: "uppercase" }}>{s}</span>
              </div>
            ))}
          </div>
          <div>
            <p style={{ fontSize: 10, fontWeight: 800, color: "#4A6275", textTransform: "uppercase", letterSpacing: 1.2, margin: "0 0 18px" }}>Dataset</p>
            <p style={{ fontSize: 13, color: "#7A94A8", margin: "0 0 8px", fontWeight: 500 }}>ZebraMap Dataset</p>
            <span style={{ fontSize: 10, fontWeight: 800, color: "#0B7B6F", background: "rgba(11,123,111,0.12)", border: "1px solid rgba(11,123,111,0.25)", padding: "3px 11px", borderRadius: 100, letterSpacing: .6, display: "inline-block", marginBottom: 18 }}>CC BY 4.0</span>
            <div style={{ background: "rgba(11,123,111,0.07)", border: "1px solid rgba(11,123,111,0.15)", borderRadius: 12, padding: "14px 16px" }}>
              <p style={{ fontSize: 11, color: "#4ABFB5", fontWeight: 700, margin: "0 0 3px" }}>36,374 patient cases</p>
              <p style={{ fontSize: 11, color: "#4A6275", margin: 0 }}>1,374 rare diseases indexed</p>
            </div>
          </div>
        </div>
        <div className="fb" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <p style={{ fontSize: 12, color: "#334155", margin: 0 }}>© 2026 AI DOC · Rare Disease Detection System</p>
          <div style={{ display: "flex", gap: 22 }}>
            {["Privacy Policy", "Terms of Use", "Contact"].map(l => (
              <a key={l} href="#" className="fl" style={{ fontSize: 12, color: "#334155", textDecoration: "none", transition: "color .15s" }}>{l}</a>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
ENDOFFILE

cat > /home/claude / PredictionCard.jsx << 'ENDOFFILE'
const P = [
  { bar: "#0B7B6F", light: "#EBF8F6", text: "#0B7B6F", border: "#B2E8E2" },
  { bar: "#1D6FA4", light: "#EBF4F9", text: "#1D6FA4", border: "#B3D8EE" },
  { bar: "#5B3DB8", light: "#F2EEF9", text: "#5B3DB8", border: "#C8B8EC" },
  { bar: "#C05B1A", light: "#FFF4EC", text: "#C05B1A", border: "#F5D8B8" },
  { bar: "#8FA5B5", light: "#F0F5F8", text: "#5A7184", border: "#C8D8E4" },
];
const PredictionCard = ({ item }) => {
  const c = P[(item.rank - 1) % P.length];
  return (
    <div style={{ background: "#fff", border: `1px solid ${c.border}`, borderRadius: 18, padding: "22px 24px", transition: "box-shadow .2s, transform .15s", cursor: "default" }}
      onMouseEnter={e => { e.currentTarget.style.boxShadow = "0 8px 28px rgba(15,28,46,0.08)"; e.currentTarget.style.transform = "translateY(-3px)" }}
      onMouseLeave={e => { e.currentTarget.style.boxShadow = "none"; e.currentTarget.style.transform = "none" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
        <span style={{ width: 32, height: 32, borderRadius: 9, background: c.light, border: `1px solid ${c.border}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 800, color: c.text, flexShrink: 0 }}>#{item.rank}</span>
        <h2 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 15, fontWeight: 700, color: "#0F1C2E", margin: 0 }}>{item.disease}</h2>
      </div>
      <div style={{ marginBottom: 13 }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 7 }}>
          <span style={{ color: "#8FA5B5", fontWeight: 500 }}>Probability</span>
          <span style={{ color: c.text, fontWeight: 800 }}>{item.probability}%</span>
        </div>
        <div style={{ height: 6, background: "#F0F5F8", borderRadius: 100 }}>
          <div style={{ height: 6, width: `${item.probability}%`, background: c.bar, borderRadius: 100, transition: "width .9s ease" }} />
        </div>
      </div>
      <span style={{ fontSize: 10, fontWeight: 800, padding: "4px 12px", borderRadius: 100, background: c.light, color: c.text, border: `1px solid ${c.border}`, textTransform: "uppercase", letterSpacing: .7 }}>{item.confidence}</span>
    </div>
  );
};
export default PredictionCard;
ENDOFFILE

cat > /home/claude / Loader.jsx << 'ENDOFFILE'
function Loader({ message = "Loading…" }) {
  return (
    <div style={{ minHeight: "100vh", background: "#F4F8FB", display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 24, fontFamily: "'Inter',sans-serif" }}>
      <div style={{ position: "relative", width: 64, height: 64 }}>
        <div style={{ position: "absolute", inset: 0, border: "3px solid #E8EFF5", borderTop: "3px solid #0B7B6F", borderRadius: "50%", animation: "spin .9s linear infinite" }} />
        <div style={{ position: "absolute", inset: 10, border: "2px solid #EDF2F6", borderTop: "2px solid #1D6FA4", borderRadius: "50%", animation: "spin 1.5s linear infinite reverse" }} />
        <div style={{ position: "absolute", inset: "50%", transform: "translate(-50%,-50%)", width: 12, height: 12, borderRadius: "50%", background: "linear-gradient(135deg,#0B7B6F,#1D6FA4)" }} />
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