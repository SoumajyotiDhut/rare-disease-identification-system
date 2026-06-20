import { Link, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";
import { useTheme } from "../context/ThemeContext";

const CSS = (c) => `
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&display=swap');
  .nav-link:hover { color:${c.teal}!important; background:${c.tealL}!important }
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

function Navbar() {
  const { dark, setDark, c } = useTheme();
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
  const links = [
    { path: "/", label: "Home" },
    { path: "/predict", label: "Predict" },
    { path: "/dashboard", label: "Dashboard" },
    { path: "/history", label: "History" },
  ];

  return (
    <>
      <style>{CSS(c)}</style>
      <header style={{
        position: "sticky", top: 0, zIndex: 300,
        background: scrolled ? `${c.navBg}F7` : c.navBg,
        backdropFilter: "blur(20px)",
        borderBottom: `1px solid ${c.border}`,
        boxShadow: scrolled ? "0 1px 24px rgba(0,0,0,0.12)" : "none",
        transition: "background .25s, box-shadow .3s, border-color .25s",
      }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", height: 68, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 24px" }}>

          {/* Logo */}
          <Link to="/" style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 36, height: 36, borderRadius: 10, background: `linear-gradient(135deg,${c.teal},${c.blue})`, display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 800, fontSize: 13, color: "#fff", boxShadow: `0 4px 12px ${c.teal}40` }}>AI</div>
            <div>
              <div style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontWeight: 800, fontSize: 16, color: c.text, letterSpacing: -.3 }}>AI DOC</div>
              <div style={{ fontSize: 9, color: c.teal, letterSpacing: 1.5, textTransform: "uppercase", fontWeight: 700 }}>Rare Disease Assistant</div>
            </div>
          </Link>

          {/* Clock */}
          <div className="clock" style={{ fontFamily: "monospace", fontSize: 12, color: c.subAlt, background: c.cardAlt, border: `1px solid ${c.border}`, padding: "5px 12px", borderRadius: 8, letterSpacing: 1, fontWeight: 500 }}>{time}</div>

          {/* Desktop nav */}
          <nav className="desktop-nav" style={{ alignItems: "center", gap: 2 }}>
            {links.map(({ path, label }) => (
              <Link key={path} to={path} className="nav-link" style={{
                textDecoration: "none", padding: "8px 15px", borderRadius: 9, fontSize: 14,
                fontWeight: isActive(path) ? 700 : 500,
                color: isActive(path) ? c.teal : c.sub,
                background: isActive(path) ? c.tealL : "transparent",
                border: isActive(path) ? `1px solid ${c.tealB}` : "1px solid transparent",
                transition: "all .15s",
              }}>{label}</Link>
            ))}
            <button onClick={() => setDark(!dark)} aria-label="Toggle dark mode" style={{
              marginLeft: 8, padding: "7px 13px", borderRadius: 9,
              border: `1px solid ${c.border}`, background: c.cardAlt,
              color: c.sub, cursor: "pointer", fontSize: 15, transition: "all .15s",
            }}>
              {dark ? "☀️" : "🌙"}
            </button>
          </nav>

          {/* Mobile hamburger */}
          <button className="mobile-toggle" onClick={() => setOpen(!open)} style={{
            flexDirection: "column", gap: 5, padding: "8px", border: `1px solid ${c.border}`,
            background: c.cardAlt, borderRadius: 9, cursor: "pointer",
          }}>
            {[0, 1, 2].map(i => (
              <span key={i} style={{
                width: 20, height: 2, background: c.sub, borderRadius: 2, display: "block",
                transform: open && i === 0 ? "rotate(45deg) translate(5px,5px)" : open && i === 2 ? "rotate(-45deg) translate(5px,-5px)" : "none",
                opacity: open && i === 1 ? 0 : 1, transition: "all .2s",
              }} />
            ))}
          </button>
        </div>

        {/* Mobile dropdown */}
        {open && (
          <div className="mobile-menu" style={{ background: c.navBg, borderTop: `1px solid ${c.border}`, padding: "12px 20px 20px" }}>
            {links.map(({ path, label }) => (
              <Link key={path} to={path} style={{
                display: "block", padding: "12px 16px", borderRadius: 10, fontSize: 15,
                fontWeight: isActive(path) ? 700 : 500, color: isActive(path) ? c.teal : c.text,
                background: isActive(path) ? c.tealL : "transparent", textDecoration: "none",
                marginBottom: 4, transition: "all .15s",
              }}>{label}</Link>
            ))}
            <div style={{ marginTop: 10, paddingTop: 14, borderTop: `1px solid ${c.border}`, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 12, color: c.muted, fontFamily: "monospace" }}>{time}</span>
              <button onClick={() => setDark(!dark)} style={{ padding: "7px 14px", borderRadius: 9, border: `1px solid ${c.border}`, background: c.cardAlt, color: c.sub, cursor: "pointer", fontSize: 15 }}>{dark ? "☀️" : "🌙"}</button>
            </div>
          </div>
        )}
      </header>
    </>
  );
}

export default Navbar;