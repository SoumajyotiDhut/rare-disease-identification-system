import { Link, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";

function Navbar({ dark, setDark }) {
  const [time, setTime] = useState("");
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date().toLocaleTimeString("en-IN")), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => { setMenuOpen(false); }, [location]);

  const isActive = (path) => location.pathname === path;

  const navLinks = [
    { path: "/", label: "Home" },
    { path: "/predict", label: "Predict" },
    { path: "/dashboard", label: "Dashboard" },
    { path: "/history", label: "History" },
  ];

  return (
    <>
      <style>{`
        .nav-link:hover { color: #0B7B6F !important; background: #F0FAF9 !important; }
        .nav-toggle { display: none; }
        .nav-clock { display: flex; }
        .nav-links-desktop { display: flex; }
        .mobile-menu { display: none; }
        @media (max-width: 768px) {
          .nav-toggle { display: flex !important; }
          .nav-clock { display: none !important; }
          .nav-links-desktop { display: none !important; }
          .mobile-menu { display: flex !important; }
          .nav-dark-btn { display: none !important; }
        }
      `}</style>
      <header style={{
        position: "sticky", top: 0, zIndex: 200,
        background: scrolled ? "rgba(255,255,255,0.97)" : "#fff",
        backdropFilter: "blur(20px)",
        borderBottom: "1px solid #EDF2F6",
        boxShadow: scrolled ? "0 1px 24px rgba(15,28,46,0.07)" : "none",
        transition: "box-shadow 0.3s",
      }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", height: 68, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 24px" }}>

          {/* Logo */}
          <Link to="/" style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
            <div style={{
              width: 36, height: 36, borderRadius: 10, background: "#0B7B6F",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontWeight: 800, fontSize: 13, color: "#fff",
            }}>AI</div>
            <div>
              <div style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontWeight: 800, fontSize: 16, color: "#0F1C2E", letterSpacing: -0.3 }}>AI DOC</div>
              <div style={{ fontSize: 9, color: "#0B7B6F", letterSpacing: 1.5, textTransform: "uppercase", fontWeight: 700 }}>Rare Disease Assistant</div>
            </div>
          </Link>

          {/* Clock */}
          <div className="nav-clock" style={{
            fontFamily: "monospace", fontSize: 12, color: "#7A94A8",
            background: "#F4F8FB", border: "1px solid #E0EBF2", padding: "5px 12px",
            borderRadius: 8, letterSpacing: 1, fontWeight: 500,
          }}>{time}</div>

          {/* Desktop nav */}
          <nav className="nav-links-desktop" style={{ alignItems: "center", gap: 2 }}>
            {navLinks.map(({ path, label }) => (
              <Link key={path} to={path} className="nav-link" style={{
                textDecoration: "none", padding: "8px 15px", borderRadius: 9,
                fontSize: 14, fontWeight: isActive(path) ? 700 : 500,
                color: isActive(path) ? "#0B7B6F" : "#5A7184",
                background: isActive(path) ? "#EBF8F6" : "transparent",
                border: isActive(path) ? "1px solid #B2E8E2" : "1px solid transparent",
                transition: "all 0.15s",
              }}>{label}</Link>
            ))}
            <button className="nav-dark-btn" onClick={() => setDark(!dark)} style={{
              marginLeft: 8, padding: "7px 13px", borderRadius: 9,
              border: "1px solid #E0EBF2", background: "#F4F8FB",
              color: "#5A7184", cursor: "pointer", fontSize: 15, transition: "all 0.15s",
            }}>{dark ? "☀️" : "🌙"}</button>
          </nav>

          {/* Mobile hamburger */}
          <button className="nav-toggle" onClick={() => setMenuOpen(!menuOpen)} style={{
            display: "flex", flexDirection: "column", gap: 5,
            padding: "8px", border: "1px solid #E0EBF2", background: "#F4F8FB",
            borderRadius: 9, cursor: "pointer",
          }}>
            {[0, 1, 2].map(i => (
              <span key={i} style={{
                width: 20, height: 2, background: "#5A7184", borderRadius: 2, display: "block",
                transform: menuOpen && i === 0 ? "rotate(45deg) translate(5px,5px)" : menuOpen && i === 2 ? "rotate(-45deg) translate(5px,-5px)" : "none",
                opacity: menuOpen && i === 1 ? 0 : 1,
                transition: "all 0.2s",
              }} />
            ))}
          </button>
        </div>

        {/* Mobile menu */}
        {menuOpen && (
          <div className="mobile-menu" style={{
            flexDirection: "column", padding: "12px 24px 20px",
            borderTop: "1px solid #EDF2F6", background: "#fff", gap: 4,
          }}>
            {navLinks.map(({ path, label }) => (
              <Link key={path} to={path} style={{
                textDecoration: "none", padding: "12px 16px", borderRadius: 10,
                fontSize: 15, fontWeight: isActive(path) ? 700 : 500,
                color: isActive(path) ? "#0B7B6F" : "#0F1C2E",
                background: isActive(path) ? "#EBF8F6" : "transparent",
                display: "block", transition: "all 0.15s",
              }}>{label}</Link>
            ))}
            <div style={{ marginTop: 8, paddingTop: 12, borderTop: "1px solid #EDF2F6", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 12, color: "#9BB8CC", fontFamily: "monospace" }}>{time}</span>
              <button onClick={() => setDark(!dark)} style={{
                padding: "7px 14px", borderRadius: 9, border: "1px solid #E0EBF2",
                background: "#F4F8FB", color: "#5A7184", cursor: "pointer", fontSize: 15,
              }}>{dark ? "☀️" : "🌙"}</button>
            </div>
          </div>
        )}
      </header>
    </>
  );
}

export default Navbar;