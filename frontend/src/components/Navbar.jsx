import { Link, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";

function Navbar({ dark, setDark }) {
  const [time, setTime] = useState("");
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date().toLocaleTimeString("en-IN"));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const isActive = (path) => location.pathname === path;

  return (
    <header style={{
      position: "sticky",
      top: 0,
      zIndex: 100,
      transition: "all 0.3s ease",
      background: scrolled ? "rgba(248,250,251,0.96)" : "#fff",
      backdropFilter: "blur(16px)",
      borderBottom: scrolled ? "1px solid #DDE8EF" : "1px solid #EDF2F6",
      boxShadow: scrolled ? "0 2px 20px rgba(15,28,46,0.06)" : "none",
    }}>
      <div style={{
        maxWidth: 1200,
        margin: "0 auto",
        height: 68,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 40px",
      }}>
        {/* Logo */}
        <Link to="/" style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 38,
            height: 38,
            borderRadius: 10,
            background: "#0B7B6F",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontWeight: 800,
            fontSize: 13,
            color: "#fff",
            letterSpacing: 0.5,
            flexShrink: 0,
          }}>
            AI
          </div>
          <div>
            <div style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif", fontWeight: 800, fontSize: 17, color: "#0F1C2E", letterSpacing: -0.3 }}>
              AI DOC
            </div>
            <div style={{ fontSize: 10, color: "#0B7B6F", letterSpacing: 1.2, textTransform: "uppercase", fontWeight: 600 }}>
              Rare Disease Assistant
            </div>
          </div>
        </Link>

        {/* Clock */}
        <div style={{
          fontFamily: "'Inter', monospace",
          fontSize: 12,
          color: "#5A7184",
          background: "#F0F5F8",
          border: "1px solid #DDE8EF",
          padding: "5px 12px",
          borderRadius: 8,
          letterSpacing: 0.8,
          fontWeight: 500,
        }}>
          {time}
        </div>

        {/* Nav */}
        <nav style={{ display: "flex", alignItems: "center", gap: 4 }}>
          {[
            { path: "/", label: "Home" },
            { path: "/predict", label: "Predict" },
            { path: "/dashboard", label: "Dashboard" },
            { path: "/history", label: "History" },
          ].map(({ path, label }) => (
            <Link
              key={path}
              to={path}
              style={{
                textDecoration: "none",
                padding: "8px 16px",
                borderRadius: 8,
                fontSize: 14,
                fontWeight: isActive(path) ? 600 : 500,
                color: isActive(path) ? "#0B7B6F" : "#5A7184",
                background: isActive(path) ? "#EBF8F6" : "transparent",
                border: isActive(path) ? "1px solid #B2E8E2" : "1px solid transparent",
                transition: "all 0.15s ease",
              }}
            >
              {label}
            </Link>
          ))}

          <button
            onClick={() => setDark(!dark)}
            style={{
              marginLeft: 8,
              padding: "7px 12px",
              borderRadius: 8,
              border: "1px solid #DDE8EF",
              background: "#F8FAFB",
              color: "#5A7184",
              cursor: "pointer",
              fontSize: 15,
              transition: "all 0.15s ease",
            }}
            onMouseEnter={e => { e.currentTarget.style.background = "#EDF2F6"; e.currentTarget.style.borderColor = "#C8D8E4"; }}
            onMouseLeave={e => { e.currentTarget.style.background = "#F8FAFB"; e.currentTarget.style.borderColor = "#DDE8EF"; }}
          >
            {dark ? "☀️" : "🌙"}
          </button>
        </nav>
      </div>
    </header>
  );
}

export default Navbar;