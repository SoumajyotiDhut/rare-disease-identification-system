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
    <header
      style={{
        position: "sticky",
        top: 0,
        zIndex: 100,
        transition: "all 0.3s ease",
        background: scrolled
          ? "rgba(10,22,40,0.95)"
          : "rgba(10,22,40,1)",
        backdropFilter: "blur(20px)",
        borderBottom: "1px solid rgba(0,212,200,0.15)",
        boxShadow: scrolled ? "0 4px 40px rgba(0,0,0,0.4)" : "none",
      }}
    >
      <div
        style={{
          maxWidth: 1280,
          margin: "0 auto",
          height: 72,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0 32px",
        }}
      >
        {/* Logo */}
        <Link to="/" style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: 12 }}>
          <div
            style={{
              width: 42,
              height: 42,
              borderRadius: 12,
              background: "linear-gradient(135deg, #00D4C8, #0066FF)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontWeight: 800,
              fontSize: 14,
              color: "#fff",
              letterSpacing: 1,
              boxShadow: "0 0 20px rgba(0,212,200,0.4)",
            }}
          >
            AI
          </div>
          <div>
            <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 700, fontSize: 18, color: "#F8FAFC", letterSpacing: 0.5 }}>
              AI DOC
            </div>
            <div style={{ fontSize: 11, color: "#00D4C8", letterSpacing: 1.5, textTransform: "uppercase" }}>
              Rare Disease Assistant
            </div>
          </div>
        </Link>

        {/* Clock */}
        <div
          style={{
            fontFamily: "'Inter', monospace",
            fontSize: 13,
            color: "#00D4C8",
            background: "rgba(0,212,200,0.08)",
            border: "1px solid rgba(0,212,200,0.2)",
            padding: "6px 14px",
            borderRadius: 8,
            letterSpacing: 1,
          }}
        >
          {time}
        </div>

        {/* Nav */}
        <nav style={{ display: "flex", alignItems: "center", gap: 8 }}>
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
                padding: "8px 18px",
                borderRadius: 8,
                fontSize: 14,
                fontWeight: isActive(path) ? 600 : 400,
                color: isActive(path) ? "#00D4C8" : "#94A3B8",
                background: isActive(path) ? "rgba(0,212,200,0.12)" : "transparent",
                border: isActive(path) ? "1px solid rgba(0,212,200,0.25)" : "1px solid transparent",
                transition: "all 0.2s ease",
              }}
            >
              {label}
            </Link>
          ))}

          <button
            onClick={() => setDark(!dark)}
            style={{
              marginLeft: 8,
              padding: "8px 14px",
              borderRadius: 8,
              border: "1px solid rgba(148,163,184,0.2)",
              background: "transparent",
              color: "#94A3B8",
              cursor: "pointer",
              fontSize: 16,
              transition: "all 0.2s ease",
            }}
          >
            {dark ? "☀️" : "🌙"}
          </button>
        </nav>
      </div>
    </header>
  );
}

export default Navbar;