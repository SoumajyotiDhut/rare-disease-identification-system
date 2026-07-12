import { Link, useLocation, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { useTheme } from "../context/ThemeContext";
import { useAuth } from "../context/AuthContext";
import { useToast } from "../context/ToastContext";

const CSS = (c) => `
  * { box-sizing: border-box; }

  .nav-link {
    position: relative; text-decoration: none;
    padding: 7px 14px; font-size: 13.5px; font-weight: 500; color: ${c.sub};
    transition: color .2s, background .2s; letter-spacing: 0.01em;
  }
  .nav-link:hover { color: ${c.text} !important; background: ${c.cardAlt} !important; }
  .nav-link.active { color: ${c.teal} !important; font-weight: 700; }
  .nav-link.active::after {
    content:''; position:absolute; bottom:-1px; left:14px; right:14px;
    height:2px; background:${c.teal};
  }

  .theme-btn {
    width:36px; height:36px; border-radius:4px;
    border:1px solid ${c.border}; background:${c.cardAlt}; color:${c.sub};
    cursor:pointer; font-size:15px; display:flex; align-items:center;
    justify-content:center; transition:all .2s; flex-shrink:0;
  }
  .theme-btn:hover { border-color:${c.borderI}; }

  .auth-outline {
    padding:8px 16px; border-radius:4px; border:1px solid ${c.border};
    background:transparent; color:${c.sub}; text-decoration:none;
    font-size:13.5px; font-weight:600; cursor:pointer;
    transition:all .2s; font-family:'Inter',sans-serif;
  }
  .auth-outline:hover { border-color:${c.teal}; color:${c.teal}; background:${c.tealL}; }

  .auth-fill {
    padding:8px 18px; border-radius:4px; border:none;
    background:${c.text}; color:${c.bg}; text-decoration:none;
    font-size:13.5px; font-weight:600; cursor:pointer;
    transition:all .2s; font-family:'Inter',sans-serif;
  }
  .auth-fill:hover { background:${c.teal}; color:#fff; }

  .logout-btn {
    padding:7px 14px; border-radius:4px;
    border:1px solid ${c.border}; background:transparent;
    color:${c.sub}; font-size:13px; font-weight:600;
    cursor:pointer; transition:all .2s; font-family:'Inter',sans-serif;
  }
  .logout-btn:hover { border-color:${c.redB}; color:${c.red}; background:${c.redL}; }

  .avatar-ring {
    width:32px; height:32px; border-radius:4px;
    background:${c.gradPrimary}; display:flex; align-items:center;
    justify-content:center; font-size:12px; font-weight:700; color:#fff;
    flex-shrink:0; font-family:'Fraunces',serif;
  }

  .mobile-toggle {
    display:none !important; flex-direction:column; gap:5px;
    padding:9px; border:1px solid ${c.border}; background:${c.cardAlt};
    border-radius:4px; cursor:pointer; transition:all .2s;
  }
  .mobile-toggle span {
    width:20px; height:2px; background:${c.sub}; border-radius:2px;
    display:block; transition:all .25s cubic-bezier(.4,0,.2,1);
  }

  .mobile-nav-link {
    display:flex; align-items:center; gap:10px; padding:11px 14px;
    border-radius:4px; font-size:15px; font-weight:500;
    text-decoration:none; transition:all .15s; margin-bottom:3px;
  }
  .mobile-nav-link:hover { background:${c.cardAlt} !important; }

  @media(max-width:768px){
    .clock-pill  { display:none !important }
    .desktop-nav { display:none !important }
    .mobile-toggle { display:flex !important }
  }
  @media(max-width:380px){
    .logo-sub { display:none !important }
  }

  @keyframes mobileSlide {
    from { opacity:0; transform:translateY(-8px) }
    to   { opacity:1; transform:translateY(0) }
  }
  .mobile-menu-open { animation: mobileSlide .2s ease both }
`;

const NAV_LINKS = [
  { path: "/", label: "Home" },
  { path: "/predict", label: "Predict" },
  { path: "/dashboard", label: "Dashboard" },
  { path: "/history", label: "History" },
];

const IconSun = () => (
  <svg width="15" height="15" viewBox="0 0 20 20" fill="none">
    <circle cx="10" cy="10" r="4" stroke="currentColor" strokeWidth="1.6" />
    <path d="M10 1.5V4M10 16v2.5M3.5 10H1M19 10h-2.5M5.05 5.05L3.3 3.3M16.7 16.7l-1.75-1.75M5.05 14.95L3.3 16.7M16.7 3.3l-1.75 1.75" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
  </svg>
);
const IconMoon = () => (
  <svg width="15" height="15" viewBox="0 0 20 20" fill="none">
    <path d="M17 11.5A7 7 0 018.5 3 7.5 7.5 0 1017 11.5z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
  </svg>
);

export default function Navbar() {
  const { dark, setDark, c } = useTheme();
  const { user, authLoading, logout } = useAuth();
  const toast = useToast();
  const navigate = useNavigate();
  const location = useLocation();

  const [time, setTime] = useState("");
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const t = setInterval(
      () => setTime(new Date().toLocaleTimeString("en-IN",
        { hour: "2-digit", minute: "2-digit", second: "2-digit" })),
      1000
    );
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 16);
    window.addEventListener("scroll", fn, { passive: true });
    return () => window.removeEventListener("scroll", fn);
  }, []);

  useEffect(() => setOpen(false), [location]);

  const isActive = (p) =>
    p === "/" ? location.pathname === "/" : location.pathname.startsWith(p);

  const handleLogout = async () => {
    try {
      await logout();
      toast.success("Signed out successfully.");
      navigate("/");
    } catch {
      toast.error("Couldn't sign out. Try again.");
    }
  };

  const initials = (user?.displayName || user?.email || "?")
    .trim().split(" ").map((w) => w[0]).slice(0, 2).join("").toUpperCase();

  const displayName = user?.displayName || user?.email?.split("@")[0] || "User";

  return (
    <>
      <style>{CSS(c)}</style>

      <header style={{
        position: "sticky", top: 0, zIndex: 300,
        background: scrolled ? c.glass : "transparent",
        backdropFilter: scrolled ? c.glassBlur : "none",
        WebkitBackdropFilter: scrolled ? c.glassBlur : "none",
        borderBottom: scrolled ? `1px solid ${c.glassBorder}` : "1px solid transparent",
        transition: "all .3s ease",
      }}>
        <div style={{
          maxWidth: 1200, margin: "0 auto", height: 64,
          display: "flex", alignItems: "center",
          justifyContent: "space-between", padding: "0 24px", gap: 16,
        }}>

          {/* Logo */}
          <Link to="/" style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
            <div style={{
              width: 34, height: 34, borderRadius: 4, background: c.gradPrimary,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontFamily: "'Fraunces',serif", fontStyle: "italic", fontWeight: 600, fontSize: 13, color: "#fff",
            }}>AI</div>
            <div>
              <div style={{
                fontFamily: "'Fraunces',serif", fontWeight: 600,
                fontSize: 15.5, color: c.text, letterSpacing: "-0.02em", lineHeight: 1.2,
              }}>AI DOC</div>
              <div className="logo-sub" style={{
                fontFamily: "'IBM Plex Mono',monospace",
                fontSize: 8.5, color: c.gold, letterSpacing: "0.1em",
                textTransform: "uppercase", fontWeight: 600, lineHeight: 1,
              }}>Rare Disease Assistant</div>
            </div>
          </Link>

          {/* Clock */}
          <div className="clock-pill" style={{
            fontFamily: "'IBM Plex Mono',monospace", fontSize: 11.5, color: c.subAlt,
            background: c.cardAlt, border: `1px solid ${c.border}`,
            padding: "5px 13px", borderRadius: 4, letterSpacing: "0.06em", fontWeight: 500,
          }}>{time}</div>

          {/* Desktop Nav */}
          <nav className="desktop-nav" style={{
            display: "flex", alignItems: "center", gap: 2,
            flex: 1, justifyContent: "center",
          }}>
            {NAV_LINKS.map(({ path, label }) => (
              <Link key={path} to={path}
                className={`nav-link${isActive(path) ? " active" : ""}`}>
                {label}
              </Link>
            ))}
          </nav>

          {/* Right controls */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
            <button className="theme-btn" onClick={() => setDark(!dark)} aria-label="Toggle theme">
              {dark ? <IconSun /> : <IconMoon />}
            </button>

            {!authLoading && (
              user ? (
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{
                    display: "flex", alignItems: "center", gap: 8,
                    background: c.cardAlt, border: `1px solid ${c.border}`,
                    borderRadius: 4, padding: "5px 12px 5px 6px",
                  }}>
                    <div className="avatar-ring">{initials}</div>
                    <span style={{
                      fontSize: 13, fontWeight: 600, color: c.text,
                      maxWidth: 100, overflow: "hidden",
                      textOverflow: "ellipsis", whiteSpace: "nowrap",
                    }}>{displayName}</span>
                  </div>
                  <button className="logout-btn" onClick={handleLogout}>Sign out</button>
                </div>
              ) : (
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <Link to="/login" className="auth-outline">Sign In</Link>
                  <Link to="/signup" className="auth-fill">Get Started</Link>
                </div>
              )
            )}
          </div>

          {/* Mobile hamburger */}
          <button className="mobile-toggle" onClick={() => setOpen(!open)} aria-label={open ? "Close menu" : "Open menu"} aria-expanded={open}>
            <span style={{ transform: open ? "rotate(45deg) translate(5px,5px)" : "none" }} />
            <span style={{ opacity: open ? 0 : 1 }} />
            <span style={{ transform: open ? "rotate(-45deg) translate(5px,-5px)" : "none" }} />
          </button>
        </div>

        {/* Mobile menu */}
        {open && (
          <div className="mobile-menu-open" style={{
            background: c.glass, backdropFilter: c.glassBlur,
            WebkitBackdropFilter: c.glassBlur,
            borderTop: `1px solid ${c.glassBorder}`,
            padding: "12px 16px 20px",
          }}>
            {NAV_LINKS.map(({ path, label }) => (
              <Link key={path} to={path} className="mobile-nav-link" style={{
                color: isActive(path) ? c.teal : c.text,
                background: isActive(path) ? c.tealL : "transparent",
                fontWeight: isActive(path) ? 700 : 500,
              }}>
                {label}
              </Link>
            ))}

            <div style={{ marginTop: 12, paddingTop: 14, borderTop: `1px solid ${c.border}` }}>
              {!authLoading && (
                user ? (
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <div className="avatar-ring">{initials}</div>
                      <div>
                        <p style={{ fontSize: 13.5, color: c.text, fontWeight: 700, margin: 0 }}>{displayName}</p>
                        <p style={{ fontSize: 11, color: c.muted, margin: 0 }}>{user.email}</p>
                      </div>
                    </div>
                    <button className="logout-btn" onClick={handleLogout}>Sign out</button>
                  </div>
                ) : (
                  <div style={{ display: "flex", gap: 8 }}>
                    <Link to="/login" className="auth-outline" style={{ flex: 1, textAlign: "center" }}>Sign In</Link>
                    <Link to="/signup" className="auth-fill" style={{ flex: 1, textAlign: "center" }}>Get Started</Link>
                  </div>
                )
              )}
            </div>

            <div style={{
              marginTop: 12, paddingTop: 12, borderTop: `1px solid ${c.border}`,
              display: "flex", justifyContent: "space-between", alignItems: "center",
            }}>
              <span style={{ fontSize: 11.5, color: c.muted, fontFamily: "'IBM Plex Mono',monospace", letterSpacing: "0.05em" }}>
                {time}
              </span>
              <button className="theme-btn" onClick={() => setDark(!dark)} aria-label="Toggle theme">{dark ? <IconSun /> : <IconMoon />}</button>
            </div>
          </div>
        )}
      </header>
    </>
  );
}