import { useState } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { useTheme } from "../context/ThemeContext";
import { useToast } from "../context/ToastContext";
import { signInWithEmail, signInWithGoogle, getAuthErrorMessage } from "../firebase";

const CSS = (c) => `
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&display=swap');
  @keyframes fadeUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
  @keyframes spin   { to{transform:rotate(360deg)} }
  .auth-card     { animation: fadeUp .5s cubic-bezier(.2,.7,.3,1) both }
  .auth-input:focus { border-color:${c.teal}!important; box-shadow:0 0 0 3px ${c.teal}18 }
  .auth-submit:hover:not(:disabled)  { transform:translateY(-1px); box-shadow:0 10px 24px ${c.teal}40 }
  .google-btn:hover  { background:${c.bgAlt}!important }
  .auth-link:hover   { text-decoration:underline }
  @media(max-width:480px){ .auth-pad { padding:32px 22px!important } }
`;

export default function Login() {
    const { c } = useTheme();
    const toast = useToast();
    const navigate = useNavigate();
    const location = useLocation();
    const from = location.state?.from?.pathname || "/predict";

    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [loading, setLoading] = useState(false);
    const [googleLoading, setGoogleLoading] = useState(false);

    const handleEmailLogin = async (e) => {
        e.preventDefault();
        if (!email.trim() || !password) {
            toast.error("Please enter both email and password.");
            return;
        }
        setLoading(true);
        try {
            await signInWithEmail(email.trim(), password);
            toast.success("Welcome back!");
            navigate(from, { replace: true });
        } catch (err) {
            toast.error(getAuthErrorMessage(err));
        } finally {
            setLoading(false);
        }
    };

    const handleGoogleLogin = async () => {
        setGoogleLoading(true);
        try {
            await signInWithGoogle();
            toast.success("Welcome back!");
            navigate(from, { replace: true });
        } catch (err) {
            toast.error(getAuthErrorMessage(err));
        } finally {
            setGoogleLoading(false);
        }
    };

    return (
        <div style={{
            minHeight: "100vh", background: c.bg, fontFamily: "'Inter',sans-serif",
            display: "flex", alignItems: "center", justifyContent: "center", padding: "40px 20px",
        }}>
            <style>{CSS(c)}</style>

            <div className="auth-card auth-pad" style={{
                width: "100%", maxWidth: 420, background: c.card,
                border: `1px solid ${c.border}`, borderRadius: 24, padding: "44px 40px",
                boxShadow: "0 24px 60px rgba(0,0,0,0.08)",
            }}>
                {/* Logo */}
                <Link to="/" style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: 10, marginBottom: 32 }}>
                    <div style={{
                        width: 38, height: 38, borderRadius: 11,
                        background: `linear-gradient(135deg,${c.teal},${c.blue})`,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontWeight: 800, fontSize: 14, color: "#fff", boxShadow: `0 4px 14px ${c.teal}40`,
                    }}>AI</div>
                    <span style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontWeight: 800, fontSize: 17, color: c.text }}>AI DOC</span>
                </Link>

                <h1 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 26, fontWeight: 800, color: c.text, margin: "0 0 6px", letterSpacing: -0.5 }}>
                    Welcome back
                </h1>
                <p style={{ fontSize: 14, color: c.sub, margin: "0 0 28px" }}>
                    Sign in to access your diagnostic history and saved predictions.
                </p>

                {/* Google button */}
                <button
                    onClick={handleGoogleLogin}
                    disabled={googleLoading}
                    className="google-btn"
                    style={{
                        width: "100%", display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
                        padding: "13px 16px", borderRadius: 12, border: `1.5px solid ${c.borderI}`,
                        background: c.card, color: c.text, fontSize: 14.5, fontWeight: 700,
                        cursor: googleLoading ? "default" : "pointer", fontFamily: "'Inter',sans-serif",
                        transition: "background .15s", marginBottom: 20, opacity: googleLoading ? 0.6 : 1,
                    }}
                >
                    {googleLoading ? (
                        <span style={{ width: 16, height: 16, border: `2px solid ${c.border}`, borderTop: `2px solid ${c.teal}`, borderRadius: "50%", animation: "spin .7s linear infinite" }} />
                    ) : (
                        <svg width="18" height="18" viewBox="0 0 18 18">
                            <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84c-.21 1.13-.84 2.07-1.8 2.71v2.26h2.9C16.6 14.2 17.64 11.94 17.64 9.2z" />
                            <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.9-2.26c-.8.55-1.84.85-3.06.85-2.36 0-4.36-1.6-5.08-3.75H.96v2.33C2.44 15.98 5.48 18 9 18z" />
                            <path fill="#FBBC05" d="M3.92 10.66A5.4 5.4 0 0 1 3.62 9c0-.58.1-1.14.3-1.66V5.01H.96A8.95 8.95 0 0 0 0 9c0 1.45.35 2.82.96 4l2.96-2.34z" />
                            <path fill="#EA4335" d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.58C13.46.89 11.43 0 9 0 5.48 0 2.44 2.02.96 5l2.96 2.34C4.64 5.18 6.64 3.58 9 3.58z" />
                        </svg>
                    )}
                    Continue with Google
                </button>

                {/* Divider */}
                <div style={{ display: "flex", alignItems: "center", gap: 12, margin: "0 0 22px" }}>
                    <div style={{ flex: 1, height: 1, background: c.border }} />
                    <span style={{ fontSize: 12, color: c.muted, fontWeight: 600 }}>OR</span>
                    <div style={{ flex: 1, height: 1, background: c.border }} />
                </div>

                {/* Email/password form */}
                <form onSubmit={handleEmailLogin}>
                    <label style={{ fontSize: 12.5, fontWeight: 700, color: c.sub, display: "block", marginBottom: 7 }}>Email</label>
                    <input
                        type="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="you@example.com"
                        className="auth-input"
                        style={{
                            width: "100%", padding: "12px 14px", borderRadius: 11, border: `1.5px solid ${c.borderI}`,
                            background: c.bgAlt, color: c.text, fontSize: 14, outline: "none",
                            fontFamily: "'Inter',sans-serif", marginBottom: 16, boxSizing: "border-box", transition: "border-color .15s, box-shadow .15s",
                        }}
                    />

                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 7 }}>
                        <label style={{ fontSize: 12.5, fontWeight: 700, color: c.sub }}>Password</label>
                        <Link to="/forgot-password" className="auth-link" style={{ fontSize: 12.5, color: c.teal, fontWeight: 600, textDecoration: "none" }}>
                            Forgot password?
                        </Link>
                    </div>
                    <input
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="••••••••"
                        className="auth-input"
                        style={{
                            width: "100%", padding: "12px 14px", borderRadius: 11, border: `1.5px solid ${c.borderI}`,
                            background: c.bgAlt, color: c.text, fontSize: 14, outline: "none",
                            fontFamily: "'Inter',sans-serif", marginBottom: 24, boxSizing: "border-box", transition: "border-color .15s, box-shadow .15s",
                        }}
                    />

                    <button
                        type="submit"
                        disabled={loading}
                        className="auth-submit"
                        style={{
                            width: "100%", padding: "14px 16px", borderRadius: 12, border: "none",
                            background: loading ? `${c.teal}99` : `linear-gradient(135deg,${c.teal},${c.blue})`,
                            color: "#fff", fontSize: 15, fontWeight: 800, cursor: loading ? "default" : "pointer",
                            fontFamily: "'Inter',sans-serif", display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
                            transition: "transform .15s, box-shadow .15s",
                        }}
                    >
                        {loading ? (
                            <>
                                <span style={{ width: 16, height: 16, border: "2px solid rgba(255,255,255,0.35)", borderTop: "2px solid #fff", borderRadius: "50%", animation: "spin .7s linear infinite" }} />
                                Signing in…
                            </>
                        ) : "Sign In"}
                    </button>
                </form>

                <p style={{ textAlign: "center", fontSize: 13.5, color: c.sub, marginTop: 26 }}>
                    Don't have an account?{" "}
                    <Link to="/signup" className="auth-link" style={{ color: c.teal, fontWeight: 700, textDecoration: "none" }}>
                        Sign up
                    </Link>
                </p>
            </div>
        </div>
    );
}