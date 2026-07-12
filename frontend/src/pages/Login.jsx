import { useState } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { useTheme } from "../context/ThemeContext";
import { useToast } from "../context/ToastContext";
import { signInWithEmail, signInWithGoogle, getAuthErrorMessage } from "../firebase";

const CSS = (c) => `

  @keyframes fadeUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
  @keyframes spin   { to{transform:rotate(360deg)} }
  @keyframes pulse  { 0%,100%{opacity:.35;transform:scale(1)} 50%{opacity:.6;transform:scale(1.04)} }
  @keyframes drawLine { from{stroke-dashoffset:600} to{stroke-dashoffset:0} }

  .auth-panel { animation: fadeUp .5s cubic-bezier(.2,.7,.3,1) both }

  .auth-input {
    width:100%; padding:13px 16px; border-radius:4px;
    border:1px solid ${c.borderI}; background:${c.bgDeep};
    color:${c.text}; font-size:14px; outline:none;
    font-family:'Inter',sans-serif; box-sizing:border-box;
    transition:border-color .2s, box-shadow .2s, background .2s;
  }
  .auth-input:focus { border-color:${c.teal} !important; box-shadow:0 0 0 3px ${c.tealL}; background:${c.card}; }
  .auth-input::placeholder { color:${c.muted} }

  .primary-btn {
    width:100%; padding:15px 16px; border-radius:4px; border:none;
    background:${c.text}; color:${c.bg}; font-size:14.5px;
    font-weight:600; cursor:pointer; font-family:'Inter',sans-serif;
    display:flex; align-items:center; justify-content:center; gap:10px;
    transition:all .25s; letter-spacing:0.01em;
  }
  .primary-btn:hover:not(:disabled) { background:${c.teal}; color:#fff; transform:translateY(-2px); box-shadow:${c.shadowTeal}; }
  .primary-btn:disabled { opacity:.6; cursor:not-allowed }

  .google-btn {
    width:100%; display:flex; align-items:center; justify-content:center; gap:10px;
    padding:13px 16px; border-radius:4px; border:1px solid ${c.borderI};
    background:${c.card}; color:${c.text}; font-size:14px; font-weight:600;
    cursor:pointer; font-family:'Inter',sans-serif; transition:all .2s;
  }
  .google-btn:hover:not(:disabled) { border-color:${c.teal}; background:${c.tealL}; }
  .google-btn:disabled { opacity:.6; cursor:not-allowed }

  .text-link { color:${c.teal}; font-weight:600; text-decoration:none; transition:opacity .15s; }
  .text-link:hover { opacity:.7; text-decoration:underline }

  .divider-line { flex:1; height:1px; background:${c.border}; }

  .blob1 { position:absolute; width:320px; height:320px; border-radius:50%; background:radial-gradient(circle, rgba(255,255,255,0.10) 0%, transparent 70%); top:-70px; left:-70px; animation:pulse 8s ease-in-out infinite; }
  .blob2 { position:absolute; width:240px; height:240px; border-radius:50%; background:radial-gradient(circle, rgba(203,175,116,0.16) 0%, transparent 70%); bottom:100px; right:-50px; animation:pulse 10s ease-in-out infinite 2s; }

  .eyebrow-light { font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:600; color:rgba(255,255,255,0.65); letter-spacing:0.16em; text-transform:uppercase; display:flex; align-items:center; gap:10px; }
  .eyebrow-light::before { content:''; width:22px; height:1px; background:rgba(255,255,255,0.5); display:inline-block; }

  .stat-chip { border-left:1px solid rgba(255,255,255,0.22); padding-left:14px; }

  @media(max-width:860px){
    .left-panel { display:none !important }
    .right-panel { max-width:100% !important; padding:40px 24px !important }
    .auth-root { grid-template-columns:1fr !important; min-height:100vh }
  }
  @media(max-width:480px){
    .right-panel { padding:32px 18px !important }
    .auth-input, .primary-btn, .google-btn { font-size:16px !important }
    .auth-title { font-size:26px !important }
  }
`;

function VitalLine({ color = "rgba(255,255,255,0.5)", width = 100, height = 20 }) {
    return (
        <svg width={width} height={height} viewBox="0 0 160 28" fill="none">
            <path d="M0 14H40L48 4L58 24L66 14H160" stroke={color} strokeWidth="1.5"
                strokeLinecap="round" strokeLinejoin="round" strokeDasharray="600"
                style={{ animation: "drawLine 1.4s ease forwards" }} />
        </svg>
    );
}

const IconEye = ({ color }) => (
    <svg width="17" height="17" viewBox="0 0 20 20" fill="none">
        <path d="M1 10s3-6 9-6 9 6 9 6-3 6-9 6-9-6-9-6z" stroke={color} strokeWidth="1.6" strokeLinejoin="round" />
        <circle cx="10" cy="10" r="2.6" stroke={color} strokeWidth="1.6" />
    </svg>
);
const IconEyeOff = ({ color }) => (
    <svg width="17" height="17" viewBox="0 0 20 20" fill="none">
        <path d="M2.5 2.5l15 15M8.3 8.4a2.6 2.6 0 003.4 3.4M5.3 5.5C3 7 1 10 1 10s3 6 9 6c1.5 0 2.8-.4 3.9-.9M12.5 4.4C11.7 4.1 10.9 4 10 4c-1 0-1.9.16-2.7.44M14.9 6.2C17 7.7 19 10 19 10s-.7 1.4-2 2.9" stroke={color} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
);

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
    const [showPass, setShowPass] = useState(false);

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

    const Spinner = () => (
        <span style={{
            width: 18, height: 18, border: "2.5px solid rgba(255,255,255,0.35)",
            borderTop: "2.5px solid #fff", borderRadius: "50%",
            animation: "spin .7s linear infinite", display: "inline-block", flexShrink: 0,
        }} />
    );

    return (
        <div style={{ minHeight: "100vh", background: c.bg, fontFamily: "'Inter',sans-serif" }}>
            <style>{CSS(c)}</style>

            <div className="auth-root" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", minHeight: "100vh" }}>

                {/* ── LEFT — Branding panel ── */}
                <div className="left-panel" style={{
                    position: "relative", background: c.gradPrimary, overflow: "hidden",
                    display: "flex", flexDirection: "column", justifyContent: "space-between", padding: "48px 52px",
                }}>
                    <div className="blob1" />
                    <div className="blob2" />

                    <Link to="/" style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: 12, position: "relative", zIndex: 2 }}>
                        <div style={{
                            width: 42, height: 42, borderRadius: 4,
                            background: "rgba(255,255,255,0.14)",
                            border: "1px solid rgba(255,255,255,0.30)",
                            display: "flex", alignItems: "center", justifyContent: "center",
                            fontFamily: "'Fraunces',serif", fontStyle: "italic", fontWeight: 600, fontSize: 17, color: "#fff",
                        }}>AI</div>
                        <span style={{ fontFamily: "'Fraunces',serif", fontWeight: 600, fontSize: 20, color: "#fff", letterSpacing: "-0.02em" }}>AI DOC</span>
                    </Link>

                    <div style={{ position: "relative", zIndex: 2 }}>
                        <span className="eyebrow-light">Rare Disease Identification</span>

                        <h2 style={{
                            fontFamily: "'Fraunces',serif", fontSize: 40, fontWeight: 600, color: "#fff",
                            margin: "22px 0 20px", lineHeight: 1.12, letterSpacing: "-0.02em",
                        }}>
                            AI-Powered<br />Diagnostics<br />
                            <span style={{ fontStyle: "italic", color: "rgba(255,255,255,0.62)", fontWeight: 500 }}>at your fingertips</span>
                        </h2>

                        <p style={{ fontSize: 15, color: "rgba(255,255,255,0.72)", lineHeight: 1.75, margin: "0 0 32px", maxWidth: 320 }}>
                            Identify rare diseases from symptoms and biomedical scans with
                            state-of-the-art multimodal AI trained on 36,487 real patient cases.
                        </p>

                        <div style={{ height: 1, background: "rgba(255,255,255,0.18)", maxWidth: 340, marginBottom: 22 }} />

                        <div style={{ display: "flex", flexWrap: "wrap", gap: 20 }}>
                            {[
                                { val: "36K+", label: "Patient Cases" },
                                { val: "1,374", label: "Rare Diseases" },
                                { val: "83.87%", label: "Top-5 Accuracy" },
                            ].map(({ val, label }, i) => (
                                <div key={label} className={i > 0 ? "stat-chip" : ""}>
                                    <div style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 18, fontWeight: 600, color: "#fff" }}>{val}</div>
                                    <div style={{ fontSize: 11, color: "rgba(255,255,255,0.60)", fontWeight: 500, marginTop: 2 }}>{label}</div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div style={{ position: "relative", zIndex: 2 }}>
                        <VitalLine width={90} height={16} />
                        <p style={{ fontSize: 11.5, color: "rgba(255,255,255,0.45)", margin: "10px 0 0", lineHeight: 1.6 }}>
                            For research and clinical support purposes only.<br />
                            Always consult a qualified healthcare professional.
                        </p>
                    </div>
                </div>

                {/* ── RIGHT — Form panel ── */}
                <div className="right-panel auth-panel" style={{
                    display: "flex", flexDirection: "column", justifyContent: "center",
                    padding: "48px 64px", background: c.card, maxWidth: 560, width: "100%", margin: "0 auto",
                }}>

                    <div style={{ marginBottom: 36 }}>
                        <h1 className="auth-title" style={{ fontFamily: "'Fraunces',serif", fontSize: 32, fontWeight: 600, color: c.text, margin: "0 0 8px", letterSpacing: "-0.02em" }}>
                            Welcome back
                        </h1>
                        <p style={{ fontSize: 14.5, color: c.sub, margin: 0, lineHeight: 1.6 }}>
                            Sign in to access your diagnostic history and predictions.
                        </p>
                    </div>

                    <button className="google-btn" onClick={handleGoogleLogin} disabled={googleLoading} style={{ marginBottom: 24 }}>
                        {googleLoading ? <Spinner /> : (
                            <svg width="18" height="18" viewBox="0 0 18 18">
                                <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84c-.21 1.13-.84 2.07-1.8 2.71v2.26h2.9C16.6 14.2 17.64 11.94 17.64 9.2z" />
                                <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.9-2.26c-.8.55-1.84.85-3.06.85-2.36 0-4.36-1.6-5.08-3.75H.96v2.33C2.44 15.98 5.48 18 9 18z" />
                                <path fill="#FBBC05" d="M3.92 10.66A5.4 5.4 0 0 1 3.62 9c0-.58.1-1.14.3-1.66V5.01H.96A8.95 8.95 0 0 0 0 9c0 1.45.35 2.82.96 4l2.96-2.34z" />
                                <path fill="#EA4335" d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.58C13.46.89 11.43 0 9 0 5.48 0 2.44 2.02.96 5l2.96 2.34C4.64 5.18 6.64 3.58 9 3.58z" />
                            </svg>
                        )}
                        Continue with Google
                    </button>

                    <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 24 }}>
                        <div className="divider-line" />
                        <span style={{ fontSize: 12, color: c.muted, fontWeight: 600, whiteSpace: "nowrap" }}>or sign in with email</span>
                        <div className="divider-line" />
                    </div>

                    <form onSubmit={handleEmailLogin}>
                        <div style={{ marginBottom: 18 }}>
                            <label style={{ fontSize: 13, fontWeight: 600, color: c.sub, display: "block", marginBottom: 8 }}>
                                Email address
                            </label>
                            <input
                                type="email" value={email} className="auth-input"
                                onChange={(e) => setEmail(e.target.value)}
                                placeholder="you@example.com"
                                autoComplete="email"
                            />
                        </div>

                        <div style={{ marginBottom: 28 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                                <label style={{ fontSize: 13, fontWeight: 600, color: c.sub }}>Password</label>
                                <Link to="/forgot-password" className="text-link" style={{ fontSize: 12.5 }}>
                                    Forgot password?
                                </Link>
                            </div>
                            <div style={{ position: "relative" }}>
                                <input
                                    type={showPass ? "text" : "password"} value={password}
                                    className="auth-input" onChange={(e) => setPassword(e.target.value)}
                                    placeholder="••••••••" autoComplete="current-password"
                                    style={{ paddingRight: 48 }}
                                />
                                <button type="button" onClick={() => setShowPass(!showPass)} aria-label={showPass ? "Hide password" : "Show password"} style={{
                                    position: "absolute", right: 14, top: "50%", transform: "translateY(-50%)",
                                    background: "none", border: "none", cursor: "pointer",
                                    color: c.muted, fontSize: 16, padding: 0, lineHeight: 1,
                                }}>
                                    {showPass ? <IconEyeOff color={c.muted} /> : <IconEye color={c.muted} />}
                                </button>
                            </div>
                        </div>

                        <button type="submit" className="primary-btn" disabled={loading}>
                            {loading ? <><Spinner /> Signing in…</> : "Sign In"}
                        </button>
                    </form>

                    <p style={{ textAlign: "center", fontSize: 13.5, color: c.sub, marginTop: 28, marginBottom: 0 }}>
                        Don't have an account?{" "}
                        <Link to="/signup" className="text-link">Create one free</Link>
                    </p>
                </div>
            </div>
        </div>
    );
}