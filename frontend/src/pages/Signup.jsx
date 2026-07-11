import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useTheme } from "../context/ThemeContext";
import { useToast } from "../context/ToastContext";
import { signUpWithEmail, signInWithGoogle, getAuthErrorMessage } from "../firebase";

const CSS = (c) => `
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
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
  .auth-input:focus { border-color:${c.teal}!important; box-shadow:0 0 0 3px ${c.tealL}; background:${c.card}; }
  .auth-input::placeholder { color:${c.muted} }

  .primary-btn {
    width:100%; padding:15px 16px; border-radius:4px; border:none;
    background:${c.text}; color:${c.bg}; font-size:14.5px; font-weight:600;
    cursor:pointer; font-family:'Inter',sans-serif;
    display:flex; align-items:center; justify-content:center; gap:10px;
    transition:all .25s;
  }
  .primary-btn:hover:not(:disabled) { background:${c.purple}; color:#fff; transform:translateY(-2px); box-shadow:0 14px 32px ${c.purpB}; }
  .primary-btn:disabled { opacity:.6; cursor:not-allowed }

  .google-btn {
    width:100%; display:flex; align-items:center; justify-content:center; gap:10px;
    padding:13px 16px; border-radius:4px; border:1px solid ${c.borderI};
    background:${c.card}; color:${c.text}; font-size:14px; font-weight:600;
    cursor:pointer; font-family:'Inter',sans-serif; transition:all .2s;
  }
  .google-btn:hover:not(:disabled) { border-color:${c.purple}; background:${c.purpL}; }
  .google-btn:disabled { opacity:.6; cursor:not-allowed }

  .text-link { color:${c.purple}; font-weight:600; text-decoration:none; transition:opacity .15s; }
  .text-link:hover { opacity:.7; text-decoration:underline }

  .blob1 { position:absolute; width:300px; height:300px; border-radius:50%; background:radial-gradient(circle,rgba(255,255,255,0.10) 0%,transparent 70%); top:-60px; right:-60px; animation:pulse 8s ease-in-out infinite; }
  .blob2 { position:absolute; width:220px; height:220px; border-radius:50%; background:radial-gradient(circle,rgba(203,175,116,0.16) 0%,transparent 70%); bottom:110px; left:-40px; animation:pulse 10s ease-in-out infinite 2s; }

  .eyebrow-light { font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:600; color:rgba(255,255,255,0.60); letter-spacing:0.16em; text-transform:uppercase; display:flex; align-items:center; gap:10px; }
  .eyebrow-light::before { content:''; width:22px; height:1px; background:rgba(255,255,255,0.5); display:inline-block; }
  .stat-chip { border-left:1px solid rgba(255,255,255,0.22); padding-left:14px; }

  @media(max-width:860px){ .left-panel{display:none!important} .right-panel{max-width:100%!important;padding:32px 24px!important} .auth-root{grid-template-columns:1fr!important;min-height:100vh} }
  @media(max-width:480px){
    .right-panel { padding:26px 18px !important }
    .auth-input, .primary-btn, .google-btn { font-size:16px !important }
    .auth-title { font-size:25px !important }
  }
`;

function strength(pw) {
    if (!pw) return { label: "", pct: 0 };
    let s = 0;
    if (pw.length >= 6) s++;
    if (pw.length >= 10) s++;
    if (/[A-Z]/.test(pw) && /[0-9]/.test(pw)) s++;
    if (/[^A-Za-z0-9]/.test(pw)) s++;
    const lvl = [{ label: "Weak", pct: 25 }, { label: "Fair", pct: 50 }, { label: "Good", pct: 75 }, { label: "Strong", pct: 100 }];
    return lvl[Math.min(s, 4) - 1] || lvl[0];
}

const Spinner = () => (
    <span style={{ width: 18, height: 18, border: "2.5px solid rgba(255,255,255,0.35)", borderTop: "2.5px solid #fff", borderRadius: "50%", animation: "spin .7s linear infinite", display: "inline-block", flexShrink: 0 }} />
);

const GIcon = () => (
    <svg width="18" height="18" viewBox="0 0 18 18">
        <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84c-.21 1.13-.84 2.07-1.8 2.71v2.26h2.9C16.6 14.2 17.64 11.94 17.64 9.2z" />
        <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.9-2.26c-.8.55-1.84.85-3.06.85-2.36 0-4.36-1.6-5.08-3.75H.96v2.33C2.44 15.98 5.48 18 9 18z" />
        <path fill="#FBBC05" d="M3.92 10.66A5.4 5.4 0 0 1 3.62 9c0-.58.1-1.14.3-1.66V5.01H.96A8.95 8.95 0 0 0 0 9c0 1.45.35 2.82.96 4l2.96-2.34z" />
        <path fill="#EA4335" d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.58C13.46.89 11.43 0 9 0 5.48 0 2.44 2.02.96 5l2.96 2.34C4.64 5.18 6.64 3.58 9 3.58z" />
    </svg>
);

function VitalLine({ color = "rgba(255,255,255,0.5)", width = 90, height = 16 }) {
    return (
        <svg width={width} height={height} viewBox="0 0 160 28" fill="none">
            <path d="M0 14H40L48 4L58 24L66 14H160" stroke={color} strokeWidth="1.5"
                strokeLinecap="round" strokeLinejoin="round" strokeDasharray="600"
                style={{ animation: "drawLine 1.4s ease forwards" }} />
        </svg>
    );
}

export default function Signup() {
    const { c } = useTheme();
    const toast = useToast();
    const navigate = useNavigate();
    const [name, setName] = useState("");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [confirm, setConfirm] = useState("");
    const [loading, setLoading] = useState(false);
    const [gLoading, setGLoading] = useState(false);
    const [showPass, setShowPass] = useState(false);
    const str = strength(password);
    const strColor = { Weak: c.red, Fair: c.amber, Good: c.blue, Strong: c.teal }[str.label] || c.border;

    const handleSignup = async (e) => {
        e.preventDefault();
        if (!name.trim() || !email.trim() || !password) { toast.error("Please fill in all fields."); return; }
        if (password.length < 6) { toast.error("Password must be at least 6 characters."); return; }
        if (password !== confirm) { toast.error("Passwords do not match."); return; }
        setLoading(true);
        try {
            await signUpWithEmail(email.trim(), password, name.trim());
            toast.success("Account created! Welcome to AI DOC.");
            navigate("/predict", { replace: true });
        } catch (err) { toast.error(getAuthErrorMessage(err)); }
        finally { setLoading(false); }
    };

    const handleGoogle = async () => {
        setGLoading(true);
        try {
            await signInWithGoogle();
            toast.success("Welcome to AI DOC!");
            navigate("/predict", { replace: true });
        } catch (err) { toast.error(getAuthErrorMessage(err)); }
        finally { setGLoading(false); }
    };

    return (
        <div style={{ minHeight: "100vh", background: c.bg, fontFamily: "'Inter',sans-serif" }}>
            <style>{CSS(c)}</style>
            <div className="auth-root" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", minHeight: "100vh" }}>

                {/* Left panel */}
                <div className="left-panel" style={{
                    position: "relative", background: c.gradPurple, overflow: "hidden",
                    display: "flex", flexDirection: "column", justifyContent: "space-between", padding: "48px 52px",
                }}>
                    <div className="blob1" /><div className="blob2" />

                    <Link to="/" style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: 12, position: "relative", zIndex: 2 }}>
                        <div style={{
                            width: 42, height: 42, borderRadius: 4, background: "rgba(255,255,255,0.14)",
                            border: "1px solid rgba(255,255,255,0.30)", display: "flex", alignItems: "center", justifyContent: "center",
                            fontFamily: "'Fraunces',serif", fontStyle: "italic", fontWeight: 600, fontSize: 17, color: "#fff",
                        }}>AI</div>
                        <span style={{ fontFamily: "'Fraunces',serif", fontWeight: 600, fontSize: 20, color: "#fff", letterSpacing: "-0.02em" }}>AI DOC</span>
                    </Link>

                    <div style={{ position: "relative", zIndex: 2 }}>
                        <span className="eyebrow-light">Join AI DOC</span>
                        <h2 style={{ fontFamily: "'Fraunces',serif", fontSize: 40, fontWeight: 600, color: "#fff", margin: "22px 0 20px", lineHeight: 1.12, letterSpacing: "-0.02em" }}>
                            Start your<br />diagnostic<br />
                            <span style={{ fontStyle: "italic", color: "rgba(255,255,255,0.60)", fontWeight: 500 }}>journey today</span>
                        </h2>
                        <p style={{ fontSize: 15, color: "rgba(255,255,255,0.72)", lineHeight: 1.75, margin: "0 0 32px", maxWidth: 320 }}>
                            Create a free account to save predictions, track history, and access advanced analytics powered by multimodal AI.
                        </p>

                        <div style={{ height: 1, background: "rgba(255,255,255,0.18)", maxWidth: 340, marginBottom: 22 }} />

                        <div style={{ display: "flex", flexWrap: "wrap", gap: 20 }}>
                            {[{ val: "Free", label: "Forever" }, { val: "Secure", label: "Firebase Auth" }, { val: "58.39%", label: "Top-1 Accuracy" }].map(({ val, label }, i) => (
                                <div key={label} className={i > 0 ? "stat-chip" : ""}>
                                    <div style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 18, fontWeight: 600, color: "#fff" }}>{val}</div>
                                    <div style={{ fontSize: 11, color: "rgba(255,255,255,0.60)", fontWeight: 500, marginTop: 2 }}>{label}</div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div style={{ position: "relative", zIndex: 2 }}>
                        <VitalLine />
                        <p style={{ fontSize: 11.5, color: "rgba(255,255,255,0.40)", margin: "10px 0 0", lineHeight: 1.6 }}>
                            By signing up you agree to our Terms of Service.<br />For research use only — not a medical device.
                        </p>
                    </div>
                </div>

                {/* Right panel */}
                <div className="right-panel auth-panel" style={{
                    display: "flex", flexDirection: "column", justifyContent: "center",
                    padding: "48px 64px", background: c.card, maxWidth: 560, width: "100%", margin: "0 auto",
                    overflowY: "auto",
                }}>
                    <div style={{ marginBottom: 28 }}>
                        <h1 className="auth-title" style={{ fontFamily: "'Fraunces',serif", fontSize: 32, fontWeight: 600, color: c.text, margin: "0 0 8px", letterSpacing: "-0.02em" }}>
                            Create your account
                        </h1>
                        <p style={{ fontSize: 14.5, color: c.sub, margin: 0 }}>Save predictions and access history from anywhere.</p>
                    </div>

                    <button className="google-btn" onClick={handleGoogle} disabled={gLoading} style={{ marginBottom: 22 }}>
                        {gLoading ? <Spinner /> : <GIcon />}
                        Continue with Google
                    </button>

                    <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 22 }}>
                        <div style={{ flex: 1, height: 1, background: c.border }} />
                        <span style={{ fontSize: 12, color: c.muted, fontWeight: 600, whiteSpace: "nowrap" }}>or sign up with email</span>
                        <div style={{ flex: 1, height: 1, background: c.border }} />
                    </div>

                    <form onSubmit={handleSignup}>
                        <div style={{ marginBottom: 16 }}>
                            <label style={{ fontSize: 13, fontWeight: 600, color: c.sub, display: "block", marginBottom: 7 }}>Full name</label>
                            <input type="text" value={name} className="auth-input" onChange={e => setName(e.target.value)} placeholder="Jane Doe" autoComplete="name" />
                        </div>

                        <div style={{ marginBottom: 16 }}>
                            <label style={{ fontSize: 13, fontWeight: 600, color: c.sub, display: "block", marginBottom: 7 }}>Email address</label>
                            <input type="email" value={email} className="auth-input" onChange={e => setEmail(e.target.value)} placeholder="you@example.com" autoComplete="email" />
                        </div>

                        <div style={{ marginBottom: 16 }}>
                            <label style={{ fontSize: 13, fontWeight: 600, color: c.sub, display: "block", marginBottom: 7 }}>Password</label>
                            <div style={{ position: "relative" }}>
                                <input type={showPass ? "text" : "password"} value={password} className="auth-input"
                                    onChange={e => setPassword(e.target.value)} placeholder="At least 6 characters"
                                    autoComplete="new-password" style={{ paddingRight: 48 }} />
                                <button type="button" onClick={() => setShowPass(!showPass)} style={{ position: "absolute", right: 14, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer", color: c.muted, fontSize: 15, padding: 0 }}>
                                    {showPass ? "🙈" : "👁"}
                                </button>
                            </div>
                            {password && (
                                <div style={{ marginTop: 8 }}>
                                    <div style={{ height: 3, background: c.border, borderRadius: 100, overflow: "hidden" }}>
                                        <div style={{ height: 3, width: `${str.pct}%`, background: strColor, borderRadius: 100, transition: "width .3s, background .3s" }} />
                                    </div>
                                    <p style={{ fontSize: 11, color: strColor, margin: "5px 0 0", fontWeight: 700 }}>{str.label} password</p>
                                </div>
                            )}
                        </div>

                        <div style={{ marginBottom: 28 }}>
                            <label style={{ fontSize: 13, fontWeight: 600, color: c.sub, display: "block", marginBottom: 7 }}>Confirm password</label>
                            <input type="password" value={confirm} className="auth-input"
                                onChange={e => setConfirm(e.target.value)} placeholder="••••••••"
                                autoComplete="new-password"
                                style={{ border: `1px solid ${confirm && confirm !== password ? c.red : c.borderI}` }} />
                            {confirm && confirm !== password && (
                                <p style={{ fontSize: 11.5, color: c.red, margin: "6px 0 0", fontWeight: 600 }}>Passwords don't match</p>
                            )}
                        </div>

                        <button type="submit" className="primary-btn" disabled={loading}>
                            {loading ? <><Spinner />Creating account…</> : "Create Free Account"}
                        </button>
                    </form>

                    <p style={{ textAlign: "center", fontSize: 13.5, color: c.sub, marginTop: 24, marginBottom: 0 }}>
                        Already have an account?{" "}
                        <Link to="/login" className="text-link">Sign in</Link>
                    </p>
                </div>
            </div>
        </div>
    );
}