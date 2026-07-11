import { useState } from "react";
import { Link } from "react-router-dom";
import { useTheme } from "../context/ThemeContext";
import { useToast } from "../context/ToastContext";
import { resetPassword, getAuthErrorMessage } from "../firebase";

const CSS = (c) => `
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Mono:wght@500;600&display=swap');
  @keyframes fadeUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
  @keyframes spin   { to{transform:rotate(360deg)} }
  @keyframes checkPop { 0%{transform:scale(0) rotate(-10deg)} 70%{transform:scale(1.15) rotate(3deg)} 100%{transform:scale(1) rotate(0deg)} }
  .auth-panel { animation: fadeUp .5s cubic-bezier(.2,.7,.3,1) both }
  .auth-input {
    width:100%; padding:13px 16px; border-radius:4px;
    border:1px solid ${c.borderI}; background:${c.bgDeep};
    color:${c.text}; font-size:14.5px; outline:none;
    font-family:'Inter',sans-serif; box-sizing:border-box;
    transition:border-color .2s, box-shadow .2s, background .2s;
  }
  .auth-input:focus { border-color:${c.teal}!important; box-shadow:0 0 0 3px ${c.tealL}; background:${c.card}; }
  .auth-input::placeholder { color:${c.muted} }
  .primary-btn {
    width:100%; padding:15px 16px; border-radius:4px; border:none;
    background:${c.text}; color:${c.bg}; font-size:15px; font-weight:600;
    cursor:pointer; font-family:'Inter',sans-serif;
    display:flex; align-items:center; justify-content:center; gap:10px;
    transition:all .25s;
  }
  .primary-btn:hover:not(:disabled) { background:${c.teal}; color:#fff; transform:translateY(-2px); box-shadow:${c.shadowTeal}; }
  .primary-btn:disabled { opacity:.6; cursor:not-allowed }
  .outline-btn {
    width:100%; padding:13px 16px; border-radius:4px;
    border:1px solid ${c.borderI}; background:transparent;
    color:${c.text}; font-size:14.5px; font-weight:600;
    cursor:pointer; font-family:'Inter',sans-serif; transition:all .2s;
  }
  .outline-btn:hover { border-color:${c.teal}; color:${c.teal}; background:${c.tealL}; }
  .text-link { color:${c.teal}; font-weight:600; text-decoration:none; }
  .text-link:hover { text-decoration:underline }
  .success-icon { animation: checkPop .5s cubic-bezier(.2,.7,.3,1) both }
  @media(max-width:480px){
    .fp-card { padding:32px 24px !important }
    .auth-input, .primary-btn, .outline-btn { font-size:16px !important }
  }
`;

const Spinner = () => (
    <span style={{ width: 18, height: 18, border: "2.5px solid rgba(255,255,255,0.35)", borderTop: "2.5px solid #fff", borderRadius: "50%", animation: "spin .7s linear infinite", display: "inline-block", flexShrink: 0 }} />
);

const IconMail = ({ color }) => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
        <rect x="2.5" y="5" width="19" height="14" rx="2" stroke={color} strokeWidth="1.6" />
        <path d="M3.5 6.5L12 13l8.5-6.5" stroke={color} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
);
const IconKey = ({ color }) => (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
        <circle cx="8" cy="15" r="4" stroke={color} strokeWidth="1.6" />
        <path d="M11 12l9-9M17 6l2.5 2.5M13.8 9.2l2 2" stroke={color} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
);

export default function ForgotPassword() {
    const { c } = useTheme();
    const toast = useToast();
    const [email, setEmail] = useState("");
    const [loading, setLoading] = useState(false);
    const [sent, setSent] = useState(false);

    const handleReset = async (e) => {
        e.preventDefault();
        if (!email.trim()) { toast.error("Please enter your email address."); return; }
        setLoading(true);
        try {
            await resetPassword(email.trim());
            setSent(true);
        } catch (err) {
            if (err?.code === "auth/user-not-found" || err?.code === "auth/invalid-email") setSent(true);
            else toast.error(getAuthErrorMessage(err));
        } finally { setLoading(false); }
    };

    return (
        <div style={{ minHeight: "100vh", background: c.bg, fontFamily: "'Inter',sans-serif", display: "flex", alignItems: "center", justifyContent: "center", padding: "40px 20px" }}>
            <style>{CSS(c)}</style>

            <div className="auth-panel" style={{ width: "100%", maxWidth: 460 }}>

                {/* Logo */}
                <Link to="/" style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: 10, marginBottom: 40, justifyContent: "center" }}>
                    <div style={{ width: 38, height: 38, borderRadius: 4, background: c.gradPrimary, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'Fraunces',serif", fontStyle: "italic", fontWeight: 600, fontSize: 15, color: "#fff" }}>AI</div>
                    <span style={{ fontFamily: "'Fraunces',serif", fontWeight: 600, fontSize: 18, color: c.text, letterSpacing: "-0.02em" }}>AI DOC</span>
                </Link>

                <div className="fp-card" style={{ background: c.card, border: `1px solid ${c.border}`, borderTop: `2px solid ${c.teal}`, padding: "44px 44px", boxShadow: c.shadowLg }}>

                    {sent ? (
                        /* ── Success state ── */
                        <div style={{ textAlign: "center" }}>
                            <div className="success-icon" style={{
                                width: 68, height: 68, borderRadius: "50%",
                                background: c.tealL, border: `1.5px solid ${c.tealB}`,
                                display: "flex", alignItems: "center", justifyContent: "center",
                                fontSize: 30, margin: "0 auto 24px",
                            }}><IconMail color={c.teal} /></div>

                            <h1 style={{ fontFamily: "'Fraunces',serif", fontSize: 25, fontWeight: 600, color: c.text, margin: "0 0 12px", letterSpacing: "-0.02em" }}>
                                Check your inbox
                            </h1>
                            <p style={{ fontSize: 14.5, color: c.sub, lineHeight: 1.7, margin: "0 0 32px" }}>
                                If an account exists for{" "}
                                <strong style={{ color: c.text }}>{email.trim()}</strong>,
                                we've sent a password reset link. Check your spam folder too.
                            </p>

                            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                                <button className="outline-btn" onClick={() => setSent(false)}>
                                    Try a different email
                                </button>
                                <Link to="/login" style={{ textAlign: "center", fontSize: 14, color: c.sub, textDecoration: "none", padding: "10px 0" }}>
                                    ← <span className="text-link">Back to Sign In</span>
                                </Link>
                            </div>
                        </div>
                    ) : (
                        /* ── Form state ── */
                        <>
                            <div style={{ marginBottom: 32 }}>
                                <div style={{
                                    width: 48, height: 48, borderRadius: 4, background: c.tealL,
                                    border: `1px solid ${c.tealB}`, display: "flex", alignItems: "center",
                                    justifyContent: "center", fontSize: 22, marginBottom: 20,
                                }}><IconKey color={c.teal} /></div>
                                <h1 style={{ fontFamily: "'Fraunces',serif", fontSize: 27, fontWeight: 600, color: c.text, margin: "0 0 8px", letterSpacing: "-0.02em" }}>
                                    Reset password
                                </h1>
                                <p style={{ fontSize: 14.5, color: c.sub, margin: 0, lineHeight: 1.6 }}>
                                    Enter your email and we'll send a link to reset your password.
                                </p>
                            </div>

                            <form onSubmit={handleReset}>
                                <div style={{ marginBottom: 24 }}>
                                    <label style={{ fontSize: 13, fontWeight: 600, color: c.sub, display: "block", marginBottom: 8 }}>
                                        Email address
                                    </label>
                                    <input
                                        type="email" value={email} className="auth-input"
                                        onChange={e => setEmail(e.target.value)}
                                        placeholder="you@example.com"
                                        autoFocus autoComplete="email"
                                    />
                                </div>

                                <button type="submit" className="primary-btn" disabled={loading} style={{ marginBottom: 16 }}>
                                    {loading ? <><Spinner />Sending…</> : "Send Reset Link"}
                                </button>
                            </form>

                            <p style={{ textAlign: "center", fontSize: 13.5, color: c.sub, margin: 0 }}>
                                Remembered it?{" "}
                                <Link to="/login" className="text-link">Back to Sign In</Link>
                            </p>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}