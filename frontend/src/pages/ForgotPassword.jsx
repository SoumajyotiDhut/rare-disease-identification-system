import { useState } from "react";
import { Link } from "react-router-dom";
import { useTheme } from "../context/ThemeContext";
import { useToast } from "../context/ToastContext";
import { resetPassword, getAuthErrorMessage } from "../firebase";

const CSS = (c) => `
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&display=swap');
  @keyframes fadeUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
  @keyframes spin   { to{transform:rotate(360deg)} }
  .auth-card     { animation: fadeUp .5s cubic-bezier(.2,.7,.3,1) both }
  .auth-input:focus { border-color:${c.teal}!important; box-shadow:0 0 0 3px ${c.teal}18 }
  .auth-submit:hover:not(:disabled)  { transform:translateY(-1px); box-shadow:0 10px 24px ${c.teal}40 }
  .auth-link:hover   { text-decoration:underline }
  @media(max-width:480px){ .auth-pad { padding:32px 22px!important } }
`;

export default function ForgotPassword() {
    const { c } = useTheme();
    const toast = useToast();

    const [email, setEmail] = useState("");
    const [loading, setLoading] = useState(false);
    const [sent, setSent] = useState(false);

    const handleReset = async (e) => {
        e.preventDefault();
        if (!email.trim()) {
            toast.error("Please enter your email address.");
            return;
        }
        setLoading(true);
        try {
            await resetPassword(email.trim());
            setSent(true);
            toast.success("Password reset email sent.");
        } catch (err) {
            // Don't reveal whether an account exists — show the same
            // success state regardless, to avoid leaking which emails are registered.
            if (err?.code === "auth/user-not-found" || err?.code === "auth/invalid-email") {
                setSent(true);
            } else {
                toast.error(getAuthErrorMessage(err));
            }
        } finally {
            setLoading(false);
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
                <Link to="/" style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: 10, marginBottom: 32 }}>
                    <div style={{
                        width: 38, height: 38, borderRadius: 11,
                        background: `linear-gradient(135deg,${c.teal},${c.blue})`,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontWeight: 800, fontSize: 14, color: "#fff", boxShadow: `0 4px 14px ${c.teal}40`,
                    }}>AI</div>
                    <span style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontWeight: 800, fontSize: 17, color: c.text }}>AI DOC</span>
                </Link>

                {sent ? (
                    <>
                        <div style={{
                            width: 52, height: 52, borderRadius: 14, background: `${c.teal}18`,
                            display: "flex", alignItems: "center", justifyContent: "center",
                            fontSize: 24, marginBottom: 18,
                        }}>
                            ✉️
                        </div>
                        <h1 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 24, fontWeight: 800, color: c.text, margin: "0 0 8px", letterSpacing: -0.5 }}>
                            Check your inbox
                        </h1>
                        <p style={{ fontSize: 14, color: c.sub, margin: "0 0 28px", lineHeight: 1.6 }}>
                            If an account exists for <strong style={{ color: c.text }}>{email.trim()}</strong>, we've sent a link to reset your password. It may take a minute to arrive — check spam too.
                        </p>
                        <button
                            onClick={() => setSent(false)}
                            className="auth-submit"
                            style={{
                                width: "100%", padding: "14px 16px", borderRadius: 12, border: `1.5px solid ${c.borderI}`,
                                background: "transparent", color: c.text, fontSize: 14.5, fontWeight: 700,
                                cursor: "pointer", fontFamily: "'Inter',sans-serif", transition: "transform .15s",
                            }}
                        >
                            Try a different email
                        </button>
                    </>
                ) : (
                    <>
                        <h1 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 26, fontWeight: 800, color: c.text, margin: "0 0 6px", letterSpacing: -0.5 }}>
                            Reset your password
                        </h1>
                        <p style={{ fontSize: 14, color: c.sub, margin: "0 0 28px", lineHeight: 1.6 }}>
                            Enter the email address linked to your account and we'll send you a link to reset your password.
                        </p>

                        <form onSubmit={handleReset}>
                            <label style={{ fontSize: 12.5, fontWeight: 700, color: c.sub, display: "block", marginBottom: 7 }}>Email</label>
                            <input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                placeholder="you@example.com"
                                className="auth-input"
                                autoFocus
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
                                        Sending…
                                    </>
                                ) : "Send reset link"}
                            </button>
                        </form>
                    </>
                )}

                <p style={{ textAlign: "center", fontSize: 13.5, color: c.sub, marginTop: 26 }}>
                    Remembered it?{" "}
                    <Link to="/login" className="auth-link" style={{ color: c.teal, fontWeight: 700, textDecoration: "none" }}>
                        Back to sign in
                    </Link>
                </p>
            </div>
        </div>
    );
}