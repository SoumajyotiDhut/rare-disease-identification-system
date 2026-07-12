import { useState, useRef, useEffect } from "react";
import { useTheme } from "../context/ThemeContext";
import { useAuth } from "../context/AuthContext";
import { useToast } from "../context/ToastContext";
import { apiUrl } from "../lib/platform";

const CSS = (c) => `
  @keyframes chatIn   { from{opacity:0;transform:translateY(16px) scale(.97)} to{opacity:1;transform:translateY(0) scale(1)} }
  @keyframes typingDot{ 0%,60%,100%{opacity:.3;transform:translateY(0)} 30%{opacity:1;transform:translateY(-3px)} }

  .chat-fab {
    position:fixed; bottom:24px; right:24px; z-index:500;
    width:56px; height:56px; border-radius:50%;
    background:${c.gradPrimary}; border:none; cursor:pointer;
    display:flex; align-items:center; justify-content:center;
    box-shadow:${c.shadowTeal}; transition:transform .2s;
  }
  .chat-fab:hover { transform:scale(1.06); }

  .chat-panel {
    position:fixed; bottom:92px; right:24px; z-index:500;
    width:380px; max-width:calc(100vw - 32px);
    height:560px; max-height:calc(100vh - 140px);
    background:${c.card}; border:1px solid ${c.border}; border-top:2px solid ${c.teal};
    border-radius:8px; box-shadow:${c.shadowXl};
    display:flex; flex-direction:column; overflow:hidden;
    animation:chatIn .25s cubic-bezier(.2,.7,.3,1) both;
  }

  .chat-msg-user {
    align-self:flex-end; background:${c.teal}; color:#fff;
    border-radius:12px 12px 2px 12px; padding:10px 14px;
    font-size:13.5px; line-height:1.55; max-width:82%;
  }
  .chat-msg-bot {
    align-self:flex-start; background:${c.cardAlt}; color:${c.text};
    border:1px solid ${c.border};
    border-radius:12px 12px 12px 2px; padding:10px 14px;
    font-size:13.5px; line-height:1.6; max-width:88%; white-space:pre-wrap;
  }

  .chat-send-btn {
    width:38px; height:38px; border-radius:4px; border:none;
    background:${c.text}; color:${c.bg}; cursor:pointer;
    display:flex; align-items:center; justify-content:center;
    flex-shrink:0; transition:background .2s;
  }
  .chat-send-btn:hover:not(:disabled) { background:${c.teal}; color:#fff; }
  .chat-send-btn:disabled { opacity:.5; cursor:not-allowed; }

  .chat-input {
    flex:1; border:1px solid ${c.borderI}; border-radius:4px;
    background:${c.bgDeep}; color:${c.text}; font-size:13.5px;
    padding:10px 12px; outline:none; font-family:'Inter',sans-serif;
    resize:none; max-height:80px;
  }
  .chat-input:focus { border-color:${c.teal}; }

  @media(max-width:480px){
    .chat-panel { right:16px; left:16px; width:auto; bottom:84px; height:min(560px, calc(100vh - 120px)); }
    .chat-fab   { right:16px; bottom:16px; }
  }
`;

const IconChat = ({ color }) => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
        <path d="M4 5h16a1 1 0 011 1v10a1 1 0 01-1 1H9l-5 4v-4H4a1 1 0 01-1-1V6a1 1 0 011-1z" stroke={color} strokeWidth="1.8" strokeLinejoin="round" />
    </svg>
);
const IconSend = ({ color }) => (
    <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
        <path d="M2 10l16-7-6 16-2.5-6.5L2 10z" stroke={color} strokeWidth="1.6" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
);
const IconClose = ({ color }) => (
    <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
        <path d="M4 4l12 12M16 4L4 16" stroke={color} strokeWidth="1.8" strokeLinecap="round" />
    </svg>
);

const GREETING = "Hi — I'm the AI DOC assistant. I can answer general health questions, but I can't diagnose you personally or replace a doctor. If this is a medical emergency, please call emergency services right away. What can I help with?";

export default function ChatWidget() {
    const { c } = useTheme();
    const { user } = useAuth();
    const toast = useToast();
    const [open, setOpen] = useState(false);
    const [messages, setMessages] = useState([{ role: "assistant", content: GREETING }]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const scrollRef = useRef(null);

    useEffect(() => {
        if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }, [messages, loading]);

    // Gate behind auth: reduces anonymous abuse of the API-key-backed endpoint.
    // This is not full protection on its own — see api/chat.js for the rest
    // of the guardrails, and the note below about hardening this further.
    if (!user) return null;

    const send = async () => {
        const text = input.trim();
        if (!text || loading) return;
        const next = [...messages, { role: "user", content: text }];
        setMessages(next);
        setInput("");
        setLoading(true);
        try {
            const res = await fetch(apiUrl("/api/chat"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ messages: next }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || "Request failed");
            setMessages((m) => [...m, { role: "assistant", content: data.reply }]);
        } catch (err) {
            toast.error("The assistant couldn't respond. Please try again.");
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            send();
        }
    };

    return (
        <>
            <style>{CSS(c)}</style>

            {open && (
                <div className="chat-panel" role="dialog" aria-label="AI DOC medical assistant chat">
                    <div style={{ padding: "16px 18px", borderBottom: `1px solid ${c.border}`, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <div>
                            <p style={{ fontFamily: "'Fraunces',serif", fontSize: 15, fontWeight: 600, color: c.text, margin: 0 }}>AI DOC Assistant</p>
                            <p style={{ fontSize: 10.5, color: c.muted, margin: "2px 0 0", fontFamily: "'IBM Plex Mono',monospace", letterSpacing: "0.04em" }}>GENERAL HEALTH INFO · NOT A DIAGNOSIS</p>
                        </div>
                        <button onClick={() => setOpen(false)} aria-label="Close chat" style={{ background: "none", border: "none", cursor: "pointer", color: c.muted, padding: 4 }}>
                            <IconClose color={c.muted} />
                        </button>
                    </div>

                    <div ref={scrollRef} style={{ flex: 1, overflowY: "auto", padding: "16px 16px", display: "flex", flexDirection: "column", gap: 10 }}>
                        {messages.map((m, i) => (
                            <div key={i} className={m.role === "user" ? "chat-msg-user" : "chat-msg-bot"}>{m.content}</div>
                        ))}
                        {loading && (
                            <div className="chat-msg-bot" style={{ display: "flex", gap: 5, padding: "12px 14px" }}>
                                {[0, 1, 2].map((i) => (
                                    <span key={i} style={{ width: 5, height: 5, borderRadius: "50%", background: c.muted, animation: `typingDot 1.1s ease-in-out ${i * 0.15}s infinite` }} />
                                ))}
                            </div>
                        )}
                    </div>

                    <div style={{ padding: "10px 6px 6px", borderTop: `1px solid ${c.border}`, background: c.ambL }}>
                        <p style={{ fontSize: 10, color: c.amber, margin: "0 0 8px", padding: "0 10px", lineHeight: 1.5 }}>
                            In a medical emergency, call your local emergency number immediately — do not wait for a chat response.
                        </p>
                    </div>

                    <div style={{ padding: "10px 12px 12px", display: "flex", gap: 8, alignItems: "flex-end" }}>
                        <textarea
                            className="chat-input"
                            rows={1}
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="Ask a general health question…"
                            disabled={loading}
                        />
                        <button className="chat-send-btn" onClick={send} disabled={loading || !input.trim()} aria-label="Send message">
                            <IconSend color={loading || !input.trim() ? c.muted : c.bg} />
                        </button>
                    </div>
                </div>
            )}

            <button className="chat-fab" onClick={() => setOpen(!open)} aria-label={open ? "Close chat" : "Open medical assistant chat"}>
                {open ? <IconClose color="#fff" /> : <IconChat color="#fff" />}
            </button>
        </>
    );
}