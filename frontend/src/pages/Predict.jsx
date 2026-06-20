import { useState, useEffect, useRef } from "react";
import { predictDisease } from "../services/Api";
import { useTheme } from "../context/ThemeContext";
import { useToast } from "../context/ToastContext";

const PALETTE_KEY = (c) => [
  { bar: c.teal, light: c.tealL, text: c.teal, border: c.tealB },
  { bar: c.blue, light: c.blueL, text: c.blue, border: c.blueB },
  { bar: c.purple, light: c.purpL, text: c.purple, border: c.purpB },
  { bar: c.amber, light: c.ambL, text: c.amber, border: c.ambB },
  { bar: c.slate, light: c.slatL, text: c.sub, border: c.slatB },
];

const CONF_KEY = (c) => ({
  High: { bg: c.tealL, color: c.teal, border: c.tealB },
  Medium: { bg: c.ambL, color: c.amber, border: c.ambB },
  Low: { bg: c.redL, color: c.red, border: c.redB },
});

/* Common symptom dictionary for autocomplete */
const SYMPTOM_BANK = [
  "fatigue", "night blindness", "skin lesions", "joint pain", "vision loss",
  "dry cough", "muscle weakness", "seizures", "hearing loss", "ataxia",
  "progressive vision loss", "angioid streaks", "skin papules", "photophobia",
  "tremor", "speech difficulty", "swallowing difficulty", "abdominal pain",
  "jaundice", "easy bruising", "frequent infections", "growth delay",
  "developmental delay", "cognitive decline", "peripheral neuropathy",
  "muscle cramps", "bone pain", "vision blurring", "double vision",
  "balance problems", "memory loss", "skin rash", "hair loss",
];

const GLOBAL_CSS = (c) => `
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

  @keyframes spin        { to { transform: rotate(360deg) } }
  @keyframes fadeUp      { from { opacity:0; transform:translateY(14px) } to { opacity:1; transform:translateY(0) } }
  @keyframes slideRight  { from { opacity:0; transform:translateX(-12px) } to { opacity:1; transform:translateX(0) } }
  @keyframes barGrow     { from { width:0 } }
  @keyframes pulseBorder { 0%,100%{box-shadow:0 0 0 0 ${c.teal}30} 50%{box-shadow:0 0 0 6px ${c.teal}00} }
  @keyframes shimmer     { 0% { background-position: -600px 0 } 100% { background-position: 600px 0 } }
  @keyframes float       { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }
  @keyframes countUp     { from{opacity:0;transform:scale(.8)} to{opacity:1;transform:scale(1)} }
  @keyframes dotBlink    { 0%,80%,100%{opacity:0} 40%{opacity:1} }
  @keyframes scanLine    { from{top:0} to{top:100%} }
  @keyframes dropdownIn  { from{opacity:0;transform:translateY(-6px)} to{opacity:1;transform:translateY(0)} }

  .chip:hover  { background:${c.tealL}!important; border-color:${c.tealB}!important; color:${c.teal}!important }
  .dropzone:hover { border-color:${c.teal}!important; background:${c.tealL}!important }
  .result-row  { animation: fadeUp .35s ease both }
  .result-row:hover { box-shadow:0 6px 24px rgba(0,0,0,0.12)!important; transform:translateY(-2px)!important }
  .predict-btn:hover:not(:disabled) { background:${c.tealDark}!important; box-shadow:0 12px 32px ${c.teal}40!important; transform:translateY(-1px) }
  .predict-btn:active:not(:disabled) { transform:scale(.98) }
  .reset-btn:hover { background:${c.bgAlt}!important; border-color:${c.faint}!important }
  .tab-btn:hover   { color:${c.teal}!important; background:${c.tealL}!important }
  .ac-item:hover   { background:${c.tealL}!important }

  .skeleton {
    background: linear-gradient(90deg, ${c.cardAlt} 25%, ${c.border} 50%, ${c.cardAlt} 75%);
    background-size: 600px 100%;
    animation: shimmer 1.4s infinite linear;
    border-radius: 10px;
  }

  @media (max-width: 900px) {
    .predict-grid    { grid-template-columns: 1fr !important }
    .predict-wrapper { padding: 36px 20px !important }
  }
  @media (max-width: 480px) {
    .predict-h1    { font-size: 30px !important }
    .chip-grid     { gap: 6px !important }
    .action-row    { flex-direction: column !important }
    .action-row button { width: 100% !important }
  }
`;

const ConfBadge = ({ conf, c }) => {
  const s = CONF_KEY(c)[conf] || CONF_KEY(c).Low;
  return (
    <span style={{
      padding: "4px 12px", borderRadius: 100, fontSize: 10, fontWeight: 800,
      background: s.bg, color: s.color, border: `1px solid ${s.border}`,
      textTransform: "uppercase", letterSpacing: .8, whiteSpace: "nowrap"
    }}>
      {conf}
    </span>
  );
};

const Dots = ({ color }) => (
  <span style={{ display: "inline-flex", gap: 3, alignItems: "center", marginLeft: 6 }}>
    {[0, 1, 2].map(i => (
      <span key={i} style={{
        width: 5, height: 5, borderRadius: "50%", background: `${color}80`,
        animation: `dotBlink 1.2s ${i * 0.2}s ease-in-out infinite`
      }} />
    ))}
  </span>
);

const ScanOverlay = ({ color }) => (
  <div style={{ position: "absolute", inset: 0, overflow: "hidden", borderRadius: 14, pointerEvents: "none" }}>
    <div style={{
      position: "absolute", left: 0, right: 0, height: 2,
      background: `linear-gradient(90deg,transparent,${color}99,transparent)`,
      animation: "scanLine 1.8s ease-in-out infinite", top: 0
    }} />
  </div>
);

const SkeletonCard = ({ delay, c }) => (
  <div style={{
    border: `1px solid ${c.border}`, borderRadius: 16, padding: "18px 20px",
    animation: `fadeUp .4s ${delay}s ease both`, opacity: 0
  }}>
    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
      <div className="skeleton" style={{ height: 18, width: "55%" }} />
      <div className="skeleton" style={{ height: 18, width: "15%" }} />
    </div>
    <div className="skeleton" style={{ height: 5, width: "100%", marginBottom: 10 }} />
    <div className="skeleton" style={{ height: 22, width: "20%", borderRadius: 100 }} />
  </div>
);

const StatPill = ({ icon, label, value, color, bg, border }) => (
  <div style={{
    background: bg, border: `1px solid ${border}`, borderRadius: 14, padding: "14px 18px",
    display: "flex", alignItems: "center", gap: 12, animation: "fadeUp .5s ease both"
  }}>
    <span style={{ fontSize: 22 }}>{icon}</span>
    <div>
      <p style={{ fontSize: 10, color, opacity: .7, textTransform: "uppercase", letterSpacing: .9, margin: "0 0 3px", fontWeight: 800 }}>{label}</p>
      <p style={{
        fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 22, fontWeight: 800, color, margin: 0,
        animation: "countUp .6s ease"
      }}>{value}</p>
    </div>
  </div>
);

function Predict() {
  const { c } = useTheme();
  const toast = useToast();
  const PALETTE = PALETTE_KEY(c);

  const [symptoms, setSymptoms] = useState("");
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [phase, setPhase] = useState("");
  const [activeTab, setActiveTab] = useState("symptoms");
  const [charCount, setCharCount] = useState(0);
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggest, setShowSuggest] = useState(false);
  const resultsRef = useRef(null);
  const textareaRef = useRef(null);

  const delay = (ms) => new Promise(r => setTimeout(r, ms));

  const runPhases = async (fn) => {
    setPhase("parsing");
    await delay(700);
    setPhase("analyzing");
    await delay(900);
    setPhase("ranking");
    const data = await fn();
    await delay(400);
    setPhase("done");
    return data;
  };

  const handlePredict = async () => {
    if (!symptoms.trim()) {
      toast.warn("Please enter at least one symptom before predicting.");
      return;
    }
    try {
      setLoading(true); setSubmitted(true); setResults([]);
      const data = await runPhases(() => predictDisease(symptoms, image));
      const preds = data?.predictions || [];
      setResults(preds);
      if (preds.length) {
        toast.success(`Found ${preds.length} potential match${preds.length > 1 ? "es" : ""}. Top result: ${preds[0].disease}.`);
      } else {
        toast.info("No predictions returned for these symptoms.");
      }
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    } catch (e) {
      console.error(e);
      setPhase("done");
      toast.error("Prediction failed. Check your API connection and try again.");
    } finally { setLoading(false); }
  };

  const handleFile = (file) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      toast.error("Please upload a valid image file (JPG or PNG).");
      return;
    }
    setImage(file); setPreview(URL.createObjectURL(file));
    toast.success("Scan uploaded successfully.");
  };

  const reset = () => {
    setSymptoms(""); setImage(null); setPreview(null);
    setResults([]); setSubmitted(false); setPhase(""); setCharCount(0);
    setSuggestions([]); setShowSuggest(false);
  };

  /* ── Autocomplete logic ── */
  const handleSymptomChange = (val) => {
    setSymptoms(val); setCharCount(val.length);
    const lastSegment = val.split(/[,\n]/).pop().trim().toLowerCase();
    if (lastSegment.length >= 2) {
      const matches = SYMPTOM_BANK.filter(s =>
        s.toLowerCase().includes(lastSegment) && s.toLowerCase() !== lastSegment
      ).slice(0, 6);
      setSuggestions(matches);
      setShowSuggest(matches.length > 0);
    } else {
      setShowSuggest(false);
    }
  };

  const applySuggestion = (s) => {
    const parts = symptoms.split(/[,\n]/);
    parts[parts.length - 1] = ` ${s}`;
    const next = parts.join(",").replace(/^,\s*/, "").trim();
    setSymptoms(next); setCharCount(next.length);
    setShowSuggest(false);
    textareaRef.current?.focus();
  };

  const phaseLabel = {
    parsing: "Parsing symptom vectors…",
    analyzing: "Running diagnostic model…",
    ranking: "Ranking differential diagnoses…",
    done: "Complete",
  };
  const phasePercent = { parsing: 20, analyzing: 60, ranking: 88, done: 100 };

  const QUICK_SYMPTOMS = [
    "fatigue", "night blindness", "skin lesions", "joint pain",
    "vision loss", "dry cough", "muscle weakness", "seizures",
    "hearing loss", "ataxia",
  ];

  return (
    <div style={{ minHeight: "100vh", background: c.bg, color: c.text, fontFamily: "'Inter',sans-serif" }}>
      <style>{GLOBAL_CSS(c)}</style>

      <div className="predict-wrapper" style={{ maxWidth: 1200, margin: "0 auto", padding: "56px 32px 0" }}>
        <div style={{ marginBottom: 40 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14, flexWrap: "wrap" }}>
            <span style={{
              fontSize: 10, fontWeight: 800, color: c.teal, background: c.tealL, border: `1px solid ${c.tealB}`,
              padding: "4px 14px", borderRadius: 100, letterSpacing: 1.2, textTransform: "uppercase"
            }}>
              Diagnostic Engine
            </span>
            <span style={{
              fontSize: 10, fontWeight: 600, color: c.muted, background: c.cardAlt, border: `1px solid ${c.border}`,
              padding: "4px 12px", borderRadius: 100, letterSpacing: .8
            }}>
              Beta · v2.4
            </span>
          </div>
          <h1 className="predict-h1" style={{
            fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 38, fontWeight: 800,
            margin: "0 0 10px", color: c.text, letterSpacing: -1
          }}>Disease Prediction</h1>
          <p style={{ color: c.sub, fontSize: 16, margin: 0, lineHeight: 1.65 }}>
            Enter patient symptoms and optionally upload a biomedical scan for AI-powered differential diagnosis across 49 rare diseases.
          </p>
        </div>

        <div className="predict-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, paddingBottom: 56 }}>

          {/* ══ LEFT PANEL ══ */}
          <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
            <div style={{
              background: c.card, border: `1px solid ${c.border}`, borderRadius: 24, overflow: "hidden",
              boxShadow: "0 2px 16px rgba(0,0,0,0.04)"
            }}>

              <div style={{ display: "flex", borderBottom: `1px solid ${c.border}`, background: c.bgAlt }}>
                {[["symptoms", "🩺", "Symptoms"], ["image", "🔬", "Scan Upload"]].map(([id, icon, label]) => (
                  <button key={id} className="tab-btn" onClick={() => setActiveTab(id)} style={{
                    flex: 1, padding: "16px 20px", border: "none", cursor: "pointer",
                    fontFamily: "'Inter',sans-serif", fontSize: 13, fontWeight: activeTab === id ? 700 : 500,
                    color: activeTab === id ? c.teal : c.sub,
                    background: activeTab === id ? c.card : "transparent",
                    borderBottom: activeTab === id ? `2px solid ${c.teal}` : "2px solid transparent",
                    transition: "all .15s", display: "flex", alignItems: "center", justifyContent: "center", gap: 7,
                  }}>
                    <span>{icon}</span>{label}
                  </button>
                ))}
              </div>

              <div style={{ padding: 28 }}>

                {activeTab === "symptoms" && (
                  <div style={{ animation: "fadeUp .3s ease" }}>
                    <label style={{ fontSize: 11, fontWeight: 800, color: c.muted, letterSpacing: .8, textTransform: "uppercase", marginBottom: 9, display: "block" }}>
                      Describe Symptoms
                    </label>
                    <div style={{ position: "relative" }}>
                      <textarea ref={textareaRef} rows={6}
                        placeholder="e.g. progressive vision loss, angioid streaks, skin papules on neck, fatigue, night blindness"
                        value={symptoms}
                        onChange={e => handleSymptomChange(e.target.value)}
                        onBlur={() => setTimeout(() => setShowSuggest(false), 150)}
                        style={{
                          width: "100%", background: c.bgAlt, border: `1.5px solid ${c.borderI}`,
                          borderRadius: 14, padding: "13px 15px", color: c.text, fontSize: 14,
                          outline: "none", fontFamily: "'Inter',sans-serif", resize: "none",
                          boxSizing: "border-box", lineHeight: 1.7, transition: "border-color .2s, box-shadow .2s"
                        }}
                        onFocus={e => { e.target.style.borderColor = c.teal; e.target.style.boxShadow = `0 0 0 3px ${c.teal}18`; }}
                      />
                      {charCount > 0 && (
                        <span style={{ position: "absolute", bottom: 10, right: 12, fontSize: 11, color: c.muted, fontWeight: 500 }}>
                          {charCount} chars
                        </span>
                      )}

                      {/* Autocomplete dropdown */}
                      {showSuggest && (
                        <div style={{
                          position: "absolute", left: 0, right: 0, top: "100%", marginTop: 6,
                          background: c.card, border: `1px solid ${c.border}`, borderRadius: 12,
                          boxShadow: "0 12px 32px rgba(0,0,0,0.14)", zIndex: 20, overflow: "hidden",
                          animation: "dropdownIn .15s ease",
                        }}>
                          <p style={{
                            fontSize: 10, color: c.muted, fontWeight: 700, textTransform: "uppercase",
                            letterSpacing: .7, padding: "9px 14px 6px", margin: 0
                          }}>Suggestions</p>
                          {suggestions.map(s => (
                            <div key={s} className="ac-item" onClick={() => applySuggestion(s)} style={{
                              padding: "9px 14px", fontSize: 13.5, color: c.text, cursor: "pointer",
                              display: "flex", alignItems: "center", gap: 8, transition: "background .12s",
                            }}>
                              <span style={{ color: c.teal, fontSize: 13 }}>+</span>{s}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                    <p style={{ fontSize: 12, color: c.muted, marginTop: 7, marginBottom: 20 }}>
                      Separate multiple symptoms with commas or new lines. Start typing for suggestions.
                    </p>

                    <label style={{ fontSize: 11, fontWeight: 800, color: c.muted, letterSpacing: .8, textTransform: "uppercase", marginBottom: 9, display: "block" }}>Quick Add</label>
                    <div className="chip-grid" style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
                      {QUICK_SYMPTOMS.map(s => {
                        const active = symptoms.toLowerCase().includes(s);
                        return (
                          <button key={s} className={active ? "" : "chip"} onClick={() => !active && setSymptoms(p => { const n = p ? `${p}, ${s}` : s; setCharCount(n.length); return n; })}
                            style={{
                              background: active ? c.tealL : c.cardAlt,
                              border: `1px solid ${active ? c.tealB : c.borderI}`,
                              color: active ? c.teal : c.sub,
                              padding: "6px 13px", borderRadius: 100, fontSize: 12, cursor: active ? "default" : "pointer",
                              fontFamily: "'Inter',sans-serif", fontWeight: active ? 700 : 500, transition: "all .15s",
                              display: "flex", alignItems: "center", gap: 5,
                            }}>
                            {active ? "✓" : "+"} {s}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}

                {activeTab === "image" && (
                  <div style={{ animation: "fadeUp .3s ease" }}>
                    <div className="dropzone"
                      onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                      onDragLeave={() => setDragOver(false)}
                      onDrop={e => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
                      onClick={() => document.getElementById("img-input").click()}
                      style={{
                        border: `2px dashed ${dragOver ? c.teal : c.faint}`,
                        borderRadius: 18, padding: preview ? "0" : "52px 24px",
                        textAlign: "center", background: dragOver ? c.tealL : c.bgAlt,
                        transition: "all .2s", overflow: "hidden", cursor: "pointer",
                        position: "relative",
                      }}>
                      {preview ? (
                        <>
                          <img src={preview} alt="scan preview" style={{ width: "100%", maxHeight: 200, objectFit: "cover", display: "block" }} />
                          <ScanOverlay color={c.teal} />
                          <div style={{
                            position: "absolute", top: 10, right: 10, background: `${c.card}EE`, backdropFilter: "blur(8px)",
                            border: `1px solid ${c.tealB}`, borderRadius: 9, padding: "5px 11px",
                            fontSize: 11, color: c.teal, fontWeight: 800, display: "flex", alignItems: "center", gap: 5
                          }}>
                            <span style={{ width: 6, height: 6, borderRadius: "50%", background: c.teal, display: "inline-block" }} />
                            Scan loaded
                          </div>
                          <button onClick={e => { e.stopPropagation(); setImage(null); setPreview(null); toast.info("Scan removed."); }}
                            style={{
                              position: "absolute", top: 10, left: 10, background: `${c.card}EE`,
                              backdropFilter: "blur(8px)", border: `1px solid ${c.border}`, borderRadius: 8,
                              padding: "5px 11px", fontSize: 11, color: c.sub, cursor: "pointer", fontWeight: 600
                            }}>
                            ✕ Remove
                          </button>
                        </>
                      ) : (
                        <div style={{ animation: "float 3s ease infinite" }}>
                          <div style={{
                            width: 64, height: 64, borderRadius: "50%", background: c.tealL, border: `1.5px solid ${c.tealB}`,
                            display: "flex", alignItems: "center", justifyContent: "center", fontSize: 28, margin: "0 auto 16px"
                          }}>🩻</div>
                          <p style={{ color: c.text, fontSize: 14, margin: "0 0 5px", fontWeight: 600 }}>Drag & drop or click to upload</p>
                          <p style={{ color: c.muted, fontSize: 12, margin: "0 0 16px" }}>JPG, PNG · MRI, CT, Dermoscopy, Histopathology</p>
                          <span style={{
                            fontSize: 11, color: c.teal, background: c.tealL, border: `1px solid ${c.tealB}`,
                            padding: "4px 14px", borderRadius: 100, fontWeight: 700
                          }}>Browse Files</span>
                        </div>
                      )}
                    </div>
                    <input id="img-input" type="file" hidden accept="image/*" onChange={e => handleFile(e.target.files[0])} />

                    <div style={{ marginTop: 18, padding: "14px 18px", background: c.bgAlt, border: `1px solid ${c.border}`, borderRadius: 12 }}>
                      <p style={{ fontSize: 12, color: c.sub, margin: 0, lineHeight: 1.6 }}>
                        <strong style={{ color: c.text }}>Supported modalities:</strong> MRI brain scans, CT abdomen/thorax, dermoscopy images, retinal fundus, and histopathology slides.
                      </p>
                    </div>
                  </div>
                )}
              </div>

              <div style={{ padding: "0 28px 28px", display: "flex", flexDirection: "column", gap: 12 }}>
                {loading && (
                  <div style={{ animation: "fadeUp .3s ease" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 7 }}>
                      <span style={{ fontSize: 12, color: c.teal, fontWeight: 700, display: "flex", alignItems: "center" }}>
                        {phaseLabel[phase]}<Dots color={c.teal} />
                      </span>
                      <span style={{ fontSize: 12, color: c.teal, fontWeight: 700 }}>{phasePercent[phase]}%</span>
                    </div>
                    <div style={{ height: 5, background: c.border, borderRadius: 100, overflow: "hidden" }}>
                      <div style={{
                        height: 5, background: `linear-gradient(90deg,${c.teal},${c.blue})`,
                        borderRadius: 100, transition: "width .6s ease",
                        width: `${phasePercent[phase]}%`
                      }} />
                    </div>
                  </div>
                )}

                <div className="action-row" style={{ display: "flex", gap: 10 }}>
                  <button className="predict-btn" onClick={handlePredict} disabled={loading} style={{
                    flex: 1, background: loading ? `${c.teal}40` : c.teal,
                    color: loading ? c.sub : "#fff", border: "none",
                    padding: "15px 20px", borderRadius: 13, fontWeight: 800, fontSize: 15,
                    cursor: loading ? "not-allowed" : "pointer",
                    fontFamily: "'Inter',sans-serif", transition: "all .2s",
                    boxShadow: loading ? "none" : `0 6px 20px ${c.teal}33`,
                    display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
                  }}>
                    {loading ? (
                      <>
                        <span style={{ width: 17, height: 17, border: `2.5px solid ${c.teal}40`, borderTop: `2.5px solid ${c.teal}`, borderRadius: "50%", animation: "spin .8s linear infinite", display: "inline-block" }} />
                        Analyzing<Dots color={c.teal} />
                      </>
                    ) : (
                      <><span>🔍</span> Predict Disease</>
                    )}
                  </button>
                  <button className="reset-btn" onClick={reset} style={{
                    padding: "15px 20px", borderRadius: 13, border: `1.5px solid ${c.borderI}`,
                    background: c.card, color: c.sub, cursor: "pointer",
                    fontFamily: "'Inter',sans-serif", fontSize: 14, fontWeight: 600, transition: "all .15s",
                  }}>Reset</button>
                </div>

                <div style={{ display: "flex", gap: 7, flexWrap: "wrap" }}>
                  {[
                    image ? "🔬 Multimodal (Symptom + Image)" : "🧬 Symptom Model",
                    "49 diseases",
                    "TF-IDF + LR",
                  ].map(t => (
                    <span key={t} style={{
                      fontSize: 11, color: c.sub, background: c.cardAlt, border: `1px solid ${c.border}`,
                      padding: "4px 11px", borderRadius: 100, fontWeight: 500
                    }}>{t}</span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* ══ RIGHT PANEL ══ */}
          <div ref={resultsRef} style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            {loading && (
              <div style={{
                background: c.card, border: `1px solid ${c.border}`, borderRadius: 24, padding: 28,
                boxShadow: "0 2px 16px rgba(0,0,0,0.04)", animation: "pulseBorder 2s ease infinite"
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <div style={{ width: 36, height: 36, borderRadius: 10, background: c.tealL, display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <span style={{ width: 20, height: 20, border: `2.5px solid ${c.teal}30`, borderTop: `2.5px solid ${c.teal}`, borderRadius: "50%", animation: "spin .8s linear infinite", display: "inline-block" }} />
                    </div>
                    <div>
                      <p style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 16, fontWeight: 800, color: c.text, margin: "0 0 2px" }}>
                        {phaseLabel[phase]}<Dots color={c.teal} />
                      </p>
                      <p style={{ fontSize: 12, color: c.muted, margin: 0 }}>AI diagnostic engine active</p>
                    </div>
                  </div>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  {[0, .1, .2].map(d => <SkeletonCard key={d} delay={d} c={c} />)}
                </div>
              </div>
            )}

            {!loading && !submitted && (
              <div style={{
                background: c.card, border: `1px solid ${c.border}`, borderRadius: 24, padding: "48px 28px",
                textAlign: "center", boxShadow: "0 2px 16px rgba(0,0,0,0.04)", animation: "fadeUp .5s ease"
              }}>
                <div style={{
                  width: 88, height: 88, borderRadius: "50%", background: c.tealL, border: `2px solid ${c.tealB}`,
                  display: "flex", alignItems: "center", justifyContent: "center", fontSize: 36,
                  margin: "0 auto 20px", animation: "float 4s ease infinite"
                }}>🧠</div>
                <h3 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 20, fontWeight: 800, color: c.text, margin: "0 0 8px" }}>
                  Ready for Diagnosis
                </h3>
                <p style={{ fontSize: 14, color: c.sub, margin: "0 0 24px", lineHeight: 1.7, maxWidth: 280, marginLeft: "auto", marginRight: "auto" }}>
                  Enter symptoms on the left. Add a scan for multimodal analysis.
                </p>
                <div style={{ display: "flex", gap: 10, justifyContent: "center", flexWrap: "wrap" }}>
                  {["49 rare diseases", "Confidence scoring", "Top-3 ranking"].map(f => (
                    <span key={f} style={{
                      fontSize: 12, color: c.teal, background: c.tealL, border: `1px solid ${c.tealB}`,
                      padding: "5px 14px", borderRadius: 100, fontWeight: 600
                    }}>✓ {f}</span>
                  ))}
                </div>
              </div>
            )}

            {!loading && submitted && results.length === 0 && (
              <div style={{
                background: c.card, border: `1px solid ${c.redB}`, borderRadius: 24, padding: "40px 28px",
                textAlign: "center", animation: "fadeUp .4s ease"
              }}>
                <div style={{ fontSize: 40, marginBottom: 14 }}>⚠️</div>
                <p style={{ fontSize: 16, color: c.sub, fontWeight: 600, margin: "0 0 6px" }}>No results returned</p>
                <p style={{ fontSize: 13, color: c.muted, margin: 0 }}>Check your API connection and try again.</p>
              </div>
            )}

            {!loading && results.length > 0 && (
              <>
                <div style={{
                  background: `linear-gradient(135deg,${c.tealL} 0%,${c.blueL} 100%)`,
                  border: `1.5px solid ${c.tealB}`, borderRadius: 24, padding: "26px 28px",
                  animation: "slideRight .5s ease", boxShadow: `0 4px 20px ${c.teal}18`
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                      <div style={{ width: 36, height: 36, borderRadius: 10, background: c.teal, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16 }}>🥇</div>
                      <div>
                        <p style={{ fontSize: 10, color: c.teal, letterSpacing: 1.2, textTransform: "uppercase", margin: "0 0 1px", fontWeight: 800 }}>Top Diagnosis</p>
                        <p style={{ fontSize: 11, color: c.sub, margin: 0 }}>Highest probability match</p>
                      </div>
                    </div>
                    <ConfBadge conf={results[0].confidence} c={c} />
                  </div>
                  <h3 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 24, fontWeight: 800, color: c.text, margin: "0 0 16px", letterSpacing: -.3 }}>
                    {results[0].disease}
                  </h3>
                  <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                    <span style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 40, fontWeight: 800, color: c.teal, animation: "countUp .6s ease" }}>
                      {results[0].probability}%
                    </span>
                    <div style={{ flex: 1 }}>
                      <div style={{ height: 8, background: `${c.teal}25`, borderRadius: 100, overflow: "hidden" }}>
                        <div style={{
                          height: 8, width: `${results[0].probability}%`, background: c.teal,
                          borderRadius: 100, animation: "barGrow .9s ease"
                        }} />
                      </div>
                      <p style={{ fontSize: 11, color: c.sub, margin: "5px 0 0", fontWeight: 500 }}>Match probability</p>
                    </div>
                  </div>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10 }}>
                  <StatPill icon="🔢" label="Matches" value={results.length} color={c.teal} bg={c.tealL} border={c.tealB} />
                  <StatPill icon="📊" label="Top Prob" value={`${results[0].probability}%`} color={c.blue} bg={c.blueL} border={c.blueB} />
                  <StatPill icon="🎯" label="Confidence" value={results[0].confidence} color={c.purple} bg={c.purpL} border={c.purpB} />
                </div>

                <div style={{
                  background: c.card, border: `1px solid ${c.border}`, borderRadius: 24, overflow: "hidden",
                  boxShadow: "0 2px 16px rgba(0,0,0,0.04)"
                }}>
                  <div style={{
                    padding: "18px 24px", background: c.bgAlt, borderBottom: `1px solid ${c.border}`,
                    display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8
                  }}>
                    <h3 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 15, fontWeight: 700, margin: 0, color: c.text }}>All Predictions</h3>
                    <span style={{
                      fontSize: 11, color: c.teal, background: c.tealL, border: `1px solid ${c.tealB}`,
                      padding: "3px 11px", borderRadius: 100, fontWeight: 700
                    }}>Ranked by probability</span>
                  </div>
                  <div style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 10 }}>
                    {results.map((item, i) => {
                      const p = PALETTE[i % PALETTE.length];
                      return (
                        <div key={i} className="result-row" style={{
                          animationDelay: `${i * 0.08}s`, border: `1px solid ${p.border}`,
                          borderRadius: 16, padding: "16px 18px", background: c.card, transition: "box-shadow .2s, transform .15s",
                        }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                              <span style={{
                                fontSize: 10, fontWeight: 800, background: p.light, color: p.text,
                                border: `1px solid ${p.border}`, padding: "3px 9px", borderRadius: 7
                              }}>#{item.rank}</span>
                              <span style={{ fontWeight: 700, fontSize: 14, color: c.text }}>{item.disease}</span>
                            </div>
                            <ConfBadge conf={item.confidence} c={c} />
                          </div>
                          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: c.muted, marginBottom: 7 }}>
                            <span>Probability</span>
                            <span style={{ color: p.text, fontWeight: 800 }}>{item.probability}%</span>
                          </div>
                          <div style={{ height: 5, background: c.border, borderRadius: 100, overflow: "hidden" }}>
                            <div style={{
                              height: 5, width: `${item.probability}%`, background: p.bar,
                              borderRadius: 100, animation: "barGrow .8s ease"
                            }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div style={{ padding: "14px 18px", background: c.ambL, border: `1px solid ${c.ambB}`, borderRadius: 14 }}>
                  <p style={{ fontSize: 12, color: c.amber, margin: 0, lineHeight: 1.65 }}>
                    <strong>⚠ Clinical Disclaimer:</strong> These predictions are AI-generated for research purposes only. Always consult a licensed clinician before making diagnostic or treatment decisions.
                  </p>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Predict;
