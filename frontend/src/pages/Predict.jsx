import { useState, useEffect, useRef } from "react";
import { predictDisease } from "../services/Api";

/* ─── Design tokens ─── */
const T = {
  teal: "#0B7B6F", tealL: "#EBF8F6", tealB: "#B2E8E2",
  blue: "#1D6FA4", blueL: "#EBF4F9", blueB: "#B3D8EE",
  purple: "#5B3DB8", purpL: "#F2EEF9", purpB: "#C8B8EC",
  amber: "#C05B1A", ambL: "#FFF4EC", ambB: "#F5D8B8",
  slate: "#8FA5B5", slatL: "#F0F5F8", slatB: "#C8D8E4",
  bg: "#F4F8FB", card: "#fff", text: "#0F1C2E",
  sub: "#7A94A8", muted: "#9BB8CC", border: "#E8EFF5",
  borderI: "#DDE8EF",
};

const PALETTE = [
  { bar: T.teal, light: T.tealL, text: T.teal, border: T.tealB },
  { bar: T.blue, light: T.blueL, text: T.blue, border: T.blueB },
  { bar: T.purple, light: T.purpL, text: T.purple, border: T.purpB },
  { bar: T.amber, light: T.ambL, text: T.amber, border: T.ambB },
  { bar: T.slate, light: T.slatL, text: T.slate, border: T.slatB },
];

const CONF_STYLES = {
  High: { bg: "#EBF8F6", color: "#0B7B6F", border: "#B2E8E2" },
  Medium: { bg: "#FFF8EC", color: "#C05B1A", border: "#F5D8B8" },
  Low: { bg: "#FDECED", color: "#B83030", border: "#F0BCBC" },
};

/* ─── Shared CSS injected once ─── */
const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

  @keyframes spin        { to { transform: rotate(360deg) } }
  @keyframes fadeUp      { from { opacity:0; transform:translateY(14px) } to { opacity:1; transform:translateY(0) } }
  @keyframes slideRight  { from { opacity:0; transform:translateX(-12px) } to { opacity:1; transform:translateX(0) } }
  @keyframes barGrow     { from { width:0 } }
  @keyframes pulseBorder { 0%,100%{box-shadow:0 0 0 0 rgba(11,123,111,0.25)} 50%{box-shadow:0 0 0 6px rgba(11,123,111,0)} }
  @keyframes shimmer     {
    0%   { background-position: -600px 0 }
    100% { background-position:  600px 0 }
  }
  @keyframes float       { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }
  @keyframes countUp     { from{opacity:0;transform:scale(.8)} to{opacity:1;transform:scale(1)} }
  @keyframes dotBlink    { 0%,80%,100%{opacity:0} 40%{opacity:1} }
  @keyframes scanLine    { from{top:0} to{top:100%} }

  .chip:hover  { background:#EBF8F6!important; border-color:#B2E8E2!important; color:#0B7B6F!important }
  .dropzone:hover { border-color:#0B7B6F!important; background:#F0FAF9!important }
  .result-row  { animation: fadeUp .35s ease both }
  .result-row:hover { box-shadow:0 6px 24px rgba(15,28,46,0.08)!important; transform:translateY(-2px)!important }
  .predict-btn:hover:not(:disabled) { background:#08635A!important; box-shadow:0 12px 32px rgba(11,123,111,0.30)!important; transform:translateY(-1px) }
  .predict-btn:active:not(:disabled) { transform:scale(.98) }
  .reset-btn:hover { background:#F4F8FB!important; border-color:#C8D8E4!important }
  .tab-btn:hover   { color:#0B7B6F!important; background:#EBF8F6!important }

  .skeleton {
    background: linear-gradient(90deg, #F0F5F8 25%, #E2EBF0 50%, #F0F5F8 75%);
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

/* ─── Subcomponents ─── */
const ConfBadge = ({ conf }) => {
  const s = CONF_STYLES[conf] || CONF_STYLES.Low;
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

/* Animated loading dots */
const Dots = () => (
  <span style={{ display: "inline-flex", gap: 3, alignItems: "center", marginLeft: 6 }}>
    {[0, 1, 2].map(i => (
      <span key={i} style={{
        width: 5, height: 5, borderRadius: "50%", background: "rgba(11,123,111,0.5)",
        animation: `dotBlink 1.2s ${i * 0.2}s ease-in-out infinite`
      }} />
    ))}
  </span>
);

/* Scanning animation overlay */
const ScanOverlay = () => (
  <div style={{ position: "absolute", inset: 0, overflow: "hidden", borderRadius: 14, pointerEvents: "none" }}>
    <div style={{
      position: "absolute", left: 0, right: 0, height: 2,
      background: "linear-gradient(90deg,transparent,rgba(11,123,111,0.6),transparent)",
      animation: "scanLine 1.8s ease-in-out infinite", top: 0
    }} />
  </div>
);

/* Skeleton card */
const SkeletonCard = ({ delay = 0 }) => (
  <div style={{
    border: "1px solid #EDF2F6", borderRadius: 16, padding: "18px 20px",
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

/* Stat counter pill */
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

/* ─── Main component ─── */
function Predict() {
  const [symptoms, setSymptoms] = useState("");
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [phase, setPhase] = useState(""); // "parsing"|"analyzing"|"ranking"|"done"
  const [activeTab, setActiveTab] = useState("symptoms"); // "symptoms"|"image"
  const [charCount, setCharCount] = useState(0);
  const resultsRef = useRef(null);

  /* Simulated multi-phase loader */
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

  const delay = (ms) => new Promise(r => setTimeout(r, ms));

  const handlePredict = async () => {
    if (!symptoms.trim()) { alert("Please enter at least one symptom."); return; }
    try {
      setLoading(true); setSubmitted(true); setResults([]);
      const data = await runPhases(() => predictDisease(symptoms, image));
      setResults(data?.predictions || []);
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    } catch (e) {
      console.error(e);
      setPhase("done");
      alert("Prediction failed. Check the API connection.");
    } finally { setLoading(false); }
  };

  const handleFile = (file) => {
    if (!file) return;
    setImage(file); setPreview(URL.createObjectURL(file));
  };

  const reset = () => {
    setSymptoms(""); setImage(null); setPreview(null);
    setResults([]); setSubmitted(false); setPhase(""); setCharCount(0);
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
    <div style={{ minHeight: "100vh", background: T.bg, color: T.text, fontFamily: "'Inter',sans-serif" }}>
      <style>{GLOBAL_CSS}</style>

      {/* ── Page header ── */}
      <div className="predict-wrapper" style={{ maxWidth: 1200, margin: "0 auto", padding: "56px 32px 0" }}>
        <div style={{ marginBottom: 40 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
            <span style={{
              fontSize: 10, fontWeight: 800, color: T.teal, background: T.tealL, border: `1px solid ${T.tealB}`,
              padding: "4px 14px", borderRadius: 100, letterSpacing: 1.2, textTransform: "uppercase"
            }}>
              Diagnostic Engine
            </span>
            <span style={{
              fontSize: 10, fontWeight: 600, color: T.muted, background: "#F0F5F8", border: `1px solid ${T.border}`,
              padding: "4px 12px", borderRadius: 100, letterSpacing: .8
            }}>
              Beta · v2.4
            </span>
          </div>
          <h1 className="predict-h1" style={{
            fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 38, fontWeight: 800,
            margin: "0 0 10px", color: T.text, letterSpacing: -1
          }}>Disease Prediction</h1>
          <p style={{ color: T.sub, fontSize: 16, margin: 0, lineHeight: 1.65 }}>
            Enter patient symptoms and optionally upload a biomedical scan for AI-powered differential diagnosis across 49 rare diseases.
          </p>
        </div>

        {/* ── Two-column grid ── */}
        <div className="predict-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, paddingBottom: 56 }}>

          {/* ══ LEFT PANEL ══ */}
          <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
            <div style={{
              background: T.card, border: `1px solid ${T.border}`, borderRadius: 24, overflow: "hidden",
              boxShadow: "0 2px 16px rgba(15,28,46,0.04)"
            }}>

              {/* Tab bar */}
              <div style={{ display: "flex", borderBottom: `1px solid ${T.border}`, background: "#F8FBFD" }}>
                {[["symptoms", "🩺", "Symptoms"], ["image", "🔬", "Scan Upload"]].map(([id, icon, label]) => (
                  <button key={id} className="tab-btn" onClick={() => setActiveTab(id)} style={{
                    flex: 1, padding: "16px 20px", border: "none", cursor: "pointer",
                    fontFamily: "'Inter',sans-serif", fontSize: 13, fontWeight: activeTab === id ? 700 : 500,
                    color: activeTab === id ? T.teal : T.sub,
                    background: activeTab === id ? T.card : "transparent",
                    borderBottom: activeTab === id ? `2px solid ${T.teal}` : "2px solid transparent",
                    transition: "all .15s", display: "flex", alignItems: "center", justifyContent: "center", gap: 7,
                  }}>
                    <span>{icon}</span>{label}
                  </button>
                ))}
              </div>

              <div style={{ padding: 28 }}>

                {/* ── TAB: Symptoms ── */}
                {activeTab === "symptoms" && (
                  <div style={{ animation: "fadeUp .3s ease" }}>
                    <label style={{ fontSize: 11, fontWeight: 800, color: "#8FA5B5", letterSpacing: .8, textTransform: "uppercase", marginBottom: 9, display: "block" }}>
                      Describe Symptoms
                    </label>
                    <div style={{ position: "relative" }}>
                      <textarea rows={6}
                        placeholder="e.g. progressive vision loss, angioid streaks, skin papules on neck, fatigue, night blindness"
                        value={symptoms}
                        onChange={e => { setSymptoms(e.target.value); setCharCount(e.target.value.length); }}
                        style={{
                          width: "100%", background: "#F8FBFD", border: `1.5px solid ${T.borderI}`,
                          borderRadius: 14, padding: "13px 15px", color: T.text, fontSize: 14,
                          outline: "none", fontFamily: "'Inter',sans-serif", resize: "none",
                          boxSizing: "border-box", lineHeight: 1.7, transition: "border-color .2s, box-shadow .2s"
                        }}
                        onFocus={e => { e.target.style.borderColor = T.teal; e.target.style.boxShadow = "0 0 0 3px rgba(11,123,111,0.09)"; }}
                        onBlur={e => { e.target.style.borderColor = T.borderI; e.target.style.boxShadow = "none"; }}
                      />
                      {charCount > 0 && (
                        <span style={{ position: "absolute", bottom: 10, right: 12, fontSize: 11, color: T.muted, fontWeight: 500 }}>
                          {charCount} chars
                        </span>
                      )}
                    </div>
                    <p style={{ fontSize: 12, color: T.muted, marginTop: 7, marginBottom: 20 }}>
                      Separate multiple symptoms with commas or new lines.
                    </p>

                    {/* Quick chips */}
                    <label style={{ fontSize: 11, fontWeight: 800, color: "#8FA5B5", letterSpacing: .8, textTransform: "uppercase", marginBottom: 9, display: "block" }}>Quick Add</label>
                    <div className="chip-grid" style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
                      {QUICK_SYMPTOMS.map(s => {
                        const active = symptoms.toLowerCase().includes(s);
                        return (
                          <button key={s} className={active ? "" : "chip"} onClick={() => !active && setSymptoms(p => p ? `${p}, ${s}` : s)}
                            style={{
                              background: active ? T.tealL : "#F0F5F8",
                              border: `1px solid ${active ? T.tealB : T.borderI}`,
                              color: active ? T.teal : T.sub,
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

                {/* ── TAB: Scan Upload ── */}
                {activeTab === "image" && (
                  <div style={{ animation: "fadeUp .3s ease" }}>
                    <div className="dropzone"
                      onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                      onDragLeave={() => setDragOver(false)}
                      onDrop={e => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
                      onClick={() => document.getElementById("img-input").click()}
                      style={{
                        border: `2px dashed ${dragOver ? T.teal : "#C8D8E4"}`,
                        borderRadius: 18, padding: preview ? "0" : "52px 24px",
                        textAlign: "center", background: dragOver ? T.tealL : "#F8FBFD",
                        transition: "all .2s", overflow: "hidden", cursor: "pointer",
                        position: "relative",
                      }}>
                      {preview ? (
                        <>
                          <img src={preview} alt="scan preview" style={{ width: "100%", maxHeight: 200, objectFit: "cover", display: "block" }} />
                          <ScanOverlay />
                          <div style={{
                            position: "absolute", top: 10, right: 10, background: "rgba(255,255,255,0.92)", backdropFilter: "blur(8px)",
                            border: `1px solid ${T.tealB}`, borderRadius: 9, padding: "5px 11px",
                            fontSize: 11, color: T.teal, fontWeight: 800, display: "flex", alignItems: "center", gap: 5
                          }}>
                            <span style={{ width: 6, height: 6, borderRadius: "50%", background: T.teal, display: "inline-block" }} />
                            Scan loaded
                          </div>
                          <button onClick={e => { e.stopPropagation(); setImage(null); setPreview(null); }}
                            style={{
                              position: "absolute", top: 10, left: 10, background: "rgba(255,255,255,0.92)",
                              backdropFilter: "blur(8px)", border: `1px solid ${T.border}`, borderRadius: 8,
                              padding: "5px 11px", fontSize: 11, color: "#5A7184", cursor: "pointer", fontWeight: 600
                            }}>
                            ✕ Remove
                          </button>
                        </>
                      ) : (
                        <div style={{ animation: "float 3s ease infinite" }}>
                          <div style={{
                            width: 64, height: 64, borderRadius: "50%", background: T.tealL, border: `1.5px solid ${T.tealB}`,
                            display: "flex", alignItems: "center", justifyContent: "center", fontSize: 28, margin: "0 auto 16px"
                          }}>🩻</div>
                          <p style={{ color: "#4A6275", fontSize: 14, margin: "0 0 5px", fontWeight: 600 }}>Drag & drop or click to upload</p>
                          <p style={{ color: T.muted, fontSize: 12, margin: "0 0 16px" }}>JPG, PNG · MRI, CT, Dermoscopy, Histopathology</p>
                          <span style={{
                            fontSize: 11, color: T.teal, background: T.tealL, border: `1px solid ${T.tealB}`,
                            padding: "4px 14px", borderRadius: 100, fontWeight: 700
                          }}>Browse Files</span>
                        </div>
                      )}
                    </div>
                    <input id="img-input" type="file" hidden accept="image/*" onChange={e => handleFile(e.target.files[0])} />

                    <div style={{ marginTop: 18, padding: "14px 18px", background: "#F8FBFD", border: `1px solid ${T.border}`, borderRadius: 12 }}>
                      <p style={{ fontSize: 12, color: T.sub, margin: 0, lineHeight: 1.6 }}>
                        <strong style={{ color: "#4A6275" }}>Supported modalities:</strong> MRI brain scans, CT abdomen/thorax, dermoscopy images, retinal fundus, and histopathology slides.
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Action bar */}
              <div style={{ padding: "0 28px 28px", display: "flex", flexDirection: "column", gap: 12 }}>
                {/* Progress bar (during loading) */}
                {loading && (
                  <div style={{ animation: "fadeUp .3s ease" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 7 }}>
                      <span style={{ fontSize: 12, color: T.teal, fontWeight: 700, display: "flex", alignItems: "center" }}>
                        {phaseLabel[phase]}<Dots />
                      </span>
                      <span style={{ fontSize: 12, color: T.teal, fontWeight: 700 }}>{phasePercent[phase]}%</span>
                    </div>
                    <div style={{ height: 5, background: "#E8EFF5", borderRadius: 100, overflow: "hidden" }}>
                      <div style={{
                        height: 5, background: `linear-gradient(90deg,${T.teal},${T.blue})`,
                        borderRadius: 100, transition: "width .6s ease",
                        width: `${phasePercent[phase]}%`
                      }} />
                    </div>
                  </div>
                )}

                <div className="action-row" style={{ display: "flex", gap: 10 }}>
                  <button className="predict-btn" onClick={handlePredict} disabled={loading} style={{
                    flex: 1, background: loading ? "#C6E9E5" : T.teal,
                    color: loading ? "#5A7184" : "#fff", border: "none",
                    padding: "15px 20px", borderRadius: 13, fontWeight: 800, fontSize: 15,
                    cursor: loading ? "not-allowed" : "pointer",
                    fontFamily: "'Inter',sans-serif", transition: "all .2s",
                    boxShadow: loading ? "none" : "0 6px 20px rgba(11,123,111,0.20)",
                    display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
                  }}>
                    {loading ? (
                      <>
                        <span style={{ width: 17, height: 17, border: "2.5px solid rgba(11,123,111,0.25)", borderTop: `2.5px solid ${T.teal}`, borderRadius: "50%", animation: "spin .8s linear infinite", display: "inline-block" }} />
                        Analyzing<Dots />
                      </>
                    ) : (
                      <><span>🔍</span> Predict Disease</>
                    )}
                  </button>
                  <button className="reset-btn" onClick={reset} style={{
                    padding: "15px 20px", borderRadius: 13, border: `1.5px solid ${T.borderI}`,
                    background: "#fff", color: T.sub, cursor: "pointer",
                    fontFamily: "'Inter',sans-serif", fontSize: 14, fontWeight: 600, transition: "all .15s",
                  }}>Reset</button>
                </div>

                {/* Model info pills */}
                <div style={{ display: "flex", gap: 7, flexWrap: "wrap" }}>
                  {[
                    image ? "🔬 Multimodal (Symptom + Image)" : "🧬 Symptom Model",
                    "49 diseases",
                    "TF-IDF + LR",
                  ].map(t => (
                    <span key={t} style={{
                      fontSize: 11, color: T.sub, background: "#F0F5F8", border: `1px solid ${T.border}`,
                      padding: "4px 11px", borderRadius: 100, fontWeight: 500
                    }}>{t}</span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* ══ RIGHT PANEL ══ */}
          <div ref={resultsRef} style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            {/* ── Loading skeleton ── */}
            {loading && (
              <div style={{
                background: T.card, border: `1px solid ${T.border}`, borderRadius: 24, padding: 28,
                boxShadow: "0 2px 16px rgba(15,28,46,0.04)", animation: "pulseBorder 2s ease infinite"
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <div style={{ width: 36, height: 36, borderRadius: 10, background: T.tealL, display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <span style={{ width: 20, height: 20, border: "2.5px solid rgba(11,123,111,0.2)", borderTop: `2.5px solid ${T.teal}`, borderRadius: "50%", animation: "spin .8s linear infinite", display: "inline-block" }} />
                    </div>
                    <div>
                      <p style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 16, fontWeight: 800, color: T.text, margin: "0 0 2px" }}>
                        {phaseLabel[phase]}<Dots />
                      </p>
                      <p style={{ fontSize: 12, color: T.muted, margin: 0 }}>AI diagnostic engine active</p>
                    </div>
                  </div>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  {[0, .1, .2].map(d => <SkeletonCard key={d} delay={d} />)}
                </div>
              </div>
            )}

            {/* ── Empty state ── */}
            {!loading && !submitted && (
              <div style={{
                background: T.card, border: `1px solid ${T.border}`, borderRadius: 24, padding: "48px 28px",
                textAlign: "center", boxShadow: "0 2px 16px rgba(15,28,46,0.04)", animation: "fadeUp .5s ease"
              }}>
                <div style={{
                  width: 88, height: 88, borderRadius: "50%", background: T.tealL, border: `2px solid ${T.tealB}`,
                  display: "flex", alignItems: "center", justifyContent: "center", fontSize: 36,
                  margin: "0 auto 20px", animation: "float 4s ease infinite"
                }}>🧠</div>
                <h3 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 20, fontWeight: 800, color: T.text, margin: "0 0 8px" }}>
                  Ready for Diagnosis
                </h3>
                <p style={{ fontSize: 14, color: T.sub, margin: "0 0 24px", lineHeight: 1.7, maxWidth: 280, marginLeft: "auto", marginRight: "auto" }}>
                  Enter symptoms on the left. Add a scan for multimodal analysis.
                </p>
                <div style={{ display: "flex", gap: 10, justifyContent: "center", flexWrap: "wrap" }}>
                  {["49 rare diseases", "Confidence scoring", "Top-3 ranking"].map(f => (
                    <span key={f} style={{
                      fontSize: 12, color: T.teal, background: T.tealL, border: `1px solid ${T.tealB}`,
                      padding: "5px 14px", borderRadius: 100, fontWeight: 600
                    }}>✓ {f}</span>
                  ))}
                </div>
              </div>
            )}

            {/* ── Error state ── */}
            {!loading && submitted && results.length === 0 && (
              <div style={{
                background: T.card, border: "1px solid #F0BCBC", borderRadius: 24, padding: "40px 28px",
                textAlign: "center", animation: "fadeUp .4s ease"
              }}>
                <div style={{ fontSize: 40, marginBottom: 14 }}>⚠️</div>
                <p style={{ fontSize: 16, color: "#7A94A8", fontWeight: 600, margin: "0 0 6px" }}>No results returned</p>
                <p style={{ fontSize: 13, color: T.muted, margin: 0 }}>Check your API connection and try again.</p>
              </div>
            )}

            {/* ── Results ── */}
            {!loading && results.length > 0 && (
              <>
                {/* Top-result hero */}
                <div style={{
                  background: "linear-gradient(135deg,#EBF8F6 0%,#EBF4F9 100%)",
                  border: `1.5px solid ${T.tealB}`, borderRadius: 24, padding: "26px 28px",
                  animation: "slideRight .5s ease", boxShadow: "0 4px 20px rgba(11,123,111,0.08)"
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                      <div style={{ width: 36, height: 36, borderRadius: 10, background: T.teal, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16 }}>🥇</div>
                      <div>
                        <p style={{ fontSize: 10, color: T.teal, letterSpacing: 1.2, textTransform: "uppercase", margin: "0 0 1px", fontWeight: 800 }}>Top Diagnosis</p>
                        <p style={{ fontSize: 11, color: T.sub, margin: 0 }}>Highest probability match</p>
                      </div>
                    </div>
                    <ConfBadge conf={results[0].confidence} />
                  </div>
                  <h3 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 24, fontWeight: 800, color: T.text, margin: "0 0 16px", letterSpacing: -.3 }}>
                    {results[0].disease}
                  </h3>
                  <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                    <span style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 40, fontWeight: 800, color: T.teal, animation: "countUp .6s ease" }}>
                      {results[0].probability}%
                    </span>
                    <div style={{ flex: 1 }}>
                      <div style={{ height: 8, background: "rgba(11,123,111,0.15)", borderRadius: 100, overflow: "hidden" }}>
                        <div style={{
                          height: 8, width: `${results[0].probability}%`, background: T.teal,
                          borderRadius: 100, animation: "barGrow .9s ease"
                        }} />
                      </div>
                      <p style={{ fontSize: 11, color: T.sub, margin: "5px 0 0", fontWeight: 500 }}>Match probability</p>
                    </div>
                  </div>
                </div>

                {/* Summary stat row */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10 }}>
                  <StatPill icon="🔢" label="Matches" value={results.length} color={T.teal} bg={T.tealL} border={T.tealB} />
                  <StatPill icon="📊" label="Top Prob" value={`${results[0].probability}%`} color={T.blue} bg={T.blueL} border={T.blueB} />
                  <StatPill icon="🎯" label="Confidence" value={results[0].confidence} color={T.purple} bg={T.purpL} border={T.purpB} />
                </div>

                {/* Result list */}
                <div style={{
                  background: T.card, border: `1px solid ${T.border}`, borderRadius: 24, overflow: "hidden",
                  boxShadow: "0 2px 16px rgba(15,28,46,0.04)"
                }}>
                  <div style={{
                    padding: "18px 24px", background: "#F8FBFD", borderBottom: `1px solid ${T.border}`,
                    display: "flex", justifyContent: "space-between", alignItems: "center"
                  }}>
                    <h3 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 15, fontWeight: 700, margin: 0, color: T.text }}>All Predictions</h3>
                    <span style={{
                      fontSize: 11, color: T.teal, background: T.tealL, border: `1px solid ${T.tealB}`,
                      padding: "3px 11px", borderRadius: 100, fontWeight: 700
                    }}>Ranked by probability</span>
                  </div>
                  <div style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 10 }}>
                    {results.map((item, i) => {
                      const c = PALETTE[i % PALETTE.length];
                      return (
                        <div key={i} className="result-row" style={{
                          animationDelay: `${i * 0.08}s`, border: `1px solid ${c.border}`,
                          borderRadius: 16, padding: "16px 18px", background: "#fff", transition: "box-shadow .2s, transform .15s",
                        }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                              <span style={{
                                fontSize: 10, fontWeight: 800, background: c.light, color: c.text,
                                border: `1px solid ${c.border}`, padding: "3px 9px", borderRadius: 7
                              }}>#{item.rank}</span>
                              <span style={{ fontWeight: 700, fontSize: 14, color: T.text }}>{item.disease}</span>
                            </div>
                            <ConfBadge conf={item.confidence} />
                          </div>
                          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#8FA5B5", marginBottom: 7 }}>
                            <span>Probability</span>
                            <span style={{ color: c.text, fontWeight: 800 }}>{item.probability}%</span>
                          </div>
                          <div style={{ height: 5, background: "#F0F5F8", borderRadius: 100, overflow: "hidden" }}>
                            <div style={{
                              height: 5, width: `${item.probability}%`, background: c.bar,
                              borderRadius: 100, animation: "barGrow .8s ease"
                            }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Disclaimer */}
                <div style={{ padding: "14px 18px", background: "#FFF8EC", border: "1px solid #F5D8B8", borderRadius: 14 }}>
                  <p style={{ fontSize: 12, color: "#C05B1A", margin: 0, lineHeight: 1.65 }}>
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