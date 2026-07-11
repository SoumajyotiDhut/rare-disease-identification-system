import { useState, useRef } from "react";
import { useTheme } from "../context/ThemeContext";
import { useToast } from "../context/ToastContext";
import { predictDisease } from "../services/Api";
import PredictionCard from "../components/PredictionCard";

const CSS = (c) => `
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
  @keyframes fadeUp   { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
  @keyframes spin     { to{transform:rotate(360deg)} }
  @keyframes pulse    { 0%,100%{opacity:.5} 50%{opacity:1} }
  @keyframes drawLine { from{stroke-dashoffset:600} to{stroke-dashoffset:0} }

  .eyebrow {
    display:inline-flex; align-items:center; gap:10px;
    font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:600;
    color:${c.gold}; letter-spacing:0.14em; text-transform:uppercase;
  }
  .eyebrow::before { content:''; width:20px; height:1px; background:${c.gold}; display:inline-block; }

  .symptom-input {
    width:100%; min-height:130px; padding:16px 18px;
    border-radius:4px; border:1px solid ${c.borderI};
    background:${c.bgDeep}; color:${c.text}; font-size:14.5px;
    font-family:'Inter',sans-serif; resize:vertical; outline:none;
    line-height:1.6; box-sizing:border-box;
    transition:border-color .2s, box-shadow .2s, background .2s;
  }
  .symptom-input:focus { border-color:${c.teal}!important; box-shadow:0 0 0 3px ${c.tealL}; background:${c.card}; }
  .symptom-input::placeholder { color:${c.muted} }

  .chip {
    padding:7px 15px; border-radius:100px; font-size:12.5px;
    font-weight:600; cursor:pointer; border:1px solid ${c.borderI};
    background:${c.card}; color:${c.sub}; transition:all .15s;
    font-family:'Inter',sans-serif; white-space:nowrap;
  }
  .chip:hover { border-color:${c.teal}; color:${c.teal}; background:${c.tealL}; }
  .chip.active { border-color:${c.teal}; color:${c.teal}; background:${c.tealL}; font-weight:700; }

  .tab-btn {
    flex:1; padding:12px 16px; border:none; cursor:pointer;
    font-size:13.5px; font-weight:600; font-family:'Inter',sans-serif;
    transition:all .2s; border-radius:2px; letter-spacing:0.01em;
  }
  .tab-btn.active { background:${c.card}; color:${c.teal}; box-shadow:${c.shadowSm}; font-weight:700; }
  .tab-btn:not(.active) { background:transparent; color:${c.sub}; }
  .tab-btn:not(.active):hover { color:${c.text}; }

  .predict-btn {
    width:100%; padding:17px; border-radius:4px; border:none;
    background:${c.text}; color:${c.bg}; font-size:15px;
    font-weight:600; cursor:pointer; font-family:'Inter',sans-serif;
    display:flex; align-items:center; justify-content:center; gap:12px;
    transition:all .25s; letter-spacing:0.01em;
  }
  .predict-btn:hover:not(:disabled) { background:${c.teal}; color:#fff; transform:translateY(-2px); box-shadow:${c.shadowTeal}; }
  .predict-btn:disabled { opacity:.6; cursor:not-allowed; transform:none; }

  .reset-btn {
    padding:14px 24px; border-radius:4px; border:1px solid ${c.borderI};
    background:transparent; color:${c.sub}; font-size:14px;
    font-weight:600; cursor:pointer; font-family:'Inter',sans-serif;
    transition:all .2s;
  }
  .reset-btn:hover { border-color:${c.red}; color:${c.red}; background:${c.redL}; }

  .upload-zone {
    border:1.5px dashed ${c.borderI}; border-radius:4px;
    padding:40px 24px; text-align:center; cursor:pointer;
    transition:all .2s; background:${c.bgDeep};
  }
  .upload-zone:hover, .upload-zone.drag { border-color:${c.teal}!important; background:${c.tealL}!important; }

  @media(max-width:1024px){ .main-grid{ grid-template-columns:1fr!important; gap:24px!important } }
  @media(max-width:600px){
    .predict-pad   { padding:32px 16px!important }
    .predict-h1    { font-size:30px!important }
    .symptom-input { min-height:110px!important; font-size:16px!important }
    .tab-btn       { padding:12px 10px!important; font-size:13px!important }
    .chip          { padding:9px 14px!important; font-size:12px!important }
    .predict-btn   { padding:15px!important; font-size:14.5px!important }
  }
`;

const QUICK_SYMPTOMS = [
  "fatigue", "night blindness", "skin lesions", "joint pain",
  "vision loss", "dry cough", "muscle weakness", "seizures",
  "hearing loss", "ataxia", "tremors", "dysphagia",
];

function VitalLine({ color, width = 70, height = 18 }) {
  return (
    <svg width={width} height={height} viewBox="0 0 160 28" fill="none">
      <path d="M0 14H40L48 4L58 24L66 14H160" stroke={color} strokeWidth="1.5"
        strokeLinecap="round" strokeLinejoin="round" strokeDasharray="600"
        style={{ animation: "drawLine 1.4s ease forwards" }} />
    </svg>
  );
}

export default function Predict() {
  const { c } = useTheme();
  const toast = useToast();
  const fileRef = useRef(null);

  const [tab, setTab] = useState("symptoms");
  const [symptoms, setSymptoms] = useState("");
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [drag, setDrag] = useState(false);
  const [topK] = useState(5);

  const activeChips = symptoms.split(",").map(s => s.trim().toLowerCase()).filter(Boolean);

  const toggleChip = (chip) => {
    const cur = symptoms.split(",").map(s => s.trim()).filter(Boolean);
    const idx = cur.findIndex(s => s.toLowerCase() === chip);
    if (idx >= 0) cur.splice(idx, 1); else cur.push(chip);
    setSymptoms(cur.join(", "));
  };

  const handleFile = (file) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) { toast.error("Please upload an image file."); return; }
    if (file.size > 10 * 1024 * 1024) { toast.error("Image must be under 10 MB."); return; }
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleDrop = (e) => {
    e.preventDefault(); setDrag(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const handlePredict = async () => {
    if (!symptoms.trim()) { toast.error("Please enter at least one symptom."); return; }
    setLoading(true); setResults(null);
    try {
      const data = await predictDisease(symptoms.trim(), image, topK);
      const preds = data.predictions || data;
      if (!preds?.length) { toast.error("No predictions returned. Try different symptoms."); return; }
      setResults({ predictions: preds, mode: data.mode || (image ? "multimodal_fusion" : "symptoms_only") });
      toast.success(`Found ${preds.length} differential diagnoses.`);
    } catch (err) {
      toast.error("Prediction failed. Check your connection and try again.");
      console.error(err);
    } finally { setLoading(false); }
  };

  const handleReset = () => {
    setSymptoms(""); setImage(null); setPreview(null); setResults(null);
  };

  const charCount = symptoms.length;

  return (
    <div className="predict-pad" style={{ minHeight: "100vh", background: c.bg, fontFamily: "'Inter',sans-serif", padding: "48px 24px" }}>
      <style>{CSS(c)}</style>

      <div style={{ maxWidth: 1200, margin: "0 auto" }}>

        {/* ── Header ─────────────────────────────────────────────────── */}
        <div style={{ marginBottom: 40, animation: "fadeUp .5s ease both" }}>
          <span className="eyebrow" style={{ marginBottom: 16, display: "inline-flex" }}>
            Diagnostic Engine · Beta v2.4
          </span>
          <h1 className="predict-h1" style={{
            fontFamily: "'Fraunces',serif",
            fontSize: 40, fontWeight: 600, color: c.text,
            margin: "16px 0 12px", letterSpacing: "-0.02em",
          }}>Disease Prediction</h1>
          <p style={{ fontSize: 15.5, color: c.sub, margin: 0, maxWidth: 600, lineHeight: 1.7 }}>
            Enter patient symptoms and optionally upload a biomedical scan for
            AI-powered differential diagnosis across 62 Tier-A rare diseases.
          </p>
        </div>

        {/* ── Main grid ──────────────────────────────────────────────── */}
        <div className="main-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 28, alignItems: "start" }}>

          {/* ── INPUT PANEL ──────────────────────────────────────────── */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

            {/* Tabs */}
            <div style={{
              background: c.bgAlt, border: `1px solid ${c.border}`,
              borderRadius: 4, padding: 6, display: "flex", gap: 4,
            }}>
              {[{ id: "symptoms", label: "Symptoms" }, { id: "image", label: "Scan Upload" }].map(t => (
                <button key={t.id} className={`tab-btn${tab === t.id ? " active" : ""}`} onClick={() => setTab(t.id)}>
                  {t.label}
                </button>
              ))}
            </div>

            {/* Symptoms tab */}
            {tab === "symptoms" && (
              <div style={{ background: c.card, border: `1px solid ${c.border}`, borderTop: `2px solid ${c.teal}`, padding: 28, animation: "fadeUp .3s ease" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                  <label style={{ fontSize: 12, fontWeight: 700, color: c.sub, textTransform: "uppercase", letterSpacing: "0.08em" }}>
                    Describe Symptoms
                  </label>
                  <span style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 11.5, color: charCount > 500 ? c.red : c.muted, fontWeight: 500 }}>
                    {charCount} chars
                  </span>
                </div>

                <textarea
                  className="symptom-input"
                  value={symptoms}
                  onChange={e => setSymptoms(e.target.value)}
                  placeholder="e.g. progressive vision loss, angioid streaks, skin papules on neck, fatigue, night blindness&#10;&#10;Separate symptoms with commas. Be as specific as possible."
                />

                {/* Quick add */}
                <div style={{ marginTop: 16 }}>
                  <div style={{ fontSize: 10.5, fontWeight: 700, color: c.muted, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 10 }}>
                    Quick Add
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
                    {QUICK_SYMPTOMS.map(s => (
                      <button key={s} className={`chip${activeChips.includes(s) ? " active" : ""}`}
                        onClick={() => toggleChip(s)}>
                        {activeChips.includes(s) ? "✓ " : ""}{s}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Model info */}
                <div style={{
                  marginTop: 18, padding: "11px 14px", borderRadius: 4,
                  background: c.bgDeep, border: `1px solid ${c.border}`,
                  display: "flex", alignItems: "center", gap: 8,
                }}>
                  <VitalLine color={c.teal} width={26} height={14} />
                  <span style={{ fontSize: 11.5, color: c.muted, fontWeight: 500 }}>
                    Symptom Model · TF-IDF + Logistic Regression · 62 Tier-A diseases
                  </span>
                </div>
              </div>
            )}

            {/* Image tab */}
            {tab === "image" && (
              <div style={{ background: c.card, border: `1px solid ${c.border}`, borderTop: `2px solid ${c.blue}`, padding: 28, animation: "fadeUp .3s ease" }}>
                <label style={{ fontSize: 12, fontWeight: 700, color: c.sub, textTransform: "uppercase", letterSpacing: "0.08em", display: "block", marginBottom: 14 }}>
                  Upload Medical Scan
                </label>

                {preview ? (
                  <div style={{ position: "relative", borderRadius: 4, overflow: "hidden", border: `1px solid ${c.tealB}` }}>
                    <img src={preview} alt="Preview" style={{ width: "100%", maxHeight: 240, objectFit: "cover", display: "block" }} />
                    <button onClick={() => { setImage(null); setPreview(null); }} style={{
                      position: "absolute", top: 10, right: 10,
                      width: 32, height: 32, borderRadius: 4,
                      background: "rgba(0,0,0,0.6)", border: "none",
                      color: "#fff", cursor: "pointer", fontSize: 14,
                      display: "flex", alignItems: "center", justifyContent: "center",
                    }}>✕</button>
                    <div style={{ padding: "12px 16px", background: c.tealL, display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ fontSize: 13 }}>✅</span>
                      <span style={{ fontSize: 12.5, color: c.teal, fontWeight: 600 }}>{image?.name}</span>
                    </div>
                  </div>
                ) : (
                  <div className={`upload-zone${drag ? " drag" : ""}`}
                    onClick={() => fileRef.current?.click()}
                    onDragOver={e => { e.preventDefault(); setDrag(true); }}
                    onDragLeave={() => setDrag(false)}
                    onDrop={handleDrop}
                  >
                    <div style={{ fontSize: 36, marginBottom: 14 }}>🩻</div>
                    <p style={{ fontFamily: "'Fraunces',serif", fontSize: 16, fontWeight: 600, color: c.text, margin: "0 0 6px" }}>Drop scan here or click to browse</p>
                    <p style={{ fontSize: 12.5, color: c.muted, margin: 0 }}>MRI · CT · Fundus · Dermoscopy · Histopathology · X-ray</p>
                    <p style={{ fontSize: 11.5, color: c.muted, margin: "8px 0 0" }}>PNG, JPG, JPEG · Max 10 MB</p>
                  </div>
                )}

                <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }}
                  onChange={e => handleFile(e.target.files[0])} />

                {!image && (
                  <div style={{ marginTop: 14, padding: "10px 14px", borderRadius: 4, background: c.blueL, border: `1px solid ${c.blueB}` }}>
                    <p style={{ fontSize: 12, color: c.blue, margin: 0, fontWeight: 500 }}>
                      ℹ️ Optional — system falls back to symptom-only mode if no image is provided.
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Action buttons */}
            <div style={{ display: "flex", gap: 12 }}>
              <button className="predict-btn" onClick={handlePredict} disabled={loading || !symptoms.trim()}>
                {loading ? (
                  <>
                    <span style={{ width: 20, height: 20, border: "2.5px solid rgba(255,255,255,0.3)", borderTop: "2.5px solid #fff", borderRadius: "50%", animation: "spin .7s linear infinite", flexShrink: 0 }} />
                    Analysing…
                  </>
                ) : (
                  <>{image ? "Predict (Multimodal)" : "Predict Disease"}</>
                )}
              </button>
              {(symptoms || image || results) && (
                <button className="reset-btn" onClick={handleReset}>Reset</button>
              )}
            </div>
          </div>

          {/* ── OUTPUT PANEL ─────────────────────────────────────────── */}
          <div style={{ animation: "fadeUp .4s ease .1s both" }}>
            {!results && !loading && (
              <div style={{
                background: c.card, border: `1px solid ${c.border}`,
                padding: "56px 32px", textAlign: "center",
              }}>
                <VitalLine color={c.teal} width={100} height={22} />
                <h3 style={{ fontFamily: "'Fraunces',serif", fontSize: 21, fontWeight: 600, color: c.text, margin: "22px 0 10px" }}>Ready for Diagnosis</h3>
                <p style={{ fontSize: 14, color: c.sub, margin: "0 0 28px", lineHeight: 1.65 }}>
                  Enter symptoms on the left, optionally upload a scan, then hit Predict.
                </p>
                <div style={{ display: "flex", justifyContent: "center", gap: 10, flexWrap: "wrap" }}>
                  {["62 rare diseases", "Confidence scoring", "Top-5 ranking"].map(s => (
                    <span key={s} style={{ fontSize: 12, color: c.teal, background: c.tealL, border: `1px solid ${c.tealB}`, padding: "6px 14px", borderRadius: 100, fontWeight: 600 }}>{s}</span>
                  ))}
                </div>
              </div>
            )}

            {loading && (
              <div style={{ background: c.card, border: `1px solid ${c.border}`, padding: "56px 32px", textAlign: "center" }}>
                <div style={{ position: "relative", width: 60, height: 60, margin: "0 auto 24px" }}>
                  <div style={{ position: "absolute", inset: 0, border: `2px solid ${c.border}`, borderTop: `2px solid ${c.teal}`, borderRadius: "50%", animation: "spin .9s linear infinite" }} />
                  <div style={{ position: "absolute", inset: 10, border: `2px solid ${c.border}`, borderTop: `2px solid ${c.blue}`, borderRadius: "50%", animation: "spin 1.5s linear infinite reverse" }} />
                </div>
                <p style={{ fontSize: 15, fontWeight: 600, color: c.text, margin: "0 0 6px" }}>Analysing…</p>
                <p style={{ fontSize: 12.5, color: c.muted, margin: 0 }}>Running {image ? "multimodal fusion" : "symptom analysis"}</p>
              </div>
            )}

            {results && (
              <div style={{ animation: "fadeUp .4s ease" }}>
                {/* Mode + summary bar */}
                <div style={{
                  background: results.mode === "multimodal_fusion"
                    ? `linear-gradient(135deg,${c.tealL},${c.blueL})`
                    : c.tealL,
                  border: `1px solid ${c.tealB}`, borderTop: `2px solid ${c.teal}`,
                  padding: "20px 22px", marginBottom: 20,
                  display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12,
                }}>
                  <div>
                    <span className="eyebrow" style={{ marginBottom: 8, display: "inline-flex" }}>
                      {results.mode === "multimodal_fusion" ? "Multimodal Fusion" : "Symptom Analysis"}
                    </span>
                    <div style={{ fontFamily: "'Fraunces',serif", fontSize: 19, fontWeight: 600, color: c.text, marginTop: 8 }}>
                      {results.predictions[0]?.disease}
                    </div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 30, fontWeight: 600, color: c.teal, lineHeight: 1 }}>
                      {results.predictions[0]?.probability}%
                    </div>
                    <div style={{ fontSize: 11, color: c.muted, fontWeight: 500, marginTop: 4 }}>Top match probability</div>
                  </div>
                </div>

                {/* Cards */}
                <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                  {results.predictions.map(p => (
                    <PredictionCard key={p.rank} item={p} />
                  ))}
                </div>

                {/* Disclaimer */}
                <div style={{ marginTop: 18, padding: "13px 16px", background: c.ambL, border: `1px solid ${c.ambB}` }}>
                  <p style={{ fontSize: 12, color: c.amber, margin: 0, lineHeight: 1.6 }}>
                    <strong>⚠ Research use only.</strong> These predictions are AI-generated and should not substitute clinical judgement. Always consult a licensed clinician.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}