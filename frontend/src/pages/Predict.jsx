import { useState } from "react";
import { predictDisease } from "../services/Api";

const PALETTE = [
  { bar: "#0B7B6F", light: "#EBF8F6", text: "#0B7B6F", border: "#B2E8E2" },
  { bar: "#1D6FA4", light: "#EBF4F9", text: "#1D6FA4", border: "#B3D8EE" },
  { bar: "#5B3DB8", light: "#F2EEF9", text: "#5B3DB8", border: "#C8B8EC" },
  { bar: "#C05B1A", light: "#FFF4EC", text: "#C05B1A", border: "#F5D8B8" },
  { bar: "#8FA5B5", light: "#F0F5F8", text: "#5A7184", border: "#C8D8E4" },
];

const ConfBadge = ({ conf }) => {
  const map = {
    High: { bg: "#EBF8F6", color: "#0B7B6F", border: "#B2E8E2" },
    Medium: { bg: "#FFF8EC", color: "#C05B1A", border: "#F5D8B8" },
    Low: { bg: "#FDECED", color: "#B83030", border: "#F0BCBC" },
  };
  const s = map[conf] || map.Low;
  return (
    <span style={{ padding: "4px 12px", borderRadius: 100, fontSize: 10, fontWeight: 800, background: s.bg, color: s.color, border: `1px solid ${s.border}`, textTransform: "uppercase", letterSpacing: 0.8 }}>{conf}</span>
  );
};

function Predict() {
  const [symptoms, setSymptoms] = useState("");
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handlePredict = async () => {
    if (!symptoms.trim()) { alert("Please enter at least one symptom."); return; }
    try {
      setLoading(true); setSubmitted(true);
      const data = await predictDisease(symptoms, image);
      setResults(data.predictions || []);
    } catch (e) {
      console.error(e);
      alert("Prediction failed. Check the API connection.");
    } finally { setLoading(false); }
  };

  const handleFile = (file) => {
    if (!file) return;
    setImage(file); setPreview(URL.createObjectURL(file));
  };

  const reset = () => { setSymptoms(""); setImage(null); setPreview(null); setResults([]); setSubmitted(false); };

  return (
    <div style={{ minHeight: "100vh", background: "#F4F8FB", color: "#0F1C2E", fontFamily: "'Inter',sans-serif", padding: "56px 32px" }}>
      <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&display=swap" rel="stylesheet" />
      <style>{`
        @keyframes spin{to{transform:rotate(360deg)}}
        @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
        .result-item{animation:fadeIn .3s ease both}
        .chip-btn:hover{background:#EBF8F6!important;border-color:#B2E8E2!important;color:#0B7B6F!important}
        .reset-btn:hover{background:#F4F8FB!important;border-color:#C8D8E4!important}
        .dropzone:hover{border-color:#0B7B6F!important;background:#F0FAF9!important}
        @media(max-width:900px){
          .predict-grid{grid-template-columns:1fr!important}
          .predict-pad{padding:40px 20px!important}
        }
      `}</style>

      <div style={{ maxWidth: 1200, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: 44 }}>
          <span style={{ fontSize: 10, fontWeight: 800, color: "#0B7B6F", background: "#EBF8F6", border: "1px solid #B2E8E2", padding: "4px 14px", borderRadius: 100, letterSpacing: 1.2, textTransform: "uppercase", display: "inline-block", marginBottom: 14 }}>Diagnostic Engine</span>
          <h1 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 38, fontWeight: 800, margin: "0 0 10px", color: "#0F1C2E", letterSpacing: -1 }}>Disease Prediction</h1>
          <p style={{ color: "#7A94A8", fontSize: 16, margin: 0, lineHeight: 1.6 }}>Enter patient symptoms and optionally upload a biomedical scan for AI-powered differential diagnosis.</p>
        </div>

        <div className="predict-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>

          {/* LEFT */}
          <div style={{ background: "#fff", border: "1px solid #E8EFF5", borderRadius: 24, padding: 36 }}>
            <h2 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 18, fontWeight: 700, margin: "0 0 28px", color: "#0F1C2E" }}>Patient Input</h2>

            {/* Drop zone */}
            <div className="dropzone"
              onDragOver={e => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={e => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
              onClick={() => document.getElementById("img-input").click()}
              style={{
                border: `2px dashed ${dragOver ? "#0B7B6F" : "#D0DDE8"}`,
                borderRadius: 16, padding: preview ? 0 : "40px 24px",
                textAlign: "center", background: dragOver ? "#EBF8F6" : "#F8FBFD",
                transition: "all 0.2s", overflow: "hidden", cursor: "pointer", marginBottom: 22,
              }}>
              {preview ? (
                <div style={{ position: "relative" }}>
                  <img src={preview} alt="preview" style={{ width: "100%", maxHeight: 180, objectFit: "cover", display: "block" }} />
                  <div style={{ position: "absolute", top: 10, right: 10, background: "#EBF8F6", border: "1px solid #B2E8E2", borderRadius: 8, padding: "4px 10px", fontSize: 11, color: "#0B7B6F", fontWeight: 700 }}>✓ Image loaded</div>
                  <button onClick={e => { e.stopPropagation(); setImage(null); setPreview(null); }} style={{ position: "absolute", top: 10, left: 10, background: "#fff", border: "1px solid #E0EBF2", borderRadius: 7, padding: "3px 10px", fontSize: 11, color: "#5A7184", cursor: "pointer", fontWeight: 600 }}>Remove</button>
                </div>
              ) : (
                <>
                  <div style={{ width: 52, height: 52, borderRadius: "50%", background: "#EBF8F6", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 24, margin: "0 auto 12px" }}>🩻</div>
                  <p style={{ color: "#5A7184", fontSize: 14, margin: "0 0 4px", fontWeight: 600 }}>Drag & drop or click to upload</p>
                  <p style={{ color: "#9BB8CC", fontSize: 12, margin: 0 }}>JPG, PNG · MRI, CT, Dermoscopy, Histopathology</p>
                </>
              )}
            </div>
            <input id="img-input" type="file" hidden accept="image/*" onChange={e => handleFile(e.target.files[0])} />

            {/* Symptoms */}
            <div style={{ marginBottom: 20 }}>
              <label style={{ fontSize: 11, fontWeight: 700, color: "#8FA5B5", letterSpacing: 0.8, textTransform: "uppercase", marginBottom: 9, display: "block" }}>Symptoms</label>
              <textarea rows={5}
                placeholder="e.g. progressive vision loss, angioid streaks, skin papules, fatigue"
                value={symptoms} onChange={e => setSymptoms(e.target.value)}
                style={{ width: "100%", background: "#F8FBFD", border: "1.5px solid #DDE8EF", borderRadius: 12, padding: "13px 15px", color: "#0F1C2E", fontSize: 15, outline: "none", fontFamily: "'Inter',sans-serif", resize: "none", boxSizing: "border-box", transition: "border-color 0.2s, box-shadow 0.2s" }}
                onFocus={e => { e.target.style.borderColor = "#0B7B6F"; e.target.style.boxShadow = "0 0 0 3px rgba(11,123,111,0.08)"; }}
                onBlur={e => { e.target.style.borderColor = "#DDE8EF"; e.target.style.boxShadow = "none"; }}
              />
              <p style={{ fontSize: 12, color: "#9BB8CC", marginTop: 6 }}>Separate multiple symptoms with commas or line breaks.</p>
            </div>

            {/* Quick chips */}
            <div style={{ marginBottom: 26 }}>
              <label style={{ fontSize: 11, fontWeight: 700, color: "#8FA5B5", letterSpacing: 0.8, textTransform: "uppercase", marginBottom: 9, display: "block" }}>Quick Add</label>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
                {["fatigue", "night blindness", "skin lesions", "joint pain", "vision loss", "dry cough", "muscle weakness", "seizures"].map(s => (
                  <button key={s} className="chip-btn" onClick={() => setSymptoms(prev => prev ? `${prev}, ${s}` : s)}
                    style={{ background: "#F0F5F8", border: "1px solid #DDE8EF", color: "#5A7184", padding: "6px 13px", borderRadius: 100, fontSize: 12, cursor: "pointer", fontFamily: "'Inter',sans-serif", fontWeight: 500, transition: "all 0.15s" }}>
                    + {s}
                  </button>
                ))}
              </div>
            </div>

            <div style={{ display: "flex", gap: 10 }}>
              <button onClick={handlePredict} disabled={loading} style={{
                flex: 1, background: loading ? "#C6E9E5" : "#0B7B6F", color: loading ? "#5A7184" : "#fff",
                border: "none", padding: "15px 20px", borderRadius: 12, fontWeight: 700, fontSize: 15,
                cursor: loading ? "not-allowed" : "pointer", fontFamily: "'Inter',sans-serif", transition: "all 0.2s",
                display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
                boxShadow: loading ? "none" : "0 6px 20px rgba(11,123,111,0.18)",
              }}
                onMouseEnter={e => { if (!loading) { e.currentTarget.style.background = "#08635A"; e.currentTarget.style.boxShadow = "0 10px 28px rgba(11,123,111,0.26)"; } }}
                onMouseLeave={e => { if (!loading) { e.currentTarget.style.background = "#0B7B6F"; e.currentTarget.style.boxShadow = "0 6px 20px rgba(11,123,111,0.18)"; } }}
              >
                {loading ? (<><span style={{ width: 16, height: 16, border: "2px solid rgba(11,123,111,0.3)", borderTop: "2px solid #0B7B6F", borderRadius: "50%", animation: "spin 0.8s linear infinite", display: "inline-block" }} /> Analyzing…</>) : "Predict Disease →"}
              </button>
              <button className="reset-btn" onClick={reset} style={{ padding: "15px 20px", borderRadius: 12, border: "1.5px solid #DDE8EF", background: "#fff", color: "#7A94A8", cursor: "pointer", fontFamily: "'Inter',sans-serif", fontSize: 14, fontWeight: 600, transition: "all 0.15s" }}>Reset</button>
            </div>
          </div>

          {/* RIGHT */}
          <div style={{ background: "#fff", border: "1px solid #E8EFF5", borderRadius: 24, padding: 36 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 28 }}>
              <h2 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 18, fontWeight: 700, margin: 0, color: "#0F1C2E" }}>Prediction Results</h2>
              {results.length > 0 && (
                <span style={{ fontSize: 11, fontWeight: 700, color: "#8FA5B5", background: "#F0F5F8", border: "1px solid #DDE8EF", padding: "4px 12px", borderRadius: 100 }}>{results.length} matches</span>
              )}
            </div>

            {/* Top result */}
            {results.length > 0 && (
              <div style={{ background: "linear-gradient(135deg,#EBF8F6,#EBF4F9)", border: "1.5px solid #B2E8E2", borderRadius: 18, padding: "22px 24px", marginBottom: 20 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
                  <p style={{ fontSize: 10, color: "#0B7B6F", letterSpacing: 1.2, textTransform: "uppercase", margin: 0, fontWeight: 800 }}>Top Diagnosis</p>
                  <ConfBadge conf={results[0].confidence} />
                </div>
                <h3 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 22, fontWeight: 800, color: "#0F1C2E", margin: "0 0 8px", letterSpacing: -0.3 }}>{results[0].disease}</h3>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <span style={{ fontSize: 28, fontWeight: 800, color: "#0B7B6F", fontFamily: "'Plus Jakarta Sans',sans-serif" }}>{results[0].probability}%</span>
                  <div style={{ flex: 1 }}>
                    <div style={{ height: 6, background: "#C6E9E5", borderRadius: 100 }}>
                      <div style={{ height: 6, width: `${results[0].probability}%`, background: "#0B7B6F", borderRadius: 100, transition: "width 0.8s ease" }} />
                    </div>
                    <p style={{ fontSize: 11, color: "#7AA89A", margin: "4px 0 0", fontWeight: 500 }}>Match probability</p>
                  </div>
                </div>
              </div>
            )}

            {/* Empty */}
            {results.length === 0 && !loading && !submitted && (
              <div style={{ textAlign: "center", padding: "52px 24px" }}>
                <div style={{ width: 80, height: 80, borderRadius: "50%", background: "#F0F5F8", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 36, margin: "0 auto 20px" }}>🧠</div>
                <p style={{ fontSize: 16, marginBottom: 6, color: "#7A94A8", fontWeight: 600 }}>No predictions yet</p>
                <p style={{ fontSize: 13, color: "#9BB8CC", margin: 0 }}>Enter symptoms and click Predict Disease</p>
              </div>
            )}

            {results.length === 0 && !loading && submitted && (
              <div style={{ textAlign: "center", padding: "52px 24px" }}>
                <div style={{ width: 80, height: 80, borderRadius: "50%", background: "#FDECED", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 36, margin: "0 auto 20px" }}>⚠️</div>
                <p style={{ fontSize: 16, marginBottom: 6, color: "#7A94A8", fontWeight: 600 }}>No results returned</p>
                <p style={{ fontSize: 13, color: "#9BB8CC", margin: 0 }}>Check API connection and try again</p>
              </div>
            )}

            {/* Result list */}
            <div style={{ display: "flex", flexDirection: "column", gap: 11 }}>
              {results.map((item, i) => {
                const c = PALETTE[i % PALETTE.length];
                return (
                  <div key={i} className="result-item" style={{ animationDelay: `${i * 0.07}s`, border: `1px solid ${c.border}`, borderRadius: 16, padding: "17px 20px", background: "#fff", transition: "box-shadow 0.2s, transform 0.15s" }}
                    onMouseEnter={e => { e.currentTarget.style.boxShadow = "0 4px 16px rgba(15,28,46,0.07)"; e.currentTarget.style.transform = "translateY(-1px)"; }}
                    onMouseLeave={e => { e.currentTarget.style.boxShadow = "none"; e.currentTarget.style.transform = "none"; }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 11 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                        <span style={{ fontSize: 10, fontWeight: 800, background: c.light, color: c.text, border: `1px solid ${c.border}`, padding: "3px 9px", borderRadius: 6 }}>#{item.rank}</span>
                        <span style={{ fontWeight: 700, fontSize: 14, color: "#0F1C2E" }}>{item.disease}</span>
                      </div>
                      <ConfBadge conf={item.confidence} />
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#8FA5B5", marginBottom: 8 }}>
                      <span>Probability</span>
                      <span style={{ color: c.text, fontWeight: 700 }}>{item.probability}%</span>
                    </div>
                    <div style={{ height: 5, background: "#F0F5F8", borderRadius: 100 }}>
                      <div style={{ height: 5, width: `${item.probability}%`, background: c.bar, borderRadius: 100, transition: "width 0.8s ease" }} />
                    </div>
                  </div>
                );
              })}
            </div>

            {results.length > 0 && (
              <div style={{ marginTop: 20, padding: "14px 18px", background: "#FFF8EC", border: "1px solid #F5D8B8", borderRadius: 12 }}>
                <p style={{ fontSize: 12, color: "#C05B1A", margin: 0, lineHeight: 1.6 }}>
                  <strong>⚠ Clinical Disclaimer:</strong> These predictions are AI-generated and intended for research purposes only. Always consult a licensed clinician for diagnosis.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Predict;