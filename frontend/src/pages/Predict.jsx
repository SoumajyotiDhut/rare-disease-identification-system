import { useState } from "react";
import { predictDisease } from "../services/Api";

const pg = {
  minHeight: "100vh",
  background: "#F8FAFB",
  color: "#0F1C2E",
  fontFamily: "'Inter', sans-serif",
  padding: "56px 40px",
};

const cardStyle = {
  background: "#fff",
  border: "1px solid #E8EFF5",
  borderRadius: 24,
  padding: 36,
};

const labelStyle = {
  fontSize: 12,
  fontWeight: 700,
  color: "#8FA5B5",
  letterSpacing: 0.8,
  textTransform: "uppercase",
  marginBottom: 10,
  display: "block",
};

const inputStyle = {
  width: "100%",
  background: "#F8FAFB",
  border: "1.5px solid #DDE8EF",
  borderRadius: 12,
  padding: "13px 16px",
  color: "#0F1C2E",
  fontSize: 15,
  outline: "none",
  fontFamily: "'Inter', sans-serif",
  resize: "none",
  boxSizing: "border-box",
  transition: "border-color 0.2s, box-shadow 0.2s",
};

const RESULT_COLORS = [
  { bar: "linear-gradient(90deg, #0B7B6F, #1AADA0)", badge: "#EBF8F6", badgeText: "#0B7B6F", border: "#B2E8E2" },
  { bar: "linear-gradient(90deg, #1D6FA4, #2D9AD4)", badge: "#EBF4F9", badgeText: "#1D6FA4", border: "#B3D8EE" },
  { bar: "linear-gradient(90deg, #5B3DB8, #8B66D4)", badge: "#F2EEF9", badgeText: "#5B3DB8", border: "#C8B8EC" },
  { bar: "linear-gradient(90deg, #C05B1A, #E07A35)", badge: "#FFF4EC", badgeText: "#C05B1A", border: "#F5D8B8" },
  { bar: "linear-gradient(90deg, #8FA5B5, #B0C5D4)", badge: "#F0F5F8", badgeText: "#5A7184", border: "#C8D8E4" },
];

const ConfBadge = ({ conf }) => {
  const map = {
    High: { bg: "#EBF8F6", color: "#0B7B6F", border: "#B2E8E2" },
    Medium: { bg: "#FFF8EC", color: "#C05B1A", border: "#F5D8B8" },
    Low: { bg: "#FDECED", color: "#B83030", border: "#F0BCBC" },
  };
  const s = map[conf] || map.Low;
  return (
    <span style={{
      padding: "4px 12px", borderRadius: 100,
      fontSize: 11, fontWeight: 700,
      background: s.bg, color: s.color,
      border: `1px solid ${s.border}`,
      textTransform: "uppercase", letterSpacing: 0.6,
    }}>{conf}</span>
  );
};

function Predict() {
  const [symptoms, setSymptoms] = useState("");
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const handlePredict = async () => {
    if (!symptoms.trim()) {
      alert("Please enter at least one symptom.");
      return;
    }
    try {
      setLoading(true);
      const data = await predictDisease(symptoms, image);
      setResults(data.predictions || []);
    } catch (e) {
      console.error(e);
      alert("Prediction failed. Check the API connection.");
    } finally {
      setLoading(false);
    }
  };

  const handleFile = (file) => {
    if (!file) return;
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const reset = () => {
    setSymptoms(""); setImage(null);
    setPreview(null); setResults([]);
  };

  return (
    <div style={pg}>
      <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&display=swap" rel="stylesheet" />
      <div style={{ maxWidth: 1200, margin: "0 auto" }}>

        {/* Header */}
        <div style={{ marginBottom: 44 }}>
          <span style={{
            fontSize: 11, fontWeight: 700, color: "#0B7B6F",
            background: "#EBF8F6", border: "1px solid #B2E8E2",
            padding: "4px 14px", borderRadius: 100, letterSpacing: 1,
            textTransform: "uppercase", display: "inline-block", marginBottom: 16,
          }}>Diagnostic Engine</span>
          <h1 style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif", fontSize: 40, fontWeight: 800, margin: "0 0 10px", color: "#0F1C2E", letterSpacing: -1 }}>
            Disease Prediction
          </h1>
          <p style={{ color: "#7A94A8", fontSize: 16, margin: 0 }}>
            Enter patient symptoms and optionally upload a biomedical scan for AI-powered differential diagnosis.
          </p>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>

          {/* LEFT — INPUT */}
          <div style={cardStyle}>
            <h2 style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif", fontSize: 18, fontWeight: 700, margin: "0 0 28px", color: "#0F1C2E" }}>
              Patient Input
            </h2>

            {/* Drop zone */}
            <div
              onDragOver={e => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={e => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
              onClick={() => document.getElementById("img-input").click()}
              style={{
                border: `2px dashed ${dragOver ? "#0B7B6F" : "#C8D8E4"}`,
                borderRadius: 16,
                padding: preview ? 0 : "44px 24px",
                textAlign: "center",
                background: dragOver ? "#EBF8F6" : "#F8FAFB",
                transition: "all 0.2s",
                overflow: "hidden",
                cursor: "pointer",
                marginBottom: 24,
              }}
            >
              {preview ? (
                <div style={{ position: "relative" }}>
                  <img src={preview} alt="preview" style={{ width: "100%", maxHeight: 190, objectFit: "cover", display: "block" }} />
                  <div style={{
                    position: "absolute", top: 10, right: 10,
                    background: "#fff", border: "1px solid #DDE8EF",
                    borderRadius: 8, padding: "4px 10px",
                    fontSize: 11, color: "#5A7184", fontWeight: 600,
                  }}>
                    ✓ Image loaded
                  </div>
                </div>
              ) : (
                <>
                  <div style={{ fontSize: 36, marginBottom: 12 }}>🩻</div>
                  <p style={{ color: "#5A7184", fontSize: 14, margin: "0 0 5px", fontWeight: 500 }}>
                    Drag & drop or click to upload
                  </p>
                  <p style={{ color: "#9BB8CC", fontSize: 12, margin: 0 }}>JPG, PNG · MRI, CT, Dermoscopy, Histopathology</p>
                </>
              )}
            </div>
            <input id="img-input" type="file" hidden accept="image/*"
              onChange={e => handleFile(e.target.files[0])} />

            {/* Symptoms */}
            <div style={{ marginBottom: 22 }}>
              <label style={labelStyle}>Symptoms</label>
              <textarea
                rows={6}
                placeholder="e.g. progressive vision loss, angioid streaks, skin papules on neck, fatigue"
                value={symptoms}
                onChange={e => setSymptoms(e.target.value)}
                style={inputStyle}
                onFocus={e => { e.target.style.borderColor = "#0B7B6F"; e.target.style.boxShadow = "0 0 0 3px rgba(11,123,111,0.08)"; }}
                onBlur={e => { e.target.style.borderColor = "#DDE8EF"; e.target.style.boxShadow = "none"; }}
              />
              <p style={{ fontSize: 12, color: "#9BB8CC", marginTop: 7 }}>
                Separate multiple symptoms with commas or line breaks.
              </p>
            </div>

            {/* Quick chips */}
            <div style={{ marginBottom: 28 }}>
              <label style={labelStyle}>Quick Add</label>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {["fatigue", "night blindness", "skin lesions", "joint pain", "vision loss", "dry cough"].map(s => (
                  <button
                    key={s}
                    onClick={() => setSymptoms(prev => prev ? `${prev}, ${s}` : s)}
                    style={{
                      background: "#F0F5F8", border: "1px solid #DDE8EF",
                      color: "#5A7184", padding: "6px 14px", borderRadius: 100,
                      fontSize: 12, cursor: "pointer", fontFamily: "'Inter', sans-serif",
                      fontWeight: 500, transition: "all 0.15s",
                    }}
                    onMouseEnter={e => { e.currentTarget.style.background = "#EBF8F6"; e.currentTarget.style.borderColor = "#B2E8E2"; e.currentTarget.style.color = "#0B7B6F"; }}
                    onMouseLeave={e => { e.currentTarget.style.background = "#F0F5F8"; e.currentTarget.style.borderColor = "#DDE8EF"; e.currentTarget.style.color = "#5A7184"; }}
                  >
                    + {s}
                  </button>
                ))}
              </div>
            </div>

            <div style={{ display: "flex", gap: 12 }}>
              <button
                onClick={handlePredict}
                disabled={loading}
                style={{
                  flex: 1,
                  background: loading ? "#C6E9E5" : "#0B7B6F",
                  color: loading ? "#5A7184" : "#fff",
                  border: "none",
                  padding: "15px 24px", borderRadius: 12,
                  fontWeight: 700, fontSize: 15,
                  cursor: loading ? "not-allowed" : "pointer",
                  fontFamily: "'Inter', sans-serif",
                  transition: "all 0.2s",
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
                }}
                onMouseEnter={e => { if (!loading) e.currentTarget.style.background = "#08635A"; }}
                onMouseLeave={e => { if (!loading) e.currentTarget.style.background = "#0B7B6F"; }}
              >
                {loading ? (
                  <>
                    <span style={{
                      width: 16, height: 16, border: "2px solid #9BB8CC",
                      borderTop: "2px solid #0B7B6F", borderRadius: "50%",
                      animation: "spin 0.8s linear infinite", display: "inline-block",
                    }} />
                    Analyzing…
                  </>
                ) : "Predict Disease →"}
              </button>
              <button
                onClick={reset}
                style={{
                  padding: "15px 22px", borderRadius: 12,
                  border: "1.5px solid #DDE8EF",
                  background: "#fff", color: "#7A94A8",
                  cursor: "pointer", fontFamily: "'Inter', sans-serif",
                  fontSize: 14, fontWeight: 600,
                  transition: "all 0.15s",
                }}
                onMouseEnter={e => { e.currentTarget.style.background = "#F8FAFB"; e.currentTarget.style.borderColor = "#C8D8E4"; }}
                onMouseLeave={e => { e.currentTarget.style.background = "#fff"; e.currentTarget.style.borderColor = "#DDE8EF"; }}
              >
                Reset
              </button>
            </div>
          </div>

          {/* RIGHT — RESULTS */}
          <div style={cardStyle}>
            <h2 style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif", fontSize: 18, fontWeight: 700, margin: "0 0 28px", color: "#0F1C2E" }}>
              Prediction Results
            </h2>

            {/* Top result highlight */}
            {results.length > 0 && (
              <div style={{
                background: "linear-gradient(135deg, #EBF8F6, #EBF4F9)",
                border: "1px solid #B2E8E2",
                borderRadius: 16, padding: "20px 24px", marginBottom: 22,
              }}>
                <p style={{ fontSize: 11, color: "#0B7B6F", letterSpacing: 1.2, textTransform: "uppercase", marginBottom: 8, fontWeight: 700 }}>
                  Top Diagnosis
                </p>
                <h3 style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif", fontSize: 22, fontWeight: 800, color: "#0F1C2E", margin: "0 0 10px", letterSpacing: -0.3 }}>
                  {results[0].disease}
                </h3>
                <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                  <span style={{ fontSize: 14, color: "#5A7184" }}>
                    Probability: <span style={{ color: "#0B7B6F", fontWeight: 700 }}>{results[0].probability}%</span>
                  </span>
                  <ConfBadge conf={results[0].confidence} />
                </div>
              </div>
            )}

            {/* Empty state */}
            {results.length === 0 && !loading && (
              <div style={{ textAlign: "center", padding: "60px 24px" }}>
                <div style={{
                  width: 72, height: 72, borderRadius: "50%",
                  background: "#F0F5F8",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 32, margin: "0 auto 20px",
                }}>🧠</div>
                <p style={{ fontSize: 16, marginBottom: 8, color: "#7A94A8", fontWeight: 500 }}>No predictions yet</p>
                <p style={{ fontSize: 13, color: "#9BB8CC" }}>Enter symptoms on the left and click Predict Disease</p>
              </div>
            )}

            {/* Results list */}
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {results.map((item, i) => {
                const c = RESULT_COLORS[i % RESULT_COLORS.length];
                return (
                  <div key={i} style={{
                    border: `1px solid ${c.border}`,
                    borderRadius: 16, padding: "18px 20px",
                    background: "#fff",
                    transition: "box-shadow 0.2s, transform 0.15s",
                  }}
                    onMouseEnter={e => { e.currentTarget.style.boxShadow = "0 4px 16px rgba(15,28,46,0.06)"; e.currentTarget.style.transform = "translateY(-1px)"; }}
                    onMouseLeave={e => { e.currentTarget.style.boxShadow = "none"; e.currentTarget.style.transform = "none"; }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <span style={{
                          fontSize: 11, fontWeight: 700,
                          background: c.badge, color: c.badgeText,
                          border: `1px solid ${c.border}`,
                          padding: "3px 9px", borderRadius: 6,
                        }}>#{item.rank}</span>
                        <span style={{ fontWeight: 600, fontSize: 15, color: "#0F1C2E" }}>{item.disease}</span>
                      </div>
                      <ConfBadge conf={item.confidence} />
                    </div>

                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#8FA5B5", marginBottom: 8 }}>
                      <span>Probability</span>
                      <span style={{ color: c.badgeText, fontWeight: 700 }}>{item.probability}%</span>
                    </div>

                    <div style={{ height: 5, background: "#F0F5F8", borderRadius: 100 }}>
                      <div style={{
                        height: 5,
                        width: `${item.probability}%`,
                        borderRadius: 100,
                        background: c.bar,
                        transition: "width 0.8s ease",
                      }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );
}

export default Predict;