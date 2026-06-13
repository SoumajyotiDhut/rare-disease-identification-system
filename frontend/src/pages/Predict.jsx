import { useState } from "react";
import { predictDisease } from "../services/Api";

const pageStyle = {
  minHeight: "100vh",
  background: "#0A1628",
  color: "#F8FAFC",
  fontFamily: "'Inter', sans-serif",
  padding: "48px 32px",
};

const card = {
  background: "rgba(255,255,255,0.03)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 24,
  padding: 36,
};

const label = {
  fontSize: 13,
  fontWeight: 600,
  color: "#94A3B8",
  letterSpacing: 0.8,
  textTransform: "uppercase",
  marginBottom: 10,
  display: "block",
};

const input = {
  width: "100%",
  background: "rgba(255,255,255,0.04)",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 12,
  padding: "14px 18px",
  color: "#F8FAFC",
  fontSize: 15,
  outline: "none",
  fontFamily: "'Inter', sans-serif",
  resize: "none",
  boxSizing: "border-box",
  transition: "border-color 0.2s",
};

const ConfidenceBadge = ({ conf }) => {
  const map = {
    High: { bg: "rgba(34,197,94,0.12)", color: "#22C55E", border: "rgba(34,197,94,0.3)" },
    Medium: { bg: "rgba(251,191,36,0.12)", color: "#FBBF24", border: "rgba(251,191,36,0.3)" },
    Low: { bg: "rgba(239,68,68,0.12)", color: "#EF4444", border: "rgba(239,68,68,0.3)" },
  };
  const s = map[conf] || map.Low;
  return (
    <span style={{
      padding: "4px 12px", borderRadius: 100,
      fontSize: 12, fontWeight: 600,
      background: s.bg, color: s.color,
      border: `1px solid ${s.border}`,
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
    <div style={pageStyle}>
      <div style={{ maxWidth: 1280, margin: "0 auto" }}>

        {/* Header */}
        <div style={{ marginBottom: 40 }}>
          <p style={{ fontSize: 12, color: "#00D4C8", letterSpacing: 2, textTransform: "uppercase", marginBottom: 12 }}>
            DIAGNOSTIC ENGINE
          </p>
          <h1 style={{ fontFamily: "'Syne', sans-serif", fontSize: 42, fontWeight: 800, margin: "0 0 10px", color: "#F8FAFC" }}>
            Disease Prediction
          </h1>
          <p style={{ color: "#64748B", fontSize: 16 }}>
            Enter patient symptoms and optionally upload a biomedical image for AI-powered differential diagnosis.
          </p>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 28 }}>

          {/* ── LEFT PANEL ── */}
          <div style={card}>
            <h2 style={{ fontFamily: "'Syne', sans-serif", fontSize: 20, fontWeight: 700, margin: "0 0 28px", color: "#F8FAFC" }}>
              Patient Input
            </h2>

            {/* Drop zone */}
            <div
              onDragOver={e => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={e => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
              style={{
                border: `2px dashed ${dragOver ? "#00D4C8" : "rgba(255,255,255,0.12)"}`,
                borderRadius: 16,
                padding: preview ? 0 : "48px 24px",
                textAlign: "center",
                background: dragOver ? "rgba(0,212,200,0.05)" : "rgba(255,255,255,0.02)",
                transition: "all 0.2s",
                overflow: "hidden",
                cursor: "pointer",
                marginBottom: 24,
              }}
              onClick={() => document.getElementById("img-input").click()}
            >
              {preview ? (
                <img src={preview} alt="preview" style={{ width: "100%", maxHeight: 200, objectFit: "cover", display: "block" }} />
              ) : (
                <>
                  <div style={{ fontSize: 40, marginBottom: 14 }}>🩻</div>
                  <p style={{ color: "#94A3B8", fontSize: 14, margin: "0 0 6px" }}>
                    Drag & drop or click to upload
                  </p>
                  <p style={{ color: "#475569", fontSize: 12 }}>JPG, PNG · MRI, CT, Dermoscopy, Histopathology</p>
                </>
              )}
            </div>
            <input id="img-input" type="file" hidden accept="image/*"
              onChange={e => handleFile(e.target.files[0])} />

            {/* Symptoms */}
            <div style={{ marginBottom: 24 }}>
              <label style={label}>Symptoms</label>
              <textarea
                rows={6}
                placeholder="e.g. progressive vision loss, angioid streaks, skin papules on neck, fatigue"
                value={symptoms}
                onChange={e => setSymptoms(e.target.value)}
                style={input}
                onFocus={e => e.target.style.borderColor = "rgba(0,212,200,0.5)"}
                onBlur={e => e.target.style.borderColor = "rgba(255,255,255,0.1)"}
              />
              <p style={{ fontSize: 12, color: "#475569", marginTop: 8 }}>
                Separate multiple symptoms with commas or line breaks.
              </p>
            </div>

            {/* Quick chips */}
            <div style={{ marginBottom: 28 }}>
              <p style={{ ...label, marginBottom: 10 }}>Quick Add</p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {["fatigue", "night blindness", "skin lesions", "joint pain", "vision loss", "dry cough"].map(s => (
                  <button
                    key={s}
                    onClick={() => setSymptoms(prev => prev ? `${prev}, ${s}` : s)}
                    style={{
                      background: "rgba(0,212,200,0.08)", border: "1px solid rgba(0,212,200,0.2)",
                      color: "#00D4C8", padding: "6px 14px", borderRadius: 100,
                      fontSize: 12, cursor: "pointer", fontFamily: "'Inter', sans-serif",
                      transition: "background 0.2s",
                    }}
                    onMouseEnter={e => e.currentTarget.style.background = "rgba(0,212,200,0.16)"}
                    onMouseLeave={e => e.currentTarget.style.background = "rgba(0,212,200,0.08)"}
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
                  background: loading ? "rgba(0,212,200,0.3)" : "linear-gradient(135deg, #00D4C8, #0066FF)",
                  color: "#fff", border: "none",
                  padding: "16px 24px", borderRadius: 12,
                  fontWeight: 700, fontSize: 15,
                  cursor: loading ? "not-allowed" : "pointer",
                  fontFamily: "'Inter', sans-serif",
                  boxShadow: loading ? "none" : "0 8px 24px rgba(0,212,200,0.3)",
                  transition: "all 0.2s",
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
                }}
              >
                {loading ? (
                  <>
                    <span style={{
                      width: 16, height: 16, border: "2px solid rgba(255,255,255,0.3)",
                      borderTop: "2px solid #fff", borderRadius: "50%",
                      animation: "spin 0.8s linear infinite", display: "inline-block",
                    }} />
                    Analyzing…
                  </>
                ) : "Predict Disease"}
              </button>
              <button
                onClick={reset}
                style={{
                  padding: "16px 24px", borderRadius: 12,
                  border: "1px solid rgba(255,255,255,0.12)",
                  background: "transparent", color: "#94A3B8",
                  cursor: "pointer", fontFamily: "'Inter', sans-serif",
                  fontSize: 14, fontWeight: 600,
                  transition: "all 0.2s",
                }}
                onMouseEnter={e => e.currentTarget.style.background = "rgba(255,255,255,0.05)"}
                onMouseLeave={e => e.currentTarget.style.background = "transparent"}
              >
                Reset
              </button>
            </div>
          </div>

          {/* ── RIGHT PANEL ── */}
          <div style={card}>
            <h2 style={{ fontFamily: "'Syne', sans-serif", fontSize: 20, fontWeight: 700, margin: "0 0 28px", color: "#F8FAFC" }}>
              Prediction Results
            </h2>

            {/* Top result highlight */}
            {results.length > 0 && (
              <div style={{
                background: "linear-gradient(135deg, rgba(0,212,200,0.12), rgba(0,102,255,0.08))",
                border: "1px solid rgba(0,212,200,0.25)",
                borderRadius: 16, padding: "20px 24px", marginBottom: 24,
              }}>
                <p style={{ fontSize: 12, color: "#00D4C8", letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 8 }}>
                  TOP DIAGNOSIS
                </p>
                <h3 style={{ fontFamily: "'Syne', sans-serif", fontSize: 24, fontWeight: 800, color: "#F8FAFC", margin: "0 0 8px" }}>
                  {results[0].disease}
                </h3>
                <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                  <span style={{ fontSize: 14, color: "#94A3B8" }}>
                    Probability: <span style={{ color: "#00D4C8", fontWeight: 700 }}>{results[0].probability}%</span>
                  </span>
                  <ConfidenceBadge conf={results[0].confidence} />
                </div>
              </div>
            )}

            {/* Empty state */}
            {results.length === 0 && !loading && (
              <div style={{ textAlign: "center", padding: "60px 24px", color: "#334155" }}>
                <div style={{ fontSize: 64, marginBottom: 20 }}>🧠</div>
                <p style={{ fontSize: 16, marginBottom: 8, color: "#475569" }}>No predictions yet</p>
                <p style={{ fontSize: 13, color: "#334155" }}>Enter symptoms on the left and click Predict Disease</p>
              </div>
            )}

            {/* Results list */}
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {results.map((item, i) => (
                <div key={i} style={{
                  border: "1px solid rgba(255,255,255,0.07)",
                  borderRadius: 16, padding: "18px 20px",
                  background: "rgba(255,255,255,0.02)",
                  transition: "border-color 0.2s",
                }}
                  onMouseEnter={e => e.currentTarget.style.borderColor = "rgba(0,212,200,0.2)"}
                  onMouseLeave={e => e.currentTarget.style.borderColor = "rgba(255,255,255,0.07)"}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
                    <div>
                      <span style={{
                        fontSize: 11, fontWeight: 700, color: "#475569",
                        background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)",
                        padding: "3px 8px", borderRadius: 6, marginRight: 10,
                      }}>#{item.rank}</span>
                      <span style={{ fontWeight: 600, fontSize: 15, color: "#F8FAFC" }}>{item.disease}</span>
                    </div>
                    <ConfidenceBadge conf={item.confidence} />
                  </div>

                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, color: "#64748B", marginBottom: 8 }}>
                    <span>Probability</span>
                    <span style={{ color: "#F8FAFC", fontWeight: 600 }}>{item.probability}%</span>
                  </div>

                  <div style={{ height: 5, background: "rgba(255,255,255,0.06)", borderRadius: 100 }}>
                    <div style={{
                      height: 5,
                      width: `${item.probability}%`,
                      borderRadius: 100,
                      background: i === 0
                        ? "linear-gradient(90deg, #00D4C8, #0066FF)"
                        : i === 1
                          ? "linear-gradient(90deg, #22C55E, #00D4C8)"
                          : "linear-gradient(90deg, #A78BFA, #6366F1)",
                      transition: "width 0.8s ease",
                    }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );
}

export default Predict;