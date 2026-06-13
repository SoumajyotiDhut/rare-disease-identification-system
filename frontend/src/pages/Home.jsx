import { useNavigate } from "react-router-dom";
import { useEffect, useRef } from "react";

/* ── tiny inline styles ── */
const S = {
  page: {
    minHeight: "100vh",
    background: "#0A1628",
    color: "#F8FAFC",
    fontFamily: "'Inter', sans-serif",
  },
  hero: {
    maxWidth: 1280,
    margin: "0 auto",
    padding: "80px 32px 60px",
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 64,
    alignItems: "center",
  },
  badge: {
    display: "inline-flex",
    alignItems: "center",
    gap: 8,
    background: "rgba(0,212,200,0.1)",
    border: "1px solid rgba(0,212,200,0.3)",
    color: "#00D4C8",
    padding: "8px 16px",
    borderRadius: 100,
    fontSize: 13,
    fontWeight: 500,
    letterSpacing: 0.5,
    marginBottom: 28,
  },
  h1: {
    fontFamily: "'Syne', sans-serif",
    fontSize: 60,
    fontWeight: 800,
    lineHeight: 1.08,
    color: "#F8FAFC",
    margin: "0 0 24px",
  },
  accent: { color: "#00D4C8" },
  sub: {
    fontSize: 18,
    color: "#94A3B8",
    lineHeight: 1.7,
    maxWidth: 480,
    margin: "0 0 40px",
  },
  btnRow: { display: "flex", gap: 16 },
  btnPrimary: {
    background: "linear-gradient(135deg, #00D4C8, #0066FF)",
    color: "#fff",
    border: "none",
    padding: "16px 36px",
    borderRadius: 12,
    fontWeight: 700,
    fontSize: 16,
    cursor: "pointer",
    boxShadow: "0 8px 32px rgba(0,212,200,0.35)",
    transition: "transform 0.2s, box-shadow 0.2s",
    fontFamily: "'Inter', sans-serif",
  },
  btnSecondary: {
    background: "transparent",
    color: "#F8FAFC",
    border: "1px solid rgba(248,250,252,0.2)",
    padding: "16px 36px",
    borderRadius: 12,
    fontWeight: 600,
    fontSize: 16,
    cursor: "pointer",
    transition: "all 0.2s",
    fontFamily: "'Inter', sans-serif",
  },
};

/* ── Pulse dot ── */
const PulseDot = () => (
  <span style={{ position: "relative", display: "inline-block", width: 10, height: 10 }}>
    <span style={{
      position: "absolute", inset: 0, borderRadius: "50%",
      background: "#00D4C8", animation: "ping 1.4s ease-in-out infinite",
    }} />
    <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: "#00D4C8" }} />
    <style>{`@keyframes ping{0%{transform:scale(1);opacity:1}70%{transform:scale(2);opacity:0}100%{transform:scale(1);opacity:0}}`}</style>
  </span>
);

/* ── Stats card ── */
const StatCard = ({ label, value, color, bg }) => (
  <div style={{
    background: bg,
    border: `1px solid ${color}22`,
    borderRadius: 16,
    padding: "20px 24px",
  }}>
    <p style={{ fontSize: 12, color: "#64748B", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>{label}</p>
    <p style={{ fontFamily: "'Syne', sans-serif", fontSize: 36, fontWeight: 800, color, margin: 0 }}>{value}</p>
  </div>
);

/* ── Feature card ── */
const FeatureCard = ({ icon, title, desc }) => (
  <div style={{
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 20,
    padding: "32px 28px",
    transition: "border-color 0.2s, transform 0.2s",
    cursor: "default",
  }}
    onMouseEnter={e => { e.currentTarget.style.borderColor = "rgba(0,212,200,0.3)"; e.currentTarget.style.transform = "translateY(-4px)"; }}
    onMouseLeave={e => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)"; e.currentTarget.style.transform = "none"; }}
  >
    <div style={{
      width: 48, height: 48, borderRadius: 12,
      background: "rgba(0,212,200,0.12)",
      display: "flex", alignItems: "center", justifyContent: "center",
      fontSize: 22, marginBottom: 20,
    }}>{icon}</div>
    <h3 style={{ fontFamily: "'Syne', sans-serif", fontSize: 18, fontWeight: 700, color: "#F8FAFC", margin: "0 0 12px" }}>{title}</h3>
    <p style={{ fontSize: 14, color: "#64748B", lineHeight: 1.7, margin: 0 }}>{desc}</p>
  </div>
);

function Home() {
  const navigate = useNavigate();

  return (
    <div style={S.page}>
      {/* ── HERO ── */}
      <div style={S.hero}>
        {/* Left */}
        <div>
          <div style={S.badge}>
            <PulseDot />
            AI-Powered Clinical Intelligence Platform
          </div>

          <h1 style={S.h1}>
            Identify Rare<br />
            Diseases with<br />
            <span style={S.accent}>Precision AI</span>
          </h1>

          <p style={S.sub}>
            Submit patient symptoms and biomedical scans.
            Our multimodal AI system cross-references 1,374 rare diseases
            to return ranked, confidence-scored differential diagnoses.
          </p>

          <div style={S.btnRow}>
            <button
              style={S.btnPrimary}
              onClick={() => navigate("/predict")}
              onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-2px)"; e.currentTarget.style.boxShadow = "0 12px 40px rgba(0,212,200,0.5)"; }}
              onMouseLeave={e => { e.currentTarget.style.transform = "none"; e.currentTarget.style.boxShadow = "0 8px 32px rgba(0,212,200,0.35)"; }}
            >
              Start Diagnosis →
            </button>
            <button
              style={S.btnSecondary}
              onClick={() => navigate("/dashboard")}
              onMouseEnter={e => { e.currentTarget.style.background = "rgba(255,255,255,0.05)"; }}
              onMouseLeave={e => { e.currentTarget.style.background = "transparent"; }}
            >
              View Dashboard
            </button>
          </div>

          {/* Trust row */}
          <div style={{ display: "flex", gap: 32, marginTop: 48 }}>
            {[
              { val: "36K+", label: "Patient Cases" },
              { val: "1,374", label: "Rare Diseases" },
              { val: "94K+", label: "Medical Images" },
            ].map(({ val, label }) => (
              <div key={label}>
                <div style={{ fontFamily: "'Syne', sans-serif", fontSize: 26, fontWeight: 800, color: "#F8FAFC" }}>{val}</div>
                <div style={{ fontSize: 12, color: "#64748B", marginTop: 4 }}>{label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Right — dashboard preview */}
        <div style={{
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(0,212,200,0.2)",
          borderRadius: 28,
          padding: 32,
          boxShadow: "0 32px 80px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05)",
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 28 }}>
            <span style={{ fontFamily: "'Syne', sans-serif", fontWeight: 700, fontSize: 18, color: "#F8FAFC" }}>
              Live Dashboard
            </span>
            <span style={{
              display: "flex", alignItems: "center", gap: 6,
              background: "rgba(0,212,200,0.12)", border: "1px solid rgba(0,212,200,0.3)",
              color: "#00D4C8", padding: "6px 14px", borderRadius: 100, fontSize: 12, fontWeight: 600,
            }}>
              <PulseDot /> Online
            </span>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
            <StatCard label="Predictions" value="1,254" color="#00D4C8" bg="rgba(0,212,200,0.06)" />
            <StatCard label="Accuracy" value="34%" color="#22C55E" bg="rgba(34,197,94,0.06)" />
            <StatCard label="Diseases" value="49" color="#A78BFA" bg="rgba(167,139,250,0.06)" />
            <StatCard label="Images" value="35K+" color="#FB923C" bg="rgba(251,146,60,0.06)" />
          </div>

          {/* Mini status */}
          <div style={{ borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: 20 }}>
            {[
              { label: "Symptom Model", pct: 34, color: "#00D4C8" },
              { label: "Image Model", pct: 62, color: "#22C55E" },
              { label: "Fusion Model", pct: 0, color: "#A78BFA", note: "In training" },
            ].map(({ label, pct, color, note }) => (
              <div key={label} style={{ marginBottom: 14 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, color: "#94A3B8", marginBottom: 6 }}>
                  <span>{label}</span>
                  <span style={{ color }}>{note || `${pct}%`}</span>
                </div>
                <div style={{ height: 4, background: "rgba(255,255,255,0.06)", borderRadius: 4 }}>
                  <div style={{ height: 4, width: `${pct}%`, background: color, borderRadius: 4, transition: "width 1s ease" }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── FEATURES ── */}
      <div style={{ maxWidth: 1280, margin: "0 auto", padding: "60px 32px" }}>
        <div style={{ textAlign: "center", marginBottom: 56 }}>
          <p style={{ fontSize: 12, color: "#00D4C8", letterSpacing: 2, textTransform: "uppercase", marginBottom: 16 }}>
            CAPABILITIES
          </p>
          <h2 style={{ fontFamily: "'Syne', sans-serif", fontSize: 44, fontWeight: 800, color: "#F8FAFC", margin: "0 0 16px" }}>
            Three Diagnostic Pathways
          </h2>
          <p style={{ fontSize: 16, color: "#64748B", maxWidth: 520, margin: "0 auto" }}>
            Use symptoms alone, images alone, or both together. The fusion model combines all signals for highest accuracy.
          </p>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 20 }}>
          <FeatureCard icon="🧬" title="Symptom Analysis" desc="TF-IDF encoded symptom vectors fed into a multinomial classifier trained on 8,568 patient cases." />
          <FeatureCard icon="🔬" title="Image Detection" desc="EfficientNet-B4 fine-tuned on 35,374 biomedical images — MRI, CT, histopathology, dermatology." />
          <FeatureCard icon="🤖" title="Fusion Prediction" desc="Cross-attention fusion of both modalities. Correct disease in top-3 predictions 53% of the time." />
          <FeatureCard icon="📊" title="Analytics" desc="Monitor prediction history, confidence distributions, and model performance metrics in real time." />
        </div>
      </div>

      {/* ── FOOTER ── */}
      <footer style={{
        borderTop: "1px solid rgba(255,255,255,0.06)",
        padding: "32px",
        textAlign: "center",
        color: "#475569",
        fontSize: 13,
      }}>
        © 2026 AI DOC · Rare Disease Detection System · ZebraMap Dataset · CC BY 4.0
      </footer>
    </div>
  );
}

export default Home;