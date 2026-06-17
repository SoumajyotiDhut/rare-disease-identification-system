import { useNavigate } from "react-router-dom";

const S = {
  page: {
    minHeight: "100vh",
    background: "#F8FAFB",
    color: "#0F1C2E",
    fontFamily: "'Inter', sans-serif",
  },
  hero: {
    maxWidth: 1200,
    margin: "0 auto",
    padding: "96px 40px 80px",
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 72,
    alignItems: "center",
  },
  eyebrow: {
    display: "inline-flex",
    alignItems: "center",
    gap: 8,
    background: "#EBF8F6",
    border: "1px solid #B2E8E2",
    color: "#0B7B6F",
    padding: "7px 16px",
    borderRadius: 100,
    fontSize: 12,
    fontWeight: 600,
    letterSpacing: 0.8,
    textTransform: "uppercase",
    marginBottom: 28,
  },
  h1: {
    fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif",
    fontSize: 58,
    fontWeight: 800,
    lineHeight: 1.06,
    color: "#0F1C2E",
    margin: "0 0 24px",
    letterSpacing: -1.5,
  },
  accent: {
    background: "linear-gradient(90deg, #0B7B6F, #1D6FA4)",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
    backgroundClip: "text",
  },
  sub: {
    fontSize: 17,
    color: "#5A7184",
    lineHeight: 1.75,
    maxWidth: 480,
    margin: "0 0 40px",
  },
  btnPrimary: {
    background: "#0B7B6F",
    color: "#fff",
    border: "none",
    padding: "15px 32px",
    borderRadius: 10,
    fontWeight: 700,
    fontSize: 15,
    cursor: "pointer",
    fontFamily: "'Inter', sans-serif",
    letterSpacing: 0.2,
    transition: "background 0.2s, transform 0.15s",
  },
  btnSecondary: {
    background: "#fff",
    color: "#0F1C2E",
    border: "1.5px solid #D6E0EA",
    padding: "15px 32px",
    borderRadius: 10,
    fontWeight: 600,
    fontSize: 15,
    cursor: "pointer",
    fontFamily: "'Inter', sans-serif",
    transition: "border-color 0.2s, background 0.2s",
  },
};

const LiveDot = () => (
  <span style={{ position: "relative", display: "inline-block", width: 8, height: 8 }}>
    <span style={{
      position: "absolute", inset: 0, borderRadius: "50%",
      background: "#0B7B6F", animation: "ping 1.6s ease-in-out infinite", opacity: 0.6,
    }} />
    <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: "#0B7B6F" }} />
    <style>{`@keyframes ping{0%{transform:scale(1);opacity:0.6}70%{transform:scale(2.2);opacity:0}100%{transform:scale(1);opacity:0}}`}</style>
  </span>
);

const FeatureCard = ({ icon, title, desc, tag }) => (
  <div style={{
    background: "#fff",
    border: "1px solid #E8EFF5",
    borderRadius: 18,
    padding: "32px 28px",
    transition: "box-shadow 0.2s, transform 0.2s",
    cursor: "default",
  }}
    onMouseEnter={e => { e.currentTarget.style.boxShadow = "0 8px 32px rgba(11,123,111,0.10)"; e.currentTarget.style.transform = "translateY(-3px)"; }}
    onMouseLeave={e => { e.currentTarget.style.boxShadow = "none"; e.currentTarget.style.transform = "none"; }}
  >
    <div style={{
      width: 48, height: 48, borderRadius: 12,
      background: "#EBF8F6",
      display: "flex", alignItems: "center", justifyContent: "center",
      fontSize: 22, marginBottom: 20,
    }}>{icon}</div>
    {tag && (
      <span style={{
        fontSize: 11, fontWeight: 700, color: "#0B7B6F",
        background: "#EBF8F6", border: "1px solid #B2E8E2",
        padding: "3px 10px", borderRadius: 100, letterSpacing: 0.6,
        textTransform: "uppercase", marginBottom: 14, display: "inline-block",
      }}>{tag}</span>
    )}
    <h3 style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif", fontSize: 17, fontWeight: 700, color: "#0F1C2E", margin: "0 0 10px" }}>{title}</h3>
    <p style={{ fontSize: 14, color: "#7A94A8", lineHeight: 1.7, margin: 0 }}>{desc}</p>
  </div>
);

const StatPill = ({ val, label }) => (
  <div style={{ textAlign: "left" }}>
    <div style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif", fontSize: 28, fontWeight: 800, color: "#0F1C2E", letterSpacing: -0.5 }}>{val}</div>
    <div style={{ fontSize: 12, color: "#8FA5B5", marginTop: 3, fontWeight: 500 }}>{label}</div>
  </div>
);

function Home() {
  const navigate = useNavigate();

  return (
    <div style={S.page}>
      <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&display=swap" rel="stylesheet" />

      {/* HERO */}
      <div style={S.hero}>
        {/* Left */}
        <div>
          <div style={S.eyebrow}>
            <LiveDot />
            AI Clinical Intelligence Platform
          </div>

          <h1 style={S.h1}>
            Identify Rare<br />
            Diseases with<br />
            <span style={S.accent}>Precision AI</span>
          </h1>

          <p style={S.sub}>
            Submit patient symptoms and biomedical scans. Our multimodal AI cross-references 1,374 rare diseases and returns ranked, confidence-scored differential diagnoses.
          </p>

          <div style={{ display: "flex", gap: 12 }}>
            <button
              style={S.btnPrimary}
              onClick={() => navigate("/predict")}
              onMouseEnter={e => { e.currentTarget.style.background = "#08635A"; }}
              onMouseLeave={e => { e.currentTarget.style.background = "#0B7B6F"; }}
            >
              Start Diagnosis →
            </button>
            <button
              style={S.btnSecondary}
              onClick={() => navigate("/dashboard")}
              onMouseEnter={e => { e.currentTarget.style.borderColor = "#9BB8CC"; e.currentTarget.style.background = "#F5F9FC"; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = "#D6E0EA"; e.currentTarget.style.background = "#fff"; }}
            >
              View Dashboard
            </button>
          </div>

          <div style={{ display: "flex", gap: 40, marginTop: 52, paddingTop: 40, borderTop: "1px solid #E8EFF5" }}>
            <StatPill val="36K+" label="Patient Cases" />
            <StatPill val="1,374" label="Rare Diseases" />
            <StatPill val="94K+" label="Medical Images" />
          </div>
        </div>

        {/* Right — dashboard preview card */}
        <div style={{
          background: "#fff",
          border: "1px solid #E0EBF2",
          borderRadius: 24,
          padding: 32,
          boxShadow: "0 20px 60px rgba(15,28,46,0.08)",
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 28 }}>
            <span style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif", fontWeight: 700, fontSize: 16, color: "#0F1C2E" }}>
              System Status
            </span>
            <span style={{
              display: "flex", alignItems: "center", gap: 6,
              background: "#EBF8F6", border: "1px solid #B2E8E2",
              color: "#0B7B6F", padding: "5px 12px", borderRadius: 100,
              fontSize: 11, fontWeight: 700, letterSpacing: 0.5, textTransform: "uppercase",
            }}>
              <LiveDot /> Online
            </span>
          </div>

          {/* Stats grid */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 24 }}>
            {[
              { label: "Predictions", value: "1,254", color: "#EBF8F6", textColor: "#0B7B6F" },
              { label: "Top-3 Accuracy", value: "52.9%", color: "#EBF4F9", textColor: "#1D6FA4" },
              { label: "Diseases Covered", value: "49", color: "#F2EEF9", textColor: "#5B3DB8" },
              { label: "Training Images", value: "35K+", color: "#FFF4EC", textColor: "#C05B1A" },
            ].map(({ label, value, color, textColor }) => (
              <div key={label} style={{
                background: color,
                borderRadius: 14,
                padding: "18px 20px",
              }}>
                <p style={{ fontSize: 11, color: textColor, opacity: 0.7, textTransform: "uppercase", letterSpacing: 0.8, margin: "0 0 8px", fontWeight: 600 }}>{label}</p>
                <p style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif", fontSize: 28, fontWeight: 800, color: textColor, margin: 0 }}>{value}</p>
              </div>
            ))}
          </div>

          {/* Model bars */}
          <div style={{ borderTop: "1px solid #EDF2F6", paddingTop: 20 }}>
            <p style={{ fontSize: 11, fontWeight: 700, color: "#8FA5B5", textTransform: "uppercase", letterSpacing: 0.8, margin: "0 0 16px" }}>Model Performance</p>
            {[
              { label: "Symptom Model", pct: 52.9, color: "#0B7B6F" },
              { label: "Image Model", pct: 35, color: "#1D6FA4", note: "Training" },
              { label: "Fusion Model", pct: 0, color: "#9B8ED6", note: "Coming soon" },
            ].map(({ label, pct, color, note }) => (
              <div key={label} style={{ marginBottom: 14 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 6 }}>
                  <span style={{ color: "#4A6275", fontWeight: 500 }}>{label}</span>
                  <span style={{ color, fontWeight: 700, fontSize: 12 }}>{note || `${pct}%`}</span>
                </div>
                <div style={{ height: 5, background: "#EDF2F6", borderRadius: 100 }}>
                  <div style={{ height: 5, width: `${pct}%`, background: color, borderRadius: 100, transition: "width 1s ease" }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* FEATURES */}
      <div style={{ background: "#fff", borderTop: "1px solid #EDF2F6", borderBottom: "1px solid #EDF2F6" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", padding: "80px 40px" }}>
          <div style={{ textAlign: "center", marginBottom: 56 }}>
            <span style={{
              fontSize: 11, fontWeight: 700, color: "#0B7B6F",
              background: "#EBF8F6", border: "1px solid #B2E8E2",
              padding: "4px 14px", borderRadius: 100, letterSpacing: 1,
              textTransform: "uppercase", display: "inline-block", marginBottom: 20,
            }}>Capabilities</span>
            <h2 style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif", fontSize: 40, fontWeight: 800, color: "#0F1C2E", margin: "0 0 16px", letterSpacing: -0.8 }}>
              Three Diagnostic Pathways
            </h2>
            <p style={{ fontSize: 16, color: "#7A94A8", maxWidth: 480, margin: "0 auto", lineHeight: 1.7 }}>
              Symptoms alone, images alone, or both combined. The fusion model achieves highest accuracy by cross-weighting all signals.
            </p>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 20 }}>
            <FeatureCard icon="🧬" tag="NLP" title="Symptom Analysis" desc="TF-IDF encoded symptom vectors fed into a multinomial classifier trained on 8,568 patient cases across 49 diseases." />
            <FeatureCard icon="🔬" tag="Computer Vision" title="Image Detection" desc="EfficientNet-B4 fine-tuned on 35,374 biomedical images — MRI, CT, histopathology, dermoscopy." />
            <FeatureCard icon="🤖" tag="Fusion" title="Multimodal AI" desc="Cross-attention fusion of both modalities. Correct diagnosis appears in top-3 predictions 53% of the time." />
            <FeatureCard icon="📊" tag="Analytics" title="Live Analytics" desc="Monitor prediction history, confidence distributions, and model performance metrics in real time." />
          </div>
        </div>
      </div>

      {/* HOW IT WORKS */}
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "80px 40px" }}>
        <div style={{ textAlign: "center", marginBottom: 56 }}>
          <span style={{
            fontSize: 11, fontWeight: 700, color: "#0B7B6F",
            background: "#EBF8F6", border: "1px solid #B2E8E2",
            padding: "4px 14px", borderRadius: 100, letterSpacing: 1,
            textTransform: "uppercase", display: "inline-block", marginBottom: 20,
          }}>Process</span>
          <h2 style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif", fontSize: 40, fontWeight: 800, color: "#0F1C2E", margin: "0 0 16px", letterSpacing: -0.8 }}>
            Diagnosis in 3 Steps
          </h2>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 0, position: "relative" }}>
          {[
            { step: "01", title: "Enter Symptoms", desc: "Describe the patient's symptoms in natural language or select from common symptom chips.", icon: "📝" },
            { step: "02", title: "Upload Scan (Optional)", desc: "Add a biomedical image — MRI, CT, dermoscopy, or histopathology — for multimodal analysis.", icon: "🩻" },
            { step: "03", title: "Receive Diagnosis", desc: "Get ranked differential diagnoses with probability scores, confidence levels, and model attribution.", icon: "🎯" },
          ].map(({ step, title, desc, icon }, i) => (
            <div key={step} style={{ position: "relative" }}>
              {i < 2 && (
                <div style={{
                  position: "absolute", top: 40, right: -1, width: 2,
                  height: 60, background: "linear-gradient(180deg, #B2E8E2, #E8EFF5)",
                  zIndex: 1,
                }} />
              )}
              <div style={{
                padding: "40px 36px",
                background: i === 1 ? "#EBF8F6" : "#fff",
                border: "1px solid #E8EFF5",
                borderRadius: i === 0 ? "18px 0 0 18px" : i === 2 ? "0 18px 18px 0" : "0",
                borderLeft: i > 0 ? "none" : "1px solid #E8EFF5",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 20 }}>
                  <span style={{
                    fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif",
                    fontSize: 12, fontWeight: 800, color: "#0B7B6F",
                    background: "#fff", border: "1.5px solid #B2E8E2",
                    width: 40, height: 40, borderRadius: "50%",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    flexShrink: 0,
                  }}>{step}</span>
                  <span style={{ fontSize: 24 }}>{icon}</span>
                </div>
                <h3 style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif", fontSize: 18, fontWeight: 700, color: "#0F1C2E", margin: "0 0 12px" }}>{title}</h3>
                <p style={{ fontSize: 14, color: "#7A94A8", lineHeight: 1.7, margin: 0 }}>{desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* FOOTER */}
      <footer style={{
        borderTop: "1px solid #EDF2F6",
        padding: "32px 40px",
        textAlign: "center",
        color: "#9BB8CC",
        fontSize: 13,
        background: "#fff",
      }}>
        © 2026 AI DOC · Rare Disease Detection System · ZebraMap Dataset · CC BY 4.0
      </footer>
    </div>
  );
}

export default Home;