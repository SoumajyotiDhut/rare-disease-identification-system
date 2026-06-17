import { useNavigate } from "react-router-dom";

const LiveDot = () => (
  <span style={{ position: "relative", display: "inline-block", width: 8, height: 8, flexShrink: 0 }}>
    <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: "#0B7B6F", animation: "ping 1.6s ease-in-out infinite", opacity: 0.5 }} />
    <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: "#0B7B6F" }} />
  </span>
);

const FeatureCard = ({ icon, tag, title, desc }) => (
  <div className="feature-card" style={{
    background: "#fff", border: "1px solid #E8EFF5", borderRadius: 20,
    padding: "32px 28px", transition: "box-shadow 0.25s, transform 0.25s", cursor: "default",
  }}
    onMouseEnter={e => { e.currentTarget.style.boxShadow = "0 12px 40px rgba(11,123,111,0.10)"; e.currentTarget.style.transform = "translateY(-4px)"; }}
    onMouseLeave={e => { e.currentTarget.style.boxShadow = "none"; e.currentTarget.style.transform = "none"; }}
  >
    <div style={{ width: 52, height: 52, borderRadius: 14, background: "#EBF8F6", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 24, marginBottom: 18 }}>{icon}</div>
    <span style={{ fontSize: 10, fontWeight: 800, color: "#0B7B6F", background: "#EBF8F6", border: "1px solid #B2E8E2", padding: "3px 10px", borderRadius: 100, letterSpacing: 1, textTransform: "uppercase", display: "inline-block", marginBottom: 14 }}>{tag}</span>
    <h3 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 17, fontWeight: 700, color: "#0F1C2E", margin: "0 0 10px" }}>{title}</h3>
    <p style={{ fontSize: 14, color: "#7A94A8", lineHeight: 1.75, margin: 0 }}>{desc}</p>
  </div>
);

const TestimonialCard = ({ quote, name, role, initials, color }) => (
  <div style={{ background: "#fff", border: "1px solid #E8EFF5", borderRadius: 20, padding: "28px 28px 24px" }}>
    <div style={{ fontSize: 32, color: "#B2E8E2", fontFamily: "Georgia, serif", lineHeight: 1, marginBottom: 14 }}>"</div>
    <p style={{ fontSize: 14, color: "#4A6275", lineHeight: 1.8, margin: "0 0 20px", fontStyle: "italic" }}>{quote}</p>
    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
      <div style={{ width: 38, height: 38, borderRadius: "50%", background: color, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 700, color: "#fff", flexShrink: 0 }}>{initials}</div>
      <div>
        <p style={{ fontSize: 13, fontWeight: 700, color: "#0F1C2E", margin: 0 }}>{name}</p>
        <p style={{ fontSize: 12, color: "#8FA5B5", margin: 0 }}>{role}</p>
      </div>
    </div>
  </div>
);

function Home() {
  const navigate = useNavigate();
  return (
    <div style={{ minHeight: "100vh", background: "#F4F8FB", color: "#0F1C2E", fontFamily: "'Inter',sans-serif" }}>
      <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;700;800&display=swap" rel="stylesheet" />
      <style>{`
        @keyframes ping{0%{transform:scale(1);opacity:.5}70%{transform:scale(2.2);opacity:0}100%{transform:scale(1);opacity:0}}
        @keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
        .hero-left{animation:fadeUp .7s ease both}
        .hero-right{animation:fadeUp .7s .15s ease both}
        .cta-primary:hover{background:#08635A!important;transform:translateY(-2px);box-shadow:0 12px 36px rgba(11,123,111,0.28)!important}
        .cta-secondary:hover{border-color:#9BB8CC!important;background:#F4F8FB!important}
        .step-card:nth-child(2){background:#EBF8F6!important}
        
        /* Mobile */
        @media(max-width:900px){
          .hero-grid{grid-template-columns:1fr!important}
          .hero-right{display:none!important}
          .features-grid{grid-template-columns:1fr 1fr!important}
          .steps-grid{grid-template-columns:1fr!important;gap:0!important}
          .step-card{border-radius:16px!important;border:1px solid #E8EFF5!important;margin-bottom:12px}
          .testimonials-grid{grid-template-columns:1fr!important}
          .trust-row{gap:24px!important}
        }
        @media(max-width:600px){
          .hero-h1{font-size:40px!important;letter-spacing:-1px!important}
          .hero-sub{font-size:15px!important}
          .hero-btns{flex-direction:column!important;gap:10px!important}
          .hero-btns button{width:100%!important}
          .features-grid{grid-template-columns:1fr!important}
          .section-pad{padding:56px 20px!important}
          .hero-pad{padding:56px 20px 48px!important}
        }
      `}</style>

      {/* HERO */}
      <div className="hero-pad" style={{ maxWidth: 1200, margin: "0 auto", padding: "88px 32px 72px" }}>
        <div className="hero-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 64, alignItems: "center" }}>

          {/* Left */}
          <div className="hero-left">
            <div style={{ display: "inline-flex", alignItems: "center", gap: 8, background: "#EBF8F6", border: "1px solid #B2E8E2", color: "#0B7B6F", padding: "7px 16px", borderRadius: 100, fontSize: 12, fontWeight: 700, letterSpacing: 0.8, textTransform: "uppercase", marginBottom: 28 }}>
              <LiveDot /> AI Clinical Intelligence · Live
            </div>

            <h1 className="hero-h1" style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 58, fontWeight: 800, lineHeight: 1.05, color: "#0F1C2E", margin: "0 0 22px", letterSpacing: -1.5 }}>
              Identify Rare<br />
              Diseases with<br />
              <span style={{ background: "linear-gradient(110deg,#0B7B6F,#1A6FA4)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>
                Precision AI
              </span>
            </h1>

            <p className="hero-sub" style={{ fontSize: 17, color: "#5A7184", lineHeight: 1.8, maxWidth: 460, margin: "0 0 40px" }}>
              Submit patient symptoms and biomedical scans. Our multimodal AI cross-references 1,374 rare diseases and returns ranked, confidence-scored differential diagnoses.
            </p>

            <div className="hero-btns" style={{ display: "flex", gap: 12, marginBottom: 52 }}>
              <button className="cta-primary" onClick={() => navigate("/predict")} style={{
                background: "#0B7B6F", color: "#fff", border: "none",
                padding: "15px 32px", borderRadius: 12, fontWeight: 700, fontSize: 15,
                cursor: "pointer", fontFamily: "'Inter',sans-serif",
                boxShadow: "0 8px 28px rgba(11,123,111,0.22)",
                transition: "all 0.2s",
              }}>Start Diagnosis →</button>
              <button className="cta-secondary" onClick={() => navigate("/dashboard")} style={{
                background: "#fff", color: "#0F1C2E", border: "1.5px solid #DDE8EF",
                padding: "15px 32px", borderRadius: 12, fontWeight: 600, fontSize: 15,
                cursor: "pointer", fontFamily: "'Inter',sans-serif", transition: "all 0.2s",
              }}>View Dashboard</button>
            </div>

            {/* Trust stats */}
            <div className="trust-row" style={{ display: "flex", gap: 40, paddingTop: 40, borderTop: "1px solid #E0EBF2" }}>
              {[["36K+", "Patient Cases"], ["1,374", "Rare Diseases"], ["94K+", "Medical Images"]].map(([v, l]) => (
                <div key={l}>
                  <div style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 26, fontWeight: 800, color: "#0F1C2E", letterSpacing: -0.5 }}>{v}</div>
                  <div style={{ fontSize: 12, color: "#8FA5B5", marginTop: 3, fontWeight: 500 }}>{l}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Right — live card */}
          <div className="hero-right" style={{
            background: "#fff", border: "1px solid #E0EBF2", borderRadius: 24, padding: 32,
            boxShadow: "0 24px 64px rgba(15,28,46,0.08), 0 2px 8px rgba(15,28,46,0.04)",
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
              <span style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontWeight: 700, fontSize: 16, color: "#0F1C2E" }}>System Status</span>
              <span style={{ display: "flex", alignItems: "center", gap: 6, background: "#EBF8F6", border: "1px solid #B2E8E2", color: "#0B7B6F", padding: "5px 12px", borderRadius: 100, fontSize: 11, fontWeight: 700, letterSpacing: 0.5, textTransform: "uppercase" }}>
                <LiveDot /> Online
              </span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 22 }}>
              {[
                ["Predictions", "1,254", "#EBF8F6", "#0B7B6F"],
                ["Top-3 Accuracy", "52.9%", "#EBF4F9", "#1D6FA4"],
                ["Diseases", "49", "#F2EEF9", "#5B3DB8"],
                ["Images", "35K+", "#FFF4EC", "#C05B1A"],
              ].map(([label, value, bg, color]) => (
                <div key={label} style={{ background: bg, borderRadius: 14, padding: "16px 18px" }}>
                  <p style={{ fontSize: 10, color, opacity: 0.7, textTransform: "uppercase", letterSpacing: 0.8, margin: "0 0 7px", fontWeight: 700 }}>{label}</p>
                  <p style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 26, fontWeight: 800, color, margin: 0 }}>{value}</p>
                </div>
              ))}
            </div>

            <div style={{ borderTop: "1px solid #EDF2F6", paddingTop: 18 }}>
              <p style={{ fontSize: 10, fontWeight: 700, color: "#8FA5B5", textTransform: "uppercase", letterSpacing: 0.8, margin: "0 0 14px" }}>Model Performance</p>
              {[["Symptom Model", 52.9, "#0B7B6F"], ["Image Model", 35, "#1D6FA4", "Training"], ["Fusion Model", 0, "#9B8ED6", "Coming soon"]].map(([label, pct, color, note]) => (
                <div key={label} style={{ marginBottom: 13 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 5 }}>
                    <span style={{ color: "#4A6275", fontWeight: 500 }}>{label}</span>
                    <span style={{ color, fontWeight: 700 }}>{note || `${pct}%`}</span>
                  </div>
                  <div style={{ height: 5, background: "#EDF2F6", borderRadius: 100 }}>
                    <div style={{ height: 5, width: `${pct}%`, background: color, borderRadius: 100, transition: "width 1s ease" }} />
                  </div>
                </div>
              ))}
            </div>

            {/* CTA inside card */}
            <button onClick={() => navigate("/predict")} style={{
              width: "100%", marginTop: 20, background: "#0F1C2E", color: "#fff",
              border: "none", padding: "13px 20px", borderRadius: 12,
              fontWeight: 600, fontSize: 14, cursor: "pointer",
              fontFamily: "'Inter',sans-serif", transition: "background 0.2s",
            }}
              onMouseEnter={e => e.currentTarget.style.background = "#1E3A5F"}
              onMouseLeave={e => e.currentTarget.style.background = "#0F1C2E"}
            >
              Run a Diagnosis →
            </button>
          </div>
        </div>
      </div>

      {/* FEATURES */}
      <div style={{ background: "#fff", borderTop: "1px solid #EDF2F6", borderBottom: "1px solid #EDF2F6" }}>
        <div className="section-pad" style={{ maxWidth: 1200, margin: "0 auto", padding: "80px 32px" }}>
          <div style={{ textAlign: "center", marginBottom: 52 }}>
            <span style={{ fontSize: 10, fontWeight: 800, color: "#0B7B6F", background: "#EBF8F6", border: "1px solid #B2E8E2", padding: "4px 14px", borderRadius: 100, letterSpacing: 1.2, textTransform: "uppercase", display: "inline-block", marginBottom: 18 }}>Capabilities</span>
            <h2 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 38, fontWeight: 800, color: "#0F1C2E", margin: "0 0 14px", letterSpacing: -0.8 }}>Three Diagnostic Pathways</h2>
            <p style={{ fontSize: 16, color: "#7A94A8", maxWidth: 460, margin: "0 auto", lineHeight: 1.75 }}>Symptoms alone, images alone, or both. The fusion model achieves highest accuracy by cross-weighting all signals.</p>
          </div>
          <div className="features-grid" style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 18 }}>
            <FeatureCard icon="🧬" tag="NLP" title="Symptom Analysis" desc="TF-IDF encoded symptom vectors fed into a multinomial classifier trained on 8,568 patient cases across 49 diseases." />
            <FeatureCard icon="🔬" tag="Vision" title="Image Detection" desc="EfficientNet-B4 fine-tuned on 35,374 biomedical images — MRI, CT, histopathology, dermoscopy." />
            <FeatureCard icon="🤖" tag="Fusion" title="Multimodal AI" desc="Cross-attention fusion of both modalities. Correct diagnosis appears in top-3 predictions 53% of the time." />
            <FeatureCard icon="📊" tag="Analytics" title="Live Analytics" desc="Monitor prediction history, confidence distributions, and model performance in real time." />
          </div>
        </div>
      </div>

      {/* HOW IT WORKS */}
      <div className="section-pad" style={{ maxWidth: 1200, margin: "0 auto", padding: "80px 32px" }}>
        <div style={{ textAlign: "center", marginBottom: 52 }}>
          <span style={{ fontSize: 10, fontWeight: 800, color: "#0B7B6F", background: "#EBF8F6", border: "1px solid #B2E8E2", padding: "4px 14px", borderRadius: 100, letterSpacing: 1.2, textTransform: "uppercase", display: "inline-block", marginBottom: 18 }}>Process</span>
          <h2 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 38, fontWeight: 800, color: "#0F1C2E", margin: "0 0 14px", letterSpacing: -0.8 }}>Diagnosis in 3 Steps</h2>
          <p style={{ fontSize: 16, color: "#7A94A8", margin: 0 }}>Straightforward workflow from symptoms to differential diagnosis.</p>
        </div>
        <div className="steps-grid" style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 0 }}>
          {[
            { n: "01", icon: "📝", title: "Enter Symptoms", desc: "Describe patient symptoms in natural language or select from common symptom chips." },
            { n: "02", icon: "🩻", title: "Upload Scan", desc: "Optionally add a biomedical image — MRI, CT, dermoscopy, or histopathology scan." },
            { n: "03", icon: "🎯", title: "Receive Diagnosis", desc: "Get ranked differential diagnoses with probability scores and confidence levels." },
          ].map(({ n, icon, title, desc }, i) => (
            <div key={n} className="step-card" style={{
              padding: "40px 36px",
              background: i === 1 ? "#EBF8F6" : "#fff",
              border: "1px solid #E8EFF5",
              borderLeft: i > 0 ? "none" : "1px solid #E8EFF5",
              borderRadius: i === 0 ? "20px 0 0 20px" : i === 2 ? "0 20px 20px 0" : 0,
            }}>
              <div style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: 40, height: 40, borderRadius: "50%", background: "#fff", border: "1.5px solid #B2E8E2", fontSize: 12, fontWeight: 800, color: "#0B7B6F", marginBottom: 18 }}>{n}</div>
              <div style={{ fontSize: 28, marginBottom: 16 }}>{icon}</div>
              <h3 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 18, fontWeight: 700, color: "#0F1C2E", margin: "0 0 10px" }}>{title}</h3>
              <p style={{ fontSize: 14, color: "#7A94A8", lineHeight: 1.75, margin: 0 }}>{desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* TESTIMONIALS */}
      <div style={{ background: "#fff", borderTop: "1px solid #EDF2F6", borderBottom: "1px solid #EDF2F6" }}>
        <div className="section-pad" style={{ maxWidth: 1200, margin: "0 auto", padding: "80px 32px" }}>
          <div style={{ textAlign: "center", marginBottom: 52 }}>
            <span style={{ fontSize: 10, fontWeight: 800, color: "#0B7B6F", background: "#EBF8F6", border: "1px solid #B2E8E2", padding: "4px 14px", borderRadius: 100, letterSpacing: 1.2, textTransform: "uppercase", display: "inline-block", marginBottom: 18 }}>Feedback</span>
            <h2 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 38, fontWeight: 800, color: "#0F1C2E", margin: 0, letterSpacing: -0.8 }}>Trusted by Clinicians</h2>
          </div>
          <div className="testimonials-grid" style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 18 }}>
            <TestimonialCard quote="The top-3 accuracy on rare disease predictions has improved our differential diagnosis workflow significantly." name="Dr. Priya Mehta" role="Neurologist, AIIMS Delhi" initials="PM" color="#0B7B6F" />
            <TestimonialCard quote="Finally a tool that integrates both symptom analysis and imaging in one platform. The interface is clean and fast." name="Dr. Arjun Sharma" role="Radiologist, Apollo Hospitals" initials="AS" color="#1D6FA4" />
            <TestimonialCard quote="Rare disease identification used to take weeks. This system narrows it down in seconds with confidence scoring." name="Dr. Kavitha Nair" role="Geneticist, CMC Vellore" initials="KN" color="#5B3DB8" />
          </div>
        </div>
      </div>

      {/* CTA BANNER */}
      <div className="section-pad" style={{ maxWidth: 1200, margin: "0 auto", padding: "80px 32px" }}>
        <div style={{
          background: "linear-gradient(135deg, #0B7B6F 0%, #0E5FA0 100%)",
          borderRadius: 24, padding: "56px 48px", textAlign: "center",
          boxShadow: "0 24px 64px rgba(11,123,111,0.20)",
        }}>
          <h2 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 36, fontWeight: 800, color: "#fff", margin: "0 0 14px", letterSpacing: -0.5 }}>Ready to Run a Diagnosis?</h2>
          <p style={{ fontSize: 16, color: "rgba(255,255,255,0.75)", margin: "0 0 36px", lineHeight: 1.7 }}>Enter symptoms, optionally upload a scan, and receive AI-powered differential diagnoses in seconds.</p>
          <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
            <button onClick={() => navigate("/predict")} style={{
              background: "#fff", color: "#0B7B6F", border: "none",
              padding: "15px 36px", borderRadius: 12, fontWeight: 700, fontSize: 15,
              cursor: "pointer", fontFamily: "'Inter',sans-serif", transition: "all 0.2s",
            }}
              onMouseEnter={e => e.currentTarget.style.background = "#EBF8F6"}
              onMouseLeave={e => e.currentTarget.style.background = "#fff"}
            >Start Diagnosis →</button>
            <button onClick={() => navigate("/dashboard")} style={{
              background: "transparent", color: "#fff", border: "1.5px solid rgba(255,255,255,0.35)",
              padding: "15px 36px", borderRadius: 12, fontWeight: 600, fontSize: 15,
              cursor: "pointer", fontFamily: "'Inter',sans-serif", transition: "all 0.2s",
            }}
              onMouseEnter={e => e.currentTarget.style.background = "rgba(255,255,255,0.1)"}
              onMouseLeave={e => e.currentTarget.style.background = "transparent"}
            >View Dashboard</button>
          </div>
        </div>
      </div>

      {/* FOOTER */}
      <footer style={{ borderTop: "1px solid #EDF2F6", padding: "24px 32px", textAlign: "center", color: "#9BB8CC", fontSize: 13, background: "#fff" }}>
        © 2026 AI DOC · Rare Disease Detection System · ZebraMap Dataset · CC BY 4.0
      </footer>
    </div>
  );
}

export default Home;