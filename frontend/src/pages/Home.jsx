import { useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";

const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
  @keyframes ping      { 0%{transform:scale(1);opacity:.5} 70%{transform:scale(2.2);opacity:0} 100%{transform:scale(1);opacity:0} }
  @keyframes fadeUp    { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }
  @keyframes slideIn   { from{opacity:0;transform:translateX(-16px)} to{opacity:1;transform:translateX(0)} }
  @keyframes float     { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
  @keyframes barGrow   { from{width:0} }
  @keyframes gradShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
  @keyframes countUp   { from{opacity:0;transform:scale(.7)} to{opacity:1;transform:scale(1)} }

  .hero-cta-p:hover  { background:#08635A!important; box-shadow:0 14px 36px rgba(11,123,111,0.30)!important; transform:translateY(-2px) }
  .hero-cta-s:hover  { border-color:#9BB8CC!important; background:#F0F5F8!important }
  .feature-card:hover { box-shadow:0 14px 44px rgba(11,123,111,0.10)!important; transform:translateY(-5px)!important; border-color:#B2E8E2!important }
  .stat-card:hover   { transform:translateY(-3px)!important; box-shadow:0 10px 30px rgba(15,28,46,0.07)!important }
  .testimonial:hover { box-shadow:0 8px 28px rgba(15,28,46,0.08)!important }
  .nav-dot           { cursor:pointer; transition:all .2s }
  .nav-dot:hover     { transform:scale(1.3) }
  .faq-item:hover    { border-color:#B2E8E2!important }

  @media(max-width:1024px) {
    .hero-grid      { grid-template-columns:1fr!important }
    .hero-preview   { display:none!important }
    .features-grid  { grid-template-columns:1fr 1fr!important }
    .steps-grid     { grid-template-columns:1fr!important; gap:12px!important }
    .step-seg       { border-left:1px solid #E8EFF5!important; border-radius:16px!important }
    .test-grid      { grid-template-columns:1fr 1fr!important }
    .faq-grid       { grid-template-columns:1fr!important }
  }
  @media(max-width:640px) {
    .hero-h1        { font-size:38px!important; letter-spacing:-1px!important }
    .hero-sub       { font-size:15px!important }
    .hero-btns      { flex-direction:column!important }
    .hero-btns button { width:100%!important }
    .trust-row      { gap:20px!important }
    .section-pad    { padding:60px 20px!important }
    .features-grid  { grid-template-columns:1fr!important }
    .test-grid      { grid-template-columns:1fr!important }
    .section-title  { font-size:30px!important }
    .cta-banner-btns{ flex-direction:column!important; align-items:stretch!important }
    .stats-row      { grid-template-columns:1fr 1fr!important }
  }
`;

const LiveDot = () => (
  <span style={{ position: "relative", display: "inline-block", width: 8, height: 8, flexShrink: 0 }}>
    <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: "#0B7B6F", animation: "ping 1.6s ease infinite", opacity: .5 }} />
    <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: "#0B7B6F" }} />
  </span>
);

const Eyebrow = ({ children }) => (
  <span style={{
    fontSize: 10, fontWeight: 800, color: "#0B7B6F", background: "#EBF8F6", border: "1px solid #B2E8E2",
    padding: "4px 14px", borderRadius: 100, letterSpacing: 1.2, textTransform: "uppercase",
    display: "inline-block", marginBottom: 18
  }}>{children}</span>
);

const FeatureCard = ({ icon, tag, tagColor, tagBg, tagBorder, title, desc, delay }) => (
  <div className="feature-card" style={{
    background: "#fff", border: "1px solid #E8EFF5", borderRadius: 22,
    padding: "32px 28px", transition: "all .25s", cursor: "default",
    animation: `fadeUp .6s ${delay}s ease both`
  }}>
    <div style={{
      width: 54, height: 54, borderRadius: 15, background: "#EBF8F6", display: "flex",
      alignItems: "center", justifyContent: "center", fontSize: 24, marginBottom: 18
    }}>{icon}</div>
    <span style={{
      fontSize: 10, fontWeight: 800, color: tagColor, background: tagBg, border: `1px solid ${tagBorder}`,
      padding: "3px 10px", borderRadius: 100, letterSpacing: 1, textTransform: "uppercase",
      display: "inline-block", marginBottom: 14
    }}>{tag}</span>
    <h3 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 17, fontWeight: 700, color: "#0F1C2E", margin: "0 0 10px" }}>{title}</h3>
    <p style={{ fontSize: 14, color: "#7A94A8", lineHeight: 1.8, margin: 0 }}>{desc}</p>
  </div>
);

const TestimonialCard = ({ quote, name, role, initials, color, bg, delay }) => (
  <div className="testimonial" style={{
    background: "#fff", border: "1px solid #E8EFF5", borderRadius: 22, padding: "28px",
    transition: "box-shadow .2s", animation: `fadeUp .6s ${delay}s ease both`
  }}>
    <div style={{ display: "flex", gap: 3, marginBottom: 14 }}>
      {[1, 2, 3, 4, 5].map(i => <span key={i} style={{ color: "#F5A623", fontSize: 14 }}>★</span>)}
    </div>
    <p style={{ fontSize: 14, color: "#4A6275", lineHeight: 1.85, margin: "0 0 22px", fontStyle: "italic" }}>"{quote}"</p>
    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
      <div style={{
        width: 40, height: 40, borderRadius: "50%", background: color, display: "flex",
        alignItems: "center", justifyContent: "center", fontSize: 14, fontWeight: 800, color: "#fff", flexShrink: 0
      }}>{initials}</div>
      <div>
        <p style={{ fontSize: 13, fontWeight: 700, color: "#0F1C2E", margin: "0 0 1px" }}>{name}</p>
        <p style={{ fontSize: 12, color: "#8FA5B5", margin: 0 }}>{role}</p>
      </div>
    </div>
  </div>
);

const FaqItem = ({ q, a }) => {
  const [open, setOpen] = useState(false);
  return (
    <div className="faq-item" onClick={() => setOpen(!open)} style={{
      background: "#fff", border: "1px solid #E8EFF5", borderRadius: 16, padding: "20px 24px",
      cursor: "pointer", transition: "border-color .2s, box-shadow .2s",
      boxShadow: open ? "0 4px 20px rgba(15,28,46,0.06)" : "none",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <p style={{ fontSize: 15, fontWeight: 700, color: "#0F1C2E", margin: 0, paddingRight: 16 }}>{q}</p>
        <span style={{
          fontSize: 18, color: "#0B7B6F", flexShrink: 0, transition: "transform .2s",
          transform: open ? "rotate(45deg)" : "none"
        }}>+</span>
      </div>
      {open && <p style={{ fontSize: 14, color: "#5A7184", lineHeight: 1.75, margin: "14px 0 0", animation: "fadeUp .25s ease" }}>{a}</p>}
    </div>
  );
};

/* Animated counter */
const Counter = ({ target, suffix = "" }) => {
  const [val, setVal] = useState(0);
  useEffect(() => {
    const num = parseInt(target.replace(/\D/g, ""));
    let start = 0;
    const step = Math.ceil(num / 40);
    const t = setInterval(() => {
      start += step;
      if (start >= num) { setVal(num); clearInterval(t); } else setVal(start);
    }, 30);
    return () => clearInterval(t);
  }, []);
  return <>{target.replace(/\d+/, val.toLocaleString())}</>;
};

function Home() {
  const navigate = useNavigate();
  return (
    <div style={{ minHeight: "100vh", background: "#F4F8FB", color: "#0F1C2E", fontFamily: "'Inter',sans-serif" }}>
      <style>{CSS}</style>

      {/* ══ HERO ══ */}
      <div className="section-pad" style={{ maxWidth: 1200, margin: "0 auto", padding: "96px 32px 80px" }}>
        <div className="hero-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 72, alignItems: "center" }}>

          {/* Left */}
          <div style={{ animation: "slideIn .7s ease" }}>
            <div style={{
              display: "inline-flex", alignItems: "center", gap: 8, background: "#EBF8F6", border: "1px solid #B2E8E2",
              color: "#0B7B6F", padding: "7px 16px", borderRadius: 100, fontSize: 12, fontWeight: 700, letterSpacing: .8,
              textTransform: "uppercase", marginBottom: 28
            }}>
              <LiveDot /> AI Clinical Intelligence · Live
            </div>

            <h1 className="hero-h1" style={{
              fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 60, fontWeight: 800,
              lineHeight: 1.04, color: "#0F1C2E", margin: "0 0 22px", letterSpacing: -2
            }}>
              Identify Rare<br />
              Diseases with<br />
              <span style={{
                background: "linear-gradient(110deg,#0B7B6F 0%,#1A6FA4 50%,#5B3DB8 100%)",
                backgroundSize: "200%", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
                backgroundClip: "text", animation: "gradShift 4s ease infinite"
              }}>
                Precision AI
              </span>
            </h1>

            <p className="hero-sub" style={{ fontSize: 17, color: "#5A7184", lineHeight: 1.8, maxWidth: 460, margin: "0 0 40px" }}>
              Submit patient symptoms and biomedical scans. Our multimodal AI cross-references 1,374 rare diseases and returns ranked, confidence-scored differential diagnoses instantly.
            </p>

            <div className="hero-btns" style={{ display: "flex", gap: 12, marginBottom: 52 }}>
              <button className="hero-cta-p" onClick={() => navigate("/predict")} style={{
                background: "#0B7B6F", color: "#fff", border: "none", padding: "16px 36px",
                borderRadius: 13, fontWeight: 800, fontSize: 15, cursor: "pointer",
                fontFamily: "'Inter',sans-serif", boxShadow: "0 8px 28px rgba(11,123,111,0.22)",
                transition: "all .2s", display: "flex", alignItems: "center", gap: 8,
              }}>
                <span>🔍</span> Start Diagnosis
              </button>
              <button className="hero-cta-s" onClick={() => navigate("/dashboard")} style={{
                background: "#fff", color: "#0F1C2E", border: "1.5px solid #DDE8EF",
                padding: "16px 36px", borderRadius: 13, fontWeight: 600, fontSize: 15,
                cursor: "pointer", fontFamily: "'Inter',sans-serif", transition: "all .2s",
              }}>
                View Dashboard
              </button>
            </div>

            <div className="trust-row" style={{ display: "flex", gap: 40, paddingTop: 40, borderTop: "1px solid #E0EBF2" }}>
              {[["36K+", "Patient Cases"], ["1,374", "Rare Diseases"], ["94K+", "Medical Images"]].map(([v, l]) => (
                <div key={l}>
                  <div style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 28, fontWeight: 800, color: "#0F1C2E", letterSpacing: -.5, animation: "countUp .7s ease" }}><Counter target={v} /></div>
                  <div style={{ fontSize: 12, color: "#8FA5B5", marginTop: 3, fontWeight: 500 }}>{l}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Right — dashboard preview */}
          <div className="hero-preview" style={{
            background: "#fff", border: "1px solid #E0EBF2", borderRadius: 26, padding: 32,
            boxShadow: "0 28px 72px rgba(15,28,46,0.09), 0 2px 8px rgba(15,28,46,0.04)",
            animation: "fadeUp .8s .15s ease both",
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 22 }}>
              <div>
                <p style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontWeight: 800, fontSize: 16, color: "#0F1C2E", margin: "0 0 2px" }}>System Status</p>
                <p style={{ fontSize: 12, color: "#8FA5B5", margin: 0 }}>Real-time overview</p>
              </div>
              <span style={{ display: "flex", alignItems: "center", gap: 6, background: "#EBF8F6", border: "1px solid #B2E8E2", color: "#0B7B6F", padding: "5px 12px", borderRadius: 100, fontSize: 11, fontWeight: 700, letterSpacing: .5, textTransform: "uppercase" }}>
                <LiveDot /> Online
              </span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 20 }}>
              {[["Predictions", "1,254", "#EBF8F6", "#0B7B6F"], ["Top-3 Accuracy", "52.9%", "#EBF4F9", "#1D6FA4"], ["Diseases", "49", "#F2EEF9", "#5B3DB8"], ["Images", "35K+", "#FFF4EC", "#C05B1A"]].map(([l, v, bg, c]) => (
                <div key={l} style={{ background: bg, borderRadius: 14, padding: "16px 18px" }}>
                  <p style={{ fontSize: 10, color: c, opacity: .7, textTransform: "uppercase", letterSpacing: .8, margin: "0 0 6px", fontWeight: 800 }}>{l}</p>
                  <p style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 26, fontWeight: 800, color: c, margin: 0 }}>{v}</p>
                </div>
              ))}
            </div>

            <div style={{ borderTop: "1px solid #EDF2F6", paddingTop: 18, marginBottom: 18 }}>
              <p style={{ fontSize: 10, fontWeight: 800, color: "#8FA5B5", textTransform: "uppercase", letterSpacing: .9, margin: "0 0 13px" }}>Model Performance</p>
              {[["Symptom Model", 52.9, "#0B7B6F"], ["Image Model", 35, "#1D6FA4", "Training"], ["Fusion Model", 0, "#9B8ED6", "Coming soon"]].map(([l, p, c, note]) => (
                <div key={l} style={{ marginBottom: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 5 }}>
                    <span style={{ color: "#4A6275", fontWeight: 500 }}>{l}</span>
                    <span style={{ color: c, fontWeight: 700 }}>{note || `${p}%`}</span>
                  </div>
                  <div style={{ height: 5, background: "#EDF2F6", borderRadius: 100 }}>
                    <div style={{ height: 5, width: `${p}%`, background: c, borderRadius: 100, animation: "barGrow 1s ease" }} />
                  </div>
                </div>
              ))}
            </div>

            <button onClick={() => navigate("/predict")} style={{
              width: "100%", background: "#0F1C2E", color: "#fff", border: "none",
              padding: "13px 20px", borderRadius: 12, fontWeight: 700, fontSize: 14,
              cursor: "pointer", fontFamily: "'Inter',sans-serif", transition: "background .2s",
            }}
              onMouseEnter={e => e.currentTarget.style.background = "#1E3A5F"}
              onMouseLeave={e => e.currentTarget.style.background = "#0F1C2E"}
            >Run a Diagnosis →</button>
          </div>
        </div>
      </div>

      {/* ══ STATS BAND ══ */}
      <div style={{ background: "#0F1C2E", padding: "40px 32px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          <div className="stats-row" style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 0 }}>
            {[
              { icon: "🔮", val: "1,254", label: "Total Predictions", sub: "All time" },
              { icon: "🎯", val: "52.9%", label: "Top-3 Accuracy", sub: "Symptom model" },
              { icon: "🧬", val: "49", label: "Diseases Covered", sub: "Tier A indexed" },
              { icon: "🩻", val: "35K+", label: "Training Images", sub: "Biomedical scans" },
            ].map(({ icon, val, label, sub }, i) => (
              <div key={label} style={{ padding: "0 32px", borderRight: i < 3 ? "1px solid rgba(255,255,255,0.08)" : "none" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                  <span style={{ fontSize: 22 }}>{icon}</span>
                  <p style={{
                    fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 32, fontWeight: 800,
                    color: "#F8FAFC", margin: 0, letterSpacing: -.5
                  }}>{val}</p>
                </div>
                <p style={{ fontSize: 13, color: "#7A94A8", margin: "0 0 3px", fontWeight: 600 }}>{label}</p>
                <p style={{ fontSize: 11, color: "#4A6275", margin: 0 }}>{sub}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ══ FEATURES ══ */}
      <div style={{ background: "#fff", borderTop: "1px solid #EDF2F6", borderBottom: "1px solid #EDF2F6" }}>
        <div className="section-pad" style={{ maxWidth: 1200, margin: "0 auto", padding: "80px 32px" }}>
          <div style={{ textAlign: "center", marginBottom: 52 }}>
            <Eyebrow>Capabilities</Eyebrow>
            <h2 className="section-title" style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 40, fontWeight: 800, color: "#0F1C2E", margin: "0 0 14px", letterSpacing: -.8 }}>Three Diagnostic Pathways</h2>
            <p style={{ fontSize: 16, color: "#7A94A8", maxWidth: 460, margin: "0 auto", lineHeight: 1.75 }}>Symptoms alone, images alone, or both combined for maximum precision.</p>
          </div>
          <div className="features-grid" style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 18 }}>
            <FeatureCard icon="🧬" tag="NLP" tagColor="#0B7B6F" tagBg="#EBF8F6" tagBorder="#B2E8E2" title="Symptom Analysis" desc="TF-IDF encoded symptom vectors fed into a multinomial classifier trained on 8,568 patient cases." delay={0} />
            <FeatureCard icon="🔬" tag="Vision" tagColor="#1D6FA4" tagBg="#EBF4F9" tagBorder="#B3D8EE" title="Image Detection" desc="EfficientNet-B4 fine-tuned on 35,374 biomedical images across MRI, CT, dermoscopy, and histopathology." delay={.08} />
            <FeatureCard icon="🤖" tag="Fusion" tagColor="#5B3DB8" tagBg="#F2EEF9" tagBorder="#C8B8EC" title="Multimodal AI" desc="Cross-attention fusion of symptom and imaging signals. Correct diagnosis in top-3 predictions 53% of the time." delay={.16} />
            <FeatureCard icon="📊" tag="Analytics" tagColor="#C05B1A" tagBg="#FFF4EC" tagBorder="#F5D8B8" title="Live Analytics" desc="Monitor prediction history, confidence distributions, and model performance metrics in real time." delay={.24} />
          </div>
        </div>
      </div>

      {/* ══ HOW IT WORKS ══ */}
      <div className="section-pad" style={{ maxWidth: 1200, margin: "0 auto", padding: "80px 32px" }}>
        <div style={{ textAlign: "center", marginBottom: 52 }}>
          <Eyebrow>Process</Eyebrow>
          <h2 className="section-title" style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 40, fontWeight: 800, color: "#0F1C2E", margin: "0 0 12px", letterSpacing: -.8 }}>Diagnosis in 3 Steps</h2>
          <p style={{ fontSize: 16, color: "#7A94A8", margin: 0 }}>Straightforward workflow from symptoms to ranked differential diagnosis.</p>
        </div>
        <div className="steps-grid" style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 0 }}>
          {[
            { n: "01", icon: "📝", title: "Enter Symptoms", desc: "Describe patient symptoms in natural language or use quick-add chips for common presentations.", color: "#0B7B6F" },
            { n: "02", icon: "🩻", title: "Upload Scan", desc: "Optionally add a biomedical image — MRI, CT, dermoscopy, or histopathology — for multimodal analysis.", color: "#1D6FA4" },
            { n: "03", icon: "🎯", title: "Get Diagnosis", desc: "Receive ranked differential diagnoses with probability scores, confidence levels, and model attribution.", color: "#5B3DB8" },
          ].map(({ n, icon, title, desc, color }, i) => (
            <div key={n} className="step-seg" style={{
              padding: "44px 36px", background: i === 1 ? "#EBF8F6" : "#fff",
              border: "1px solid #E8EFF5", borderLeft: i > 0 ? "none" : "1px solid #E8EFF5",
              borderRadius: i === 0 ? "20px 0 0 20px" : i === 2 ? "0 20px 20px 0" : 0,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
                <div style={{
                  width: 42, height: 42, borderRadius: "50%", background: "#fff", border: `2px solid ${color}44`,
                  display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 800, fontSize: 14, color
                }}>
                  {n}
                </div>
                <span style={{ fontSize: 28 }}>{icon}</span>
              </div>
              <h3 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 19, fontWeight: 700, color: "#0F1C2E", margin: "0 0 12px" }}>{title}</h3>
              <p style={{ fontSize: 14, color: "#7A94A8", lineHeight: 1.8, margin: 0 }}>{desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* ══ TESTIMONIALS ══ */}
      <div style={{ background: "#fff", borderTop: "1px solid #EDF2F6", borderBottom: "1px solid #EDF2F6" }}>
        <div className="section-pad" style={{ maxWidth: 1200, margin: "0 auto", padding: "80px 32px" }}>
          <div style={{ textAlign: "center", marginBottom: 52 }}>
            <Eyebrow>Testimonials</Eyebrow>
            <h2 className="section-title" style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 40, fontWeight: 800, color: "#0F1C2E", margin: 0, letterSpacing: -.8 }}>Trusted by Clinicians</h2>
          </div>
          <div className="test-grid" style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 18 }}>
            <TestimonialCard quote="The top-3 accuracy on rare disease predictions has measurably improved our differential diagnosis workflow." name="Dr. Priya Mehta" role="Neurologist, AIIMS Delhi" initials="PM" color="#0B7B6F" delay={0} />
            <TestimonialCard quote="Finally a tool combining symptom analysis and imaging in a single, clean, fast interface." name="Dr. Arjun Sharma" role="Radiologist, Apollo Hospitals" initials="AS" color="#1D6FA4" delay={.1} />
            <TestimonialCard quote="Rare disease identification used to take weeks of specialist referrals. This system narrows it down in seconds." name="Dr. Kavitha Nair" role="Geneticist, CMC Vellore" initials="KN" color="#5B3DB8" delay={.2} />
          </div>
        </div>
      </div>

      {/* ══ FAQ ══ */}
      <div className="section-pad" style={{ maxWidth: 1200, margin: "0 auto", padding: "80px 32px" }}>
        <div style={{ textAlign: "center", marginBottom: 52 }}>
          <Eyebrow>FAQ</Eyebrow>
          <h2 className="section-title" style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 40, fontWeight: 800, color: "#0F1C2E", margin: 0, letterSpacing: -.8 }}>Common Questions</h2>
        </div>
        <div className="faq-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, maxWidth: 960, margin: "0 auto" }}>
          {[
            { q: "How accurate is the symptom model?", a: "The symptom model achieves 52.9% top-3 accuracy across 49 rare diseases, trained on 8,568 patient cases using TF-IDF encoding and logistic regression." },
            { q: "What image types are supported?", a: "MRI brain scans, CT abdomen/thorax, dermoscopy images, retinal fundus photographs, and histopathology slides. The model is built on EfficientNet-B4." },
            { q: "Is this for clinical use?", a: "AI DOC is intended for research and educational purposes. All predictions should be reviewed by a licensed clinician before influencing treatment decisions." },
            { q: "How is patient data handled?", a: "No patient data is stored. Inputs are processed in-session and are not retained after the session ends. No personal identifiers should be submitted." },
            { q: "What is the fusion model?", a: "The fusion model will combine symptom and imaging signals via cross-attention. It is currently in development and expected to significantly improve accuracy." },
            { q: "How many diseases are covered?", a: "The current Tier A release covers 49 rare diseases with high-quality symptom and imaging data. Coverage will expand as additional datasets are validated." },
          ].map(({ q, a }) => <FaqItem key={q} q={q} a={a} />)}
        </div>
      </div>

      {/* ══ CTA BANNER ══ */}
      <div className="section-pad" style={{ maxWidth: 1200, margin: "0 auto", padding: "0 32px 80px" }}>
        <div style={{
          background: "linear-gradient(135deg,#0F1C2E 0%,#0B7B6F 60%,#1D6FA4 100%)",
          backgroundSize: "200%", borderRadius: 26, padding: "64px 48px", textAlign: "center",
          boxShadow: "0 28px 72px rgba(11,123,111,0.18)", animation: "gradShift 6s ease infinite"
        }}>
          <Eyebrow>Get Started</Eyebrow>
          <h2 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 38, fontWeight: 800, color: "#fff", margin: "0 0 14px", letterSpacing: -.5 }}>Ready to Run a Diagnosis?</h2>
          <p style={{ fontSize: 16, color: "rgba(255,255,255,0.7)", margin: "0 0 36px", lineHeight: 1.75, maxWidth: 480, marginLeft: "auto", marginRight: "auto" }}>Enter symptoms, upload a scan, and receive AI-powered differential diagnoses in under 3 seconds.</p>
          <div className="cta-banner-btns" style={{ display: "flex", gap: 12, justifyContent: "center" }}>
            <button onClick={() => navigate("/predict")} style={{ background: "#fff", color: "#0B7B6F", border: "none", padding: "15px 36px", borderRadius: 13, fontWeight: 800, fontSize: 15, cursor: "pointer", fontFamily: "'Inter',sans-serif", transition: "all .2s" }}
              onMouseEnter={e => e.currentTarget.style.background = "#EBF8F6"}
              onMouseLeave={e => e.currentTarget.style.background = "#fff"}>
              Start Diagnosis →
            </button>
            <button onClick={() => navigate("/dashboard")} style={{ background: "transparent", color: "#fff", border: "1.5px solid rgba(255,255,255,0.3)", padding: "15px 36px", borderRadius: 13, fontWeight: 600, fontSize: 15, cursor: "pointer", fontFamily: "'Inter',sans-serif", transition: "all .2s" }}
              onMouseEnter={e => e.currentTarget.style.background = "rgba(255,255,255,0.08)"}
              onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
              View Dashboard
            </button>
          </div>
        </div>
      </div>

      {/* footer handled by Footer.jsx */}
    </div>
  );
}

export default Home;