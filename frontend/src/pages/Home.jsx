import { Link } from "react-router-dom";
import { useEffect, useState } from "react";
import { useTheme } from "../context/ThemeContext";
import { useAuth } from "../context/AuthContext";
import { getAnalytics } from "../services/Api";

const CSS = (c) => `
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700;9..144,800&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

  @keyframes fadeUp   { from{opacity:0;transform:translateY(24px)} to{opacity:1;transform:translateY(0)} }
  @keyframes fadeIn   { from{opacity:0} to{opacity:1} }
  @keyframes pulse    { 0%,100%{opacity:.45;transform:scale(1)} 50%{opacity:.9;transform:scale(1.03)} }
  @keyframes drawLine { from{stroke-dashoffset:600} to{stroke-dashoffset:0} }

  .eyebrow {
    display:inline-flex; align-items:center; gap:10px;
    font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:600;
    color:${c.gold}; letter-spacing:0.16em; text-transform:uppercase;
  }
  .eyebrow::before { content:''; width:22px; height:1px; background:${c.gold}; display:inline-block; }

  .hero-cta-primary {
    display:inline-flex; align-items:center; gap:10px;
    padding:16px 30px; border-radius:4px; border:none;
    background:${c.text}; color:${c.bg};
    font-size:14.5px; font-weight:600; cursor:pointer;
    text-decoration:none; font-family:'Inter',sans-serif;
    transition:all .25s cubic-bezier(.2,.7,.3,1); letter-spacing:0.01em;
  }
  .hero-cta-primary:hover { background:${c.teal}; color:#fff; transform:translateY(-2px); box-shadow:${c.shadowTeal}; }

  .hero-cta-outline {
    display:inline-flex; align-items:center; gap:10px;
    padding:15px 26px; border-radius:4px;
    border:1px solid ${c.borderI}; background:transparent; color:${c.text};
    font-size:14px; font-weight:600; cursor:pointer;
    text-decoration:none; font-family:'Inter',sans-serif;
    transition:all .25s;
  }
  .hero-cta-outline:hover { border-color:${c.teal}; color:${c.teal}; }

  .stat-card {
    background:${c.card}; border:1px solid ${c.border}; border-top:2px solid transparent;
    padding:30px 26px; transition:transform .2s, box-shadow .2s, border-color .2s;
  }
  .stat-card:hover { transform:translateY(-3px); box-shadow:${c.shadowLg}; border-top-color:${c.gold}; }

  .feature-card {
    background:${c.card}; border:1px solid ${c.border};
    padding:34px 30px; transition:transform .2s, box-shadow .2s, border-color .2s;
    position:relative;
  }
  .feature-card:hover { transform:translateY(-4px); box-shadow:${c.shadowLg}; border-color:${c.tealB}; }

  .step-card { padding:30px 4px 0; position:relative; }
  .step-num {
    font-family:'Fraunces',serif; font-style:italic; font-weight:500;
    font-size:15px; color:${c.gold};
  }

  .blob { position:absolute; border-radius:50%; pointer-events:none; }

  .cta-link-light {
    display:inline-flex; align-items:center; gap:10px;
    padding:15px 30px; border-radius:4px; border:none;
    background:${c.bg}; color:${c.teal}; font-size:14.5px; font-weight:700;
    text-decoration:none; transition:all .25s; font-family:'Inter',sans-serif;
  }
  .cta-link-light:hover { transform:translateY(-2px); box-shadow:0 14px 30px rgba(0,0,0,0.25); }

  .cta-link-ghost {
    display:inline-flex; align-items:center; gap:10px;
    padding:14px 26px; border-radius:4px;
    border:1px solid rgba(255,255,255,0.35);
    background:transparent; color:#fff; font-size:14px; font-weight:600;
    text-decoration:none; transition:all .25s; font-family:'Inter',sans-serif;
  }
  .cta-link-ghost:hover { background:rgba(255,255,255,0.10); }

  @media(max-width:900px){
    .hero-grid   { grid-template-columns:1fr!important }
    .stats-grid  { grid-template-columns:1fr 1fr!important }
    .feat-grid   { grid-template-columns:1fr!important }
    .steps-grid  { grid-template-columns:1fr 1fr!important; gap:36px!important }
    .hero-visual { display:none!important }
  }
  @media(max-width:600px){
    .stats-grid  { grid-template-columns:1fr!important }
    .steps-grid  { grid-template-columns:1fr!important }
    .hero-title  { font-size:34px!important }
    .section-pad { padding:56px 18px!important }
    .hero-cta-row { flex-direction:column!important; align-items:stretch!important }
    .hero-cta-primary, .hero-cta-outline { width:100%!important; justify-content:center!important; padding:15px 20px!important }
    .mini-stats  { gap:22px!important }
    .cta-banner-pad { padding:44px 26px!important }
    .cta-link-row { flex-direction:column!important; align-items:stretch!important }
    .cta-link-light, .cta-link-ghost { width:100%!important; justify-content:center!important }
  }
  @media(max-width:420px){
    .hero-title    { font-size:29px!important }
    .stat-card     { padding:22px 18px!important }
    .feature-card  { padding:26px 20px!important }
    .step-card     { padding-top:14px!important }
  }
`;

const STATS = [
  { val: "36,487", label: "Patient Cases", sub: "ZebraMap dataset" },
  { val: "1,374", label: "Rare Diseases", sub: "ORPHA coded" },
  { val: "83.87%", label: "Top-5 Accuracy", sub: "Fusion model" },
  { val: "8,948", label: "Synthetic Images", sub: "FastGAN generated" },
];

const FEATURES = [
  {
    mark: "01", title: "Symptom Analysis",
    desc: "Enter free-text symptoms and our TF-IDF + Logistic Regression classifier maps them to 62 rare diseases with probability scores.",
    color: "teal", badge: "34.73% Top-1",
  },
  {
    mark: "02", title: "Image Recognition",
    desc: "Upload MRI, CT, fundus, dermoscopy, or histopathology scans. EfficientNet-B4 fine-tuned on 35,000+ biomedical images.",
    color: "blue", badge: "EfficientNet-B4",
  },
  {
    mark: "03", title: "Multimodal Fusion",
    desc: "Late-weighted fusion combines both signals at optimal 0.9/0.1 weights, boosting Top-1 accuracy to 58.39% — a +23.66% improvement.",
    color: "purple", badge: "58.39% Top-1",
  },
  {
    mark: "04", title: "GAN Augmentation",
    desc: "FastGAN generates synthetic rare disease images for ultra-rare classes, achieving 87.97% accuracy — beating the full-data upper bound.",
    color: "amber", badge: "87.97% Acc.",
  },
];

const STEPS = [
  { n: "I", title: "Describe Symptoms", desc: "Type patient symptoms in natural language. Quick-add chips help you select common presentations fast." },
  { n: "II", title: "Upload Medical Scan", desc: "Optionally upload any biomedical image — MRI, CT, dermoscopy, fundus or histopathology scan." },
  { n: "III", title: "AI Analyses Input", desc: "Models run in parallel: TF-IDF+LR for symptoms, EfficientNet-B4 for images, then fused at 0.9/0.1 weights." },
  { n: "IV", title: "Review Diagnoses", desc: "Ranked top-5 differential diagnoses appear with probability scores, confidence levels, and disease details." },
];

/** Thin "vital sign" line — the page's recurring signature mark */
function VitalLine({ color, width = 160, height = 28 }) {
  return (
    <svg width={width} height={height} viewBox="0 0 160 28" fill="none" style={{ display: "block" }}>
      <path
        d="M0 14H40L48 4L58 24L66 14H160"
        stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
        strokeDasharray="600" style={{ animation: "drawLine 1.4s ease forwards" }}
      />
    </svg>
  );
}

export default function Home() {
  const { c } = useTheme();
  const { user } = useAuth();
  const [analytics, setAnalytics] = useState(null);
  const [photoFailed, setPhotoFailed] = useState(false);

  useEffect(() => {
    getAnalytics().then(setAnalytics).catch(() => { });
  }, []);

  const colorMap = {
    teal: { text: c.teal, bg: c.tealL, border: c.tealB },
    blue: { text: c.blue, bg: c.blueL, border: c.blueB },
    purple: { text: c.purple, bg: c.purpL, border: c.purpB },
    amber: { text: c.amber, bg: c.ambL, border: c.ambB },
  };

  return (
    <div style={{ background: c.bg, fontFamily: "'Inter',sans-serif" }}>
      <style>{CSS(c)}</style>

      {/* ── HERO ──────────────────────────────────────────────────────────── */}
      <section className="section-pad" style={{
        padding: "96px 32px 100px", position: "relative", overflow: "hidden",
        background: c.gradHero,
        backgroundImage: `${c.gradHero}, radial-gradient(${c.borderI} 1px, transparent 1px)`,
        backgroundSize: "auto, 28px 28px",
      }}>
        <div className="blob" style={{ width: 480, height: 480, background: `radial-gradient(circle,${c.teal}12 0%,transparent 70%)`, top: -100, right: -100, animation: "pulse 9s ease-in-out infinite" }} />
        <div className="blob" style={{ width: 380, height: 380, background: `radial-gradient(circle,${c.gold}10 0%,transparent 70%)`, bottom: -80, left: -80, animation: "pulse 11s ease-in-out infinite 3s" }} />

        <div style={{ maxWidth: 1200, margin: "0 auto", position: "relative", zIndex: 2 }}>
          <div className="hero-grid" style={{ display: "grid", gridTemplateColumns: "1.05fr 0.95fr", gap: 80, alignItems: "center" }}>

            {/* Left */}
            <div style={{ animation: "fadeUp .6s cubic-bezier(.2,.7,.3,1) both" }}>
              <span className="eyebrow" style={{ marginBottom: 22, display: "inline-flex" }}>
                Multimodal Diagnostic Intelligence
              </span>

              <h1 className="hero-title" style={{
                fontFamily: "'Fraunces',serif",
                fontSize: 58, fontWeight: 600, color: c.text,
                margin: "20px 0 26px", lineHeight: 1.06, letterSpacing: "-0.02em",
              }}>
                Identify rare<br />
                <span style={{ fontStyle: "italic", color: c.teal, fontWeight: 500 }}>diseases</span> with
                <br />precision AI
              </h1>

              <p style={{
                fontSize: 16.5, color: c.sub, lineHeight: 1.75,
                margin: "0 0 36px", maxWidth: 480,
              }}>
                Multimodal AI combining symptom analysis and biomedical image recognition,
                trained on <strong style={{ color: c.text, fontWeight: 700 }}>36,487 real patient cases</strong> across
                1,374 rare diseases from the ZebraMap dataset.
              </p>

              <div className="hero-cta-row" style={{ display: "flex", gap: 14, flexWrap: "wrap", marginBottom: 44 }}>
                <Link to={user ? "/predict" : "/signup"} className="hero-cta-primary">
                  {user ? "Start Predicting" : "Get Started Free"}
                </Link>
                <Link to="/dashboard" className="hero-cta-outline">
                  View Analytics
                </Link>
              </div>

              <div style={{ height: 1, background: c.border, maxWidth: 480, marginBottom: 28 }} />

              {/* Mini stats — third stat shows real live usage once analytics loads,
                  otherwise falls back to a real static dataset fact (never a fabricated number) */}
              <div className="mini-stats" style={{ display: "flex", gap: 32, flexWrap: "wrap" }}>
                {[
                  { val: "58.39%", label: "Top-1 Fusion Acc." },
                  { val: "83.87%", label: "Top-5 Accuracy" },
                  analytics?.total_predictions
                    ? { val: Number(analytics.total_predictions).toLocaleString(), label: "Predictions Made" }
                    : { val: "1,374", label: "Diseases Modeled" },
                ].map(({ val, label }) => (
                  <div key={label}>
                    <div style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 21, fontWeight: 600, color: c.text, letterSpacing: "-0.01em" }}>{val}</div>
                    <div style={{ fontSize: 11.5, color: c.muted, fontWeight: 500, marginTop: 4 }}>{label}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Right — visual */}
            <div className="hero-visual" style={{ animation: "fadeUp .6s cubic-bezier(.2,.7,.3,1) .15s both" }}>
              <div style={{
                background: c.card, border: `1px solid ${c.border}`,
                borderTop: `2px solid ${c.teal}`,
                borderRadius: 6, padding: "36px 32px 34px", boxShadow: c.shadowXl,
                position: "relative", overflow: "hidden",
              }}>
                {/* Faint dot-grid backdrop behind the illustration */}
                <div style={{
                  position: "absolute", inset: 0,
                  backgroundImage: `radial-gradient(${c.borderI} 1px, transparent 1px)`,
                  backgroundSize: "22px 22px",
                  opacity: 0.6, pointerEvents: "none",
                }} />

                <div style={{ position: "relative", zIndex: 2 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                    <span className="eyebrow" style={{ fontSize: 10 }}>Clinical Intelligence</span>
                    <VitalLine color={c.teal} width={70} height={18} />
                  </div>

                  {/* Clinician photo — expects file at /public/assets/doctor.jpg
                      (falls back to the line illustration automatically if missing) */}
                  {!photoFailed ? (
                    <div style={{
                      position: "relative", width: "100%", aspectRatio: "9 / 10",
                      borderRadius: 4, overflow: "hidden", margin: "8px 0 4px",
                      background: c.bgDeep,
                    }}>
                      <img
                        src="/assets/doctor.jpg"
                        alt="Clinician reviewing an AI-assisted diagnostic report"
                        onError={() => setPhotoFailed(true)}
                        style={{
                          width: "100%", height: "100%", objectFit: "cover",
                          display: "block", filter: "grayscale(12%) contrast(1.05) saturate(1.04)",
                        }}
                      />
                      {/* brand-tint wash so the photo sits inside the teal/gold palette */}
                      <div style={{
                        position: "absolute", inset: 0, pointerEvents: "none",
                        background: `linear-gradient(180deg, transparent 55%, ${c.teal}30 100%)`,
                        mixBlendMode: "multiply",
                      }} />
                      <div style={{
                        position: "absolute", inset: 0, pointerEvents: "none",
                        boxShadow: `inset 0 0 0 1px ${c.borderI}`,
                      }} />
                    </div>
                  ) : (
                    <svg viewBox="0 0 360 400" width="100%" height="auto" style={{ display: "block", margin: "8px 0 4px" }}>
                      <defs>
                        <radialGradient id="docHalo" cx="50%" cy="42%" r="60%">
                          <stop offset="0%" stopColor={c.teal} stopOpacity="0.16" />
                          <stop offset="100%" stopColor={c.teal} stopOpacity="0" />
                        </radialGradient>
                        <linearGradient id="docFace" x1="0.15" y1="0" x2="0.9" y2="1">
                          <stop offset="0%" stopColor="#EAF6F3" />
                          <stop offset="60%" stopColor="#BFE0D8" />
                          <stop offset="100%" stopColor="#5FA79A" />
                        </linearGradient>
                        <linearGradient id="docCoat" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#FFFFFF" />
                          <stop offset="100%" stopColor="#D9E5E2" />
                        </linearGradient>
                        <linearGradient id="docHair" x1="0" y1="0" x2="1" y2="1">
                          <stop offset="0%" stopColor="#1D343D" />
                          <stop offset="100%" stopColor="#0E1A20" />
                        </linearGradient>
                      </defs>

                      {/* ambient halo */}
                      <ellipse cx="180" cy="170" rx="170" ry="170" fill="url(#docHalo)" />

                      {/* coat + shoulders */}
                      <path
                        d="M38 402 C38 314 82 254 136 236 L136 206 L224 206 L224 236 C278 254 322 314 322 402 Z"
                        fill="url(#docCoat)" stroke="#A9C2BC" strokeWidth="1.4" strokeLinejoin="round"
                      />
                      {/* v-neck collar */}
                      <path d="M136 206 L180 252 L224 206" fill="none" stroke="#9FB8B2" strokeWidth="1.6" strokeLinejoin="round" />
                      <path d="M158 210 L180 244 L202 210 Z" fill="#EDF3F1" />
                      {/* lapel fold lines */}
                      <path d="M152 212 L134 296" stroke="#B9CCC7" strokeWidth="1.2" strokeLinecap="round" />
                      <path d="M208 212 L226 296" stroke="#B9CCC7" strokeWidth="1.2" strokeLinecap="round" />
                      {/* breast pocket + pen */}
                      <rect x="150" y="266" width="27" height="21" rx="2" fill="none" stroke="#B9CCC7" strokeWidth="1.3" />
                      <line x1="159" y1="266" x2="157" y2="246" stroke={c.gold} strokeWidth="2.2" strokeLinecap="round" />
                      {/* front buttons */}
                      <circle cx="180" cy="300" r="2.6" fill="#B9CCC7" />
                      <circle cx="180" cy="332" r="2.6" fill="#B9CCC7" />
                      <circle cx="180" cy="364" r="2.6" fill="#B9CCC7" />

                      {/* neck */}
                      <path d="M160 188 L160 222 Q180 232 200 222 L200 188 Z" fill="url(#docFace)" />

                      {/* ears */}
                      <ellipse cx="126" cy="150" rx="7.5" ry="13" fill="url(#docFace)" stroke="#5FA79A" strokeWidth="0.8" />
                      <ellipse cx="234" cy="150" rx="7.5" ry="13" fill="url(#docFace)" stroke="#5FA79A" strokeWidth="0.8" />

                      {/* face */}
                      <path
                        d="M180 76 C210 76 230 102 230 138 C230 168 220 192 202 206 C194 212 187 214 180 214 C173 214 166 212 158 206 C140 192 130 168 130 138 C130 102 150 76 180 76 Z"
                        fill="url(#docFace)" stroke="#5FA79A" strokeWidth="1"
                      />
                      {/* subtle cheek/jaw shading */}
                      <path d="M150 172 Q160 196 180 206" fill="none" stroke="#3F8C7E" strokeWidth="1" opacity="0.35" strokeLinecap="round" />

                      {/* hair — short, professional, side part */}
                      <path
                        d="M129 136 C127 96 150 66 180 66 C210 66 233 96 231 136 C231 118 224 104 214 98 C206 94 200 100 192 96 C185 93 182 98 175 95 C167 92 160 97 152 96 C140 98 129 112 129 136 Z"
                        fill="url(#docHair)" stroke={c.teal} strokeWidth="0.8" strokeLinejoin="round"
                      />
                      {/* sideburns */}
                      <path d="M132 118 C129 128 128 138 129 148" fill="none" stroke="#0E1A20" strokeWidth="6" strokeLinecap="round" />
                      <path d="M228 118 C231 128 232 138 231 148" fill="none" stroke="#0E1A20" strokeWidth="6" strokeLinecap="round" />

                      {/* eyebrows */}
                      <path d="M152 132 Q162 125 173 130" fill="none" stroke="#1D343D" strokeWidth="2.6" strokeLinecap="round" />
                      <path d="M187 130 Q198 125 208 132" fill="none" stroke="#1D343D" strokeWidth="2.6" strokeLinecap="round" />

                      {/* eyes */}
                      <path d="M155 145 Q163 141 171 145" fill="none" stroke="#1D343D" strokeWidth="2.2" strokeLinecap="round" />
                      <path d="M189 145 Q197 141 205 145" fill="none" stroke="#1D343D" strokeWidth="2.2" strokeLinecap="round" />

                      {/* nose shadow */}
                      <path d="M180 150 L175 174 Q180 179 185 174" fill="none" stroke="#3F8C7E" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" opacity="0.55" />

                      {/* calm confident mouth */}
                      <path d="M162 187 Q180 196 198 187" fill="none" stroke="#1D343D" strokeWidth="2.4" strokeLinecap="round" />

                      {/* stethoscope draped over shoulders */}
                      <path
                        d="M146 214 C146 246 156 262 180 262 C204 262 214 246 214 214"
                        fill="none" stroke={c.gold} strokeWidth="3" strokeLinecap="round"
                      />
                      <circle cx="146" cy="212" r="4.5" fill={c.gold} />
                      <circle cx="214" cy="212" r="4.5" fill={c.gold} />
                      <path d="M180 262 L180 284" stroke={c.gold} strokeWidth="3" strokeLinecap="round" />
                      <circle cx="180" cy="294" r="10" fill="none" stroke={c.gold} strokeWidth="3" />
                      <circle cx="180" cy="294" r="4" fill={c.gold} opacity="0.4" />
                    </svg>
                  )}

                  <div style={{ textAlign: "center", marginTop: 4 }}>
                    <p style={{ fontFamily: "'Fraunces',serif", fontSize: 17, fontWeight: 600, color: c.text, margin: "0 0 6px", letterSpacing: "-0.01em" }}>
                      Built alongside clinical judgment
                    </p>
                    <p style={{ fontSize: 13, color: c.sub, margin: "0 auto", maxWidth: 300, lineHeight: 1.65 }}>
                      AI DOC ranks differential diagnoses for review — it supports the clinician, it doesn't replace them.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

      </section>

      {/* ── STATS ─────────────────────────────────────────────────────────── */}
      <section className="section-pad" style={{ padding: "88px 32px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          <div style={{ textAlign: "center", marginBottom: 56 }}>
            <span className="eyebrow" style={{ justifyContent: "center" }}>By The Numbers</span>
            <h2 style={{ fontFamily: "'Fraunces',serif", fontSize: 36, fontWeight: 600, color: c.text, margin: "18px 0 0", letterSpacing: "-0.02em" }}>
              Built on real patient data
            </h2>
          </div>
          <div className="stats-grid" style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 1, background: c.border, border: `1px solid ${c.border}` }}>
            {STATS.map(({ val, label, sub }) => (
              <div key={label} className="stat-card">
                <div style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 32, fontWeight: 600, color: c.text, letterSpacing: "-0.02em", lineHeight: 1, marginBottom: 10 }}>{val}</div>
                <div style={{ fontSize: 13.5, fontWeight: 700, color: c.text, marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: 12, color: c.muted }}>{sub}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── FEATURES ──────────────────────────────────────────────────────── */}
      <section className="section-pad" style={{ padding: "88px 32px", background: c.bgAlt }}>
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          <div style={{ textAlign: "center", marginBottom: 56 }}>
            <span className="eyebrow" style={{ justifyContent: "center" }}>Core Technology</span>
            <h2 style={{ fontFamily: "'Fraunces',serif", fontSize: 36, fontWeight: 600, color: c.text, margin: "18px 0 14px", letterSpacing: "-0.02em" }}>
              Three models, one diagnosis
            </h2>
            <p style={{ fontSize: 15.5, color: c.sub, maxWidth: 520, margin: "0 auto", lineHeight: 1.7 }}>
              Symptom NLP, biomedical image CNN, and GAN augmentation — combined via late-weighted fusion.
            </p>
          </div>

          <div className="feat-grid" style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 1, background: c.border, border: `1px solid ${c.border}` }}>
            {FEATURES.map(({ mark, title, desc, color, badge }) => {
              const col = colorMap[color];
              return (
                <div key={title} className="feature-card">
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 22 }}>
                    <span style={{ fontFamily: "'Fraunces',serif", fontStyle: "italic", fontSize: 22, fontWeight: 600, color: col.text }}>{mark}</span>
                    <span style={{ fontSize: 10.5, fontWeight: 700, color: col.text, background: col.bg, border: `1px solid ${col.border}`, padding: "4px 12px", borderRadius: 100, letterSpacing: "0.05em", textTransform: "uppercase" }}>{badge}</span>
                  </div>
                  <h3 style={{ fontFamily: "'Fraunces',serif", fontSize: 19, fontWeight: 600, color: c.text, margin: "0 0 12px", letterSpacing: "-0.01em" }}>{title}</h3>
                  <p style={{ fontSize: 14, color: c.sub, lineHeight: 1.75, margin: 0 }}>{desc}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* ── HOW IT WORKS ──────────────────────────────────────────────────── */}
      <section className="section-pad" style={{ padding: "88px 32px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          <div style={{ textAlign: "center", marginBottom: 56 }}>
            <span className="eyebrow" style={{ justifyContent: "center" }}>Process</span>
            <h2 style={{ fontFamily: "'Fraunces',serif", fontSize: 36, fontWeight: 600, color: c.text, margin: "18px 0 0", letterSpacing: "-0.02em" }}>
              Diagnosis in four steps
            </h2>
          </div>

          <div className="steps-grid" style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 20 }}>
            {STEPS.map(({ n, title, desc }, i) => (
              <div key={n} className="step-card" style={{ borderTop: `2px solid ${i === 0 ? c.teal : c.border}` }}>
                <div className="step-num" style={{ marginTop: 18, marginBottom: 14 }}>{n}</div>
                <h3 style={{ fontFamily: "'Fraunces',serif", fontSize: 16.5, fontWeight: 600, color: c.text, margin: "0 0 10px" }}>{title}</h3>
                <p style={{ fontSize: 13.5, color: c.sub, lineHeight: 1.7, margin: 0 }}>{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA BANNER ────────────────────────────────────────────────────── */}
      <section className="section-pad" style={{ padding: "88px 32px" }}>
        <div style={{ maxWidth: 900, margin: "0 auto" }}>
          <div className="cta-banner-pad" style={{
            background: c.text, borderRadius: 8, padding: "64px 52px",
            textAlign: "center", position: "relative", overflow: "hidden",
          }}>
            <div style={{ position: "absolute", inset: 0, background: `radial-gradient(circle at 25% 30%, ${c.teal}30, transparent 60%)` }} />
            <div style={{ position: "relative", zIndex: 2 }}>
              <div style={{ display: "flex", justifyContent: "center", marginBottom: 20 }}>
                <VitalLine color={c.gold} width={140} height={24} />
              </div>
              <h2 style={{ fontFamily: "'Fraunces',serif", fontSize: 36, fontWeight: 600, color: "#fff", margin: "0 0 16px", letterSpacing: "-0.02em", lineHeight: 1.15 }}>
                Ready to try AI DOC?
              </h2>
              <p style={{ fontSize: 15.5, color: "rgba(255,255,255,0.68)", margin: "0 0 36px", lineHeight: 1.7, maxWidth: 460, marginLeft: "auto", marginRight: "auto" }}>
                Enter symptoms, upload a scan, and get ranked rare disease predictions in seconds. No special hardware required.
              </p>
              <div className="cta-link-row" style={{ display: "flex", gap: 14, justifyContent: "center", flexWrap: "wrap" }}>
                <Link to={user ? "/predict" : "/signup"} className="cta-link-light">
                  {user ? "Start Predicting →" : "Get Started Free →"}
                </Link>
                <Link to="/dashboard" className="cta-link-ghost">
                  View Model Results
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}