import { useEffect, useState } from "react";
import { getAnalytics } from "../services/Api";
import { useTheme } from "../context/ThemeContext";
import { useToast } from "../context/ToastContext";

const CSS = (c) => `
  @keyframes fadeUp   { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
  @keyframes spin     { to{transform:rotate(360deg)} }
  @keyframes shimmer  { 0%{opacity:.5} 50%{opacity:1} 100%{opacity:.5} }
  @keyframes drawLine { from{stroke-dashoffset:600} to{stroke-dashoffset:0} }

  .eyebrow {
    display:inline-flex; align-items:center; gap:10px;
    font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:600;
    color:${c.gold}; letter-spacing:0.14em; text-transform:uppercase;
  }
  .eyebrow::before { content:''; width:20px; height:1px; background:${c.gold}; display:inline-block; }

  .kpi-card { background:${c.card}; border:1px solid ${c.border}; padding:24px 24px; transition:transform .2s, box-shadow .2s, border-color .2s; }
  .kpi-card:hover { transform:translateY(-3px); box-shadow:${c.shadowLg}; }

  .model-row { border:1px solid ${c.border}; transition:border-color .2s, box-shadow .2s; }
  .model-row:hover { border-color:${c.tealB}; box-shadow:${c.shadowMd}; }

  .disease-row:hover { background:${c.bgAlt}!important; }

  .skeleton { background:linear-gradient(90deg, ${c.border} 25%, ${c.borderI} 50%, ${c.border} 75%); background-size:200% 100%; animation:shimmer 1.4s ease-in-out infinite; }

  .refresh-btn {
    padding:9px 16px; border-radius:4px; border:1px solid ${c.borderI};
    background:${c.card}; color:${c.sub}; font-size:13px; font-weight:600;
    cursor:pointer; font-family:'Inter',sans-serif; transition:all .2s;
    display:flex; align-items:center; gap:8px;
  }
  .refresh-btn:hover:not(:disabled) { border-color:${c.teal}; color:${c.teal}; background:${c.tealL}; }
  .refresh-btn:disabled { opacity:.6; cursor:not-allowed; }

  @media(max-width:980px){
    .kpi-grid    { grid-template-columns:1fr 1fr!important }
    .split-grid  { grid-template-columns:1fr!important }
    .model-grid  { grid-template-columns:1fr 1fr!important }
    .dash-h1     { font-size:30px!important }
  }
  @media(max-width:560px){
    .kpi-grid    { grid-template-columns:1fr!important }
    .model-grid  { grid-template-columns:1fr!important }
    .dash-pad    { padding:40px 20px!important }
  }
  @media(max-width:420px){
    .dash-pad  { padding:32px 16px!important }
    .dash-h1   { font-size:25px!important }
    .kpi-card  { padding:20px 18px!important }
    .refresh-btn { padding:8px 12px!important; font-size:12px!important }
  }
`;

/** Fixed benchmark results from model evaluation — not user-usage data */
const MODEL_BENCHMARKS = [
  { name: "TF-IDF + Logistic Regression", metric: "Top-1 Accuracy", value: 34.73, color: "teal", note: "Symptoms only · 62 Tier-A diseases" },
  { name: "EfficientNet-B4", metric: "Image Classification", value: 61.2, color: "blue", note: "Fine-tuned on 35K+ biomedical images" },
  { name: "Late-Weighted Fusion", metric: "Top-1 Accuracy", value: 58.39, color: "purple", note: "0.9 / 0.1 symptom-image weighting" },
  { name: "Late-Weighted Fusion", metric: "Top-5 Accuracy", value: 83.87, color: "purple", note: "Ranked differential diagnosis" },
  { name: "FastGAN Augmentation", metric: "Accuracy (Ultra-rare)", value: 87.97, color: "amber", note: "Synthetic data beats full-data baseline" },
];

function VitalLine({ color, width = 90, height = 20 }) {
  return (
    <svg width={width} height={height} viewBox="0 0 160 28" fill="none">
      <path d="M0 14H40L48 4L58 24L66 14H160" stroke={color} strokeWidth="1.5"
        strokeLinecap="round" strokeLinejoin="round" strokeDasharray="600"
        style={{ animation: "drawLine 1.4s ease forwards" }} />
    </svg>
  );
}

const IconAlert = ({ color }) => (
  <svg width="16" height="16" viewBox="0 0 20 20" fill="none" style={{ flexShrink: 0 }}>
    <path d="M10 2.5L18.5 17H1.5L10 2.5z" stroke={color} strokeWidth="1.6" strokeLinejoin="round" />
    <path d="M10 8v4M10 14.5v.1" stroke={color} strokeWidth="1.8" strokeLinecap="round" />
  </svg>
);

function Skeleton({ w = "100%", h = 14 }) {
  return <div className="skeleton" style={{ width: w, height: h, borderRadius: 3 }} />;
}

export default function Dashboard() {
  const { c } = useTheme();
  const toast = useToast();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(false);

  const load = () => {
    setLoading(true); setErr(false);
    getAnalytics()
      .then(setData)
      .catch(e => {
        console.error(e);
        setErr(true);
        toast.error("Could not load analytics data.");
      })
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const colorMap = {
    teal: { text: c.teal, bg: c.tealL, border: c.tealB },
    blue: { text: c.blue, bg: c.blueL, border: c.blueB },
    purple: { text: c.purple, bg: c.purpL, border: c.purpB },
    amber: { text: c.amber, bg: c.ambL, border: c.ambB },
  };

  // Defensive field access — analytics API shape may evolve
  const totalPredictions = data?.total_predictions ?? data?.total ?? 0;
  const avgProbability = data?.avg_probability ?? data?.average_confidence ?? null;
  const uniqueDiseases = data?.unique_diseases ?? data?.disease_count ?? 0;
  const confBreakdown = data?.confidence_breakdown ?? data?.confidence_distribution ?? null;
  const modeBreakdown = data?.mode_breakdown ?? data?.mode_distribution ?? null;
  const topDiseases = data?.top_diseases ?? data?.most_common_diseases ?? [];

  const confHigh = confBreakdown?.High ?? confBreakdown?.high ?? 0;
  const confMed = confBreakdown?.Medium ?? confBreakdown?.medium ?? 0;
  const confLow = confBreakdown?.Low ?? confBreakdown?.low ?? 0;
  const confTotal = confHigh + confMed + confLow || 1;

  const symptomOnly = modeBreakdown?.symptoms_only ?? modeBreakdown?.symptom_only ?? 0;
  const fusion = modeBreakdown?.multimodal_fusion ?? modeBreakdown?.fusion ?? 0;
  const modeTotal = symptomOnly + fusion || 1;

  const hasUsageData = !loading && !err && totalPredictions > 0;

  return (
    <div style={{ minHeight: "100vh", background: c.bg, fontFamily: "'Inter',sans-serif" }}>
      <style>{CSS(c)}</style>
      <div className="dash-pad" style={{ maxWidth: 1200, margin: "0 auto", padding: "56px 32px 80px" }}>

        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: 16, marginBottom: 40 }}>
          <div>
            <span className="eyebrow" style={{ marginBottom: 16, display: "inline-flex" }}>System Analytics</span>
            <h1 className="dash-h1" style={{ fontFamily: "'Fraunces',serif", fontSize: 36, fontWeight: 600, margin: "16px 0 10px", color: c.text, letterSpacing: "-0.02em" }}>
              Model &amp; Usage Dashboard
            </h1>
            <p style={{ color: c.sub, fontSize: 15, margin: 0, maxWidth: 560, lineHeight: 1.65 }}>
              Live prediction statistics alongside fixed benchmark results from model evaluation on the ZebraMap test set.
            </p>
          </div>
          <button className="refresh-btn" onClick={load} disabled={loading}>
            {loading ? (
              <span style={{ width: 13, height: 13, border: `2px solid ${c.border}`, borderTop: `2px solid ${c.teal}`, borderRadius: "50%", animation: "spin .7s linear infinite", display: "inline-block" }} />
            ) : "↻"}
            Refresh
          </button>
        </div>

        {/* ── KPI ROW (live usage data) ─────────────────────────────── */}
        <div className="kpi-grid" style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 1, background: c.border, border: `1px solid ${c.border}`, marginBottom: 48 }}>
          {[
            { label: "Total Predictions", value: totalPredictions, color: "teal" },
            { label: "Avg. Top Probability", value: avgProbability != null ? `${Math.round(avgProbability)}%` : "—", color: "blue" },
            { label: "Unique Diseases Seen", value: uniqueDiseases, color: "purple" },
            { label: "High Confidence Rate", value: confBreakdown ? `${Math.round((confHigh / confTotal) * 100)}%` : "—", color: "amber" },
          ].map(({ label, value, color }) => (
            <div key={label} className="kpi-card">
              <p style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 10, color: c.muted, textTransform: "uppercase", letterSpacing: "0.1em", margin: "0 0 12px", fontWeight: 700 }}>{label}</p>
              {loading ? <Skeleton w="60%" h={30} /> : (
                <p style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 30, fontWeight: 600, color: colorMap[color].text, margin: 0, letterSpacing: "-0.01em" }}>{value}</p>
              )}
            </div>
          ))}
        </div>

        {err && (
          <div style={{ background: c.redL, border: `1px solid ${c.redB}`, padding: "16px 20px", marginBottom: 40, display: "flex", alignItems: "center", gap: 12 }}>
            <IconAlert color={c.red} />
            <p style={{ fontSize: 13.5, color: c.red, margin: 0, fontWeight: 500 }}>
              Couldn't reach the analytics endpoint. Showing model benchmark data only.
            </p>
          </div>
        )}

        {!loading && !err && !hasUsageData && (
          <div style={{ background: c.card, border: `1px solid ${c.border}`, padding: "20px 24px", marginBottom: 40 }}>
            <p style={{ fontSize: 13.5, color: c.sub, margin: 0, lineHeight: 1.6 }}>
              No predictions have been logged yet — usage statistics will appear here once diagnoses start coming in.
            </p>
          </div>
        )}

        {/* ── SPLIT: confidence distribution + mode breakdown ─────────── */}
        <div className="split-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginBottom: 48 }}>

          {/* Confidence distribution */}
          <div style={{ background: c.card, border: `1px solid ${c.border}`, borderTop: `2px solid ${c.teal}`, padding: 28 }}>
            <h3 style={{ fontFamily: "'Fraunces',serif", fontSize: 17, fontWeight: 600, color: c.text, margin: "0 0 20px" }}>Confidence Distribution</h3>
            {loading ? (
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                <Skeleton h={10} /><Skeleton h={10} /><Skeleton h={10} />
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                {[
                  { label: "High", val: confHigh, color: c.teal, bg: c.tealL },
                  { label: "Medium", val: confMed, color: c.amber, bg: c.ambL },
                  { label: "Low", val: confLow, color: c.red, bg: c.redL },
                ].map(({ label, val, color }) => {
                  const pct = Math.round((val / confTotal) * 100);
                  return (
                    <div key={label}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                        <span style={{ fontSize: 13, fontWeight: 600, color: c.text }}>{label}</span>
                        <span style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 13, color, fontWeight: 600 }}>{confBreakdown ? `${pct}%` : "—"}</span>
                      </div>
                      <div style={{ height: 6, background: c.border, borderRadius: 100, overflow: "hidden" }}>
                        <div style={{ height: 6, width: confBreakdown ? `${pct}%` : "0%", background: color, borderRadius: 100, transition: "width .8s ease" }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Mode breakdown */}
          <div style={{ background: c.card, border: `1px solid ${c.border}`, borderTop: `2px solid ${c.blue}`, padding: 28 }}>
            <h3 style={{ fontFamily: "'Fraunces',serif", fontSize: 17, fontWeight: 600, color: c.text, margin: "0 0 20px" }}>Prediction Mode Split</h3>
            {loading ? (
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                <Skeleton h={10} /><Skeleton h={10} />
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                {[
                  { label: "Symptom-Only", val: symptomOnly, color: c.slate },
                  { label: "Multimodal Fusion", val: fusion, color: c.purple },
                ].map(({ label, val, color }) => {
                  const pct = Math.round((val / modeTotal) * 100);
                  return (
                    <div key={label}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                        <span style={{ fontSize: 13, fontWeight: 600, color: c.text }}>{label}</span>
                        <span style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 13, color, fontWeight: 600 }}>{modeBreakdown ? `${pct}% · ${val}` : "—"}</span>
                      </div>
                      <div style={{ height: 6, background: c.border, borderRadius: 100, overflow: "hidden" }}>
                        <div style={{ height: 6, width: modeBreakdown ? `${pct}%` : "0%", background: color, borderRadius: 100, transition: "width .8s ease" }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* ── TOP DISEASES ─────────────────────────────────────────────── */}
        <div style={{ marginBottom: 48 }}>
          <span className="eyebrow" style={{ marginBottom: 18, display: "inline-flex" }}>Most Frequently Identified</span>
          <div style={{ background: c.card, border: `1px solid ${c.border}`, marginTop: 18 }}>
            {loading ? (
              <div style={{ padding: 24, display: "flex", flexDirection: "column", gap: 16 }}>
                {[1, 2, 3, 4].map(i => <Skeleton key={i} h={16} />)}
              </div>
            ) : topDiseases.length ? (
              topDiseases.slice(0, 8).map((d, i) => {
                const name = d.disease || d.name || "—";
                const count = d.count ?? d.total ?? 0;
                const maxCount = Math.max(...topDiseases.map(x => x.count ?? x.total ?? 0), 1);
                const pct = Math.round((count / maxCount) * 100);
                return (
                  <div key={i} className="disease-row" style={{
                    display: "flex", alignItems: "center", gap: 16, padding: "16px 24px",
                    borderBottom: i < Math.min(topDiseases.length, 8) - 1 ? `1px solid ${c.border}` : "none",
                    transition: "background .12s",
                  }}>
                    <span style={{ fontFamily: "'Fraunces',serif", fontStyle: "italic", fontWeight: 600, fontSize: 16, color: c.gold, width: 24, flexShrink: 0 }}>{i + 1}</span>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <p style={{ fontSize: 13.5, fontWeight: 700, color: c.text, margin: "0 0 6px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{name}</p>
                      <div style={{ height: 4, background: c.border, borderRadius: 100, maxWidth: 320 }}>
                        <div style={{ height: 4, width: `${pct}%`, background: c.teal, borderRadius: 100 }} />
                      </div>
                    </div>
                    <span style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 13, color: c.muted, fontWeight: 600, flexShrink: 0 }}>{count}×</span>
                  </div>
                );
              })
            ) : (
              <p style={{ padding: "28px 24px", fontSize: 13.5, color: c.muted, margin: 0, fontStyle: "italic" }}>No disease frequency data available yet.</p>
            )}
          </div>
        </div>

        {/* ── MODEL BENCHMARKS (fixed evaluation results) ─────────────── */}
        <div>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 18, flexWrap: "wrap", gap: 12 }}>
            <span className="eyebrow">Model Benchmarks · ZebraMap Test Set</span>
            <VitalLine color={c.teal} width={70} height={16} />
          </div>

          <div className="model-grid" style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 14 }}>
            {MODEL_BENCHMARKS.map(({ name, metric, value, color, note }, i) => {
              const col = colorMap[color];
              return (
                <div key={i} className="model-row" style={{ padding: "22px 22px", borderTop: `2px solid ${col.text}` }}>
                  <p style={{ fontSize: 10.5, fontWeight: 700, color: c.muted, textTransform: "uppercase", letterSpacing: "0.08em", margin: "0 0 10px" }}>{metric}</p>
                  <p style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 30, fontWeight: 600, color: col.text, margin: "0 0 12px", letterSpacing: "-0.01em" }}>{value}%</p>
                  <p style={{ fontFamily: "'Fraunces',serif", fontSize: 14.5, fontWeight: 600, color: c.text, margin: "0 0 8px" }}>{name}</p>
                  <p style={{ fontSize: 12, color: c.muted, margin: 0, lineHeight: 1.6 }}>{note}</p>
                </div>
              );
            })}
          </div>
        </div>

        {/* Disclaimer */}
        <div style={{ marginTop: 40, padding: "14px 18px", background: c.ambL, border: `1px solid ${c.ambB}`, display: "flex", gap: 10, alignItems: "flex-start" }}>
          <div style={{ marginTop: 2 }}><IconAlert color={c.amber} /></div>
          <p style={{ fontSize: 12, color: c.amber, margin: 0, lineHeight: 1.6 }}>
            <strong>Research use only.</strong> Benchmark figures reflect offline evaluation on the ZebraMap test split and may not generalize to all populations.
          </p>
        </div>
      </div>
    </div>
  );
}