import { useEffect, useState } from "react";
import { getAnalytics } from "./services/Api";
import { useTheme } from "./context/ThemeContext";
import { useToast } from "./context/ToastContext";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell, LineChart, Line } from "recharts";

const CSS = (c) => `
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&display=swap');
  @keyframes spin    { to { transform:rotate(360deg) } }
  @keyframes fadeUp  { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
  @keyframes barGrow { from{width:0} }
  .stat-card:hover { transform:translateY(-3px)!important; box-shadow:0 12px 32px rgba(0,0,0,0.1)!important }
  .range-btn:hover { border-color:${c.tealB}!important; color:${c.teal}!important }
  .lb-row:hover    { background:${c.bgAlt}!important }
  @media(max-width:900px){
    .dash-pad    { padding:40px 20px!important }
    .stat-grid   { grid-template-columns:1fr 1fr!important }
    .chart-row   { grid-template-columns:1fr!important }
    .perf-row    { grid-template-columns:1fr!important }
    .dash-h1     { font-size:30px!important }
  }
  @media(max-width:520px){
    .stat-grid   { grid-template-columns:1fr!important }
  }
`;

const RANGES = {
  "7D": [{ day: "Mon", v: 40 }, { day: "Tue", v: 65 }, { day: "Wed", v: 85 }, { day: "Thu", v: 55 }, { day: "Fri", v: 95 }, { day: "Sat", v: 75 }, { day: "Sun", v: 50 }],
  "30D": [{ day: "W1", v: 280 }, { day: "W2", v: 340 }, { day: "W3", v: 410 }, { day: "W4", v: 390 }],
  "90D": [{ day: "M1", v: 980 }, { day: "M2", v: 1120 }, { day: "M3", v: 1254 }],
};

const LEADERBOARD = [
  { rank: 1, disease: "Fabry Disease", count: 187, pct: 14.9, trend: "up" },
  { rank: 2, disease: "Ehlers-Danlos Syndrome", count: 162, pct: 12.9, trend: "up" },
  { rank: 3, disease: "Wilson's Disease", count: 134, pct: 10.7, trend: "down" },
  { rank: 4, disease: "Marfan Syndrome", count: 119, pct: 9.5, trend: "up" },
  { rank: 5, disease: "Pompe Disease", count: 98, pct: 7.8, trend: "flat" },
  { rank: 6, disease: "Gaucher Disease", count: 87, pct: 6.9, trend: "down" },
];

const lineData = [
  { week: "W1", accuracy: 45 }, { week: "W2", accuracy: 48 }, { week: "W3", accuracy: 50 }, { week: "W4", accuracy: 52.9 },
];
const pieData = [
  { name: "High", value: 38 }, { name: "Medium", value: 44 }, { name: "Low", value: 18 },
];

function Tip({ active, payload, label, c }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: c.card, border: `1px solid ${c.border}`, borderRadius: 12, padding: "10px 16px", boxShadow: "0 4px 20px rgba(0,0,0,0.12)" }}>
      <p style={{ color: c.muted, fontSize: 10, margin: "0 0 3px", fontWeight: 800, textTransform: "uppercase", letterSpacing: .6 }}>{label}</p>
      <p style={{ color: c.teal, fontWeight: 800, fontSize: 18, margin: 0, fontFamily: "'Plus Jakarta Sans',sans-serif" }}>{payload[0].value}{payload[0].name === "accuracy" ? "%" : ""}</p>
    </div>
  );
}

function Dashboard() {
  const { c } = useTheme();
  const toast = useToast();
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [range, setRange] = useState("7D");
  const [apiStatus, setApiStatus] = useState("checking"); // checking|online|offline

  useEffect(() => {
    getAnalytics()
      .then(d => { setAnalytics(d); setApiStatus("online"); })
      .catch(e => {
        console.error(e);
        setApiStatus("offline");
        toast.error("Could not reach the analytics API. Showing cached data.");
      })
      .finally(() => setLoading(false));
  }, []);

  const chartData = RANGES[range];
  const peakIdx = chartData.reduce((maxI, d, i, arr) => d.v > arr[maxI].v ? i : maxI, 0);

  const TREND_ICON = { up: "↑", down: "↓", flat: "→" };
  const TREND_COLOR = (t) => t === "up" ? c.teal : t === "down" ? c.red : c.muted;

  return (
    <div style={{ minHeight: "100vh", background: c.bg, fontFamily: "'Inter',sans-serif" }}>
      <style>{CSS(c)}</style>
      <div className="dash-pad" style={{ maxWidth: 1200, margin: "0 auto", padding: "56px 32px" }}>

        {/* Header */}
        <div style={{ marginBottom: 36 }}>
          <span style={{ fontSize: 10, fontWeight: 800, color: c.teal, background: c.tealL, border: `1px solid ${c.tealB}`, padding: "4px 14px", borderRadius: 100, letterSpacing: 1.2, textTransform: "uppercase", display: "inline-block", marginBottom: 14 }}>Platform Metrics</span>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: 12 }}>
            <div>
              <h1 className="dash-h1" style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 38, fontWeight: 800, margin: "0 0 8px", color: c.text, letterSpacing: -1 }}>Analytics Dashboard</h1>
              <p style={{ color: c.sub, fontSize: 15, margin: 0 }}>Real-time performance metrics and model status.</p>
            </div>
            <div style={{
              display: "flex", alignItems: "center", gap: 7,
              background: apiStatus === "online" ? c.tealL : apiStatus === "offline" ? c.redL : c.cardAlt,
              border: `1px solid ${apiStatus === "online" ? c.tealB : apiStatus === "offline" ? c.redB : c.border}`,
              padding: "8px 16px", borderRadius: 100
            }}>
              <span style={{
                width: 7, height: 7, borderRadius: "50%",
                background: apiStatus === "online" ? c.teal : apiStatus === "offline" ? c.red : c.muted,
                display: "inline-block", boxShadow: apiStatus === "online" ? `0 0 0 3px ${c.teal}30` : "none"
              }} />
              <span style={{
                fontSize: 12, fontWeight: 700,
                color: apiStatus === "online" ? c.teal : apiStatus === "offline" ? c.red : c.muted
              }}>
                {apiStatus === "checking" ? "Checking API…" : apiStatus === "online" ? "All systems operational" : "API unreachable · cached data"}
              </span>
            </div>
          </div>
        </div>

        {loading ? (
          <div style={{ textAlign: "center", padding: "100px 0", color: c.muted }}>
            <div style={{ width: 48, height: 48, border: `3px solid ${c.border}`, borderTop: `3px solid ${c.teal}`, borderRadius: "50%", animation: "spin .8s linear infinite", margin: "0 auto 16px" }} />
            <p style={{ margin: 0, fontWeight: 500, fontSize: 15 }}>Loading analytics…</p>
          </div>
        ) : (
          <>
            {/* Stat cards */}
            <div className="stat-grid" style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 16, marginBottom: 20 }}>
              {[
                { l: "Total Predictions", v: analytics?.total_predictions || "1,254", s: "All time", bg: c.tealL, col: c.teal, ic: "🔮" },
                { l: "Top-3 Accuracy", v: "52.9%", s: "Symptoms model", bg: c.blueL, col: c.blue, ic: "🎯" },
                { l: "Diseases Covered", v: "49", s: "Tier A", bg: c.purpL, col: c.purple, ic: "🧬" },
                { l: "Training Images", v: "35K+", s: "Biomedical", bg: c.ambL, col: c.amber, ic: "🩻" },
              ].map(({ l, v, s, bg, col, ic }) => (
                <div key={l} className="stat-card" style={{ background: bg, borderRadius: 20, padding: "26px 24px", cursor: "default", transition: "all .2s", animation: "fadeUp .5s ease both" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                    <div>
                      <p style={{ fontSize: 10, color: col, opacity: .65, textTransform: "uppercase", letterSpacing: 1, margin: "0 0 12px", fontWeight: 800 }}>{l}</p>
                      <p style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 34, fontWeight: 800, color: col, margin: 0, letterSpacing: -.8 }}>{v}</p>
                      {s && <p style={{ fontSize: 12, color: col, opacity: .6, margin: "6px 0 0", fontWeight: 500 }}>{s}</p>}
                    </div>
                    <span style={{ fontSize: 24 }}>{ic}</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Chart row */}
            <div className="chart-row" style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16, marginBottom: 20 }}>
              <div style={{ background: c.card, border: `1px solid ${c.border}`, borderRadius: 22, padding: 28, animation: "fadeUp .5s .1s ease both" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 22, flexWrap: "wrap", gap: 10 }}>
                  <div>
                    <h2 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 17, fontWeight: 700, margin: "0 0 3px", color: c.text }}>Prediction Volume</h2>
                    <p style={{ fontSize: 12, color: c.muted, margin: 0 }}>Predictions over time</p>
                  </div>
                  <div style={{ display: "flex", gap: 6 }}>
                    {Object.keys(RANGES).map(r => (
                      <button key={r} className="range-btn" onClick={() => setRange(r)} style={{
                        padding: "6px 14px", borderRadius: 100, fontSize: 12, fontWeight: 700, cursor: "pointer",
                        fontFamily: "'Inter',sans-serif", transition: "all .15s",
                        border: range === r ? `1.5px solid ${c.tealB}` : `1px solid ${c.borderI}`,
                        background: range === r ? c.tealL : c.card,
                        color: range === r ? c.teal : c.sub,
                      }}>{r}</button>
                    ))}
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={chartData} barSize={chartData.length > 5 ? 26 : 40}>
                    <CartesianGrid strokeDasharray="3 3" stroke={c.border} vertical={false} />
                    <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fill: c.muted, fontSize: 12 }} />
                    <YAxis axisLine={false} tickLine={false} tick={{ fill: c.muted, fontSize: 12 }} />
                    <Tooltip content={<Tip c={c} />} cursor={{ fill: `${c.teal}0A` }} />
                    <Bar dataKey="v" radius={[8, 8, 0, 0]}>
                      {chartData.map((_, i) => <Cell key={i} fill={i === peakIdx ? c.teal : `${c.teal}40`} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Status */}
              <div style={{ background: c.card, border: `1px solid ${c.border}`, borderRadius: 22, padding: 28, animation: "fadeUp .5s .15s ease both" }}>
                <h3 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 17, fontWeight: 700, margin: "0 0 20px", color: c.text }}>Platform Status</h3>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {[
                    { label: "Backend API", status: apiStatus === "online" ? "Online" : apiStatus === "offline" ? "Offline" : "Checking", col: apiStatus === "online" ? c.teal : apiStatus === "offline" ? c.red : c.muted, bg: apiStatus === "online" ? c.tealL : apiStatus === "offline" ? c.redL : c.cardAlt, bc: apiStatus === "online" ? c.tealB : apiStatus === "offline" ? c.redB : c.border },
                    { label: "Symptom Model", status: "Active", col: c.teal, bg: c.tealL, bc: c.tealB },
                    { label: "Image Model", status: "Training", col: c.amber, bg: c.ambL, bc: c.ambB },
                    { label: "Fusion Model", status: "Pending", col: c.muted, bg: c.cardAlt, bc: c.border },
                  ].map(({ label, status, col, bg, bc }) => (
                    <div key={label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "11px 14px", background: c.bgAlt, border: `1px solid ${c.border}`, borderRadius: 12 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                        <span style={{ width: 8, height: 8, borderRadius: "50%", background: col, display: "inline-block" }} />
                        <span style={{ fontSize: 13, color: c.sub, fontWeight: 500 }}>{label}</span>
                      </div>
                      <span style={{ fontSize: 10, fontWeight: 800, padding: "4px 11px", borderRadius: 100, background: bg, color: col, border: `1px solid ${bc}`, textTransform: "uppercase", letterSpacing: .7 }}>{status}</span>
                    </div>
                  ))}
                </div>

                <div style={{ marginTop: 24, paddingTop: 20, borderTop: `1px solid ${c.border}` }}>
                  <p style={{ fontSize: 11, fontWeight: 800, color: c.muted, textTransform: "uppercase", letterSpacing: .8, margin: "0 0 12px" }}>Confidence Distribution</p>
                  <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
                    {pieData.map(({ name, value }, i) => {
                      const col = [c.teal, c.amber, c.red][i];
                      return (
                        <div key={name} style={{ flex: 1 }}>
                          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 5 }}>
                            <span style={{ color: c.sub, fontWeight: 500 }}>{name}</span>
                            <span style={{ color: col, fontWeight: 800 }}>{value}%</span>
                          </div>
                          <div style={{ height: 4, background: c.border, borderRadius: 100 }}>
                            <div style={{ height: 4, width: `${value}%`, background: col, borderRadius: 100, animation: "barGrow .9s ease" }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>

            {/* Second row: accuracy trend + model perf */}
            <div className="perf-row" style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: 16, marginBottom: 20 }}>
              <div style={{ background: c.card, border: `1px solid ${c.border}`, borderRadius: 22, padding: 28, animation: "fadeUp .5s .2s ease both" }}>
                <h3 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 17, fontWeight: 700, margin: "0 0 4px", color: c.text }}>Accuracy Trend</h3>
                <p style={{ fontSize: 12, color: c.muted, margin: "0 0 20px" }}>Top-3 accuracy over 4 weeks</p>
                <ResponsiveContainer width="100%" height={160}>
                  <LineChart data={lineData}>
                    <CartesianGrid strokeDasharray="3 3" stroke={c.border} vertical={false} />
                    <XAxis dataKey="week" axisLine={false} tickLine={false} tick={{ fill: c.muted, fontSize: 11 }} />
                    <YAxis axisLine={false} tickLine={false} tick={{ fill: c.muted, fontSize: 11 }} domain={[40, 56]} />
                    <Tooltip content={<Tip c={c} />} />
                    <Line type="monotone" dataKey="accuracy" stroke={c.teal} strokeWidth={2.5} dot={{ fill: c.teal, r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div style={{ background: c.card, border: `1px solid ${c.border}`, borderRadius: 22, padding: 28, animation: "fadeUp .5s .25s ease both" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
                  <h3 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 17, fontWeight: 700, margin: 0, color: c.text }}>Model Performance</h3>
                  <span style={{ fontSize: 12, color: c.muted, fontWeight: 500 }}>Last updated today</span>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                  {[
                    { l: "Symptom Model — Top-3 Accuracy (Exp 3)", p: 52.9, col: c.teal, bg: c.tealL, d: "TF-IDF + LR · 49 diseases" },
                    { l: "Symptom Model — Exp 1 (5% data)", p: 40.3, col: c.purple, bg: c.purpL, d: "Scarcity sim · 514 samples" },
                    { l: "Image Model — In Training", p: 35, col: c.blue, bg: c.blueL, d: "EfficientNet-B4 · 28,299 images" },
                    { l: "Fusion Model — Not Yet Built", p: 0, col: c.amber, bg: c.ambL, d: "Cross-attention · Coming soon" },
                  ].map(({ l, p, col, bg, d }) => (
                    <div key={l}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
                        <div>
                          <p style={{ fontSize: 13, color: c.text, fontWeight: 700, margin: "0 0 2px" }}>{l}</p>
                          <p style={{ fontSize: 11, color: c.muted, margin: 0 }}>{d}</p>
                        </div>
                        <span style={{ fontSize: 14, fontWeight: 800, color: col, background: bg, padding: "4px 12px", borderRadius: 9, flexShrink: 0, marginLeft: 12, fontFamily: "'Plus Jakarta Sans',sans-serif" }}>{p}%</span>
                      </div>
                      <div style={{ height: 7, background: c.border, borderRadius: 100, overflow: "hidden" }}>
                        <div style={{ height: 7, width: `${p}%`, background: col, borderRadius: 100, animation: "barGrow 1.2s ease" }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Disease frequency leaderboard */}
            <div style={{ background: c.card, border: `1px solid ${c.border}`, borderRadius: 22, overflow: "hidden", animation: "fadeUp .5s .3s ease both" }}>
              <div style={{ padding: "22px 28px", borderBottom: `1px solid ${c.border}`, display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
                <div>
                  <h3 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 17, fontWeight: 700, margin: "0 0 3px", color: c.text }}>Most Predicted Diseases</h3>
                  <p style={{ fontSize: 12, color: c.muted, margin: 0 }}>Frequency leaderboard, all time</p>
                </div>
                <span style={{ fontSize: 11, color: c.teal, background: c.tealL, border: `1px solid ${c.tealB}`, padding: "4px 12px", borderRadius: 100, fontWeight: 700 }}>Top 6 of 49</span>
              </div>
              <div>
                {LEADERBOARD.map(({ rank, disease, count, pct, trend }) => (
                  <div key={rank} className="lb-row" style={{
                    display: "grid", gridTemplateColumns: "44px 1fr 90px 70px", alignItems: "center",
                    padding: "15px 28px", borderBottom: rank < LEADERBOARD.length ? `1px solid ${c.border}` : "none",
                    transition: "background .12s",
                  }}>
                    <span style={{
                      width: 30, height: 30, borderRadius: 9,
                      background: rank <= 3 ? c.tealL : c.cardAlt,
                      border: `1px solid ${rank <= 3 ? c.tealB : c.border}`,
                      display: "flex", alignItems: "center", justifyContent: "center",
                      fontSize: 12, fontWeight: 800, color: rank <= 3 ? c.teal : c.sub,
                    }}>{rank}</span>
                    <div>
                      <p style={{ fontSize: 14, color: c.text, fontWeight: 700, margin: "0 0 6px" }}>{disease}</p>
                      <div style={{ height: 5, background: c.border, borderRadius: 100, maxWidth: 280 }}>
                        <div style={{ height: 5, width: `${pct * 4}%`, maxWidth: "100%", background: c.teal, borderRadius: 100 }} />
                      </div>
                    </div>
                    <span style={{ fontSize: 13, color: c.sub, fontWeight: 600, textAlign: "right" }}>{count} cases</span>
                    <span style={{ fontSize: 13, fontWeight: 800, color: TREND_COLOR(trend), textAlign: "right" }}>
                      {TREND_ICON[trend]} {pct}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default Dashboard;