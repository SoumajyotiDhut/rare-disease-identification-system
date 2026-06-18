import { useEffect, useState } from "react";
import { getAnalytics } from "../services/Api";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell, LineChart, Line, PieChart, Pie, Legend } from "recharts";

const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&display=swap');
  @keyframes spin    { to { transform:rotate(360deg) } }
  @keyframes fadeUp  { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
  @keyframes barGrow { from{width:0} }
  .stat-card:hover { transform:translateY(-3px)!important; box-shadow:0 12px 32px rgba(15,28,46,0.07)!important }
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

const chartData = [
  { day: "Mon", v: 40 }, { day: "Tue", v: 65 }, { day: "Wed", v: 85 },
  { day: "Thu", v: 55 }, { day: "Fri", v: 95 }, { day: "Sat", v: 75 }, { day: "Sun", v: 50 },
];
const lineData = [
  { week: "W1", accuracy: 45 }, { week: "W2", accuracy: 48 }, { week: "W3", accuracy: 50 }, { week: "W4", accuracy: 52.9 },
];
const pieData = [
  { name: "High", value: 38, fill: "#0B7B6F" }, { name: "Medium", value: 44, fill: "#C05B1A" }, { name: "Low", value: 18, fill: "#B83030" },
];

const Tip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#fff", border: "1px solid #E0EBF2", borderRadius: 12, padding: "10px 16px", boxShadow: "0 4px 20px rgba(15,28,46,0.09)" }}>
      <p style={{ color: "#8FA5B5", fontSize: 10, margin: "0 0 3px", fontWeight: 800, textTransform: "uppercase", letterSpacing: .6 }}>{label}</p>
      <p style={{ color: "#0B7B6F", fontWeight: 800, fontSize: 18, margin: 0, fontFamily: "'Plus Jakarta Sans',sans-serif" }}>{payload[0].value}{payload[0].name === "accuracy" ? "%" : ""}</p>
    </div>
  );
};

function Dashboard() {
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getAnalytics().then(setAnalytics).catch(console.error).finally(() => setLoading(false));
  }, []);

  return (
    <div style={{ minHeight: "100vh", background: "#F4F8FB", fontFamily: "'Inter',sans-serif" }}>
      <style>{CSS}</style>
      <div className="dash-pad" style={{ maxWidth: 1200, margin: "0 auto", padding: "56px 32px" }}>

        {/* Header */}
        <div style={{ marginBottom: 44 }}>
          <span style={{ fontSize: 10, fontWeight: 800, color: "#0B7B6F", background: "#EBF8F6", border: "1px solid #B2E8E2", padding: "4px 14px", borderRadius: 100, letterSpacing: 1.2, textTransform: "uppercase", display: "inline-block", marginBottom: 14 }}>Platform Metrics</span>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: 12 }}>
            <div>
              <h1 className="dash-h1" style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 38, fontWeight: 800, margin: "0 0 8px", color: "#0F1C2E", letterSpacing: -1 }}>Analytics Dashboard</h1>
              <p style={{ color: "#7A94A8", fontSize: 15, margin: 0 }}>Real-time performance metrics and model status.</p>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 7, background: "#EBF8F6", border: "1px solid #B2E8E2", padding: "8px 16px", borderRadius: 100 }}>
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#0B7B6F", display: "inline-block", boxShadow: "0 0 0 3px rgba(11,123,111,0.2)" }} />
              <span style={{ fontSize: 12, color: "#0B7B6F", fontWeight: 700 }}>All systems operational</span>
            </div>
          </div>
        </div>

        {loading ? (
          <div style={{ textAlign: "center", padding: "100px 0", color: "#8FA5B5" }}>
            <div style={{ width: 48, height: 48, border: "3px solid #E8EFF5", borderTop: "3px solid #0B7B6F", borderRadius: "50%", animation: "spin .8s linear infinite", margin: "0 auto 16px" }} />
            <p style={{ margin: 0, fontWeight: 500, fontSize: 15 }}>Loading analytics…</p>
          </div>
        ) : (
          <>
            {/* Stat cards */}
            <div className="stat-grid" style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 16, marginBottom: 20 }}>
              {[
                { l: "Total Predictions", v: analytics?.total_predictions || "1,254", s: "All time", bg: "#EBF8F6", c: "#0B7B6F", ic: "🔮" },
                { l: "Top-3 Accuracy", v: "52.9%", s: "Symptoms model", bg: "#EBF4F9", c: "#1D6FA4", ic: "🎯" },
                { l: "Diseases Covered", v: "49", s: "Tier A", bg: "#F2EEF9", c: "#5B3DB8", ic: "🧬" },
                { l: "Training Images", v: "35K+", s: "Biomedical", bg: "#FFF4EC", c: "#C05B1A", ic: "🩻" },
              ].map(({ l, v, s, bg, c, ic }) => (
                <div key={l} className="stat-card" style={{ background: bg, borderRadius: 20, padding: "26px 24px", cursor: "default", transition: "all .2s", animation: "fadeUp .5s ease both" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                    <div>
                      <p style={{ fontSize: 10, color: c, opacity: .65, textTransform: "uppercase", letterSpacing: 1, margin: "0 0 12px", fontWeight: 800 }}>{l}</p>
                      <p style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 34, fontWeight: 800, color: c, margin: 0, letterSpacing: -.8 }}>{v}</p>
                      {s && <p style={{ fontSize: 12, color: c, opacity: .6, margin: "6px 0 0", fontWeight: 500 }}>{s}</p>}
                    </div>
                    <span style={{ fontSize: 24 }}>{ic}</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Chart row */}
            <div className="chart-row" style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16, marginBottom: 20 }}>
              {/* Bar chart */}
              <div style={{ background: "#fff", border: "1px solid #E8EFF5", borderRadius: 22, padding: 28, animation: "fadeUp .5s .1s ease both" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 22 }}>
                  <div>
                    <h2 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 17, fontWeight: 700, margin: "0 0 3px", color: "#0F1C2E" }}>Weekly Prediction Trends</h2>
                    <p style={{ fontSize: 12, color: "#8FA5B5", margin: 0 }}>Predictions per day this week</p>
                  </div>
                  <span style={{ fontSize: 11, fontWeight: 700, color: "#0B7B6F", background: "#EBF8F6", border: "1px solid #B2E8E2", padding: "4px 12px", borderRadius: 100 }}>This Week</span>
                </div>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={chartData} barSize={26}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#F0F5F8" vertical={false} />
                    <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fill: "#8FA5B5", fontSize: 12 }} />
                    <YAxis axisLine={false} tickLine={false} tick={{ fill: "#8FA5B5", fontSize: 12 }} />
                    <Tooltip content={<Tip />} cursor={{ fill: "rgba(11,123,111,0.04)" }} />
                    <Bar dataKey="v" radius={[8, 8, 0, 0]}>
                      {chartData.map((_, i) => <Cell key={i} fill={i === 4 ? "#0B7B6F" : "#C6E9E5"} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Status */}
              <div style={{ background: "#fff", border: "1px solid #E8EFF5", borderRadius: 22, padding: 28, animation: "fadeUp .5s .15s ease both" }}>
                <h3 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 17, fontWeight: 700, margin: "0 0 20px", color: "#0F1C2E" }}>Platform Status</h3>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {[
                    { label: "Backend API", status: "Online", c: "#0B7B6F", bg: "#EBF8F6", bc: "#B2E8E2", dot: "#0B7B6F" },
                    { label: "Symptom Model", status: "Active", c: "#0B7B6F", bg: "#EBF8F6", bc: "#B2E8E2", dot: "#0B7B6F" },
                    { label: "Image Model", status: "Training", c: "#C05B1A", bg: "#FFF8EC", bc: "#F5D8B8", dot: "#C05B1A" },
                    { label: "Fusion Model", status: "Pending", c: "#8FA5B5", bg: "#F0F5F8", bc: "#DDE8EF", dot: "#C8D8E4" },
                  ].map(({ label, status, c, bg, bc, dot }) => (
                    <div key={label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "11px 14px", background: "#F8FBFD", border: "1px solid #EDF2F6", borderRadius: 12 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                        <span style={{ width: 8, height: 8, borderRadius: "50%", background: dot, display: "inline-block" }} />
                        <span style={{ fontSize: 13, color: "#4A6275", fontWeight: 500 }}>{label}</span>
                      </div>
                      <span style={{ fontSize: 10, fontWeight: 800, padding: "4px 11px", borderRadius: 100, background: bg, color: c, border: `1px solid ${bc}`, textTransform: "uppercase", letterSpacing: .7 }}>{status}</span>
                    </div>
                  ))}
                </div>

                {/* Confidence distribution mini pie */}
                <div style={{ marginTop: 24, paddingTop: 20, borderTop: "1px solid #EDF2F6" }}>
                  <p style={{ fontSize: 11, fontWeight: 800, color: "#8FA5B5", textTransform: "uppercase", letterSpacing: .8, margin: "0 0 12px" }}>Confidence Distribution</p>
                  <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
                    {pieData.map(({ name, value, fill }) => (
                      <div key={name} style={{ flex: 1 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 5 }}>
                          <span style={{ color: "#5A7184", fontWeight: 500 }}>{name}</span>
                          <span style={{ color: fill, fontWeight: 800 }}>{value}%</span>
                        </div>
                        <div style={{ height: 4, background: "#F0F5F8", borderRadius: 100 }}>
                          <div style={{ height: 4, width: `${value}%`, background: fill, borderRadius: 100, animation: "barGrow .9s ease" }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Second row: accuracy trend + model perf */}
            <div className="perf-row" style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: 16, marginBottom: 20 }}>
              {/* Line chart */}
              <div style={{ background: "#fff", border: "1px solid #E8EFF5", borderRadius: 22, padding: 28, animation: "fadeUp .5s .2s ease both" }}>
                <h3 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 17, fontWeight: 700, margin: "0 0 4px", color: "#0F1C2E" }}>Accuracy Trend</h3>
                <p style={{ fontSize: 12, color: "#8FA5B5", margin: "0 0 20px" }}>Top-3 accuracy over 4 weeks</p>
                <ResponsiveContainer width="100%" height={160}>
                  <LineChart data={lineData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#F0F5F8" vertical={false} />
                    <XAxis dataKey="week" axisLine={false} tickLine={false} tick={{ fill: "#8FA5B5", fontSize: 11 }} />
                    <YAxis axisLine={false} tickLine={false} tick={{ fill: "#8FA5B5", fontSize: 11 }} domain={[40, 56]} />
                    <Tooltip content={<Tip />} />
                    <Line type="monotone" dataKey="accuracy" stroke="#0B7B6F" strokeWidth={2.5} dot={{ fill: "#0B7B6F", r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Model performance */}
              <div style={{ background: "#fff", border: "1px solid #E8EFF5", borderRadius: 22, padding: 28, animation: "fadeUp .5s .25s ease both" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
                  <h3 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 17, fontWeight: 700, margin: 0, color: "#0F1C2E" }}>Model Performance</h3>
                  <span style={{ fontSize: 12, color: "#8FA5B5", fontWeight: 500 }}>Last updated today</span>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                  {[
                    { l: "Symptom Model — Top-3 Accuracy (Exp 3)", p: 52.9, c: "#0B7B6F", bg: "#EBF8F6", d: "TF-IDF + LR · 49 diseases" },
                    { l: "Symptom Model — Exp 1 (5% data)", p: 40.3, c: "#5B3DB8", bg: "#F2EEF9", d: "Scarcity sim · 514 samples" },
                    { l: "Image Model — In Training", p: 35, c: "#1D6FA4", bg: "#EBF4F9", d: "EfficientNet-B4 · 28,299 images" },
                    { l: "Fusion Model — Not Yet Built", p: 0, c: "#C05B1A", bg: "#FFF4EC", d: "Cross-attention · Coming soon" },
                  ].map(({ l, p, c, bg, d }) => (
                    <div key={l}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
                        <div>
                          <p style={{ fontSize: 13, color: "#0F1C2E", fontWeight: 700, margin: "0 0 2px" }}>{l}</p>
                          <p style={{ fontSize: 11, color: "#8FA5B5", margin: 0 }}>{d}</p>
                        </div>
                        <span style={{ fontSize: 14, fontWeight: 800, color: c, background: bg, padding: "4px 12px", borderRadius: 9, flexShrink: 0, marginLeft: 12, fontFamily: "'Plus Jakarta Sans',sans-serif" }}>{p}%</span>
                      </div>
                      <div style={{ height: 7, background: "#F0F5F8", borderRadius: 100, overflow: "hidden" }}>
                        <div style={{ height: 7, width: `${p}%`, background: c, borderRadius: 100, animation: "barGrow 1.2s ease", transition: "width .3s" }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default Dashboard;