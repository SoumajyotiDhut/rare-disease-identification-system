import { useEffect, useState } from "react";
import { getAnalytics } from "../services/Api";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Cell,
} from "recharts";

const pg = {
  minHeight: "100vh",
  background: "#0A1628",
  color: "#F8FAFC",
  fontFamily: "'Inter', sans-serif",
  padding: "48px 32px",
};

const card = (extra = {}) => ({
  background: "rgba(255,255,255,0.03)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 20,
  padding: 28,
  ...extra,
});

const StatCard = ({ label, value, sub, color, icon }) => (
  <div style={{
    background: `rgba(${color},0.06)`,
    border: `1px solid rgba(${color},0.18)`,
    borderRadius: 20, padding: 28,
  }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
      <div>
        <p style={{ fontSize: 12, color: "#64748B", textTransform: "uppercase", letterSpacing: 1, marginBottom: 12 }}>{label}</p>
        <p style={{ fontFamily: "'Syne', sans-serif", fontSize: 40, fontWeight: 800, color: "#F8FAFC", margin: 0 }}>{value}</p>
        {sub && <p style={{ fontSize: 12, color: `rgb(${color})`, marginTop: 8 }}>{sub}</p>}
      </div>
      <span style={{ fontSize: 28 }}>{icon}</span>
    </div>
  </div>
);

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#0F2040", border: "1px solid rgba(0,212,200,0.3)",
      borderRadius: 10, padding: "12px 16px",
    }}>
      <p style={{ color: "#94A3B8", fontSize: 12, margin: "0 0 4px" }}>{label}</p>
      <p style={{ color: "#00D4C8", fontWeight: 700, fontSize: 18, margin: 0 }}>
        {payload[0].value} predictions
      </p>
    </div>
  );
};

const chartData = [
  { day: "Mon", predictions: 40 },
  { day: "Tue", predictions: 65 },
  { day: "Wed", predictions: 85 },
  { day: "Thu", predictions: 55 },
  { day: "Fri", predictions: 95 },
  { day: "Sat", predictions: 75 },
  { day: "Sun", predictions: 50 },
];

const TEAL = "#00D4C8";

function Dashboard() {
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getAnalytics()
      .then(setAnalytics)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  return (
    <div style={pg}>
      <div style={{ maxWidth: 1280, margin: "0 auto" }}>

        {/* Header */}
        <div style={{ marginBottom: 40 }}>
          <p style={{ fontSize: 12, color: "#00D4C8", letterSpacing: 2, textTransform: "uppercase", marginBottom: 12 }}>PLATFORM METRICS</p>
          <h1 style={{ fontFamily: "'Syne', sans-serif", fontSize: 42, fontWeight: 800, margin: "0 0 8px" }}>
            Analytics Dashboard
          </h1>
          <p style={{ color: "#64748B", fontSize: 16 }}>
            Real-time performance metrics and system status for AI DOC.
          </p>
        </div>

        {loading ? (
          <div style={{ textAlign: "center", padding: "80px 0", color: "#475569" }}>
            <div style={{
              width: 40, height: 40, border: "3px solid rgba(0,212,200,0.2)",
              borderTop: "3px solid #00D4C8", borderRadius: "50%",
              animation: "spin 0.8s linear infinite", margin: "0 auto 20px",
            }} />
            Loading analytics…
          </div>
        ) : (
          <>
            {/* Stat cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 20, marginBottom: 28 }}>
              <StatCard label="Total Predictions" value={analytics?.total_predictions || "1,254"} sub="All time" color="0,212,200" icon="🔮" />
              <StatCard label="Top-3 Accuracy" value="52.9%" sub="Symptoms model" color="34,197,94" icon="🎯" />
              <StatCard label="Diseases Covered" value="49" sub="Tier A diseases" color="167,139,250" icon="🧬" />
              <StatCard label="Training Images" value="35K+" sub="Biomedical scans" color="251,146,60" icon="🩻" />
            </div>

            {/* Chart + status */}
            <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 20, marginBottom: 28 }}>

              {/* Bar chart */}
              <div style={card()}>
                <h2 style={{ fontFamily: "'Syne', sans-serif", fontSize: 20, fontWeight: 700, margin: "0 0 24px" }}>
                  Weekly Prediction Trends
                </h2>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={chartData} barSize={32}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                    <XAxis dataKey="day" axisLine={false} tickLine={false}
                      tick={{ fill: "#64748B", fontSize: 12 }} />
                    <YAxis axisLine={false} tickLine={false}
                      tick={{ fill: "#64748B", fontSize: 12 }} />
                    <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
                    <Bar dataKey="predictions" radius={[8, 8, 0, 0]}>
                      {chartData.map((_, i) => (
                        <Cell key={i} fill={i === 4 ? TEAL : "rgba(0,212,200,0.25)"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Platform status */}
              <div style={card()}>
                <h3 style={{ fontFamily: "'Syne', sans-serif", fontSize: 18, fontWeight: 700, margin: "0 0 24px" }}>
                  Platform Status
                </h3>
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                  {[
                    { label: "Backend API", status: "Online", ok: true },
                    { label: "Symptom Model", status: "Active", ok: true },
                    { label: "Image Model", status: "Training", ok: false },
                    { label: "Fusion Model", status: "Pending", ok: false },
                  ].map(({ label, status, ok }) => (
                    <div key={label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{ fontSize: 14, color: "#94A3B8" }}>{label}</span>
                      <span style={{
                        fontSize: 12, fontWeight: 600,
                        padding: "4px 12px", borderRadius: 100,
                        background: ok ? "rgba(34,197,94,0.1)" : "rgba(251,191,36,0.1)",
                        color: ok ? "#22C55E" : "#FBBF24",
                        border: `1px solid ${ok ? "rgba(34,197,94,0.3)" : "rgba(251,191,36,0.3)"}`,
                      }}>{status}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Model performance */}
            <div style={card()}>
              <h3 style={{ fontFamily: "'Syne', sans-serif", fontSize: 20, fontWeight: 700, margin: "0 0 28px" }}>
                Model Performance Overview
              </h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                {[
                  { label: "Symptom Model — Top-3 Accuracy (Exp 3)", pct: 52.9, color: "#00D4C8", detail: "TF-IDF + Logistic Regression · 49 diseases" },
                  { label: "Symptom Model — Top-3 Accuracy (Exp 1, 5% data)", pct: 40.3, color: "#A78BFA", detail: "Scarcity simulation · 514 training samples" },
                  { label: "Image Model — In Training", pct: 35, color: "#22C55E", detail: "EfficientNet-B4 · 28,299 training images" },
                  { label: "Fusion Model — Not Yet Built", pct: 0, color: "#FB923C", detail: "Cross-attention · Coming soon" },
                ].map(({ label, pct, color, detail }) => (
                  <div key={label}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                      <div>
                        <p style={{ fontSize: 14, color: "#F8FAFC", fontWeight: 600, margin: "0 0 2px" }}>{label}</p>
                        <p style={{ fontSize: 12, color: "#64748B", margin: 0 }}>{detail}</p>
                      </div>
                      <span style={{ fontSize: 18, fontWeight: 700, color, alignSelf: "center" }}>{pct}%</span>
                    </div>
                    <div style={{ height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 100 }}>
                      <div style={{
                        height: 6, width: `${pct}%`, background: color,
                        borderRadius: 100, transition: "width 1s ease",
                        boxShadow: `0 0 8px ${color}66`,
                      }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );
}

export default Dashboard;