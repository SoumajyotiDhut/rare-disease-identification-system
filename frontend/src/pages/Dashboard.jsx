import { useEffect, useState } from "react";
import { getAnalytics } from "../services/Api";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Cell,
} from "recharts";

const pg = {
  minHeight: "100vh",
  background: "#F8FAFB",
  color: "#0F1C2E",
  fontFamily: "'Inter', sans-serif",
  padding: "56px 40px",
};

const card = (extra = {}) => ({
  background: "#fff",
  border: "1px solid #E8EFF5",
  borderRadius: 20,
  padding: 28,
  ...extra,
});

const StatCard = ({ label, value, sub, colorBg, colorText, icon }) => (
  <div style={{
    background: colorBg,
    borderRadius: 20,
    padding: 28,
    border: "none",
  }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
      <div>
        <p style={{ fontSize: 11, color: colorText, opacity: 0.65, textTransform: "uppercase", letterSpacing: 1, marginBottom: 12, fontWeight: 700 }}>{label}</p>
        <p style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif", fontSize: 38, fontWeight: 800, color: colorText, margin: 0, letterSpacing: -0.8 }}>{value}</p>
        {sub && <p style={{ fontSize: 12, color: colorText, opacity: 0.65, marginTop: 8, fontWeight: 500 }}>{sub}</p>}
      </div>
      <span style={{ fontSize: 26, opacity: 0.8 }}>{icon}</span>
    </div>
  </div>
);

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#fff", border: "1px solid #DDE8EF",
      borderRadius: 12, padding: "12px 18px",
      boxShadow: "0 4px 20px rgba(15,28,46,0.1)",
    }}>
      <p style={{ color: "#7A94A8", fontSize: 12, margin: "0 0 4px", fontWeight: 600 }}>{label}</p>
      <p style={{ color: "#0B7B6F", fontWeight: 700, fontSize: 18, margin: 0 }}>
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
      <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&display=swap" rel="stylesheet" />
      <div style={{ maxWidth: 1200, margin: "0 auto" }}>

        {/* Header */}
        <div style={{ marginBottom: 44 }}>
          <span style={{
            fontSize: 11, fontWeight: 700, color: "#0B7B6F",
            background: "#EBF8F6", border: "1px solid #B2E8E2",
            padding: "4px 14px", borderRadius: 100, letterSpacing: 1,
            textTransform: "uppercase", display: "inline-block", marginBottom: 16,
          }}>Platform Metrics</span>
          <h1 style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif", fontSize: 40, fontWeight: 800, margin: "0 0 10px", color: "#0F1C2E", letterSpacing: -1 }}>
            Analytics Dashboard
          </h1>
          <p style={{ color: "#7A94A8", fontSize: 16, margin: 0, lineHeight: 1.6 }}>
            Real-time performance metrics and system status for AI DOC.
          </p>
        </div>

        {loading ? (
          <div style={{ textAlign: "center", padding: "100px 0", color: "#8FA5B5" }}>
            <div style={{
              width: 40, height: 40, border: "3px solid #E8EFF5",
              borderTop: "3px solid #0B7B6F", borderRadius: "50%",
              animation: "spin 0.8s linear infinite", margin: "0 auto 20px",
            }} />
            Loading analytics…
          </div>
        ) : (
          <>
            {/* Stat cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 18, marginBottom: 24 }}>
              <StatCard label="Total Predictions" value={analytics?.total_predictions || "1,254"} sub="All time" colorBg="#EBF8F6" colorText="#0B7B6F" icon="🔮" />
              <StatCard label="Top-3 Accuracy" value="52.9%" sub="Symptoms model" colorBg="#EBF4F9" colorText="#1D6FA4" icon="🎯" />
              <StatCard label="Diseases Covered" value="49" sub="Tier A diseases" colorBg="#F2EEF9" colorText="#5B3DB8" icon="🧬" />
              <StatCard label="Training Images" value="35K+" sub="Biomedical scans" colorBg="#FFF4EC" colorText="#C05B1A" icon="🩻" />
            </div>

            {/* Chart + status */}
            <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 18, marginBottom: 24 }}>
              <div style={card()}>
                <h2 style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif", fontSize: 18, fontWeight: 700, margin: "0 0 24px", color: "#0F1C2E" }}>
                  Weekly Prediction Trends
                </h2>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={chartData} barSize={30}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#F0F5F8" vertical={false} />
                    <XAxis dataKey="day" axisLine={false} tickLine={false}
                      tick={{ fill: "#8FA5B5", fontSize: 12 }} />
                    <YAxis axisLine={false} tickLine={false}
                      tick={{ fill: "#8FA5B5", fontSize: 12 }} />
                    <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(11,123,111,0.04)" }} />
                    <Bar dataKey="predictions" radius={[8, 8, 0, 0]}>
                      {chartData.map((_, i) => (
                        <Cell key={i} fill={i === 4 ? "#0B7B6F" : "#C6E9E5"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Platform status */}
              <div style={card()}>
                <h3 style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif", fontSize: 17, fontWeight: 700, margin: "0 0 24px", color: "#0F1C2E" }}>
                  Platform Status
                </h3>
                <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                  {[
                    { label: "Backend API", status: "Online", ok: true },
                    { label: "Symptom Model", status: "Active", ok: true },
                    { label: "Image Model", status: "Training", ok: false },
                    { label: "Fusion Model", status: "Pending", ok: false },
                  ].map(({ label, status, ok }) => (
                    <div key={label} style={{
                      display: "flex", justifyContent: "space-between", alignItems: "center",
                      padding: "12px 16px", borderRadius: 12,
                      background: "#F8FAFB", border: "1px solid #EDF2F6",
                    }}>
                      <span style={{ fontSize: 14, color: "#4A6275", fontWeight: 500 }}>{label}</span>
                      <span style={{
                        fontSize: 11, fontWeight: 700,
                        padding: "4px 12px", borderRadius: 100,
                        background: ok ? "#EBF8F6" : "#FFF8EC",
                        color: ok ? "#0B7B6F" : "#C05B1A",
                        border: `1px solid ${ok ? "#B2E8E2" : "#F5D8B8"}`,
                        textTransform: "uppercase", letterSpacing: 0.6,
                      }}>{status}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Model performance */}
            <div style={card()}>
              <h3 style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif", fontSize: 18, fontWeight: 700, margin: "0 0 28px", color: "#0F1C2E" }}>
                Model Performance Overview
              </h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
                {[
                  { label: "Symptom Model — Top-3 Accuracy (Exp 3)", pct: 52.9, color: "#0B7B6F", bg: "#EBF8F6", detail: "TF-IDF + Logistic Regression · 49 diseases" },
                  { label: "Symptom Model — Top-3 Accuracy (Exp 1, 5% data)", pct: 40.3, color: "#5B3DB8", bg: "#F2EEF9", detail: "Scarcity simulation · 514 training samples" },
                  { label: "Image Model — In Training", pct: 35, color: "#1D6FA4", bg: "#EBF4F9", detail: "EfficientNet-B4 · 28,299 training images" },
                  { label: "Fusion Model — Not Yet Built", pct: 0, color: "#C05B1A", bg: "#FFF4EC", detail: "Cross-attention · Coming soon" },
                ].map(({ label, pct, color, bg, detail }) => (
                  <div key={label}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
                      <div>
                        <p style={{ fontSize: 14, color: "#0F1C2E", fontWeight: 600, margin: "0 0 3px" }}>{label}</p>
                        <p style={{ fontSize: 12, color: "#8FA5B5", margin: 0 }}>{detail}</p>
                      </div>
                      <span style={{
                        fontSize: 16, fontWeight: 800, color,
                        background: bg, padding: "4px 12px", borderRadius: 8,
                        alignSelf: "center", flexShrink: 0, marginLeft: 16,
                        fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif",
                      }}>{pct}%</span>
                    </div>
                    <div style={{ height: 7, background: "#F0F5F8", borderRadius: 100 }}>
                      <div style={{
                        height: 7, width: `${pct}%`, background: color,
                        borderRadius: 100, transition: "width 1s ease",
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