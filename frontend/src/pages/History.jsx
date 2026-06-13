import { useEffect, useState } from "react";
import { getHistory } from "../services/Api";

const pg = {
  minHeight: "100vh",
  background: "#0A1628",
  color: "#F8FAFC",
  fontFamily: "'Inter', sans-serif",
  padding: "48px 32px",
};

const ConfBadge = ({ conf }) => {
  const m = {
    High: { bg: "rgba(34,197,94,0.12)", color: "#22C55E", border: "rgba(34,197,94,0.3)" },
    Medium: { bg: "rgba(251,191,36,0.12)", color: "#FBBF24", border: "rgba(251,191,36,0.3)" },
    Low: { bg: "rgba(239,68,68,0.12)", color: "#EF4444", border: "rgba(239,68,68,0.3)" },
  };
  const s = m[conf] || m.Low;
  return (
    <span style={{
      padding: "4px 12px", borderRadius: 100,
      fontSize: 12, fontWeight: 600,
      background: s.bg, color: s.color,
      border: `1px solid ${s.border}`,
    }}>{conf || "—"}</span>
  );
};

function History() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("All");

  useEffect(() => {
    getHistory()
      .then(data => {
        if (Array.isArray(data)) setHistory(data);
        else if (data.history) setHistory(data.history);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const filtered = history.filter(item => {
    const disease = item.predictions?.[0]?.disease || item.disease || "";
    const matchSearch = disease.toLowerCase().includes(search.toLowerCase());
    const conf = item.predictions?.[0]?.confidence || item.confidence || "";
    const matchFilter = filter === "All" || conf === filter;
    return matchSearch && matchFilter;
  });

  return (
    <div style={pg}>
      <div style={{ maxWidth: 1280, margin: "0 auto" }}>

        {/* Header */}
        <div style={{ marginBottom: 40 }}>
          <p style={{ fontSize: 12, color: "#00D4C8", letterSpacing: 2, textTransform: "uppercase", marginBottom: 12 }}>
            AUDIT TRAIL
          </p>
          <h1 style={{ fontFamily: "'Syne', sans-serif", fontSize: 42, fontWeight: 800, margin: "0 0 8px" }}>
            Prediction History
          </h1>
          <p style={{ color: "#64748B", fontSize: 16 }}>
            Browse all previous AI diagnostic sessions and their outcomes.
          </p>
        </div>

        {/* Controls */}
        <div style={{ display: "flex", gap: 16, marginBottom: 28, flexWrap: "wrap" }}>
          {/* Search */}
          <div style={{ position: "relative", flex: 1, minWidth: 240 }}>
            <span style={{
              position: "absolute", left: 16, top: "50%",
              transform: "translateY(-50%)", fontSize: 16, color: "#475569",
            }}>🔍</span>
            <input
              type="text"
              placeholder="Search by disease name…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={{
                width: "100%",
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 12,
                padding: "14px 18px 14px 44px",
                color: "#F8FAFC",
                fontSize: 15,
                outline: "none",
                fontFamily: "'Inter', sans-serif",
                boxSizing: "border-box",
                transition: "border-color 0.2s",
              }}
              onFocus={e => e.target.style.borderColor = "rgba(0,212,200,0.5)"}
              onBlur={e => e.target.style.borderColor = "rgba(255,255,255,0.1)"}
            />
          </div>

          {/* Filter pills */}
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            {["All", "High", "Medium", "Low"].map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                style={{
                  padding: "10px 20px", borderRadius: 100,
                  border: filter === f
                    ? "1px solid rgba(0,212,200,0.5)"
                    : "1px solid rgba(255,255,255,0.1)",
                  background: filter === f ? "rgba(0,212,200,0.12)" : "transparent",
                  color: filter === f ? "#00D4C8" : "#64748B",
                  fontSize: 13, fontWeight: 600,
                  cursor: "pointer", fontFamily: "'Inter', sans-serif",
                  transition: "all 0.2s",
                }}
              >
                {f}
              </button>
            ))}
          </div>

          {/* Count badge */}
          <div style={{
            display: "flex", alignItems: "center",
            background: "rgba(0,212,200,0.08)",
            border: "1px solid rgba(0,212,200,0.2)",
            borderRadius: 10, padding: "10px 18px",
            fontSize: 13, color: "#00D4C8", fontWeight: 600,
            whiteSpace: "nowrap",
          }}>
            {filtered.length} record{filtered.length !== 1 ? "s" : ""}
          </div>
        </div>

        {/* Table */}
        {loading ? (
          <div style={{ textAlign: "center", padding: "80px 0", color: "#475569" }}>
            <div style={{
              width: 40, height: 40,
              border: "3px solid rgba(0,212,200,0.2)",
              borderTop: "3px solid #00D4C8",
              borderRadius: "50%",
              animation: "spin 0.8s linear infinite",
              margin: "0 auto 20px",
            }} />
            Loading history…
          </div>
        ) : filtered.length === 0 ? (
          <div style={{
            textAlign: "center", padding: "80px 0",
            background: "rgba(255,255,255,0.02)",
            border: "1px solid rgba(255,255,255,0.06)",
            borderRadius: 20,
          }}>
            <div style={{ fontSize: 56, marginBottom: 16 }}>📋</div>
            <p style={{ color: "#475569", fontSize: 16, margin: "0 0 8px" }}>
              No records found
            </p>
            <p style={{ color: "#334155", fontSize: 13 }}>
              {search ? "Try a different search term" : "Make a prediction to see history here"}
            </p>
          </div>
        ) : (
          <div style={{
            background: "rgba(255,255,255,0.02)",
            border: "1px solid rgba(255,255,255,0.07)",
            borderRadius: 20,
            overflow: "hidden",
          }}>
            {/* Table head */}
            <div style={{
              display: "grid",
              gridTemplateColumns: "160px 1fr 200px 120px 120px",
              padding: "14px 24px",
              background: "rgba(255,255,255,0.03)",
              borderBottom: "1px solid rgba(255,255,255,0.06)",
            }}>
              {["Date & Time", "Symptoms", "Top Diagnosis", "Probability", "Confidence"].map(h => (
                <span key={h} style={{
                  fontSize: 11, fontWeight: 700, color: "#475569",
                  textTransform: "uppercase", letterSpacing: 1,
                }}>{h}</span>
              ))}
            </div>

            {/* Table rows */}
            {filtered.map((item, i) => {
              const disease = item.predictions?.[0]?.disease || item.disease || "—";
              const conf = item.predictions?.[0]?.confidence || item.confidence || "—";
              const prob = item.predictions?.[0]?.probability || "—";
              const symptoms = Array.isArray(item.symptoms)
                ? item.symptoms.join(", ")
                : item.symptoms || "—";

              return (
                <div
                  key={i}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "160px 1fr 200px 120px 120px",
                    padding: "18px 24px",
                    borderBottom: i < filtered.length - 1
                      ? "1px solid rgba(255,255,255,0.04)"
                      : "none",
                    alignItems: "center",
                    transition: "background 0.15s",
                    cursor: "default",
                  }}
                  onMouseEnter={e => e.currentTarget.style.background = "rgba(0,212,200,0.03)"}
                  onMouseLeave={e => e.currentTarget.style.background = "transparent"}
                >
                  {/* Date */}
                  <div>
                    <p style={{ fontSize: 13, color: "#F8FAFC", margin: "0 0 2px", fontWeight: 500 }}>
                      {item.timestamp
                        ? new Date(item.timestamp).toLocaleDateString("en-IN")
                        : "—"}
                    </p>
                    <p style={{ fontSize: 11, color: "#475569", margin: 0 }}>
                      {item.timestamp
                        ? new Date(item.timestamp).toLocaleTimeString("en-IN")
                        : ""}
                    </p>
                  </div>

                  {/* Symptoms */}
                  <div style={{
                    fontSize: 13, color: "#94A3B8",
                    overflow: "hidden", textOverflow: "ellipsis",
                    whiteSpace: "nowrap", paddingRight: 16,
                  }}>
                    {symptoms}
                  </div>

                  {/* Disease */}
                  <div style={{ fontSize: 14, color: "#F8FAFC", fontWeight: 600, paddingRight: 12 }}>
                    {disease}
                  </div>

                  {/* Probability */}
                  <div>
                    {prob !== "—" ? (
                      <div>
                        <p style={{ fontSize: 14, color: "#00D4C8", fontWeight: 700, margin: "0 0 4px" }}>
                          {prob}%
                        </p>
                        <div style={{ height: 3, background: "rgba(255,255,255,0.06)", borderRadius: 100 }}>
                          <div style={{
                            height: 3, width: `${prob}%`,
                            background: "#00D4C8", borderRadius: 100,
                          }} />
                        </div>
                      </div>
                    ) : (
                      <span style={{ color: "#334155" }}>—</span>
                    )}
                  </div>

                  {/* Confidence */}
                  <div>
                    <ConfBadge conf={conf} />
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Summary footer */}
        {!loading && history.length > 0 && (
          <div style={{
            display: "flex", gap: 32, marginTop: 24,
            padding: "20px 28px",
            background: "rgba(255,255,255,0.02)",
            border: "1px solid rgba(255,255,255,0.06)",
            borderRadius: 14,
          }}>
            {[
              { label: "Total Sessions", val: history.length },
              {
                label: "High Confidence",
                val: history.filter(h =>
                  (h.predictions?.[0]?.confidence || h.confidence) === "High"
                ).length,
              },
              {
                label: "Unique Diseases",
                val: new Set(history.map(h =>
                  h.predictions?.[0]?.disease || h.disease
                )).size,
              },
            ].map(({ label, val }) => (
              <div key={label}>
                <p style={{ fontSize: 11, color: "#475569", textTransform: "uppercase", letterSpacing: 1, margin: "0 0 4px" }}>{label}</p>
                <p style={{ fontFamily: "'Syne', sans-serif", fontSize: 24, fontWeight: 800, color: "#F8FAFC", margin: 0 }}>{val}</p>
              </div>
            ))}
          </div>
        )}
      </div>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );
}

export default History;