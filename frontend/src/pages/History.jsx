import { useEffect, useState } from "react";
import { getHistory } from "../services/Api";

const pg = {
  minHeight: "100vh",
  background: "#F8FAFB",
  color: "#0F1C2E",
  fontFamily: "'Inter', sans-serif",
  padding: "56px 40px",
};

const ConfBadge = ({ conf }) => {
  const m = {
    High: { bg: "#EBF8F6", color: "#0B7B6F", border: "#B2E8E2" },
    Medium: { bg: "#FFF8EC", color: "#C05B1A", border: "#F5D8B8" },
    Low: { bg: "#FDECED", color: "#B83030", border: "#F0BCBC" },
  };
  const s = m[conf] || m.Low;
  return (
    <span style={{
      padding: "4px 12px", borderRadius: 100,
      fontSize: 11, fontWeight: 700,
      background: s.bg, color: s.color,
      border: `1px solid ${s.border}`,
      textTransform: "uppercase", letterSpacing: 0.6,
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
      <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&display=swap" rel="stylesheet" />
      <div style={{ maxWidth: 1200, margin: "0 auto" }}>

        {/* Header */}
        <div style={{ marginBottom: 44 }}>
          <span style={{
            fontSize: 11, fontWeight: 700, color: "#0B7B6F",
            background: "#EBF8F6", border: "1px solid #B2E8E2",
            padding: "4px 14px", borderRadius: 100, letterSpacing: 1,
            textTransform: "uppercase", display: "inline-block", marginBottom: 16,
          }}>Audit Trail</span>
          <h1 style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif", fontSize: 40, fontWeight: 800, margin: "0 0 10px", color: "#0F1C2E", letterSpacing: -1 }}>
            Prediction History
          </h1>
          <p style={{ color: "#7A94A8", fontSize: 16, margin: 0 }}>
            Browse all previous AI diagnostic sessions and their outcomes.
          </p>
        </div>

        {/* Controls */}
        <div style={{ display: "flex", gap: 12, marginBottom: 24, flexWrap: "wrap", alignItems: "center" }}>
          {/* Search */}
          <div style={{ position: "relative", flex: 1, minWidth: 240 }}>
            <span style={{
              position: "absolute", left: 14, top: "50%",
              transform: "translateY(-50%)", fontSize: 15, color: "#9BB8CC",
              pointerEvents: "none",
            }}>🔍</span>
            <input
              type="text"
              placeholder="Search by disease name…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={{
                width: "100%",
                background: "#fff",
                border: "1.5px solid #DDE8EF",
                borderRadius: 12,
                padding: "12px 16px 12px 42px",
                color: "#0F1C2E",
                fontSize: 14,
                outline: "none",
                fontFamily: "'Inter', sans-serif",
                boxSizing: "border-box",
                transition: "border-color 0.2s, box-shadow 0.2s",
              }}
              onFocus={e => { e.target.style.borderColor = "#0B7B6F"; e.target.style.boxShadow = "0 0 0 3px rgba(11,123,111,0.08)"; }}
              onBlur={e => { e.target.style.borderColor = "#DDE8EF"; e.target.style.boxShadow = "none"; }}
            />
          </div>

          {/* Filter pills */}
          <div style={{ display: "flex", gap: 6 }}>
            {["All", "High", "Medium", "Low"].map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                style={{
                  padding: "10px 18px", borderRadius: 100,
                  border: filter === f ? "1.5px solid #B2E8E2" : "1.5px solid #DDE8EF",
                  background: filter === f ? "#EBF8F6" : "#fff",
                  color: filter === f ? "#0B7B6F" : "#7A94A8",
                  fontSize: 13, fontWeight: 600,
                  cursor: "pointer", fontFamily: "'Inter', sans-serif",
                  transition: "all 0.15s",
                }}
              >
                {f}
              </button>
            ))}
          </div>

          {/* Count badge */}
          <div style={{
            background: "#F0F5F8",
            border: "1px solid #DDE8EF",
            borderRadius: 10, padding: "10px 16px",
            fontSize: 13, color: "#5A7184", fontWeight: 600,
            whiteSpace: "nowrap",
          }}>
            {filtered.length} record{filtered.length !== 1 ? "s" : ""}
          </div>
        </div>

        {/* Table */}
        {loading ? (
          <div style={{ textAlign: "center", padding: "100px 0", color: "#9BB8CC" }}>
            <div style={{
              width: 40, height: 40, border: "3px solid #E8EFF5",
              borderTop: "3px solid #0B7B6F", borderRadius: "50%",
              animation: "spin 0.8s linear infinite", margin: "0 auto 20px",
            }} />
            Loading history…
          </div>
        ) : filtered.length === 0 ? (
          <div style={{
            textAlign: "center", padding: "80px 0",
            background: "#fff", border: "1px solid #E8EFF5", borderRadius: 20,
          }}>
            <div style={{
              width: 72, height: 72, borderRadius: "50%",
              background: "#F0F5F8",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 32, margin: "0 auto 20px",
            }}>📋</div>
            <p style={{ color: "#7A94A8", fontSize: 16, margin: "0 0 6px", fontWeight: 500 }}>
              No records found
            </p>
            <p style={{ color: "#9BB8CC", fontSize: 13 }}>
              {search ? "Try a different search term" : "Make a prediction to see history here"}
            </p>
          </div>
        ) : (
          <div style={{
            background: "#fff",
            border: "1px solid #E8EFF5",
            borderRadius: 20,
            overflow: "hidden",
          }}>
            {/* Table head */}
            <div style={{
              display: "grid",
              gridTemplateColumns: "155px 1fr 210px 120px 120px",
              padding: "14px 24px",
              background: "#F8FAFB",
              borderBottom: "1px solid #EDF2F6",
            }}>
              {["Date & Time", "Symptoms", "Top Diagnosis", "Probability", "Confidence"].map(h => (
                <span key={h} style={{
                  fontSize: 11, fontWeight: 700, color: "#8FA5B5",
                  textTransform: "uppercase", letterSpacing: 0.8,
                }}>{h}</span>
              ))}
            </div>

            {/* Rows */}
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
                    gridTemplateColumns: "155px 1fr 210px 120px 120px",
                    padding: "18px 24px",
                    borderBottom: i < filtered.length - 1 ? "1px solid #F0F5F8" : "none",
                    alignItems: "center",
                    transition: "background 0.15s",
                    cursor: "default",
                  }}
                  onMouseEnter={e => e.currentTarget.style.background = "#FAFCFD"}
                  onMouseLeave={e => e.currentTarget.style.background = "transparent"}
                >
                  <div>
                    <p style={{ fontSize: 13, color: "#0F1C2E", margin: "0 0 2px", fontWeight: 500 }}>
                      {item.timestamp ? new Date(item.timestamp).toLocaleDateString("en-IN") : "—"}
                    </p>
                    <p style={{ fontSize: 11, color: "#9BB8CC", margin: 0 }}>
                      {item.timestamp ? new Date(item.timestamp).toLocaleTimeString("en-IN") : ""}
                    </p>
                  </div>

                  <div style={{
                    fontSize: 13, color: "#7A94A8",
                    overflow: "hidden", textOverflow: "ellipsis",
                    whiteSpace: "nowrap", paddingRight: 16,
                  }}>
                    {symptoms}
                  </div>

                  <div style={{ fontSize: 14, color: "#0F1C2E", fontWeight: 600, paddingRight: 12 }}>
                    {disease}
                  </div>

                  <div>
                    {prob !== "—" ? (
                      <div>
                        <p style={{ fontSize: 14, color: "#0B7B6F", fontWeight: 700, margin: "0 0 5px" }}>
                          {prob}%
                        </p>
                        <div style={{ height: 4, background: "#EDF2F6", borderRadius: 100 }}>
                          <div style={{ height: 4, width: `${prob}%`, background: "#0B7B6F", borderRadius: 100 }} />
                        </div>
                      </div>
                    ) : <span style={{ color: "#C8D8E4" }}>—</span>}
                  </div>

                  <div><ConfBadge conf={conf} /></div>
                </div>
              );
            })}
          </div>
        )}

        {/* Summary footer */}
        {!loading && history.length > 0 && (
          <div style={{
            display: "flex", gap: 0, marginTop: 20,
            background: "#fff",
            border: "1px solid #E8EFF5",
            borderRadius: 16,
            overflow: "hidden",
          }}>
            {[
              { label: "Total Sessions", val: history.length },
              {
                label: "High Confidence",
                val: history.filter(h => (h.predictions?.[0]?.confidence || h.confidence) === "High").length,
              },
              {
                label: "Unique Diseases",
                val: new Set(history.map(h => h.predictions?.[0]?.disease || h.disease)).size,
              },
            ].map(({ label, val }, i) => (
              <div key={label} style={{
                padding: "20px 32px",
                borderRight: i < 2 ? "1px solid #EDF2F6" : "none",
                flex: 1,
              }}>
                <p style={{ fontSize: 11, color: "#8FA5B5", textTransform: "uppercase", letterSpacing: 0.8, margin: "0 0 6px", fontWeight: 700 }}>{label}</p>
                <p style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif", fontSize: 28, fontWeight: 800, color: "#0F1C2E", margin: 0, letterSpacing: -0.5 }}>{val}</p>
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