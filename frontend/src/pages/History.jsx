import { useEffect, useState } from "react";
import { getHistory } from "../services/Api";

const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&display=swap');
  @keyframes spin   { to { transform:rotate(360deg) } }
  @keyframes fadeUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
  .hrow:hover       { background:#FAFCFD!important }
  .filter-btn:hover { border-color:#B2E8E2!important; color:#0B7B6F!important }
  @media(max-width:900px){
    .hist-pad  { padding:40px 20px!important }
    .ctrl-row  { flex-direction:column!important; align-items:stretch!important }
    .sum-grid  { grid-template-columns:1fr 1fr!important }
    .t-wrap    { overflow-x:auto }
    .hist-h1   { font-size:30px!important }
    .hist-table{ min-width:700px }
  }
  @media(max-width:520px){
    .sum-grid  { grid-template-columns:1fr!important }
    .pills     { flex-wrap:wrap!important }
  }
`;

const ConfBadge = ({ conf }) => {
  const m = { High: { bg: "#EBF8F6", c: "#0B7B6F", b: "#B2E8E2" }, Medium: { bg: "#FFF8EC", c: "#C05B1A", b: "#F5D8B8" }, Low: { bg: "#FDECED", c: "#B83030", b: "#F0BCBC" } };
  const s = m[conf] || m.Low;
  return <span style={{ padding: "4px 12px", borderRadius: 100, fontSize: 10, fontWeight: 800, background: s.bg, color: s.c, border: `1px solid ${s.b}`, textTransform: "uppercase", letterSpacing: .7 }}>{conf || "—"}</span>;
};

function History() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("All");
  const [sortDesc, setSortDesc] = useState(true);
  const [page, setPage] = useState(1);
  const PER_PAGE = 10;

  useEffect(() => {
    getHistory()
      .then(data => { if (Array.isArray(data)) setHistory(data); else if (data.history) setHistory(data.history); })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const filtered = history
    .filter(item => {
      const d = item.predictions?.[0]?.disease || item.disease || "";
      const c = item.predictions?.[0]?.confidence || item.confidence || "";
      return d.toLowerCase().includes(search.toLowerCase()) && (filter === "All" || c === filter);
    })
    .sort((a, b) => {
      const ta = a.timestamp ? new Date(a.timestamp) : 0;
      const tb = b.timestamp ? new Date(b.timestamp) : 0;
      return sortDesc ? tb - ta : ta - tb;
    });

  const pages = Math.ceil(filtered.length / PER_PAGE);
  const paginated = filtered.slice((page - 1) * PER_PAGE, page * PER_PAGE);

  const highConf = history.filter(h => (h.predictions?.[0]?.confidence || h.confidence) === "High").length;
  const avgProb = history.length ? Math.round(history.reduce((a, h) => a + (h.predictions?.[0]?.probability || 0), 0) / history.length) : 0;
  const uniqCount = new Set(history.map(h => h.predictions?.[0]?.disease || h.disease)).size;

  return (
    <div style={{ minHeight: "100vh", background: "#F4F8FB", fontFamily: "'Inter',sans-serif" }}>
      <style>{CSS}</style>
      <div className="hist-pad" style={{ maxWidth: 1200, margin: "0 auto", padding: "56px 32px" }}>

        {/* Header */}
        <div style={{ marginBottom: 40 }}>
          <span style={{ fontSize: 10, fontWeight: 800, color: "#0B7B6F", background: "#EBF8F6", border: "1px solid #B2E8E2", padding: "4px 14px", borderRadius: 100, letterSpacing: 1.2, textTransform: "uppercase", display: "inline-block", marginBottom: 14 }}>Audit Trail</span>
          <h1 className="hist-h1" style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 38, fontWeight: 800, margin: "0 0 10px", color: "#0F1C2E", letterSpacing: -1 }}>Prediction History</h1>
          <p style={{ color: "#7A94A8", fontSize: 15, margin: 0 }}>Browse all previous AI diagnostic sessions and outcomes.</p>
        </div>

        {/* Summary stats */}
        {!loading && history.length > 0 && (
          <div className="sum-grid" style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 14, marginBottom: 24 }}>
            {[
              { l: "Total Sessions", v: history.length, ic: "📋", bg: "#EBF8F6", c: "#0B7B6F" },
              { l: "High Confidence", v: highConf, ic: "🎯", bg: "#EBF4F9", c: "#1D6FA4" },
              { l: "Unique Diseases", v: uniqCount, ic: "🧬", bg: "#F2EEF9", c: "#5B3DB8" },
              { l: "Avg Probability", v: `${avgProb}%`, ic: "📊", bg: "#FFF4EC", c: "#C05B1A" },
            ].map(({ l, v, ic, bg, c }) => (
              <div key={l} style={{ background: bg, borderRadius: 16, padding: "20px 22px", animation: "fadeUp .5s ease both" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                  <div>
                    <p style={{ fontSize: 10, color: c, opacity: .65, textTransform: "uppercase", letterSpacing: 1, margin: "0 0 8px", fontWeight: 800 }}>{l}</p>
                    <p style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 28, fontWeight: 800, color: c, margin: 0, letterSpacing: -.5 }}>{v}</p>
                  </div>
                  <span style={{ fontSize: 22 }}>{ic}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Controls */}
        <div className="ctrl-row" style={{ display: "flex", gap: 10, marginBottom: 18, alignItems: "center", flexWrap: "wrap" }}>
          <div style={{ position: "relative", flex: 1, minWidth: 200 }}>
            <span style={{ position: "absolute", left: 13, top: "50%", transform: "translateY(-50%)", fontSize: 14, color: "#9BB8CC", pointerEvents: "none" }}>🔍</span>
            <input type="text" placeholder="Search disease name…" value={search}
              onChange={e => { setSearch(e.target.value); setPage(1); }}
              style={{ width: "100%", background: "#fff", border: "1.5px solid #DDE8EF", borderRadius: 12, padding: "11px 14px 11px 38px", color: "#0F1C2E", fontSize: 14, outline: "none", fontFamily: "'Inter',sans-serif", boxSizing: "border-box", transition: "border-color .2s, box-shadow .2s" }}
              onFocus={e => { e.target.style.borderColor = "#0B7B6F"; e.target.style.boxShadow = "0 0 0 3px rgba(11,123,111,0.08)"; }}
              onBlur={e => { e.target.style.borderColor = "#DDE8EF"; e.target.style.boxShadow = "none"; }}
            />
          </div>

          <div className="pills" style={{ display: "flex", gap: 6 }}>
            {["All", "High", "Medium", "Low"].map(f => (
              <button key={f} className="filter-btn" onClick={() => { setFilter(f); setPage(1); }} style={{
                padding: "10px 16px", borderRadius: 100,
                border: filter === f ? "1.5px solid #B2E8E2" : "1.5px solid #DDE8EF",
                background: filter === f ? "#EBF8F6" : "#fff",
                color: filter === f ? "#0B7B6F" : "#7A94A8",
                fontSize: 13, fontWeight: 700, cursor: "pointer", fontFamily: "'Inter',sans-serif", transition: "all .15s",
              }}>{f}</button>
            ))}
          </div>

          <button onClick={() => setSortDesc(!sortDesc)} style={{ padding: "10px 14px", borderRadius: 10, border: "1px solid #DDE8EF", background: "#fff", color: "#5A7184", fontSize: 13, fontWeight: 600, cursor: "pointer", fontFamily: "'Inter',sans-serif", display: "flex", alignItems: "center", gap: 5, whiteSpace: "nowrap", transition: "all .15s" }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = "#B2E8E2"; e.currentTarget.style.color = "#0B7B6F"; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = "#DDE8EF"; e.currentTarget.style.color = "#5A7184"; }}>
            {sortDesc ? "↓" : "↑"} Date
          </button>

          <div style={{ background: "#EBF8F6", border: "1px solid #B2E8E2", borderRadius: 10, padding: "10px 15px", fontSize: 13, color: "#0B7B6F", fontWeight: 700, whiteSpace: "nowrap" }}>
            {filtered.length} record{filtered.length !== 1 ? "s" : ""}
          </div>
        </div>

        {/* Table */}
        {loading ? (
          <div style={{ textAlign: "center", padding: "100px 0", color: "#8FA5B5" }}>
            <div style={{ width: 44, height: 44, border: "3px solid #E8EFF5", borderTop: "3px solid #0B7B6F", borderRadius: "50%", animation: "spin .8s linear infinite", margin: "0 auto 16px" }} />
            <p style={{ margin: 0, fontWeight: 500, fontSize: 15 }}>Loading history…</p>
          </div>
        ) : filtered.length === 0 ? (
          <div style={{ textAlign: "center", padding: "80px 0", background: "#fff", border: "1px solid #E8EFF5", borderRadius: 22, animation: "fadeUp .4s ease" }}>
            <div style={{ width: 76, height: 76, borderRadius: "50%", background: "#F0F5F8", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 32, margin: "0 auto 18px" }}>📋</div>
            <p style={{ color: "#7A94A8", fontSize: 16, margin: "0 0 6px", fontWeight: 600 }}>No records found</p>
            <p style={{ color: "#9BB8CC", fontSize: 13, margin: 0 }}>{search ? "Try a different search term" : "Make a prediction to see history here"}</p>
          </div>
        ) : (
          <>
            <div className="t-wrap" style={{ background: "#fff", border: "1px solid #E8EFF5", borderRadius: 22, overflow: "hidden", animation: "fadeUp .4s ease" }}>
              <div className="hist-table">
                {/* Head */}
                <div style={{ display: "grid", gridTemplateColumns: "148px 1fr 200px 110px 110px", padding: "13px 24px", background: "#F8FBFD", borderBottom: "1px solid #EDF2F6" }}>
                  {["Date & Time", "Symptoms", "Top Diagnosis", "Probability", "Confidence"].map(h => (
                    <span key={h} style={{ fontSize: 10, fontWeight: 800, color: "#8FA5B5", textTransform: "uppercase", letterSpacing: .8 }}>{h}</span>
                  ))}
                </div>
                {/* Rows */}
                {paginated.map((item, i) => {
                  const disease = item.predictions?.[0]?.disease || item.disease || "—";
                  const conf = item.predictions?.[0]?.confidence || item.confidence || "—";
                  const prob = item.predictions?.[0]?.probability || "—";
                  const symptoms = Array.isArray(item.symptoms) ? item.symptoms.join(", ") : item.symptoms || "—";
                  return (
                    <div key={i} className="hrow" style={{ display: "grid", gridTemplateColumns: "148px 1fr 200px 110px 110px", padding: "17px 24px", borderBottom: i < paginated.length - 1 ? "1px solid #F0F5F8" : "none", alignItems: "center", transition: "background .12s", cursor: "default", animationDelay: `${i * .04}s` }}>
                      <div>
                        <p style={{ fontSize: 13, color: "#0F1C2E", margin: "0 0 2px", fontWeight: 600 }}>{item.timestamp ? new Date(item.timestamp).toLocaleDateString("en-IN") : "—"}</p>
                        <p style={{ fontSize: 11, color: "#9BB8CC", margin: 0 }}>{item.timestamp ? new Date(item.timestamp).toLocaleTimeString("en-IN") : ""}</p>
                      </div>
                      <div style={{ fontSize: 13, color: "#7A94A8", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", paddingRight: 16 }}>{symptoms}</div>
                      <div style={{ fontSize: 13, color: "#0F1C2E", fontWeight: 700, paddingRight: 12 }}>{disease}</div>
                      <div>
                        {prob !== "—" ? (
                          <>
                            <p style={{ fontSize: 14, color: "#0B7B6F", fontWeight: 800, margin: "0 0 5px", fontFamily: "'Plus Jakarta Sans',sans-serif" }}>{prob}%</p>
                            <div style={{ height: 4, background: "#EDF2F6", borderRadius: 100 }}>
                              <div style={{ height: 4, width: `${prob}%`, background: "#0B7B6F", borderRadius: 100 }} />
                            </div>
                          </>
                        ) : <span style={{ color: "#C8D8E4" }}>—</span>}
                      </div>
                      <div><ConfBadge conf={conf} /></div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Pagination */}
            {pages > 1 && (
              <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: 8, marginTop: 20 }}>
                <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1} style={{ padding: "8px 16px", borderRadius: 10, border: "1px solid #DDE8EF", background: "#fff", color: page === 1 ? "#C8D8E4" : "#5A7184", cursor: page === 1 ? "default" : "pointer", fontWeight: 600, fontSize: 13, fontFamily: "'Inter',sans-serif", transition: "all .15s" }}>← Prev</button>
                {Array.from({ length: pages }, (_, i) => i + 1).map(p => (
                  <button key={p} onClick={() => setPage(p)} style={{ width: 36, height: 36, borderRadius: 9, border: p === page ? "1.5px solid #B2E8E2" : "1px solid #DDE8EF", background: p === page ? "#EBF8F6" : "#fff", color: p === page ? "#0B7B6F" : "#7A94A8", cursor: "pointer", fontWeight: 700, fontSize: 13, fontFamily: "'Inter',sans-serif", transition: "all .15s" }}>{p}</button>
                ))}
                <button onClick={() => setPage(p => Math.min(pages, p + 1))} disabled={page === pages} style={{ padding: "8px 16px", borderRadius: 10, border: "1px solid #DDE8EF", background: "#fff", color: page === pages ? "#C8D8E4" : "#5A7184", cursor: page === pages ? "default" : "pointer", fontWeight: 600, fontSize: 13, fontFamily: "'Inter',sans-serif", transition: "all .15s" }}>Next →</button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default History;