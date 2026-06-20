import { useEffect, useState } from "react";
import { getHistory } from "../services/Api";
import { useTheme } from "../context/ThemeContext";
import { useToast } from "../context/ToastContext";
import HistoryDrawer from "../components/HistoryDrawer";

const CSS = (c) => `
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@700;800&display=swap');
  @keyframes spin   { to { transform:rotate(360deg) } }
  @keyframes fadeUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
  .hrow:hover       { background:${c.bgAlt}!important }
  .filter-btn:hover { border-color:${c.tealB}!important; color:${c.teal}!important }
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

function ConfBadge({ conf, c }) {
  const m = { High: { bg: c.tealL, col: c.teal, b: c.tealB }, Medium: { bg: c.ambL, col: c.amber, b: c.ambB }, Low: { bg: c.redL, col: c.red, b: c.redB } };
  const s = m[conf] || m.Low;
  return <span style={{ padding: "4px 12px", borderRadius: 100, fontSize: 10, fontWeight: 800, background: s.bg, color: s.col, border: `1px solid ${s.b}`, textTransform: "uppercase", letterSpacing: .7 }}>{conf || "—"}</span>;
}

function History() {
  const { c } = useTheme();
  const toast = useToast();
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("All");
  const [sortDesc, setSortDesc] = useState(true);
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState(null);
  const PER_PAGE = 10;

  useEffect(() => {
    getHistory()
      .then(data => {
        if (Array.isArray(data)) setHistory(data);
        else if (data.history) setHistory(data.history);
      })
      .catch(e => { console.error(e); toast.error("Could not load prediction history."); })
      .finally(() => setLoading(false));
  }, []);

  const filtered = history
    .filter(item => {
      const d = item.predictions?.[0]?.disease || item.disease || "";
      const cf = item.predictions?.[0]?.confidence || item.confidence || "";
      return d.toLowerCase().includes(search.toLowerCase()) && (filter === "All" || cf === filter);
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
    <div style={{ minHeight: "100vh", background: c.bg, fontFamily: "'Inter',sans-serif" }}>
      <style>{CSS(c)}</style>
      <div className="hist-pad" style={{ maxWidth: 1200, margin: "0 auto", padding: "56px 32px" }}>

        <div style={{ marginBottom: 40 }}>
          <span style={{ fontSize: 10, fontWeight: 800, color: c.teal, background: c.tealL, border: `1px solid ${c.tealB}`, padding: "4px 14px", borderRadius: 100, letterSpacing: 1.2, textTransform: "uppercase", display: "inline-block", marginBottom: 14 }}>Audit Trail</span>
          <h1 className="hist-h1" style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 38, fontWeight: 800, margin: "0 0 10px", color: c.text, letterSpacing: -1 }}>Prediction History</h1>
          <p style={{ color: c.sub, fontSize: 15, margin: 0 }}>Browse all previous AI diagnostic sessions. Click any row for full details.</p>
        </div>

        {!loading && history.length > 0 && (
          <div className="sum-grid" style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 14, marginBottom: 24 }}>
            {[
              { l: "Total Sessions", v: history.length, ic: "📋", bg: c.tealL, col: c.teal },
              { l: "High Confidence", v: highConf, ic: "🎯", bg: c.blueL, col: c.blue },
              { l: "Unique Diseases", v: uniqCount, ic: "🧬", bg: c.purpL, col: c.purple },
              { l: "Avg Probability", v: `${avgProb}%`, ic: "📊", bg: c.ambL, col: c.amber },
            ].map(({ l, v, ic, bg, col }) => (
              <div key={l} style={{ background: bg, borderRadius: 16, padding: "20px 22px", animation: "fadeUp .5s ease both" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                  <div>
                    <p style={{ fontSize: 10, color: col, opacity: .65, textTransform: "uppercase", letterSpacing: 1, margin: "0 0 8px", fontWeight: 800 }}>{l}</p>
                    <p style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 28, fontWeight: 800, color: col, margin: 0, letterSpacing: -.5 }}>{v}</p>
                  </div>
                  <span style={{ fontSize: 22 }}>{ic}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="ctrl-row" style={{ display: "flex", gap: 10, marginBottom: 18, alignItems: "center", flexWrap: "wrap" }}>
          <div style={{ position: "relative", flex: 1, minWidth: 200 }}>
            <span style={{ position: "absolute", left: 13, top: "50%", transform: "translateY(-50%)", fontSize: 14, color: c.muted, pointerEvents: "none" }}>🔍</span>
            <input type="text" placeholder="Search disease name…" value={search}
              onChange={e => { setSearch(e.target.value); setPage(1); }}
              style={{ width: "100%", background: c.card, border: `1.5px solid ${c.borderI}`, borderRadius: 12, padding: "11px 14px 11px 38px", color: c.text, fontSize: 14, outline: "none", fontFamily: "'Inter',sans-serif", boxSizing: "border-box", transition: "border-color .2s, box-shadow .2s" }}
              onFocus={e => { e.target.style.borderColor = c.teal; e.target.style.boxShadow = `0 0 0 3px ${c.teal}18`; }}
              onBlur={e => { e.target.style.borderColor = c.borderI; e.target.style.boxShadow = "none"; }}
            />
          </div>

          <div className="pills" style={{ display: "flex", gap: 6 }}>
            {["All", "High", "Medium", "Low"].map(f => (
              <button key={f} className="filter-btn" onClick={() => { setFilter(f); setPage(1); }} style={{
                padding: "10px 16px", borderRadius: 100,
                border: filter === f ? `1.5px solid ${c.tealB}` : `1.5px solid ${c.borderI}`,
                background: filter === f ? c.tealL : c.card,
                color: filter === f ? c.teal : c.sub,
                fontSize: 13, fontWeight: 700, cursor: "pointer", fontFamily: "'Inter',sans-serif", transition: "all .15s",
              }}>{f}</button>
            ))}
          </div>

          <button onClick={() => setSortDesc(!sortDesc)} style={{ padding: "10px 14px", borderRadius: 10, border: `1px solid ${c.borderI}`, background: c.card, color: c.sub, fontSize: 13, fontWeight: 600, cursor: "pointer", fontFamily: "'Inter',sans-serif", display: "flex", alignItems: "center", gap: 5, whiteSpace: "nowrap", transition: "all .15s" }}>
            {sortDesc ? "↓" : "↑"} Date
          </button>

          <div style={{ background: c.tealL, border: `1px solid ${c.tealB}`, borderRadius: 10, padding: "10px 15px", fontSize: 13, color: c.teal, fontWeight: 700, whiteSpace: "nowrap" }}>
            {filtered.length} record{filtered.length !== 1 ? "s" : ""}
          </div>
        </div>

        {loading ? (
          <div style={{ textAlign: "center", padding: "100px 0", color: c.muted }}>
            <div style={{ width: 44, height: 44, border: `3px solid ${c.border}`, borderTop: `3px solid ${c.teal}`, borderRadius: "50%", animation: "spin .8s linear infinite", margin: "0 auto 16px" }} />
            <p style={{ margin: 0, fontWeight: 500, fontSize: 15 }}>Loading history…</p>
          </div>
        ) : filtered.length === 0 ? (
          <div style={{ textAlign: "center", padding: "80px 0", background: c.card, border: `1px solid ${c.border}`, borderRadius: 22, animation: "fadeUp .4s ease" }}>
            <div style={{ width: 76, height: 76, borderRadius: "50%", background: c.cardAlt, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 32, margin: "0 auto 18px" }}>📋</div>
            <p style={{ color: c.sub, fontSize: 16, margin: "0 0 6px", fontWeight: 600 }}>No records found</p>
            <p style={{ color: c.muted, fontSize: 13, margin: 0 }}>{search ? "Try a different search term" : "Make a prediction to see history here"}</p>
          </div>
        ) : (
          <>
            <div className="t-wrap" style={{ background: c.card, border: `1px solid ${c.border}`, borderRadius: 22, overflow: "hidden", animation: "fadeUp .4s ease" }}>
              <div className="hist-table">
                <div style={{ display: "grid", gridTemplateColumns: "148px 1fr 200px 110px 110px", padding: "13px 24px", background: c.bgAlt, borderBottom: `1px solid ${c.border}` }}>
                  {["Date & Time", "Symptoms", "Top Diagnosis", "Probability", "Confidence"].map(h => (
                    <span key={h} style={{ fontSize: 10, fontWeight: 800, color: c.muted, textTransform: "uppercase", letterSpacing: .8 }}>{h}</span>
                  ))}
                </div>
                {paginated.map((item, i) => {
                  const disease = item.predictions?.[0]?.disease || item.disease || "—";
                  const conf = item.predictions?.[0]?.confidence || item.confidence || "—";
                  const prob = item.predictions?.[0]?.probability || "—";
                  const symptoms = Array.isArray(item.symptoms) ? item.symptoms.join(", ") : item.symptoms || "—";
                  return (
                    <div key={i} className="hrow" onClick={() => setSelected(item)} style={{
                      display: "grid", gridTemplateColumns: "148px 1fr 200px 110px 110px", padding: "17px 24px",
                      borderBottom: i < paginated.length - 1 ? `1px solid ${c.border}` : "none", alignItems: "center",
                      transition: "background .12s", cursor: "pointer",
                    }}>
                      <div>
                        <p style={{ fontSize: 13, color: c.text, margin: "0 0 2px", fontWeight: 600 }}>{item.timestamp ? new Date(item.timestamp).toLocaleDateString("en-IN") : "—"}</p>
                        <p style={{ fontSize: 11, color: c.muted, margin: 0 }}>{item.timestamp ? new Date(item.timestamp).toLocaleTimeString("en-IN") : ""}</p>
                      </div>
                      <div style={{ fontSize: 13, color: c.sub, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", paddingRight: 16 }}>{symptoms}</div>
                      <div style={{ fontSize: 13, color: c.text, fontWeight: 700, paddingRight: 12 }}>{disease}</div>
                      <div>
                        {prob !== "—" ? (
                          <>
                            <p style={{ fontSize: 14, color: c.teal, fontWeight: 800, margin: "0 0 5px", fontFamily: "'Plus Jakarta Sans',sans-serif" }}>{prob}%</p>
                            <div style={{ height: 4, background: c.border, borderRadius: 100 }}>
                              <div style={{ height: 4, width: `${prob}%`, background: c.teal, borderRadius: 100 }} />
                            </div>
                          </>
                        ) : <span style={{ color: c.faint }}>—</span>}
                      </div>
                      <div><ConfBadge conf={conf} c={c} /></div>
                    </div>
                  );
                })}
              </div>
            </div>

            {pages > 1 && (
              <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: 8, marginTop: 20 }}>
                <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1} style={{ padding: "8px 16px", borderRadius: 10, border: `1px solid ${c.borderI}`, background: c.card, color: page === 1 ? c.faint : c.sub, cursor: page === 1 ? "default" : "pointer", fontWeight: 600, fontSize: 13, fontFamily: "'Inter',sans-serif" }}>← Prev</button>
                {Array.from({ length: pages }, (_, i) => i + 1).map(p => (
                  <button key={p} onClick={() => setPage(p)} style={{ width: 36, height: 36, borderRadius: 9, border: p === page ? `1.5px solid ${c.tealB}` : `1px solid ${c.borderI}`, background: p === page ? c.tealL : c.card, color: p === page ? c.teal : c.sub, cursor: "pointer", fontWeight: 700, fontSize: 13, fontFamily: "'Inter',sans-serif" }}>{p}</button>
                ))}
                <button onClick={() => setPage(p => Math.min(pages, p + 1))} disabled={page === pages} style={{ padding: "8px 16px", borderRadius: 10, border: `1px solid ${c.borderI}`, background: c.card, color: page === pages ? c.faint : c.sub, cursor: page === pages ? "default" : "pointer", fontWeight: 600, fontSize: 13, fontFamily: "'Inter',sans-serif" }}>Next →</button>
              </div>
            )}
          </>
        )}
      </div>

      {selected && <HistoryDrawer item={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}

export default History;