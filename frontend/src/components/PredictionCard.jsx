import { useTheme } from "../context/ThemeContext";

const PredictionCard = ({ item }) => {
  const { c } = useTheme();
  const PALETTE = [
    { bar: c.teal, light: c.tealL, text: c.teal, border: c.tealB },
    { bar: c.blue, light: c.blueL, text: c.blue, border: c.blueB },
    { bar: c.purple, light: c.purpL, text: c.purple, border: c.purpB },
    { bar: c.amber, light: c.ambL, text: c.amber, border: c.ambB },
    { bar: c.slate, light: c.slatL, text: c.sub, border: c.slatB },
  ];
  const p = PALETTE[(item.rank - 1) % PALETTE.length];
  return (
    <div style={{ background: c.card, border: `1px solid ${p.border}`, borderRadius: 18, padding: "22px 24px", transition: "box-shadow .2s, transform .15s", cursor: "default" }}
      onMouseEnter={e => { e.currentTarget.style.boxShadow = "0 8px 28px rgba(0,0,0,0.12)"; e.currentTarget.style.transform = "translateY(-3px)" }}
      onMouseLeave={e => { e.currentTarget.style.boxShadow = "none"; e.currentTarget.style.transform = "none" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
        <span style={{ width: 32, height: 32, borderRadius: 9, background: p.light, border: `1px solid ${p.border}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 800, color: p.text, flexShrink: 0 }}>#{item.rank}</span>
        <h2 style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", fontSize: 15, fontWeight: 700, color: c.text, margin: 0 }}>{item.disease}</h2>
      </div>
      <div style={{ marginBottom: 13 }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 7 }}>
          <span style={{ color: c.muted, fontWeight: 500 }}>Probability</span>
          <span style={{ color: p.text, fontWeight: 800 }}>{item.probability}%</span>
        </div>
        <div style={{ height: 6, background: c.border, borderRadius: 100 }}>
          <div style={{ height: 6, width: `${item.probability}%`, background: p.bar, borderRadius: 100, transition: "width .9s ease" }} />
        </div>
      </div>
      <span style={{ fontSize: 10, fontWeight: 800, padding: "4px 12px", borderRadius: 100, background: p.light, color: p.text, border: `1px solid ${p.border}`, textTransform: "uppercase", letterSpacing: .7 }}>{item.confidence}</span>
    </div>
  );
};
export default PredictionCard;