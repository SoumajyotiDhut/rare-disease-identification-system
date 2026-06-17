const PALETTE = [
  { bar: "#0B7B6F", badge: "#EBF8F6", badgeText: "#0B7B6F", border: "#B2E8E2" },
  { bar: "#1D6FA4", badge: "#EBF4F9", badgeText: "#1D6FA4", border: "#B3D8EE" },
  { bar: "#5B3DB8", badge: "#F2EEF9", badgeText: "#5B3DB8", border: "#C8B8EC" },
  { bar: "#C05B1A", badge: "#FFF4EC", badgeText: "#C05B1A", border: "#F5D8B8" },
  { bar: "#8FA5B5", badge: "#F0F5F8", badgeText: "#5A7184", border: "#C8D8E4" },
];

const PredictionCard = ({ item }) => {
  const c = PALETTE[(item.rank - 1) % PALETTE.length];

  return (
    <div
      style={{
        background: "#fff",
        border: `1px solid ${c.border}`,
        borderRadius: 16,
        padding: "20px 24px",
        transition: "box-shadow 0.2s, transform 0.15s",
        cursor: "default",
      }}
      onMouseEnter={e => {
        e.currentTarget.style.boxShadow = "0 6px 24px rgba(15,28,46,0.08)";
        e.currentTarget.style.transform = "translateY(-2px)";
      }}
      onMouseLeave={e => {
        e.currentTarget.style.boxShadow = "none";
        e.currentTarget.style.transform = "none";
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
        <span style={{
          width: 30, height: 30, borderRadius: 8,
          background: c.badge,
          border: `1px solid ${c.border}`,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 11, fontWeight: 800, color: c.badgeText,
          flexShrink: 0,
        }}>#{item.rank}</span>
        <h2 style={{
          fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif",
          fontSize: 15, fontWeight: 700,
          color: "#0F1C2E", margin: 0,
        }}>{item.disease}</h2>
      </div>

      <div style={{ marginBottom: 12 }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#8FA5B5", marginBottom: 7, fontWeight: 500 }}>
          <span>Probability</span>
          <span style={{ color: c.badgeText, fontWeight: 700 }}>{item.probability}%</span>
        </div>
        <div style={{ height: 5, background: "#F0F5F8", borderRadius: 100 }}>
          <div style={{
            height: 5, width: `${item.probability}%`,
            background: c.bar, borderRadius: 100,
            transition: "width 0.8s ease",
          }} />
        </div>
      </div>

      <span style={{
        fontSize: 11, fontWeight: 700,
        padding: "4px 12px", borderRadius: 100,
        background: c.badge,
        color: c.badgeText,
        border: `1px solid ${c.border}`,
        textTransform: "uppercase", letterSpacing: 0.6,
      }}>
        {item.confidence}
      </span>
    </div>
  );
};

export default PredictionCard;