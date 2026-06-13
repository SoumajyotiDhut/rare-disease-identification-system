const colors = ["#00D4C8", "#22C55E", "#A78BFA", "#FB923C", "#F472B6"];

const PredictionCard = ({ item }) => {
  const color = colors[(item.rank - 1) % colors.length];

  return (
    <div style={{
      background: "rgba(255,255,255,0.03)",
      border: `1px solid ${color}22`,
      borderRadius: 16,
      padding: "20px 24px",
      transition: "border-color 0.2s, transform 0.2s",
    }}
      onMouseEnter={e => {
        e.currentTarget.style.borderColor = `${color}55`;
        e.currentTarget.style.transform = "translateY(-2px)";
      }}
      onMouseLeave={e => {
        e.currentTarget.style.borderColor = `${color}22`;
        e.currentTarget.style.transform = "none";
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
        <span style={{
          width: 28, height: 28, borderRadius: 8,
          background: `${color}22`,
          border: `1px solid ${color}44`,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 12, fontWeight: 700, color,
        }}>#{item.rank}</span>
        <h2 style={{
          fontFamily: "'Syne', sans-serif",
          fontSize: 16, fontWeight: 700,
          color: "#F8FAFC", margin: 0,
        }}>{item.disease}</h2>
      </div>

      <div style={{ marginBottom: 10 }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, color: "#64748B", marginBottom: 6 }}>
          <span>Probability</span>
          <span style={{ color, fontWeight: 700 }}>{item.probability}%</span>
        </div>
        <div style={{ height: 4, background: "rgba(255,255,255,0.06)", borderRadius: 100 }}>
          <div style={{
            height: 4, width: `${item.probability}%`,
            background: color, borderRadius: 100,
          }} />
        </div>
      </div>

      <span style={{
        fontSize: 11, fontWeight: 600,
        padding: "3px 10px", borderRadius: 100,
        background: `${color}15`,
        color, border: `1px solid ${color}33`,
      }}>
        {item.confidence}
      </span>
    </div>
  );
};

export default PredictionCard;