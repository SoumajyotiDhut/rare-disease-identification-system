import { useTheme } from "../context/ThemeContext";

const PALETTE = (c) => [
  { bar: c.teal, light: c.tealL, text: c.teal, border: c.tealB },
  { bar: c.blue, light: c.blueL, text: c.blue, border: c.blueB },
  { bar: c.purple, light: c.purpL, text: c.purple, border: c.purpB },
  { bar: c.amber, light: c.ambL, text: c.amber, border: c.ambB },
  { bar: c.slate, light: c.slatL, text: c.sub, border: c.slatB },
];

export default function PredictionCard({ item }) {
  const { c } = useTheme();
  const pal = PALETTE(c)[(item.rank - 1) % 5];
  const isTop = item.rank === 1;

  return (
    <div style={{
      position: "relative",
      background: c.card,
      border: `1px solid ${isTop ? c.tealB : c.border}`,
      borderTop: `2px solid ${isTop ? c.teal : "transparent"}`,
      padding: "22px 24px",
      transition: "transform .2s ease, box-shadow .2s ease",
      overflow: "hidden",
      cursor: "default",
    }}
      onMouseEnter={e => {
        e.currentTarget.style.transform = "translateY(-3px)";
        e.currentTarget.style.boxShadow = isTop ? c.shadowTeal : c.shadowMd;
      }}
      onMouseLeave={e => {
        e.currentTarget.style.transform = "translateY(0)";
        e.currentTarget.style.boxShadow = "none";
      }}
    >
      {/* Top row: rank + disease name */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: 14, marginBottom: 16 }}>
        <span style={{
          fontFamily: "'Fraunces',serif", fontStyle: "italic", fontWeight: 600,
          fontSize: 20, color: pal.text, width: 30, flexShrink: 0, lineHeight: 1.2,
        }}>
          {item.rank}
        </span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <h2 style={{
            fontFamily: "'Fraunces',serif",
            fontSize: 16, fontWeight: 600, color: c.text,
            margin: 0, lineHeight: 1.3,
            overflow: "hidden", textOverflow: "ellipsis",
            display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical",
          }}>{item.disease}</h2>
          {item.orpha_code && (
            <span style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 10.5, color: c.muted, fontWeight: 500 }}>
              ORPHA:{item.orpha_code}
            </span>
          )}
        </div>

        {isTop && (
          <span style={{
            fontFamily: "'IBM Plex Mono',monospace",
            fontSize: 9, fontWeight: 600, color: c.gold,
            border: `1px solid ${c.goldB}`,
            padding: "3px 9px",
            letterSpacing: "0.08em", textTransform: "uppercase", flexShrink: 0,
          }}>Top Match</span>
        )}
      </div>

      {/* Probability bar */}
      <div style={{ marginBottom: 14 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 8 }}>
          <span style={{ fontSize: 11, color: c.muted, fontWeight: 500, letterSpacing: "0.05em", textTransform: "uppercase" }}>
            Probability
          </span>
          <span style={{
            fontFamily: "'IBM Plex Mono',monospace",
            fontSize: 19, fontWeight: 600, color: pal.text, lineHeight: 1,
          }}>{item.probability}%</span>
        </div>
        <div style={{ height: 4, background: c.border, borderRadius: 100, overflow: "hidden" }}>
          <div style={{
            height: 4, width: `${item.probability}%`,
            background: pal.bar,
            borderRadius: 100,
            transition: "width 1.2s cubic-bezier(.2,.7,.3,1)",
          }} />
        </div>
      </div>

      {/* Confidence + mode */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{
          fontSize: 10.5, fontWeight: 700, padding: "5px 13px", borderRadius: 100,
          background: pal.light, color: pal.text,
          border: `1px solid ${pal.border}`,
          textTransform: "uppercase", letterSpacing: "0.06em",
        }}>{item.confidence}</span>

        {item.mode && (
          <span style={{
            fontFamily: "'IBM Plex Mono',monospace",
            fontSize: 10, color: c.muted, fontWeight: 500,
            background: c.cardAlt, border: `1px solid ${c.border}`,
            padding: "4px 10px", borderRadius: 100, letterSpacing: "0.03em",
          }}>
            {item.mode === "multimodal_fusion" ? "Fusion" : "Symptoms"}
          </span>
        )}
      </div>
    </div>
  );
}