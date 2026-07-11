/**
 * AI DOC — Premium Clinical Design System
 * ----------------------------------------
 * Palette concept: "Clinical Editorial" — a warm porcelain / deep ink base
 * (like a well-printed medical journal) with a refined deep-teal as the
 * working clinical color, and a muted brass/gold as the premium accent
 * reserved for hairlines, dividers and signature marks. Display type is
 * a serif (Fraunces) for editorial gravitas; body stays on Inter for
 * clarity; data/stats use a mono face for a precise, instrument-panel feel.
 *
 * All existing token names are preserved so components don't need to change
 * their `c.*` references — only the values changed.
 */

export const FONT_DISPLAY = "'Fraunces', 'Georgia', serif";
export const FONT_BODY = "'Inter', sans-serif";
export const FONT_MONO = "'IBM Plex Mono', 'SFMono-Regular', monospace";

export const LIGHT = {
    // Surfaces
    bg: "#F7F4EE",
    bgAlt: "#F1ECE2",
    bgDeep: "#EBE5D8",
    card: "#FFFFFF",
    cardAlt: "#F6F2E9",

    // Borders
    border: "rgba(15,23,35,0.09)",
    borderI: "rgba(15,23,35,0.15)",

    // Text
    text: "#12181F",
    sub: "#4B5563",
    subAlt: "#5B6472",
    muted: "#8B8C86",
    faint: "#C9C4B5",

    // Clinical teal (primary)
    teal: "#0B6B5F",
    tealL: "rgba(11,107,95,0.09)",
    tealB: "rgba(11,107,95,0.28)",

    // Blue
    blue: "#2C4F73",
    blueL: "rgba(44,79,115,0.09)",
    blueB: "rgba(44,79,115,0.26)",

    // Purple
    purple: "#5F4B7F",
    purpL: "rgba(95,75,127,0.09)",
    purpB: "rgba(95,75,127,0.26)",

    // Amber
    amber: "#9C6B18",
    ambL: "rgba(156,107,24,0.11)",
    ambB: "rgba(156,107,24,0.28)",

    // Red
    red: "#A63B34",
    redL: "rgba(166,59,52,0.09)",
    redB: "rgba(166,59,52,0.26)",

    // Slate
    slate: "#767F8C",
    slatL: "rgba(118,127,140,0.09)",
    slatB: "rgba(118,127,140,0.24)",

    // Gold — premium signature accent, used sparingly
    gold: "#A5813F",
    goldL: "rgba(165,129,63,0.12)",
    goldB: "rgba(165,129,63,0.36)",

    // Gradients
    gradPrimary: "linear-gradient(135deg,#0E8577 0%,#0B6B5F 55%,#0A4F47 100%)",
    gradPurple: "linear-gradient(135deg,#6C5590 0%,#4A3866 100%)",
    gradAmber: "linear-gradient(135deg,#B4832E 0%,#8A5F16 100%)",
    gradGold: "linear-gradient(135deg,#C9A961 0%,#9C7B3F 100%)",
    gradHero: "linear-gradient(180deg,#F9F7F1 0%,#F1ECE2 100%)",

    // Shadows
    shadowSm: "0 2px 8px rgba(15,23,35,0.06)",
    shadowMd: "0 10px 28px rgba(15,23,35,0.08)",
    shadowLg: "0 24px 56px rgba(15,23,35,0.10)",
    shadowXl: "0 36px 80px rgba(15,23,35,0.13)",
    shadowTeal: "0 14px 34px rgba(11,107,95,0.24)",

    // Glass
    glass: "rgba(255,255,255,0.70)",
    glassBlur: "blur(16px)",
    glassBorder: "rgba(15,23,35,0.08)",

    fontDisplay: FONT_DISPLAY,
    fontBody: FONT_BODY,
    fontMono: FONT_MONO,
};

export const DARK = {
    bg: "#0A1219",
    bgAlt: "#0E1720",
    bgDeep: "#070D13",
    card: "#101B24",
    cardAlt: "#15212B",

    border: "rgba(255,255,255,0.07)",
    borderI: "rgba(255,255,255,0.13)",

    text: "#F3F1EA",
    sub: "#AAB2B8",
    subAlt: "#8B959E",
    muted: "#67707A",
    faint: "#3A4249",

    teal: "#33B39F",
    tealL: "rgba(51,179,159,0.13)",
    tealB: "rgba(51,179,159,0.32)",

    blue: "#7DA3C9",
    blueL: "rgba(125,163,201,0.13)",
    blueB: "rgba(125,163,201,0.30)",

    purple: "#A692C9",
    purpL: "rgba(166,146,201,0.13)",
    purpB: "rgba(166,146,201,0.30)",

    amber: "#D6A24C",
    ambL: "rgba(214,162,76,0.13)",
    ambB: "rgba(214,162,76,0.30)",

    red: "#D67F79",
    redL: "rgba(214,127,121,0.13)",
    redB: "rgba(214,127,121,0.30)",

    slate: "#8D96A1",
    slatL: "rgba(141,150,161,0.13)",
    slatB: "rgba(141,150,161,0.28)",

    gold: "#CBAF74",
    goldL: "rgba(203,175,116,0.13)",
    goldB: "rgba(203,175,116,0.34)",

    gradPrimary: "linear-gradient(135deg,#1FA592 0%,#0F7C6C 55%,#0A5A50 100%)",
    gradPurple: "linear-gradient(135deg,#A692C9 0%,#6C5590 100%)",
    gradAmber: "linear-gradient(135deg,#D6A24C 0%,#9C6B18 100%)",
    gradGold: "linear-gradient(135deg,#E2C989 0%,#A5813F 100%)",
    gradHero: "linear-gradient(180deg,#0A1219 0%,#0E1720 100%)",

    shadowSm: "0 2px 10px rgba(0,0,0,0.30)",
    shadowMd: "0 12px 28px rgba(0,0,0,0.36)",
    shadowLg: "0 26px 58px rgba(0,0,0,0.42)",
    shadowXl: "0 38px 84px rgba(0,0,0,0.50)",
    shadowTeal: "0 16px 38px rgba(31,165,146,0.30)",

    glass: "rgba(10,18,25,0.66)",
    glassBlur: "blur(16px)",
    glassBorder: "rgba(255,255,255,0.08)",

    fontDisplay: FONT_DISPLAY,
    fontBody: FONT_BODY,
    fontMono: FONT_MONO,
};