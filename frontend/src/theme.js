// Centralized design tokens for light + dark mode.
// Import: import { useTheme } from "../context/ThemeContext";  const { c } = useTheme();

export const LIGHT = {
    bg: "#F4F8FB",
    bgAlt: "#F8FBFD",
    card: "#fff",
    cardAlt: "#F0F5F8",
    text: "#0F1C2E",
    sub: "#5A7184",
    subAlt: "#7A94A8",
    muted: "#9BB8CC",
    faint: "#C8D8E4",
    border: "#E8EFF5",
    borderI: "#DDE8EF",
    navBg: "#fff",
    footerBg: "#0F1C2E",

    teal: "#0B7B6F", tealDark: "#08635A", tealL: "#EBF8F6", tealB: "#B2E8E2",
    blue: "#1D6FA4", blueL: "#EBF4F9", blueB: "#B3D8EE",
    purple: "#5B3DB8", purpL: "#F2EEF9", purpB: "#C8B8EC",
    amber: "#C05B1A", ambL: "#FFF4EC", ambB: "#F5D8B8",
    red: "#B83030", redL: "#FDECED", redB: "#F0BCBC",
    slate: "#8FA5B5", slatL: "#F0F5F8", slatB: "#C8D8E4",
};

export const DARK = {
    bg: "#0A1420",
    bgAlt: "#0E1A28",
    card: "#11202F",
    cardAlt: "#16273A",
    text: "#EAF2F8",
    sub: "#9AB0C2",
    subAlt: "#7E96AB",
    muted: "#5E7A90",
    faint: "#39526A",
    border: "#1E3349",
    borderI: "#24405A",
    navBg: "#0E1A28",
    footerBg: "#060D16",

    teal: "#1FCBB8", tealDark: "#17A695", tealL: "#13302C", tealB: "#1F5C52",
    blue: "#4CA8DE", blueL: "#132A3A", blueB: "#1F4A63",
    purple: "#9D85E8", purpL: "#241C3D", purpB: "#3D2F66",
    amber: "#E8924A", ambL: "#332210", ambB: "#5A3C18",
    red: "#E36767", redL: "#341818", redB: "#5C2A2A",
    slate: "#7E96AB", slatL: "#16273A", slatB: "#2A415A",
};

export const FONTS = `@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');`;