import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider, useTheme } from "./context/ThemeContext";
import { ToastProvider } from "./context/ToastContext";
import PageTransition from "./components/PageTransition";

import Navbar from "./components/Navbar";
import Footer from "./components/Footer";

import Home from "./pages/Home";
import Predict from "./pages/Predict";
import Dashboard from "./pages/Dashboard";
import History from "./pages/History";

/**
 * Inner component so it can call useTheme() — must be inside ThemeProvider.
 * Sets the page background to match the active theme so there's no
 * white/dark flash around the routed content.
 */
function AppShell() {
  const { c } = useTheme();
  return (
    <div style={{ background: c.bg, minHeight: "100vh", transition: "background .25s ease" }}>
      <Navbar />
      <PageTransition>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </PageTransition>
      <Footer />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <ThemeProvider>
        <ToastProvider>
          <AppShell />
        </ToastProvider>
      </ThemeProvider>
    </BrowserRouter>
  );
}