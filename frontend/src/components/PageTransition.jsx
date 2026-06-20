import { useLocation } from "react-router-dom";
import { useEffect, useState } from "react";

/**
 * Wrap your <Routes> children (or each page) with this to get a soft
 * fade + rise transition whenever the route changes.
 *
 * Usage in App.jsx:
 *   <PageTransition><Routes>...</Routes></PageTransition>
 * or wrap each individual page element.
 */
export default function PageTransition({ children }) {
    const location = useLocation();
    const [visible, setVisible] = useState(true);

    useEffect(() => {
        setVisible(false);
        const t = setTimeout(() => setVisible(true), 20);
        return () => clearTimeout(t);
    }, [location.pathname]);

    return (
        <div
            key={location.pathname}
            style={{
                animation: visible ? "pageIn .38s cubic-bezier(.2,.7,.3,1) both" : "none",
            }}
        >
            <style>{`
        @keyframes pageIn {
          from { opacity: 0; transform: translateY(10px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
            {children}
        </div>
    );
}