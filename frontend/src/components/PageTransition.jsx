import { useLocation } from "react-router-dom";
import { useEffect, useState } from "react";

export default function PageTransition({ children }) {
    const location = useLocation();
    const [key, setKey] = useState(location.pathname);

    useEffect(() => {
        const t = setTimeout(() => setKey(location.pathname), 10);
        return () => clearTimeout(t);
    }, [location.pathname]);

    return (
        <div key={key} style={{ animation: "pageIn .35s cubic-bezier(.2,.7,.3,1) both" }}>
            <style>{`
        @keyframes pageIn {
          from { opacity:0; transform:translateY(12px) scale(0.99) }
          to   { opacity:1; transform:translateY(0)   scale(1)    }
        }
      `}</style>
            {children}
        </div>
    );
}