import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import Loader from "./Loader";

export default function ProtectedRoute({ children }) {
    const { user, authLoading } = useAuth();
    const location = useLocation();

    if (authLoading) return <Loader message="Verifying your session…" />;
    if (!user) return <Navigate to="/login" state={{ from: location }} replace />;
    return children;
}