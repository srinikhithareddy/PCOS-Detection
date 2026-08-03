import { Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

// Usage: <ProtectedRoute role="admin"><Admin /></ProtectedRoute>
// Omit `role` to just require any authenticated user.
function ProtectedRoute({ children, role }) {
  const { isAuthenticated, user, loading } = useAuth();

  if (loading) {
    return (
      <div className="page-container">
        <p style={{ color: "var(--color-text-muted)" }}>Checking session...</p>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  if (role && user?.role !== role) {
    return (
      <div className="page-container">
        <h1 className="page-title">Access Denied</h1>
        <p className="form-error" style={{ display: "inline-block" }}>
          You do not have permission to view this page.
        </p>
      </div>
    );
  }

  return children;
}

export default ProtectedRoute;
