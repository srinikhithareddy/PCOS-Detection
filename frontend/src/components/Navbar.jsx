import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

function Navbar() {
  const linkClass = ({ isActive }) => (isActive ? "active" : "");
  const { isAuthenticated, user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <nav className="navbar">
      <span className="navbar-brand">PCOS Detect</span>
      <NavLink to="/" className={linkClass} end>
        Home
      </NavLink>
      <NavLink to="/patient-entry" className={linkClass}>
        Patient Entry
      </NavLink>
      <NavLink to="/prediction" className={linkClass}>
        Prediction
      </NavLink>
      <NavLink to="/result" className={linkClass}>
        Result
      </NavLink>
      <NavLink to="/report" className={linkClass}>
        Report
      </NavLink>
      <NavLink to="/feedback" className={linkClass}>
        Feedback
      </NavLink>
      <div className="navbar-right" style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
        {isAuthenticated ? (
          <>
            <span style={{ fontSize: "0.85rem", color: "var(--color-text-muted)" }}>
              {user?.name}
            </span>
            <button className="btn-secondary" onClick={handleLogout} style={{ padding: "0.4rem 0.9rem" }}>
              Logout
            </button>
          </>
        ) : (
          <NavLink to="/login" className={linkClass}>
            Login
          </NavLink>
        )}
      </div>
    </nav>
  );
}

export default Navbar;
