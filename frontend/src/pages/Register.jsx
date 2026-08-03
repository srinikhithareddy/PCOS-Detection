import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { registerUser } from "../services/api";

function Register() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!name || !email || !password) {
      setError("Please fill in all fields.");
      return;
    }
    if (password.length < 4) {
      setError("Password must be at least 4 characters.");
      return;
    }

    setLoading(true);
    try {
      await registerUser({ name, email, password, role: "clinician" });
      setSuccess(true);
      setTimeout(() => navigate("/login"), 1500);
    } catch (err) {
      if (err.code === "ERR_NETWORK" || !err.response) {
        setError("Cannot reach the server. Please make sure the backend is running.");
      } else if (err.response.status === 400) {
        setError("An account with this email already exists.");
      } else {
        setError("Registration failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <div className="page-container" style={{ maxWidth: "420px" }}>
        <h1 className="page-title">Account Created</h1>
        <div className="card" style={{ padding: "1.5rem" }}>
          <p className="form-success" style={{ display: "inline-block" }}>
            ✓ Your account has been created. Redirecting to login...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container" style={{ maxWidth: "420px" }}>
      <h1 className="page-title">Create Clinician Account</h1>

      <div className="card" style={{ padding: "1.5rem" }}>
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: "1rem" }}>
            <label className="field-label">Full Name</label>
            <input
              type="text"
              className="field-input"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>

          <div style={{ marginBottom: "1rem" }}>
            <label className="field-label">Email</label>
            <input
              type="email"
              className="field-input"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>

          <div style={{ marginBottom: "1rem" }}>
            <label className="field-label">Password</label>
            <input
              type="password"
              className="field-input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          {error && <p className="form-error">{error}</p>}

          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? "Creating account..." : "Create Account"}
          </button>
        </form>

        <p style={{ marginTop: "1rem", fontSize: "0.9rem" }}>
          Already have an account? <Link to="/login">Login here</Link>
        </p>
      </div>
    </div>
  );
}

export default Register;
