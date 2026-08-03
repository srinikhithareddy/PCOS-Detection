import { useState } from "react";
import { useNavigate } from "react-router-dom";

function Feedback() {
  const [verifiedLabel, setVerifiedLabel] = useState("");
  const [comments, setComments] = useState("");
  const [error, setError] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    setError("");

    if (!verifiedLabel) {
      setError("Please select the verified assessment before submitting.");
      return;
    }

    // TEMPORARY: no backend yet — log locally
    console.log("Clinician feedback:", { verifiedLabel, comments });
    setSubmitted(true);
  };

  if (submitted) {
    return (
      <div className="page-container">
        <h1 className="page-title">Feedback Submitted</h1>
        <div className="card" style={{ padding: "1.5rem" }}>
          <p className="form-success" style={{ display: "inline-block" }}>
            ✓ Thank you. Your verified feedback has been recorded and linked to this
            prediction for future model review.
          </p>
          <div style={{ marginTop: "1rem" }}>
            <button className="btn-primary" onClick={() => navigate("/")}>
              Return to Home
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container">
      <h1 className="page-title">Clinician Feedback</h1>
      <p style={{ color: "var(--color-text-muted)", marginBottom: "1.5rem" }}>
        Review the AI prediction and provide your verified clinical assessment. This
        feedback helps validate and improve the system over time.
      </p>

      <div className="card" style={{ padding: "1.5rem" }}>
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: "1.25rem" }}>
            <label className="field-label">Verified Assessment</label>
            <select
              className="field-select"
              value={verifiedLabel}
              onChange={(e) => setVerifiedLabel(e.target.value)}
            >
              <option value="">Select...</option>
              <option value="agree_pcos">Agree — Confirmed PCOS</option>
              <option value="agree_normal">Agree — Confirmed Normal</option>
              <option value="disagree_pcos">Disagree — Actually PCOS</option>
              <option value="disagree_normal">Disagree — Actually Normal</option>
              <option value="inconclusive">Inconclusive — Needs further testing</option>
            </select>
          </div>

          <div style={{ marginBottom: "1.25rem" }}>
            <label className="field-label">Comments (optional)</label>
            <textarea
              className="field-input"
              rows={5}
              placeholder="Add any clinical notes, discrepancies, or observations..."
              value={comments}
              onChange={(e) => setComments(e.target.value)}
              style={{ resize: "vertical", fontFamily: "inherit" }}
            />
          </div>

          {error && <p className="form-error">{error}</p>}

          <button type="submit" className="btn-primary">
            Submit Feedback
          </button>
        </form>
      </div>
    </div>
  );
}

export default Feedback;
