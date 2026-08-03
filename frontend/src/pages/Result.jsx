import { Link } from "react-router-dom";

// TEMPORARY: mock prediction data — replace with real API response later
const mockResult = {
  label: "PCOS",
  probability: 0.82,
  confidence: 0.88,
  uncertainty: 0.12,
  topClinicalFactors: [
    { name: "LH/FSH Ratio", impact: 0.24 },
    { name: "BMI", impact: 0.19 },
    { name: "Fasting Insulin", impact: 0.15 },
    { name: "Menstrual Irregularity", impact: 0.11 },
  ],
};

function Bar({ value, color }) {
  return (
    <div
      style={{
        background: "var(--color-border)",
        borderRadius: "4px",
        height: "10px",
        width: "100%",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          width: `${Math.round(value * 100)}%`,
          background: color,
          height: "100%",
        }}
      />
    </div>
  );
}

function Result() {
  const { label, probability, confidence, uncertainty, topClinicalFactors } = mockResult;
  const isPCOS = label === "PCOS";

  return (
    <div className="page-container">
      <h1 className="page-title">Analysis Result</h1>

      {/* Prediction summary card */}
      <div className="card" style={{ padding: "1.5rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "1rem" }}>
          <span
            style={{
              padding: "0.4rem 1rem",
              borderRadius: "20px",
              fontWeight: 700,
              fontSize: "1.1rem",
              background: isPCOS ? "#fdecea" : "#e8f5ec",
              color: isPCOS ? "var(--color-danger)" : "var(--color-success)",
            }}
          >
            {label}
          </span>
          <span style={{ color: "var(--color-text-muted)", fontSize: "0.9rem" }}>
            Prediction based on fused ultrasound + clinical analysis
          </span>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
          <div>
            <label className="field-label">PCOS Probability</label>
            <Bar value={probability} color="var(--color-accent)" />
            <span style={{ fontSize: "0.85rem", color: "var(--color-text-muted)" }}>
              {Math.round(probability * 100)}%
            </span>
          </div>

          <div>
            <label className="field-label">Confidence Score</label>
            <Bar value={confidence} color="var(--color-primary)" />
            <span style={{ fontSize: "0.85rem", color: "var(--color-text-muted)" }}>
              {Math.round(confidence * 100)}%
            </span>
          </div>

          <div>
            <label className="field-label">Uncertainty Score</label>
            <Bar value={uncertainty} color="#e0a800" />
            <span style={{ fontSize: "0.85rem", color: "var(--color-text-muted)" }}>
              {Math.round(uncertainty * 100)}% (lower is better)
            </span>
          </div>
        </div>
      </div>

      {/* Image explainability placeholder */}
      <div className="card" style={{ padding: "1.5rem" }}>
        <h3 style={{ marginTop: 0 }}>Image Explainability — Grad-CAM++</h3>
        <p style={{ color: "var(--color-text-muted)", fontSize: "0.9rem" }}>
          Heatmap highlighting ovarian regions that most influenced the prediction.
        </p>
        <div
          style={{
            height: "200px",
            background: "var(--color-primary-light)",
            border: "1px dashed var(--color-border)",
            borderRadius: "6px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "var(--color-text-muted)",
            fontSize: "0.9rem",
          }}
        >
          [ Grad-CAM++ heatmap will render here ]
        </div>
      </div>

      {/* Clinical explainability placeholder */}
      <div className="card" style={{ padding: "1.5rem" }}>
        <h3 style={{ marginTop: 0 }}>Clinical Explainability — SHAP Feature Importance</h3>
        <p style={{ color: "var(--color-text-muted)", fontSize: "0.9rem" }}>
          Clinical features that contributed most to this prediction.
        </p>
        {topClinicalFactors.map((factor) => (
          <div key={factor.name} style={{ marginBottom: "0.75rem" }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "0.85rem",
                marginBottom: "0.25rem",
              }}
            >
              <span>{factor.name}</span>
              <span style={{ color: "var(--color-text-muted)" }}>
                {Math.round(factor.impact * 100)}%
              </span>
            </div>
            <Bar value={factor.impact} color="var(--color-accent)" />
          </div>
        ))}
      </div>

      <div style={{ display: "flex", gap: "1rem", marginTop: "1.5rem" }}>
        <Link to="/report">
          <button className="btn-primary">View Full Clinical Report</button>
        </Link>
        <Link to="/feedback">
          <button className="btn-secondary">Submit Feedback</button>
        </Link>
      </div>
    </div>
  );
}

export default Result;
