import { Link } from "react-router-dom";

function Home() {
  return (
    <div className="page-container">
      <h1 className="page-title">PCOS Detection & Clinical Decision Support</h1>
      <p style={{ color: "var(--color-text-muted)", maxWidth: "640px", marginBottom: "2rem" }}>
        An explainable, reliability-aware multimodal AI system that combines ovarian
        ultrasound imaging with structured clinical and laboratory data to assist
        doctors in PCOS screening — providing prediction confidence, uncertainty
        scores, and visual explanations to support faster, evidence-based clinical
        decisions.
      </p>

      <div className="card" style={{ padding: "1.5rem" }}>
        <h3 style={{ marginTop: 0 }}>Get Started</h3>
        <p style={{ color: "var(--color-text-muted)", fontSize: "0.9rem" }}>
          Begin a new patient screening by entering clinical data and uploading an
          ultrasound image.
        </p>
        <Link to="/patient-entry">
          <button className="btn-primary" style={{ marginTop: "0.5rem" }}>
            Start New Patient Screening
          </button>
        </Link>
      </div>
    </div>
  );
}

export default Home;
