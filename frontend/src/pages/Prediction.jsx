import { useState } from "react";
import { useNavigate } from "react-router-dom";

function Prediction() {
  const [imageFile, setImageFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [qualityStatus, setQualityStatus] = useState(null); // null | "checking" | "good" | "poor"
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setError("");
    setQualityStatus(null);

    if (!file) return;

    const validTypes = ["image/jpeg", "image/png", "image/jpg"];
    if (!validTypes.includes(file.type)) {
      setError("Please upload a JPG or PNG ultrasound image.");
      setImageFile(null);
      setPreviewUrl(null);
      return;
    }

    setImageFile(file);
    setPreviewUrl(URL.createObjectURL(file));

    // TEMPORARY: simulate quality check (BRISQUE/NIQE will run on backend later)
    setQualityStatus("checking");
    setTimeout(() => {
      // Mock result — replace with real API response later
      setQualityStatus("good");
    }, 1200);
  };

  const handleAnalyze = () => {
    if (!imageFile) {
      setError("Please upload an ultrasound image before analyzing.");
      return;
    }
    if (qualityStatus === "poor") {
      setError("Image quality is poor. Please re-upload a clearer ultrasound image.");
      return;
    }

    // TEMPORARY: no backend yet — navigate straight to Result with mock trigger
    console.log("Analyzing image:", imageFile.name);
    navigate("/result");
  };

  return (
    <div className="page-container">
      <h1 className="page-title">Ultrasound Upload &amp; Analysis</h1>
      <p style={{ color: "var(--color-text-muted)", marginBottom: "1.5rem" }}>
        Upload an ovarian ultrasound image. The system will check image quality
        before running the multimodal PCOS analysis.
      </p>

      <div className="card" style={{ padding: "1.5rem" }}>
        <label className="field-label">Ultrasound Image (JPG/PNG)</label>
        <input
          type="file"
          accept="image/png, image/jpeg"
          onChange={handleFileChange}
          className="field-input"
          style={{ padding: "0.4rem" }}
        />

        {error && <p className="form-error">{error}</p>}

        {previewUrl && (
          <div style={{ marginTop: "1.5rem" }}>
            <img
              src={previewUrl}
              alt="Ultrasound preview"
              style={{
                maxWidth: "100%",
                maxHeight: "360px",
                borderRadius: "6px",
                border: "1px solid var(--color-border)",
                display: "block",
              }}
            />

            <div style={{ marginTop: "1rem" }}>
              {qualityStatus === "checking" && (
                <span style={{ color: "var(--color-text-muted)", fontSize: "0.9rem" }}>
                  Checking image quality (BRISQUE/NIQE)...
                </span>
              )}
              {qualityStatus === "good" && (
                <span className="form-success" style={{ display: "inline-block" }}>
                  ✓ Image quality: Good
                </span>
              )}
              {qualityStatus === "poor" && (
                <span className="form-error" style={{ display: "inline-block" }}>
                  ✗ Image quality: Poor — please re-upload a clearer image
                </span>
              )}
            </div>
          </div>
        )}

        <button
          className="btn-primary"
          style={{ marginTop: "1.5rem" }}
          onClick={handleAnalyze}
        >
          Analyze
        </button>
      </div>
    </div>
  );
}

export default Prediction;
