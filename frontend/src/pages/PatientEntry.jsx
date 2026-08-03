import { useState } from "react";
import { useNavigate } from "react-router-dom";

// Field groups matching the clinical dataset columns
const FIELD_GROUPS = [
  {
    title: "Demographics",
    fields: [
      { name: "Age", label: "Age (years)", type: "number" },
      { name: "Height_cm", label: "Height (cm)", type: "number" },
      { name: "Weight_kg", label: "Weight (kg)", type: "number" },
      { name: "BMI", label: "BMI", type: "number", step: "0.01" },
      { name: "Waist_Circumference_cm", label: "Waist Circumference (cm)", type: "number" },
      { name: "Hip_Circumference_cm", label: "Hip Circumference (cm)", type: "number" },
      { name: "Waist_Hip_Ratio", label: "Waist-Hip Ratio", type: "number", step: "0.01" },
    ],
  },
  {
    title: "Menstrual & Reproductive History",
    fields: [
      { name: "Age_at_Menarche", label: "Age at Menarche (years)", type: "number" },
      { name: "Menstrual_Cycle_Length_days", label: "Menstrual Cycle Length (days)", type: "number" },
      {
        name: "Menstrual_Irregularity",
        label: "Menstrual Irregularity",
        type: "select",
        options: [
          { value: "0", label: "Regular" },
          { value: "1", label: "Irregular" },
        ],
      },
      { name: "Gravidity", label: "Gravidity (number of pregnancies)", type: "number" },
      { name: "Parity", label: "Parity (number of live births)", type: "number" },
    ],
  },
  {
    title: "Physical Signs",
    fields: [
      { name: "Hirsutism_Score_FG", label: "Hirsutism Score (Ferriman-Gallwey)", type: "number" },
      {
        name: "Acne_Severity",
        label: "Acne Severity",
        type: "select",
        options: [
          { value: "0", label: "None" },
          { value: "1", label: "Mild" },
          { value: "2", label: "Moderate" },
          { value: "3", label: "Severe" },
        ],
      },
      {
        name: "Alopecia",
        label: "Alopecia (hair thinning/loss)",
        type: "select",
        options: [
          { value: "0", label: "No" },
          { value: "1", label: "Yes" },
        ],
      },
      {
        name: "Skin_Darkening_Acanthosis",
        label: "Skin Darkening (Acanthosis Nigricans)",
        type: "select",
        options: [
          { value: "0", label: "No" },
          { value: "1", label: "Yes" },
        ],
      },
    ],
  },
  {
    title: "Vitals & Lifestyle",
    fields: [
      { name: "Blood_Pressure_Systolic", label: "Blood Pressure - Systolic (mmHg)", type: "number" },
      { name: "Blood_Pressure_Diastolic", label: "Blood Pressure - Diastolic (mmHg)", type: "number" },
      {
        name: "Physical_Activity_Level",
        label: "Physical Activity Level",
        type: "select",
        options: [
          { value: "0", label: "Sedentary" },
          { value: "1", label: "Light" },
          { value: "2", label: "Moderate" },
          { value: "3", label: "Active" },
        ],
      },
      {
        name: "Smoking_Status",
        label: "Smoking Status",
        type: "select",
        options: [
          { value: "0", label: "No" },
          { value: "1", label: "Yes" },
        ],
      },
      {
        name: "Alcohol_Intake",
        label: "Alcohol Intake",
        type: "select",
        options: [
          { value: "0", label: "No" },
          { value: "1", label: "Yes" },
        ],
      },
      {
        name: "Dietary_Sugar_Intake",
        label: "Dietary Sugar Intake",
        type: "select",
        options: [
          { value: "0", label: "Low" },
          { value: "1", label: "Moderate" },
          { value: "2", label: "High" },
        ],
      },
      { name: "Sleep_Hours", label: "Sleep (hours/night)", type: "number", step: "0.1" },
    ],
  },
  {
    title: "Hormonal Panel",
    fields: [
      { name: "FSH_mIU_mL", label: "FSH (mIU/mL)", type: "number", step: "0.01" },
      { name: "LH_mIU_mL", label: "LH (mIU/mL)", type: "number", step: "0.01" },
      { name: "LH_FSH_Ratio", label: "LH/FSH Ratio", type: "number", step: "0.01" },
      { name: "Total_Testosterone_ng_dL", label: "Total Testosterone (ng/dL)", type: "number", step: "0.01" },
      { name: "Free_Testosterone_pg_mL", label: "Free Testosterone (pg/mL)", type: "number", step: "0.01" },
      { name: "DHEAS_ug_dL", label: "DHEAS (µg/dL)", type: "number", step: "0.01" },
      { name: "Prolactin_ng_mL", label: "Prolactin (ng/mL)", type: "number", step: "0.01" },
      { name: "Estradiol_pg_mL", label: "Estradiol (pg/mL)", type: "number", step: "0.01" },
      { name: "Progesterone_ng_mL", label: "Progesterone (ng/mL)", type: "number", step: "0.01" },
      { name: "SHBG_nmol_L", label: "SHBG (nmol/L)", type: "number", step: "0.01" },
    ],
  },
  {
    title: "Metabolic Panel",
    fields: [
      { name: "Fasting_Glucose_mg_dL", label: "Fasting Glucose (mg/dL)", type: "number" },
      { name: "Fasting_Insulin_uIU_mL", label: "Fasting Insulin (µIU/mL)", type: "number", step: "0.01" },
      { name: "HOMA_IR", label: "HOMA-IR", type: "number", step: "0.01" },
      { name: "HbA1c_percent", label: "HbA1c (%)", type: "number", step: "0.01" },
      { name: "Total_Cholesterol_mg_dL", label: "Total Cholesterol (mg/dL)", type: "number" },
      { name: "HDL_mg_dL", label: "HDL (mg/dL)", type: "number" },
      { name: "LDL_mg_dL", label: "LDL (mg/dL)", type: "number" },
      { name: "Triglycerides_mg_dL", label: "Triglycerides (mg/dL)", type: "number" },
    ],
  },
  {
    title: "Other Labs",
    fields: [
      { name: "CRP_mg_L", label: "CRP (mg/L)", type: "number", step: "0.01" },
      { name: "ALT_U_L", label: "ALT (U/L)", type: "number" },
      { name: "AST_U_L", label: "AST (U/L)", type: "number" },
      { name: "TSH_uIU_mL", label: "TSH (µIU/mL)", type: "number", step: "0.01" },
      { name: "Vitamin_D_ng_mL", label: "Vitamin D (ng/mL)", type: "number", step: "0.01" },
      { name: "Hemoglobin_g_dL", label: "Hemoglobin (g/dL)", type: "number", step: "0.01" },
    ],
  },
];

// Build initial empty state from field groups
const buildInitialState = () => {
  const state = {};
  FIELD_GROUPS.forEach((group) => {
    group.fields.forEach((field) => {
      state[field.name] = "";
    });
  });
  return state;
};

function PatientEntry() {
  const [formData, setFormData] = useState(buildInitialState());
  const [openSections, setOpenSections] = useState(
    Object.fromEntries(FIELD_GROUPS.map((g) => [g.title, true]))
  );
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleChange = (name, value) => {
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const toggleSection = (title) => {
    setOpenSections((prev) => ({ ...prev, [title]: !prev[title] }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setError("");

    const emptyFields = Object.entries(formData).filter(([, v]) => v === "");
    if (emptyFields.length > 0) {
      setError(`Please fill in all fields. ${emptyFields.length} field(s) remaining.`);
      return;
    }

    // TEMPORARY: no backend yet — store in local state / console for now
    console.log("Patient clinical data:", formData);

    // Later: save to Context or send to backend, then navigate to Prediction page
    navigate("/prediction");
  };

  return (
    <div className="page-container">
      <h1 className="page-title">Patient Clinical &amp; Laboratory Data</h1>
      <form onSubmit={handleSubmit}>
        {FIELD_GROUPS.map((group) => (
          <div key={group.title} className="card">
            <button
              type="button"
              onClick={() => toggleSection(group.title)}
              className="section-header"
            >
              <span>{openSections[group.title] ? "▾" : "▸"}</span>
              {group.title}
            </button>

            {openSections[group.title] && (
              <div className="section-body">
                {group.fields.map((field) => (
                  <div key={field.name}>
                    <label className="field-label">{field.label}</label>
                    {field.type === "select" ? (
                      <select
                        className="field-select"
                        value={formData[field.name]}
                        onChange={(e) => handleChange(field.name, e.target.value)}
                      >
                        <option value="">Select...</option>
                        {field.options.map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        className="field-input"
                        type={field.type}
                        step={field.step || "1"}
                        value={formData[field.name]}
                        onChange={(e) => handleChange(field.name, e.target.value)}
                      />
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}

        {error && <p className="form-error">{error}</p>}

        <button type="submit" className="btn-primary">
          Continue to Ultrasound Upload
        </button>
      </form>
    </div>
  );
}

export default PatientEntry;
