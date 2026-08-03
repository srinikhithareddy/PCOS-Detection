import axios from "axios";

// Change this to your actual backend URL once it's running
const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

const api = axios.create({
  baseURL: BASE_URL,
  headers: { "Content-Type": "application/json" },
});

// Attach JWT token to every request if present
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// If the backend ever returns 401, clear the stale token
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      localStorage.removeItem("token");
      localStorage.removeItem("user");
    }
    return Promise.reject(error);
  }
);

// ===== Auth Endpoints =====
// Backend expected contract (adjust once implemented):
// POST /auth/login  { email, password } -> { token, user: { id, name, email, role } }
export const loginUser = (email, password) =>
  api.post("/auth/login", { email, password });

// POST /auth/register (if you build signup later)
export const registerUser = (payload) => api.post("/auth/register", payload);

// ===== Prediction Endpoints (for later use) =====
// POST /predict  (multipart: image + clinical JSON) -> prediction result
export const submitPrediction = (formData) =>
  api.post("/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

// POST /feedback -> stores clinician feedback
export const submitFeedback = (payload) => api.post("/feedback", payload);

export default api;
