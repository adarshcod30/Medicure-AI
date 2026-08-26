import axios from 'axios';

// Single FastAPI service. The Node/Express gateway that used to sit in front
// of the ML service is gone — one backend, one base URL.
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/v1';

const api = axios.create({
  baseURL: API_BASE,
  // OCR over several DIP renditions plus retrieval takes ~1s locally, but a
  // cold start has to load a 125 MB index first.
  timeout: 120000,
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('medicure_token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

/** Photo of packaging -> grounded, cited result. */
export async function scanImage(imageFile, { explain = true } = {}) {
  const formData = new FormData();
  formData.append('file', imageFile);

  const { data } = await api.post(`/scan?explain=${explain}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

/** Typed name or composition -> the same contract as scanImage. */
export async function searchMedicine(query, { explain = true } = {}) {
  const { data } = await api.post('/search', { query, explain });
  return data;
}

/** Type-ahead over brand names. Lexical only, no claims attached. */
export async function suggest(q, limit = 8) {
  if (!q || q.trim().length < 2) return { suggestions: [] };
  const { data } = await api.get('/suggest', { params: { q, limit } });
  return data;
}

/** Readiness, including which capabilities are degraded and why. */
export async function getHealth() {
  const { data } = await api.get('/health');
  return data;
}

/**
 * Index coverage, calibration report and known data gaps.
 * Surfaced in the UI on purpose — a system whose main claim is that it knows
 * when to stop should be willing to show its own limits.
 */
export async function getMetrics() {
  const { data } = await api.get('/metrics');
  return data;
}

export function logout() {
  localStorage.removeItem('medicure_token');
  localStorage.removeItem('medicure_user');
}

export function getStoredUser() {
  try {
    return JSON.parse(localStorage.getItem('medicure_user'));
  } catch {
    return null;
  }
}

export function isAuthenticated() {
  return !!localStorage.getItem('medicure_token');
}

export default api;
