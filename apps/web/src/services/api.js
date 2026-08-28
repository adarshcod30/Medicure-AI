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

/* ── Accounts, history, cabinet ──────────────────────────────────────────
 *
 * Persistence is optional by design. A deployment without MongoDB serves
 * identification exactly as before, and every route below answers 503 with a
 * clear detail instead. Callers must render that as "accounts are disabled
 * on this deployment" — a statement of capability, not an error screen.
 */

/** True when the backend said "no storage on this deployment" (HTTP 503). */
export function isStorageDisabled(err) {
  return err?.response?.status === 503;
}

/** The backend's own words for what went wrong, whatever shape `detail` took. */
export function errorDetail(err, fallback = 'Something went wrong.') {
  const detail = err?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (detail?.error) return detail.error;
  if (detail) return JSON.stringify(detail);
  return err?.message || fallback;
}

function storeSession({ token, user }) {
  localStorage.setItem('medicure_token', token);
  localStorage.setItem('medicure_user', JSON.stringify(user));
}

/** Create an account. Stores the session on success. */
export async function register(email, password, name) {
  const { data } = await api.post('/auth/register', { email, password, name });
  storeSession(data);
  return data;
}

/** Sign in. Stores the session on success. */
export async function login(email, password) {
  const { data } = await api.post('/auth/login', { email, password });
  storeSession(data);
  return data;
}

/** The signed-in user according to the server, not localStorage. */
export async function fetchMe() {
  const { data } = await api.get('/auth/me');
  return data;
}

/** Past scans and searches for the signed-in user, newest first. */
export async function getHistory(limit = 50) {
  const { data } = await api.get('/history', { params: { limit } });
  return data;
}

/** One saved result, in the same shape /scan and /search return. */
export async function getHistoryItem(id) {
  const { data } = await api.get(`/history/${id}`);
  return data;
}

export async function deleteHistoryItem(id) {
  const { data } = await api.delete(`/history/${id}`);
  return data;
}

/** The medicine cabinet: what this user takes, as composition signatures. */
export async function getCabinet() {
  const { data } = await api.get('/cabinet');
  return data;
}

/**
 * Save an identified medicine to the cabinet. `signature` must be the
 * signature from a result payload, passed through untouched — the cabinet
 * holds only what retrieval identified, never free text typed by anyone.
 */
export async function addToCabinet({ display_name, signature, source }) {
  const { data } = await api.post('/cabinet', { display_name, signature, source });
  return data;
}

export async function removeFromCabinet(id) {
  const { data } = await api.delete(`/cabinet/${id}`);
  return data;
}

/**
 * Interaction findings across the cabinet, each with its source. An empty
 * list is a real answer (dataset coverage is partial), not a failure.
 */
export async function getCabinetInteractions() {
  const { data } = await api.get('/cabinet/interactions');
  return data;
}

/** Look-alike/sound-alike names confusable with `name`. Deterministic. */
export async function getLasa(name) {
  const { data } = await api.get('/lasa', { params: { name } });
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

/**
 * Follow-up question about one medicine.
 *
 * `subject` is the brand name, not the whole result: the server re-resolves it
 * so the answer grounds against a fact sheet it produced itself this request.
 *
 * The reply's `grounded` flag is the important part. True means every claim
 * traces to the retrieved records; false means the databases did not cover the
 * question and the model answered from its own training. The UI must show
 * those differently — that distinction is the entire point of the feature.
 */
export async function askAboutMedicine(question, subject, history = []) {
  const { data } = await api.post('/chat', { question, subject, history });
  return data;
}
