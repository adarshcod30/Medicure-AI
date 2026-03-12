import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:3001/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000, // 2 min (OCR + LLM can be slow)
});

// Attach JWT to every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('medicure_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ── Scan endpoints ──────────────────────────────────
export async function scanImage(imageFile, targetLanguage = 'en-US') {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('target_language', targetLanguage);
  
  const { data } = await api.post('/scan', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

// ── Chat endpoints ──────────────────────────────────
export async function sendChatMessage(scanId, message, targetLanguage = 'en-US') {
  const { data } = await api.post('/chat', { 
    scan_id: scanId, 
    message,
    target_language: targetLanguage 
  });
  return data;
}

// ── Search endpoints ────────────────────────────────
export async function searchMedicine(name, targetLanguage = 'en-US') {
  const { data } = await api.post('/search', { 
    name,
    target_language: targetLanguage 
  });
  return data;
}

// ── History endpoints ───────────────────────────────
export async function getHistory(page = 1, limit = 20) {
  const { data } = await api.get('/history', { params: { page, limit } });
  return data;
}

export async function deleteScan(scanId) {
  const { data } = await api.delete(`/history/${scanId}`);
  return data;
}

// ── Auth endpoints ──────────────────────────────────
export async function loginWithGoogle(credential) {
  const { data } = await api.post('/auth/google', { credential });
  if (data.token) {
    localStorage.setItem('medicure_token', data.token);
    localStorage.setItem('medicure_user', JSON.stringify(data.user));
  }
  return data;
}

export async function devLogin() {
  const { data } = await api.post('/auth/dev-login');
  if (data.token) {
    localStorage.setItem('medicure_token', data.token);
    localStorage.setItem('medicure_user', JSON.stringify(data.user));
  }
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
