import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FiActivity } from 'react-icons/fi';
import { register, login, isStorageDisabled, errorDetail } from '../services/api';

/**
 * Email + password against the real backend (/v1/auth). The Google button
 * that used to live here talked to a Node gateway that no longer exists.
 *
 * Two states get special, honest treatment:
 *   - a 503 means the deployment runs without MongoDB. That is a supported
 *     configuration, so it renders as a calm capability notice, not a failure;
 *   - any other backend error is shown in the backend's own words rather
 *     than a generic "please try again".
 */

const inputStyle = {
  width: '100%',
  padding: '0.75rem 1rem',
  borderRadius: '8px',
  border: '1px solid var(--border)',
  background: 'var(--bg-input)',
  color: 'var(--text-primary)',
  fontSize: '0.95rem',
  fontFamily: 'inherit',
  outline: 'none',
  boxSizing: 'border-box',
};

export default function LoginPage() {
  const navigate = useNavigate();
  const [mode, setMode] = useState('login'); // 'login' | 'register'
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [storageDisabled, setStorageDisabled] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setStorageDisabled(false);
    setBusy(true);
    try {
      if (mode === 'register') {
        await register(email, password, name);
      } else {
        await login(email, password);
      }
      navigate('/', { replace: true });
    } catch (err) {
      if (isStorageDisabled(err)) setStorageDisabled(true);
      else setError(errorDetail(err, 'Could not reach the server.'));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-page">
      <div className="card login-card">
        <FiActivity style={{ fontSize: '3rem', color: 'var(--accent)', marginBottom: '1.5rem' }} />
        <h2>{mode === 'register' ? 'Create your account' : 'Welcome back'}</h2>
        <p>
          Sign in to keep a history of your scans and a medicine cabinet that
          is checked for interactions. Scanning itself needs no account.
        </p>

        <form onSubmit={handleSubmit}
              style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', textAlign: 'left' }}>
          {mode === 'register' && (
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Your name"
              autoComplete="name"
              required
              disabled={busy}
              style={inputStyle}
            />
          )}
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Email"
            autoComplete="email"
            required
            disabled={busy}
            style={inputStyle}
          />
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            autoComplete={mode === 'register' ? 'new-password' : 'current-password'}
            required
            disabled={busy}
            style={inputStyle}
          />

          {storageDisabled && (
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: 1.5,
                        background: 'var(--bg-input)', border: '1px solid var(--border)',
                        borderRadius: '8px', padding: '0.75rem 1rem', margin: 0 }}>
              Accounts are disabled on this deployment — it runs without a
              database. Scanning and searching work exactly the same without
              signing in.
            </p>
          )}

          {error && (
            <p style={{ color: 'var(--danger)', fontSize: '0.85rem', margin: 0 }}>{error}</p>
          )}

          <button type="submit" className="btn btn-primary" disabled={busy}
                  style={{ width: '100%', marginTop: '0.25rem' }}>
            {busy ? 'Please wait…' : mode === 'register' ? 'Create account' : 'Sign in'}
          </button>
        </form>

        <div className="dev-skip">
          {mode === 'register' ? 'Already have an account?' : 'New here?'}{' '}
          <button
            type="button"
            onClick={() => { setMode(mode === 'register' ? 'login' : 'register'); setError(null); }}
          >
            {mode === 'register' ? 'Sign in instead' : 'Create an account'}
          </button>
        </div>
      </div>
    </div>
  );
}
