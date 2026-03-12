import { FiActivity } from 'react-icons/fi';
import { devLogin, loginWithGoogle } from '../services/api';
import { GoogleLogin } from '@react-oauth/google';

export default function LoginPage({ onLogin }) {
  const handleDevLogin = async () => {
    try {
      await devLogin();
      onLogin();
    } catch (err) {
      console.error('Dev login failed:', err);
    }
  };

  const handleGoogleSuccess = async (credentialResponse) => {
    try {
      await loginWithGoogle(credentialResponse.credential);
      onLogin();
    } catch (err) {
      console.error('Google login failed:', err);
      alert('Google login failed. Please try again.');
    }
  };

  const handleGoogleError = () => {
    console.error('Google Login Failed');
    alert('Google Login Failed. Please try Dev Login.');
  };

  return (
    <div className="login-page">
      <div className="card login-card">
        <FiActivity style={{ fontSize: '3rem', color: 'var(--accent)', marginBottom: '1.5rem' }} />
        <h2>Welcome to Medicure Plus</h2>
        <p>
          Your AI-powered medicine assistant. Scan any medicine packaging and
          get instant, easy-to-understand information.
        </p>

        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1rem', marginTop: '1rem' }}>
          <GoogleLogin
            onSuccess={handleGoogleSuccess}
            onError={handleGoogleError}
            useOneTap
            theme="filled_black"
            shape="rectangular"
          />
        </div>

        <div className="dev-skip">
          Development mode?{' '}
          <button onClick={handleDevLogin}>Skip to app →</button>
        </div>
      </div>
    </div>
  );
}
