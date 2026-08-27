import { Link, useLocation, useNavigate } from 'react-router-dom';
import { FiActivity, FiLogOut } from 'react-icons/fi';
import { getStoredUser, logout } from '../services/api';

export default function Navbar() {
  const location = useLocation();
  const navigate = useNavigate();
  const user = getStoredUser();

  const handleLogout = () => {
    logout();
    // Leave whatever account-gated page we were on; the dashboard works
    // anonymously.
    navigate('/');
  };

  const isActive = (path) => location.pathname === path ? 'active' : '';

  return (
    <nav className="navbar">
      <Link to="/" className="navbar-brand">
        <FiActivity className="brand-icon" />
        Medicure Plus
      </Link>

      <div className="navbar-links">
        <Link to="/" className={isActive('/')}>Scan</Link>
        <Link to="/history" className={isActive('/history')}>History</Link>
        <Link to="/cabinet" className={isActive('/cabinet')}>Cabinet</Link>
        {user ? (
          <button onClick={handleLogout} title={`Sign out ${user.name || user.email || ''}`.trim()}>
            <FiLogOut />
          </button>
        ) : (
          <Link to="/login" className={isActive('/login')}>Sign in</Link>
        )}
      </div>
    </nav>
  );
}
