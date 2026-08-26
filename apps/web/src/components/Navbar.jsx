import { Link, useLocation } from 'react-router-dom';
import { FiActivity, FiLogOut } from 'react-icons/fi';
import { getStoredUser, logout } from '../services/api';

export default function Navbar({ onLogout }) {
  const location = useLocation();
  const user = getStoredUser();

  const handleLogout = () => {
    logout();
    if (onLogout) onLogout();
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
        {user && (
          <button onClick={handleLogout} title="Sign out">
            <FiLogOut />
          </button>
        )}
      </div>
    </nav>
  );
}
