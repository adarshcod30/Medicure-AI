import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import LoginPage from './components/LoginPage';
import Dashboard from './pages/Dashboard';
import History from './pages/History';
import { isAuthenticated } from './services/api';

function App() {
  const [authStatus, setAuthStatus] = useState(isAuthenticated());

  // Listen to storage events (e.g. login from another tab)
  useEffect(() => {
    const handleStorage = () => {
      setAuthStatus(isAuthenticated());
    };
    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, []);

  const handleLogin = () => setAuthStatus(true);
  const handleLogout = () => setAuthStatus(false);

  // Protected Route wrapper
  const ProtectedRoute = ({ children }) => {
    if (!authStatus) return <Navigate to="/login" replace />;
    return children;
  };

  return (
    <Router>
      {authStatus && <Navbar onLogout={handleLogout} />}
      <Routes>
        <Route 
          path="/login" 
          element={!authStatus ? <LoginPage onLogin={handleLogin} /> : <Navigate to="/" replace />} 
        />
        <Route 
          path="/" 
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/history" 
          element={
            <ProtectedRoute>
              <History />
            </ProtectedRoute>
          } 
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}

export default App;
