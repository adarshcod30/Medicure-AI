import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import LoginPage from './components/LoginPage';
import Dashboard from './pages/Dashboard';
import History from './pages/History';
import Cabinet from './pages/Cabinet';

/**
 * Routing.
 *
 * There is deliberately no global login gate. Scanning and searching work
 * anonymously, and on a deployment without MongoDB they are the only parts
 * that CAN work. The pages that need an account — history and the cabinet —
 * gate themselves: each renders a sign-in prompt when anonymous, and a calm
 * "accounts are disabled on this deployment" notice when the backend answers
 * 503, instead of hiding the working system behind the optional one.
 */
export default function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/history" element={<History />} />
        <Route path="/cabinet" element={<Cabinet />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}
