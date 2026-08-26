import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';

/**
 * Routing.
 *
 * The login gate is deliberately absent in this milestone. Authentication and
 * per-user history (the medicine cabinet that drives interaction checking)
 * arrive with the MongoDB layer. Until then, requiring a sign-in would gate the
 * part of the system that actually works behind a part that does not yet exist.
 */
export default function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}
