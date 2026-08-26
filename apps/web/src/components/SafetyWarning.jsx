import { FiAlertTriangle } from 'react-icons/fi';
import { locales } from '../locales';

export default function SafetyWarning({ lang = 'en-US' }) {
  const t = locales[lang] || locales['en-US'];
  
  return (
    <div className="safety-warning">
      <FiAlertTriangle className="icon" />
      <p>
        <strong>Medical Disclaimer:</strong> {t.safety_disclaimer}
      </p>
    </div>
  );
}
