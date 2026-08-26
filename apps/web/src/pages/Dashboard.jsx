import { useState } from 'react';
import { FiCamera, FiSearch } from 'react-icons/fi';
import ImageUpload from '../components/ImageUpload';
import ResultsDisplay from '../components/ResultsDisplay';
import ChatInterface from '../components/ChatInterface';
import SafetyWarning from '../components/SafetyWarning';
import { scanImage, searchMedicine } from '../services/api';
import { locales } from '../locales';

export default function Dashboard() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [targetLanguage, setTargetLanguage] = useState('en-US');
  const [searchQuery, setSearchQuery] = useState('');

  const t = locales[targetLanguage] || locales['en-US'];

  const indianLanguages = [
    { code: 'en-US', name: 'English' },
    { code: 'hi-IN', name: 'हिंदी (Hindi)' },
    { code: 'ta-IN', name: 'தமிழ் (Tamil)' },
    { code: 'te-IN', name: 'తెలుగు (Telugu)' },
    { code: 'bn-IN', name: 'বাংলা (Bengali)' },
    { code: 'mr-IN', name: 'मराठी (Marathi)' },
  ];

  const handleImageSelected = async (selectedFile) => {
    setFile(selectedFile);
    setResult(null);
    setError(null);

    if (!selectedFile) return;

    setLoading(true);
    try {
      const res = await scanImage(selectedFile, targetLanguage);
      if (res.success) {
        setResult({ ...res.data, scan_id: res.scan_id });
      } else {
        setError(res.error || 'Failed to analyze the image.');
      }
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Network error.');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    const name = searchQuery.trim();
    if (!name) return;

    setFile(null);
    setResult(null);
    setError(null);
    setLoading(true);

    try {
      const res = await searchMedicine(name, targetLanguage);
      if (res.success) {
        setResult({ ...res.data, scan_id: res.scan_id });
      } else {
        setError(res.error || 'Failed to find medicine info.');
      }
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Network error.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      {!file && !result && (
        <section className="hero">
          <h1>{t.title}</h1>
          <p>{t.description}</p>
        </section>
      )}

      {/* Safety Warning */}
      <SafetyWarning lang={targetLanguage} />

      <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
        {/* Step 1: Upload & Language */}
        <section>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            {(!file || loading) && (
              <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <FiCamera style={{ color: 'var(--accent)' }} /> 
                {t.upload_title}
              </h3>
            )}
            
            {/* Language Selector */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <label htmlFor="lang-select" style={{ fontSize: '0.9rem', fontWeight: 'bold' }}>{t.translate_to}</label>
              <select 
                id="lang-select"
                value={targetLanguage} 
                onChange={(e) => setTargetLanguage(e.target.value)}
                disabled={loading}
                style={{
                  padding: '0.4rem 0.8rem',
                  borderRadius: '6px',
                  border: '1px solid var(--border)',
                  background: 'var(--surface)',
                  color: 'var(--text)',
                  outline: 'none',
                  cursor: 'pointer'
                }}
              >
                {indianLanguages.map(lang => (
                  <option key={lang.code} value={lang.code}>{lang.name}</option>
                ))}
              </select>
            </div>
          </div>
          <ImageUpload onImageSelected={handleImageSelected} disabled={loading} lang={targetLanguage} />

          {/* Medicine Text Search Bar */}
          <div style={{
            display: 'flex',
            gap: '0.5rem',
            marginTop: '1rem',
            alignItems: 'center'
          }}>
            <div style={{ position: 'relative', flex: 1 }}>
              <FiSearch style={{
                position: 'absolute',
                left: '0.75rem',
                top: '50%',
                transform: 'translateY(-50%)',
                color: 'var(--text-muted)',
                fontSize: '1.1rem'
              }} />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') handleSearch(); }}
                placeholder="Or type a medicine name (e.g., Crocin, Paracetamol)..."
                disabled={loading}
                style={{
                  width: '100%',
                  padding: '0.75rem 0.75rem 0.75rem 2.5rem',
                  borderRadius: '8px',
                  border: '1px solid var(--border)',
                  background: 'var(--surface)',
                  color: 'var(--text)',
                  fontSize: '0.95rem',
                  outline: 'none',
                  boxSizing: 'border-box'
                }}
              />
            </div>
            <button
              className="btn btn-primary"
              onClick={handleSearch}
              disabled={loading || !searchQuery.trim()}
              style={{ padding: '0.75rem 1.5rem', whiteSpace: 'nowrap' }}
            >
              <FiSearch style={{ marginRight: '0.35rem' }} />
              Search
            </button>
          </div>

          {loading && (
            <div className="loading-overlay">
              <div className="spinner" />
              <p>{t.analyzing}</p>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                {t.analysis_time}
              </p>
            </div>
          )}

          {error && (
            <div className="safety-warning" style={{ marginTop: '1rem', background: 'var(--danger-bg)', borderColor: 'var(--danger)' }}>
              <p style={{ color: 'var(--danger)' }}>{error}</p>
            </div>
          )}
        </section>

        {/* Step 2: Results & Chat (Side-by-side on desktop) */}
        {result && !loading && (
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', 
            gap: '2rem',
            alignItems: 'start' 
          }}>
            {/* Left: Structured Data */}
            <section className="card">
              <ResultsDisplay data={result} lang={targetLanguage} />
            </section>

            {/* Right: Chatbot */}
            <section style={{ position: 'sticky', top: '100px' }}>
              <h3 style={{ marginBottom: '1rem' }}>{t.follow_up_title}</h3>
              <ChatInterface scanId={result.scan_id} targetLanguage={targetLanguage} />
            </section>
          </div>
        )}
      </div>
    </div>
  );
}
