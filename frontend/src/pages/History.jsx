import { useState, useEffect } from 'react';
import { FiTrash2, FiClock } from 'react-icons/fi';
import { getHistory, deleteScan } from '../services/api';
import ResultsDisplay from '../components/ResultsDisplay';

export default function History() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedScan, setSelectedScan] = useState(null);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await getHistory();
      if (res.scans) {
        setHistory(res.scans);
      } else {
        setError(res.error || 'Failed to load history.');
      }
    } catch (err) {
      setError('Failed to load history.');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (e, id) => {
    e.stopPropagation(); // prevent opening the scan
    if (!window.confirm('Delete this scan history?')) return;
    
    try {
      const res = await deleteScan(id);
      if (res.success) {
        setHistory(prev => prev.filter(s => s.id !== id));
        if (selectedScan?.id === id) setSelectedScan(null);
      }
    } catch (err) {
      alert('Failed to delete scan.');
    }
  };

  if (loading) return <div className="page" style={{ textAlign: 'center', marginTop: '4rem' }}><div className="spinner" /></div>;

  return (
    <div className="page" style={{ display: 'grid', gridTemplateColumns: '350px 1fr', gap: '2rem' }}>
      {/* Sidebar: History List */}
      <div className="card" style={{ padding: '1.5rem', maxHeight: 'calc(100vh - 120px)', overflowY: 'auto' }}>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
          <FiClock style={{ color: 'var(--accent)' }}/> Your Scans
        </h2>
        
        {error ? (
          <p style={{ color: 'var(--danger)' }}>{error}</p>
        ) : history.length === 0 ? (
          <p style={{ color: 'var(--text-muted)' }}>No previous scans found.</p>
        ) : (
          <div className="history-list">
            {history.map(scan => (
              <div 
                key={scan.id} 
                className="history-item"
                style={{
                  borderColor: selectedScan?.id === scan.id ? 'var(--accent)' : ''
                }}
                onClick={() => setSelectedScan(scan)}
              >
                <img 
                  src={`http://localhost:3001${scan.image_path}`} 
                  alt="Scan thumbnail" 
                  className="thumb" 
                  onError={(e) => { e.target.src = 'https://placehold.co/100x100/111827/34ebc6?text=Photo'; }}
                />
                <div className="info">
                  <h4>{scan.result_json?.brand_name || scan.result_json?.generic_name || 'Unknown Medicine'}</h4>
                  <p>{new Date(scan.created_at).toLocaleDateString()}</p>
                </div>
                <button 
                  onClick={(e) => handleDelete(e, scan.id)}
                  style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: '0.5rem' }}
                >
                  <FiTrash2 />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Main Area: Scan Details */}
      <div>
        {selectedScan ? (
          <div className="card">
            <h2 style={{ marginBottom: '1rem', paddingBottom: '1rem', borderBottom: '1px solid var(--border)' }}>
              Scan Details — {new Date(selectedScan.created_at).toLocaleString()}
            </h2>
            <div style={{ marginBottom: '2rem', textAlign: 'center' }}>
               <img 
                  src={`http://localhost:3001${selectedScan.image_path}`} 
                  alt="Original scan" 
                  style={{ maxHeight: '300px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}
                />
            </div>
            <ResultsDisplay data={selectedScan.result_json} />
          </div>
        ) : (
          <div className="card" style={{ textAlign: 'center', padding: '4rem 2rem', color: 'var(--text-muted)' }}>
            <FiClock style={{ fontSize: '3rem', color: 'var(--border)', marginBottom: '1rem' }} />
            <h3>Select a scan from history</h3>
            <p>Click on an item in the list to view its analysis.</p>
          </div>
        )}
      </div>
    </div>
  );
}
