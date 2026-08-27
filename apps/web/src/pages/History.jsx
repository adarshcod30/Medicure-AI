import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { FiCamera, FiClock, FiSearch, FiTrash2 } from 'react-icons/fi';
import {
  getHistory, getHistoryItem, deleteHistoryItem,
  isAuthenticated, isStorageDisabled, errorDetail,
} from '../services/api';
import ResultsDisplay from '../components/ResultsDisplay';

/**
 * Per-user scan history against the real backend (/v1/history).
 *
 * Every identification status is shown as what it is. "Abstained" and
 * "unreadable" get chips like any other outcome — a scan where the system
 * declined to guess is a result worth keeping, not an error to hide.
 */

// Chip styling per identification status. Abstention is deliberately NOT
// rendered in error red: declining to guess is the system working.
const STATUS_CHIP = {
  confident: { label: 'Identified', color: 'var(--success)', bg: 'var(--success-bg)' },
  ambiguous: { label: 'Ambiguous', color: 'var(--warning)', bg: 'var(--warning-bg)' },
  abstained: { label: 'Abstained', color: 'var(--text-secondary)', bg: 'var(--bg-input)' },
  unreadable: { label: 'Unreadable photo', color: 'var(--warning)', bg: 'var(--warning-bg)' },
};

// The list endpoint is built against the same storage seam as this page
// (scans docs: {_id, user_id, kind, query, result, created_at}). Accept both
// a bare array and an {items: [...]} envelope so neither side has to guess
// the other's framing, and read ids/statuses from wherever they landed.
const normaliseList = (res) =>
  Array.isArray(res) ? res : res?.items || res?.scans || res?.history || [];
const itemId = (item) => item.id ?? item._id;
const itemStatus = (item) =>
  item.status ?? item.result?.identification?.status ?? item.identification?.status;
const itemResult = (item) => item.result ?? item;

function StatusChip({ status }) {
  const chip = STATUS_CHIP[status];
  if (!chip) return null;
  return (
    <span style={{ display: 'inline-block', padding: '0.15rem 0.6rem', borderRadius: 50,
                   fontSize: '0.7rem', fontWeight: 600, textTransform: 'uppercase',
                   letterSpacing: '0.5px', color: chip.color, background: chip.bg,
                   whiteSpace: 'nowrap' }}>
      {chip.label}
    </span>
  );
}

function SignInPrompt() {
  return (
    <div className="page">
      <div className="card" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
        <FiClock style={{ fontSize: '3rem', color: 'var(--border)', marginBottom: '1rem' }} />
        <h3>History is tied to an account</h3>
        <p style={{ color: 'var(--text-muted)', margin: '0.5rem 0 1.5rem' }}>
          Sign in and every scan and search is kept here for you.
        </p>
        <Link to="/login" className="btn btn-primary">Sign in</Link>
      </div>
    </div>
  );
}

function StorageDisabledNotice() {
  return (
    <div className="page">
      <div className="card" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
        <FiClock style={{ fontSize: '3rem', color: 'var(--border)', marginBottom: '1rem' }} />
        <h3>Accounts are disabled on this deployment</h3>
        <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem', maxWidth: 480, marginInline: 'auto' }}>
          This server runs without a database, so history is not kept.
          Scanning and searching work exactly the same.
        </p>
      </div>
    </div>
  );
}

export default function History() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [storageDisabled, setStorageDisabled] = useState(false);
  const [authRequired, setAuthRequired] = useState(!isAuthenticated());
  const [selected, setSelected] = useState(null);
  const [selectedLoading, setSelectedLoading] = useState(false);

  useEffect(() => {
    if (authRequired) { setLoading(false); return; }
    (async () => {
      try {
        const res = await getHistory();
        setHistory(normaliseList(res));
      } catch (err) {
        if (isStorageDisabled(err)) setStorageDisabled(true);
        else if (err.response?.status === 401) setAuthRequired(true);
        else setError(errorDetail(err, 'Failed to load history.'));
      } finally {
        setLoading(false);
      }
    })();
  }, [authRequired]);

  const handleSelect = async (item) => {
    setSelectedLoading(true);
    try {
      // The list may carry only a summary; the detail endpoint returns the
      // full stored payload in the same shape /scan and /search produce.
      const full = await getHistoryItem(itemId(item));
      setSelected({ ...item, ...full });
    } catch (err) {
      setError(errorDetail(err, 'Failed to load this scan.'));
    } finally {
      setSelectedLoading(false);
    }
  };

  const handleDelete = async (e, id) => {
    e.stopPropagation(); // prevent opening the scan
    if (!window.confirm('Delete this entry from your history?')) return;
    try {
      await deleteHistoryItem(id);
      setHistory((prev) => prev.filter((s) => itemId(s) !== id));
      if (selected && itemId(selected) === id) setSelected(null);
    } catch (err) {
      setError(errorDetail(err, 'Failed to delete.'));
    }
  };

  if (authRequired) return <SignInPrompt />;
  if (storageDisabled) return <StorageDisabledNotice />;
  if (loading) {
    return (
      <div className="page" style={{ textAlign: 'center', marginTop: '4rem' }}>
        <div className="spinner" />
      </div>
    );
  }

  return (
    <div className="page" style={{ display: 'grid', gridTemplateColumns: '350px 1fr', gap: '2rem' }}>
      {/* Sidebar: history list */}
      <div className="card" style={{ padding: '1.5rem', maxHeight: 'calc(100vh - 120px)', overflowY: 'auto' }}>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
          <FiClock style={{ color: 'var(--accent)' }} /> Your scans
        </h2>

        {error && <p style={{ color: 'var(--danger)', fontSize: '0.85rem' }}>{error}</p>}

        {history.length === 0 && !error ? (
          <p style={{ color: 'var(--text-muted)' }}>
            Nothing here yet. Scans and searches you run while signed in will
            be kept in this list.
          </p>
        ) : (
          <div className="history-list">
            {history.map((item) => {
              const id = itemId(item);
              const status = itemStatus(item);
              const result = itemResult(item);
              const title =
                result?.identification?.composition || item.query || 'Photo scan';
              return (
                <div
                  key={id}
                  className="history-item"
                  style={{ borderColor: selected && itemId(selected) === id ? 'var(--accent)' : '' }}
                  onClick={() => handleSelect(item)}
                >
                  {item.kind === 'search'
                    ? <FiSearch style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
                    : <FiCamera style={{ color: 'var(--text-muted)', flexShrink: 0 }} />}
                  <div className="info">
                    <h4 style={{ overflowWrap: 'anywhere' }}>{title}</h4>
                    <p>
                      {item.created_at ? new Date(item.created_at).toLocaleDateString() : ''}
                    </p>
                    <div style={{ marginTop: '0.25rem' }}><StatusChip status={status} /></div>
                  </div>
                  <button
                    onClick={(e) => handleDelete(e, id)}
                    title="Delete"
                    style={{ background: 'none', border: 'none', color: 'var(--text-muted)',
                             cursor: 'pointer', padding: '0.5rem' }}
                  >
                    <FiTrash2 />
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Main area: one saved result, rendered by the same view as a live scan */}
      <div>
        {selectedLoading ? (
          <div className="card" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
            <div className="spinner" />
          </div>
        ) : selected ? (
          <div className="card">
            <h2 style={{ marginBottom: '1rem', paddingBottom: '1rem', borderBottom: '1px solid var(--border)' }}>
              {selected.kind === 'search' ? 'Search' : 'Scan'}
              {selected.created_at ? ` — ${new Date(selected.created_at).toLocaleString()}` : ''}
            </h2>
            {selected.query && (
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '1rem' }}>
                Query: {selected.query}
              </p>
            )}
            <ResultsDisplay data={itemResult(selected)} />
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
