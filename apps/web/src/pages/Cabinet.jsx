import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { FiAlertTriangle, FiCopy, FiPackage, FiTag, FiTrash2 } from 'react-icons/fi';
import {
  getCabinet, getCabinetInteractions, removeFromCabinet,
  isAuthenticated, isStorageDisabled, errorDetail,
} from '../services/api';

/**
 * The medicine cabinet (/v1/cabinet) and its interaction check.
 *
 * Two things this page must never do:
 *   - dress an empty interaction list up as safety. The dataset is partial,
 *     so "no interactions on record" is worded as exactly that — absence of
 *     evidence, not evidence of absence;
 *   - invent severity. Every finding is rendered with the severity, the two
 *     ingredient names, the description and the source the backend retrieved.
 */

// Same dataset labels the results view uses.
const DATASET_LABEL = {
  a_z_medicines_india: 'A–Z Medicines of India',
  jan_aushadhi_pmbjp: 'Jan Aushadhi (PMBJP)',
  master_medicines_final: 'NPPA ceiling price data',
  ddinter: 'DDInter',
};

function SourceLine({ source }) {
  if (!source) return null;
  const label = DATASET_LABEL[source.dataset] || source.dataset;
  return (
    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
      <FiTag style={{ verticalAlign: '-1px' }} /> {label}
      {source.record_id ? ` · record ${source.record_id}` : ''}
      {source.url ? (
        <> · <a href={source.url} target="_blank" rel="noreferrer">{source.url}</a></>
      ) : ''}
    </span>
  );
}

// Severity is a dataset string, not an enum we control. Map the common words
// to a tone and fall back to neutral rather than guessing danger.
function severityTone(severity) {
  const s = String(severity || '').toLowerCase();
  if (/major|high|severe|serious|contraindicat/.test(s)) {
    return { color: 'var(--danger)', bg: 'var(--danger-bg)' };
  }
  if (/moderate|medium/.test(s)) {
    return { color: 'var(--warning)', bg: 'var(--warning-bg)' };
  }
  return { color: 'var(--text-secondary)', bg: 'var(--bg-input)' };
}

// The interactions endpoint is built against the same seam as this page.
// Tolerate list-vs-envelope framing and read the two ingredient names from
// wherever they landed, so neither side has to guess the other's field names.
const normaliseItems = (res) => (Array.isArray(res) ? res : res?.items || res?.cabinet || []);
const normaliseFindings = (res) =>
  Array.isArray(res) ? res : res?.interactions || res?.findings || [];
const normaliseDuplicates = (res) =>
  res?.duplicate_therapy || res?.duplicates || res?.duplicate_warnings || [];
const findingPair = (f) => {
  if (Array.isArray(f.ingredients) && f.ingredients.length >= 2) return f.ingredients.slice(0, 2);
  if (Array.isArray(f.pair) && f.pair.length >= 2) return f.pair.slice(0, 2);
  return [f.ingredient_a ?? f.a, f.ingredient_b ?? f.b].filter(Boolean);
};
const itemId = (item) => item.id ?? item._id;

function SignInPrompt() {
  return (
    <div className="page">
      <div className="card" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
        <FiPackage style={{ fontSize: '3rem', color: 'var(--border)', marginBottom: '1rem' }} />
        <h3>The cabinet is tied to an account</h3>
        <p style={{ color: 'var(--text-muted)', margin: '0.5rem 0 1.5rem' }}>
          Sign in to keep the medicines you take in one place and have every
          pair checked against the interaction dataset.
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
        <FiPackage style={{ fontSize: '3rem', color: 'var(--border)', marginBottom: '1rem' }} />
        <h3>Accounts are disabled on this deployment</h3>
        <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem', maxWidth: 480, marginInline: 'auto' }}>
          This server runs without a database, so there is no cabinet to keep.
          Scanning and searching work exactly the same.
        </p>
      </div>
    </div>
  );
}

function InteractionsPanel({ findings, duplicates, itemCount }) {
  return (
    <div className="card" style={{ padding: '1.5rem' }}>
      <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
        <FiAlertTriangle style={{ color: 'var(--accent)' }} /> Interactions
      </h2>

      {itemCount < 2 && findings.length === 0 ? (
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
          Interactions are checked between pairs — add at least two medicines.
        </p>
      ) : findings.length === 0 ? (
        // Absence of evidence, worded as absence of evidence.
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: 1.6 }}>
          No interactions on record between these items (dataset coverage is
          partial). This is not a guarantee of safety — confirm with a
          pharmacist.
        </p>
      ) : (
        <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
          {findings.map((f, i) => {
            const [a, b] = findingPair(f);
            const tone = severityTone(f.severity);
            return (
              <li key={`${a}-${b}-${i}`}
                  style={{ padding: '0.85rem 1rem', border: '1px solid var(--border)',
                           borderRadius: 'var(--radius-md)', marginBottom: '0.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
                  {f.severity && (
                    <span style={{ padding: '0.15rem 0.6rem', borderRadius: 50, fontSize: '0.7rem',
                                   fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px',
                                   color: tone.color, background: tone.bg }}>
                      {f.severity}
                    </span>
                  )}
                  <strong style={{ fontSize: '0.9rem' }}>{a} + {b}</strong>
                </div>
                {(f.description || f.note) && (
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem',
                              margin: '0.5rem 0 0', lineHeight: 1.5 }}>
                    {f.description || f.note}
                  </p>
                )}
                <div style={{ marginTop: '0.4rem' }}><SourceLine source={f.source} /></div>
              </li>
            );
          })}
        </ul>
      )}

      {duplicates.length > 0 && (
        <div style={{ marginTop: '1.25rem' }}>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8rem',
                       textTransform: 'uppercase', letterSpacing: '0.06em',
                       color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            <FiCopy /> Duplicate therapy
          </h3>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
            {duplicates.map((dup, i) => {
              const ingredient = dup.ingredient || dup.name || '';
              const members = dup.items || dup.members || dup.medicines || [];
              return (
                <li key={`${ingredient}-${i}`}
                    style={{ padding: '0.75rem 1rem', border: '1px solid rgba(245, 158, 11, 0.3)',
                             background: 'var(--warning-bg)', borderRadius: 'var(--radius-md)',
                             marginBottom: '0.5rem', fontSize: '0.85rem',
                             color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                  {ingredient && <strong style={{ color: 'var(--warning)' }}>{ingredient}</strong>}
                  {members.length > 0 && (
                    <> appears in {members.map((m) => m.display_name || m.name || m).join(', ')}</>
                  )}
                  {(dup.message || dup.note || dup.description) && (
                    <div style={{ marginTop: '0.25rem' }}>
                      {dup.message || dup.note || dup.description}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}

export default function Cabinet() {
  const [items, setItems] = useState([]);
  const [interactions, setInteractions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [storageDisabled, setStorageDisabled] = useState(false);
  const [authRequired, setAuthRequired] = useState(!isAuthenticated());

  useEffect(() => {
    if (authRequired) { setLoading(false); return; }
    (async () => {
      try {
        const [cabinetRes, interactionsRes] = await Promise.all([
          getCabinet(),
          getCabinetInteractions(),
        ]);
        setItems(normaliseItems(cabinetRes));
        setInteractions(interactionsRes);
      } catch (err) {
        if (isStorageDisabled(err)) setStorageDisabled(true);
        else if (err.response?.status === 401) setAuthRequired(true);
        else setError(errorDetail(err, 'Failed to load the cabinet.'));
      } finally {
        setLoading(false);
      }
    })();
  }, [authRequired]);

  const handleRemove = async (id) => {
    if (!window.confirm('Remove this medicine from your cabinet?')) return;
    try {
      await removeFromCabinet(id);
      setItems((prev) => prev.filter((item) => itemId(item) !== id));
      // The pairwise findings just changed; re-ask rather than recompute here.
      setInteractions(await getCabinetInteractions());
    } catch (err) {
      setError(errorDetail(err, 'Failed to remove.'));
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
      {/* Sidebar: what is in the cabinet */}
      <div className="card" style={{ padding: '1.5rem', maxHeight: 'calc(100vh - 120px)', overflowY: 'auto' }}>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
          <FiPackage style={{ color: 'var(--accent)' }} /> Your cabinet
        </h2>

        {error && <p style={{ color: 'var(--danger)', fontSize: '0.85rem' }}>{error}</p>}

        {items.length === 0 && !error ? (
          <p style={{ color: 'var(--text-muted)' }}>
            Nothing here yet. Identify a medicine on the{' '}
            <Link to="/">scan page</Link> and add it — only confidently
            identified medicines can be added.
          </p>
        ) : (
          <div className="history-list">
            {items.map((item) => (
              <div key={itemId(item)} className="history-item" style={{ cursor: 'default' }}>
                <FiPackage style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
                <div className="info">
                  <h4 style={{ overflowWrap: 'anywhere' }}>{item.display_name}</h4>
                  <p>
                    {item.added_at ? new Date(item.added_at).toLocaleDateString() : ''}
                  </p>
                  <div style={{ marginTop: '0.25rem' }}><SourceLine source={item.source} /></div>
                </div>
                <button
                  onClick={() => handleRemove(itemId(item))}
                  title="Remove"
                  style={{ background: 'none', border: 'none', color: 'var(--text-muted)',
                           cursor: 'pointer', padding: '0.5rem' }}
                >
                  <FiTrash2 />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Main area: interaction findings across the cabinet */}
      <InteractionsPanel
        findings={normaliseFindings(interactions)}
        duplicates={normaliseDuplicates(interactions)}
        itemCount={items.length}
      />
    </div>
  );
}
