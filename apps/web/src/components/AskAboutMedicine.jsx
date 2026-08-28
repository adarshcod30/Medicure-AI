import { useState, useRef, useEffect } from 'react';
import { FiSend, FiCheckCircle, FiAlertTriangle } from 'react-icons/fi';
import { askAboutMedicine } from '../services/api';

/**
 * Follow-up questions about the medicine currently on screen.
 *
 * Two answer kinds, and they must never look alike. A grounded answer traces
 * every claim to the retrieved records; an unverified one is the model talking
 * from memory because the databases had nothing. Rendering them identically
 * would undo the whole architecture, so they get different borders, different
 * icons and — for unverified — the disclaimer inline rather than in a footnote
 * nobody reads.
 */
export default function AskAboutMedicine({ result }) {
  const [turns, setTurns] = useState([]);
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const endRef = useRef(null);

  const id = result?.identification || {};
  const subject = id.closest_brand || id.composition || '';
  const answerable = Boolean(subject) && id.status !== 'unreadable';

  useEffect(() => { setTurns([]); }, [subject]);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [turns]);

  const ask = async (e) => {
    e.preventDefault();
    const q = question.trim();
    if (!q || busy) return;
    setQuestion('');
    setBusy(true);
    // History is what the server uses to resolve pronouns ("its side effects"),
    // so it carries both sides of the conversation.
    const history = turns.flatMap((t) => [
      { role: 'user', text: t.question },
      { role: 'assistant', text: t.answer?.text || '' },
    ]).filter((m) => m.text);

    try {
      const data = await askAboutMedicine(q, subject, history);
      setTurns((prev) => [...prev, { question: q, answer: data.answer }]);
    } catch (err) {
      setTurns((prev) => [...prev, {
        question: q,
        answer: {
          text: null,
          grounded: true,
          reason: err?.response?.data?.detail
            || 'Could not reach the answering service. The facts above are unaffected.',
        },
      }]);
    } finally {
      setBusy(false);
    }
  };

  if (!answerable) return null;

  return (
    <section style={{ marginTop: '1.5rem' }}>
      <h3 style={{ fontSize: '1rem', margin: '0 0 0.15rem' }}>Ask about this medicine</h3>
      <p style={{ margin: '0 0 0.75rem', fontSize: '0.82rem', color: '#64748b' }}>
        Answers come from the retrieved records above. Anything they do not cover
        is answered by the model and clearly marked unverified.
      </p>

      {turns.map((turn, i) => {
        const a = turn.answer || {};
        const grounded = a.grounded !== false;
        return (
          <div key={i} style={{ marginBottom: '0.85rem' }}>
            <div style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: '0.3rem' }}>
              {turn.question}
            </div>
            <div style={{
              padding: '0.7rem 0.85rem',
              borderRadius: 10,
              fontSize: '0.9rem',
              lineHeight: 1.55,
              background: grounded ? '#f0fdf4' : '#fffbeb',
              border: `1px solid ${grounded ? '#bbf7d0' : '#fde68a'}`,
            }}>
              <div style={{
                display: 'flex', alignItems: 'center', gap: '0.4rem',
                fontSize: '0.75rem', fontWeight: 600, letterSpacing: '0.02em',
                color: grounded ? '#15803d' : '#b45309', marginBottom: '0.35rem',
              }}>
                {grounded ? <FiCheckCircle size={13} /> : <FiAlertTriangle size={13} />}
                {grounded ? 'FROM THE RECORDS ABOVE' : 'NOT VERIFIED — MODEL KNOWLEDGE'}
              </div>
              <div style={{ color: '#1e293b' }}>{a.text || a.reason}</div>
              {!grounded && a.disclaimer && (
                <div style={{ marginTop: '0.45rem', fontSize: '0.78rem', color: '#92400e' }}>
                  {a.disclaimer}
                </div>
              )}
            </div>
          </div>
        );
      })}
      <div ref={endRef} />

      <form onSubmit={ask} style={{ display: 'flex', gap: '0.5rem' }}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. what are the side effects?"
          maxLength={400}
          disabled={busy}
          style={{
            flex: 1, padding: '0.6rem 0.8rem', borderRadius: 8,
            border: '1px solid #cbd5e1', fontSize: '0.9rem',
          }}
        />
        <button type="submit" disabled={busy || !question.trim()}
          style={{
            padding: '0.6rem 0.9rem', borderRadius: 8, border: 'none',
            background: busy ? '#94a3b8' : '#0f766e', color: 'white',
            cursor: busy ? 'default' : 'pointer', display: 'flex',
            alignItems: 'center', gap: '0.4rem', fontSize: '0.9rem',
          }}>
          <FiSend size={14} />{busy ? 'Asking…' : 'Ask'}
        </button>
      </form>
    </section>
  );
}
