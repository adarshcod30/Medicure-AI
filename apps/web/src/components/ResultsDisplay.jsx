import { useState } from 'react';
import {
  FiAlertTriangle, FiCamera, FiCheckCircle, FiChevronDown, FiChevronRight,
  FiHelpCircle, FiPlus, FiSlash, FiTag, FiTrendingDown,
} from 'react-icons/fi';
import {
  addToCabinet, isAuthenticated, isStorageDisabled, errorDetail,
} from '../services/api';

/**
 * Renders the grounded response contract.
 *
 * The design goal is that a user can see WHY the system believes what it says,
 * and — more importantly — see clearly when it does not believe anything. The
 * abstention states are given the most visual weight on the page, because
 * "I am not sure which medicine this is" is the most important thing this
 * system can tell someone, and burying it under a confident-looking summary
 * would defeat the entire architecture behind it.
 *
 * Every number carries its source. Prices show their arithmetic. Alternatives
 * name the dataset they came from.
 */

const STATUS = {
  confident: {
    icon: FiCheckCircle,
    color: 'var(--success)',
    bg: 'var(--success-bg)',
    border: 'var(--border-accent)',
    label: 'Identified',
  },
  ambiguous: {
    icon: FiHelpCircle,
    color: 'var(--warning)',
    bg: 'var(--warning-bg)',
    border: 'rgba(245, 158, 11, 0.35)',
    label: 'More than one possibility',
  },
  abstained: {
    icon: FiSlash,
    color: 'var(--danger)',
    bg: 'var(--danger-bg)',
    border: 'rgba(239, 68, 68, 0.35)',
    label: 'Not confident enough to say',
  },
  unreadable: {
    icon: FiCamera,
    color: 'var(--danger)',
    bg: 'var(--danger-bg)',
    border: 'rgba(239, 68, 68, 0.35)',
    label: 'Photo could not be read',
  },
};

function Section({ title, children, right }) {
  return (
    <section style={{ marginTop: '1.25rem' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
        <h3 style={{ fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.06em',
                     color: 'var(--text-secondary)', margin: '0 0 0.5rem' }}>{title}</h3>
        {right}
      </div>
      {children}
    </section>
  );
}

function Collapsible({ label, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{ marginTop: '0.5rem' }}>
      <button
        onClick={() => setOpen(!open)}
        style={{ background: 'none', border: 'none', padding: 0, cursor: 'pointer',
                 color: 'var(--text-secondary)', fontSize: '0.82rem', display: 'flex', alignItems: 'center',
                 gap: '0.3rem' }}
      >
        {open ? <FiChevronDown /> : <FiChevronRight />} {label}
      </button>
      {open && <div style={{ marginTop: '0.5rem' }}>{children}</div>}
    </div>
  );
}

function Source({ source }) {
  if (!source) return null;
  const label = {
    a_z_medicines_india: 'A–Z Medicines of India',
    jan_aushadhi_pmbjp: 'Jan Aushadhi (PMBJP)',
    master_medicines_final: 'NPPA ceiling price data',
  }[source.dataset] || source.dataset;

  return (
    <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
      <FiTag style={{ verticalAlign: '-1px' }} /> {label}
      {source.record_id ? ` · record ${source.record_id}` : ''}
      {source.caveat ? ` · ${source.caveat}` : ''}
    </span>
  );
}

/** Confidence bar. Shows the abstention threshold as a marked line. */
function Confidence({ probability, calibrated, threshold = 0.83 }) {
  const pct = Math.round(probability * 100);
  return (
    <div style={{ marginTop: '0.6rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem',
                    color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
        <span>{calibrated ? 'Calibrated confidence' : 'Raw similarity (not calibrated)'}</span>
        <strong>{pct}%</strong>
      </div>
      <div style={{ position: 'relative', height: 8, background: 'var(--border)', borderRadius: 4 }}>
        <div style={{ width: `${pct}%`, height: '100%', borderRadius: 4,
                      background: probability >= threshold ? 'var(--success)' : 'var(--warning)' }} />
        <div title={`answers above ${Math.round(threshold * 100)}%`}
             style={{ position: 'absolute', left: `${threshold * 100}%`, top: -3, bottom: -3,
                      width: 2, background: 'var(--text-secondary)' }} />
      </div>
      {calibrated && (
        <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', margin: '0.35rem 0 0' }}>
          Fitted on held-out data: answers scored {pct}% are correct about {pct}% of the time.
          The marker is the threshold below which this system declines to answer.
        </p>
      )}
    </div>
  );
}

function PriceCheck({ price }) {
  if (!price) return null;

  const tone = {
    verified_over_ceiling: { color: 'var(--danger)', icon: FiAlertTriangle },
    verified_within_ceiling: { color: 'var(--success)', icon: FiCheckCircle },
  }[price.status] || { color: 'var(--text-secondary)', icon: FiHelpCircle };
  const Icon = tone.icon;

  return (
    <Section title="Price check">
      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
        <Icon style={{ color: tone.color, flexShrink: 0, marginTop: 3 }} />
        <p style={{ margin: 0, color: 'var(--text-primary)', lineHeight: 1.5 }}>{price.message}</p>
      </div>

      {price.workings?.length > 0 && (
        <Collapsible label="Show the arithmetic">
          <pre style={{ margin: 0, padding: '0.6rem 0.75rem', background: 'var(--bg-secondary)',
                        border: '1px solid #e2e8f0', borderRadius: 6, fontSize: '0.76rem',
                        color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' }}>
            {price.workings.join('\n')}
          </pre>
          <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.4rem' }}>
            Computed from retrieved records, not generated by a language model.
          </p>
        </Collapsible>
      )}

      {price.ceiling_source && (
        <div style={{ marginTop: '0.5rem' }}><Source source={price.ceiling_source.source} /></div>
      )}
    </Section>
  );
}

function Alternatives({ alternatives }) {
  if (!alternatives) return null;

  const items = alternatives.alternatives || [];
  const plausible = items.filter((a) => !a.implausible);

  return (
    <Section
      title="Cheaper equivalents"
      right={
        alternatives.jan_aushadhi_available ? (
          <span style={{ fontSize: '0.72rem', color: 'var(--success)' }}>Jan Aushadhi available</span>
        ) : null
      }
    >
      <p style={{ margin: 0, color: 'var(--text-primary)', lineHeight: 1.5 }}>{alternatives.message}</p>

      {plausible.length === 0 ? (
        <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
          Nothing is listed here because nothing was found. This system does not suggest
          substitutes it cannot point to in a real dataset.
        </p>
      ) : (
        <ul style={{ listStyle: 'none', padding: 0, margin: '0.75rem 0 0' }}>
          {plausible.map((alt) => (
            <li key={`${alt.kind}-${alt.name}`}
                style={{ padding: '0.6rem 0.75rem', border: '1px solid #e2e8f0',
                         borderRadius: 8, marginBottom: '0.5rem', background: 'var(--bg-card)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem' }}>
                <div>
                  <strong style={{ color: 'var(--text-primary)' }}>{alt.name}</strong>
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', marginTop: 2 }}>
                    {alt.kind === 'jan_aushadhi' ? 'Jan Aushadhi Kendra' : alt.manufacturer}
                    {alt.pack?.label ? ` · ${alt.pack.label}` : ''}
                    {alt.form_differs ? ' · different dosage form' : ''}
                  </div>
                </div>
                <div style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                  <div style={{ color: 'var(--success)', fontWeight: 600 }}>
                    <FiTrendingDown style={{ verticalAlign: '-2px' }} />{' '}
                    {Math.round(alt.saving_percent)}% less
                  </div>
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)' }}>
                    ₹{alt.price_per_unit?.toFixed(2)} per unit
                  </div>
                </div>
              </div>
              <div style={{ marginTop: '0.4rem' }}><Source source={alt.source} /></div>
            </li>
          ))}
        </ul>
      )}

      {items.length > plausible.length && (
        <p style={{ fontSize: '0.72rem', color: 'var(--warning)', marginTop: '0.5rem' }}>
          {items.length - plausible.length} further listing(s) were hidden: their prices imply
          savings above 90%, which usually means the source pack size does not match the
          recorded price rather than a genuine bargain.
        </p>
      )}
    </Section>
  );
}

/**
 * "Add to cabinet" — rendered only when there is something real to add: the
 * user is signed in, the identification is confident, and the payload carries
 * the composition signature. The signature is passed through byte-for-byte;
 * this component never rebuilds one from display text, because display text
 * is for people and the signature is the identity the resolver computed.
 */
function AddToCabinet({ data }) {
  const [state, setState] = useState('idle'); // idle | busy | added
  const [notice, setNotice] = useState(null);

  const id = data.identification || {};
  const signature = id.signature;
  if (!isAuthenticated()) return null;
  if (id.status !== 'confident') return null;
  if (!Array.isArray(signature) || signature.length === 0) return null;

  const handleAdd = async () => {
    setState('busy');
    setNotice(null);
    try {
      await addToCabinet({
        display_name: id.closest_brand || id.composition,
        signature,
        source: data.reference_product?.source ?? null,
      });
      setState('added');
    } catch (err) {
      setState('idle');
      setNotice(
        isStorageDisabled(err)
          ? 'Accounts are disabled on this deployment, so there is no cabinet to add to.'
          : errorDetail(err, 'Could not add to cabinet.'),
      );
    }
  };

  return (
    <div style={{ marginTop: '0.9rem' }}>
      {state === 'added' ? (
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem',
                       color: 'var(--success)', fontSize: '0.85rem', fontWeight: 600 }}>
          <FiCheckCircle /> In your cabinet — interactions are checked on the cabinet page
        </span>
      ) : (
        <button
          onClick={handleAdd}
          disabled={state === 'busy'}
          style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem',
                   padding: '0.45rem 0.9rem', borderRadius: 8, cursor: 'pointer',
                   border: '1px solid #bbf7d0', background: 'var(--success-bg)', color: 'var(--success)',
                   fontSize: '0.85rem', fontWeight: 600,
                   opacity: state === 'busy' ? 0.6 : 1 }}
        >
          <FiPlus /> {state === 'busy' ? 'Adding…' : 'Add to cabinet'}
        </button>
      )}
      {notice && (
        <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', margin: '0.4rem 0 0' }}>{notice}</p>
      )}
    </div>
  );
}

function ImageQuality({ quality }) {
  if (!quality) return null;
  const good = quality.verdict === 'good';

  return (
    <Collapsible label={`Image quality: ${quality.verdict}`}>
      <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
        <div>Focus (variance of Laplacian): {quality.blur_variance}</div>
        <div>Glare coverage: {(quality.glare_fraction * 100).toFixed(1)}%</div>
        <div>Text-scale contrast: {quality.text_contrast}</div>
        <div>Skew corrected: {quality.skew_deg}°</div>
        <div>Packet boundary: {quality.boundary_method}</div>
        <div>Perspective rectified: {quality.rectified ? 'yes' : 'no'}</div>
        <div>Denoising: {quality.denoise_method}</div>
        {!good && quality.reasons?.length > 0 && (
          <div style={{ marginTop: '0.4rem', color: 'var(--warning)' }}>
            Issues: {quality.reasons.join('; ')}
          </div>
        )}
        <div style={{ marginTop: '0.4rem', color: 'var(--text-secondary)' }}>
          Processing stages: {(quality.dip_stages_applied || []).join(' → ')}
        </div>
      </div>
    </Collapsible>
  );
}

export default function ResultsDisplay({ data }) {
  if (!data) return null;

  const id = data.identification || {};
  const facts = data.facts;
  const fallback = data.fallback;
  const style = STATUS[id.status] || STATUS.abstained;
  const Icon = style.icon;
  const answered = id.status === 'confident' || id.status === 'ambiguous';

  return (
    <div>
      {/* The verdict leads. When the system is unsure, that is the headline. */}
      <div style={{ padding: '1rem 1.15rem', borderRadius: 12, background: style.bg,
                    border: `1px solid ${style.border}` }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: style.color }}>
          <Icon size={18} />
          <strong style={{ fontSize: '0.9rem', letterSpacing: '0.01em' }}>{style.label}</strong>
        </div>

        {/* The composition is a HEADLINE only when the system is actually
            answering. Rendering it unconditionally was a real defect: a
            crumpled Becosules strip abstained at 1% confidence and still put
            "ibandronic acid 150mg" on screen at 1.35rem, directly under the
            words "Not confident enough to say". Someone glancing at that sees
            a drug name, not a refusal — which is precisely the confident-wrong
            impression the abstention exists to prevent.

            When abstaining, `id.reason` already names the closest match in a
            sentence that frames it ("the closest match is X, but confidence is
            only 1%"). That is the honest place for it: in context, at body
            size, hedged. */}
        {id.composition && answered && (
          <h2 style={{ margin: '0.6rem 0 0', fontSize: '1.35rem', color: 'var(--text-primary)' }}>
            {id.composition}
          </h2>
        )}

        {id.closest_brand && answered && (
          <p style={{ margin: '0.3rem 0 0', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
            Closest product: {id.closest_brand}
            {id.brands_sharing_composition > 1 &&
              ` — ${id.brands_sharing_composition} products share this exact composition`}
          </p>
        )}

        {id.reason && (
          <p style={{ margin: '0.7rem 0 0', color: 'var(--text-primary)', lineHeight: 1.55 }}>{id.reason}</p>
        )}

        {id.status !== 'unreadable' && (
          <Confidence probability={id.probability} calibrated={id.calibrated} />
        )}

        <AddToCabinet data={data} />
      </div>

      {/* Uses and side effects, straight from the dataset. Rendered inside the
          verdict card's sibling so they read as part of the identified result,
          and each carries the source the API supplied. */}
      {facts && (facts.uses?.length > 0 || facts.side_effects?.length > 0) && (
        <section style={{ marginTop: '1.1rem' }}>
          {facts.uses?.length > 0 && (
            <div style={{ marginBottom: '0.7rem' }}>
              <h4 style={{ margin: '0 0 0.3rem', fontSize: '0.9rem' }}>What it is used for</h4>
              <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>{facts.uses.join(' · ')}</div>
            </div>
          )}
          {facts.side_effects?.length > 0 && (
            <div style={{ marginBottom: '0.7rem' }}>
              <h4 style={{ margin: '0 0 0.3rem', fontSize: '0.9rem' }}>Possible side effects</h4>
              <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                {facts.side_effects.slice(0, 10).join(' · ')}
                {facts.side_effects.length > 10 && ` (+${facts.side_effects.length - 10} more)`}
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                Possible effects recorded in a dataset — not effects you will have.
              </div>
            </div>
          )}
          {facts.source?.dataset && (
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              source: {facts.source.dataset}
            </div>
          )}
        </section>
      )}

      {/* The model answering for a product the catalogue does not carry. Amber,
          never green, and never merged with the cited fields above. */}
      {fallback?.available && (
        <section style={{
          marginTop: '1.1rem', padding: '0.8rem 0.9rem', borderRadius: 10,
          background: 'var(--warning-bg)', border: '1px solid #fde68a',
        }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--warning)', marginBottom: '0.35rem' }}>
            NOT VERIFIED — MODEL KNOWLEDGE
          </div>
          <div style={{ fontSize: '0.9rem', color: 'var(--text-primary)', lineHeight: 1.55 }}>{fallback.text}</div>
          <div style={{ fontSize: '0.78rem', color: 'var(--warning)', marginTop: '0.45rem' }}>
            {fallback.disclaimer}
          </div>
        </section>
      )}

      <div style={{ display: 'none' }}>
      </div>

      {data.explanation?.text && (
        <Section title="In plain words">
          <p style={{ margin: 0, lineHeight: 1.65, color: 'var(--text-primary)' }}>{data.explanation.text}</p>
          <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
            Written by a language model from the retrieved facts above, and checked against
            them. It states nothing that was not retrieved.
          </p>
        </Section>
      )}

      {data.explanation && !data.explanation.available && data.explanation.note && (
        <Section title="In plain words">
          <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
            {data.explanation.note}
          </p>
        </Section>
      )}

      {answered && <PriceCheck price={data.price_check} />}
      {answered && <Alternatives alternatives={data.alternatives} />}

      {id.candidates_considered?.length > 1 && (
        <Section title="Other possibilities considered">
          <ul style={{ margin: 0, paddingLeft: '1.1rem', color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
            {id.candidates_considered.slice(1, 5).map((c) => (
              <li key={c.composition} style={{ marginBottom: '0.2rem' }}>
                {c.composition} — similarity {c.top_similarity.toFixed(2)} (e.g. {c.closest_brand})
              </li>
            ))}
          </ul>
        </Section>
      )}

      <Section title="How this was worked out">
        <ImageQuality quality={data.image_quality} />
        {data.timing_ms && (
          <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
            {Object.entries(data.timing_ms)
              .map(([k, v]) => `${k} ${v}ms`)
              .join(' · ')}
          </div>
        )}
      </Section>

      <p style={{ marginTop: '1.5rem', padding: '0.75rem 0.9rem', background: 'var(--bg-secondary)',
                  borderRadius: 8, fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
        {data.disclaimer}
      </p>
    </div>
  );
}
