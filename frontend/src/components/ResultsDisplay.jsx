import { useState } from 'react';
import {
  FiChevronDown, FiHeart, FiAlertCircle, FiShield,
  FiZap, FiPackage, FiDollarSign, FiInfo, FiActivity, FiX
} from 'react-icons/fi';
import { locales } from '../locales';

function AccordionItem({ title, icon: Icon, items, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);

  if (!items || items.length === 0) return null;

  return (
    <div className={`accordion-item ${open ? 'open' : ''}`}>
      <button className="accordion-header" onClick={() => setOpen(!open)}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Icon className="icon" />
          {title}
        </span>
        <FiChevronDown className="chevron" />
      </button>
      {open && (
        <div className="accordion-body">
          <ul>
            {items.map((item, i) => (
              <li key={i}>{typeof item === 'object' ? item.name || JSON.stringify(item) : item}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default function ResultsDisplay({ data, lang = 'en-US' }) {
  if (!data) return null;

  const t = locales[lang] || locales['en-US'];

  const {
    brand_name, generic_name, composition, form, manufacturer,
    indications, uses, side_effects, precautions, contraindications, interactions, 
    dosage, storage, warnings, simplified_explanation, confidence, method,
    price, nppa_ceiling_price, price_flag, cheaper_alternatives, schedule_type
  } = data;

  return (
    <div>
      {/* Simplified explanation box */}
      {simplified_explanation && (
        <div className="simplified-box">
          <h3>✨ {t.simplified_explanation || 'In Simple Words'}</h3>
          <p>{simplified_explanation}</p>
        </div>
      )}

      {/* Medicine header */}
      <div className="medicine-header">
        <div>
          <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem', flexWrap: 'wrap' }}>
            {form && <span className="badge badge-generic">{form}</span>}
            {schedule_type && <span className="badge" style={{ background: 'var(--accent)', color: 'white' }}>{schedule_type}</span>}
            {price_flag === 'overpriced' && <span className="badge badge-overpriced">{t.overpriced}</span>}
            {price_flag === 'ok' && <span className="badge badge-ok">{t.fair_price}</span>}
            {method === 'vision_fallback' && <span className="badge badge-branded">Vision AI</span>}
          </div>
          <h2>{brand_name || t.brand_name || 'Unknown Medicine'}</h2>
          {generic_name && <p className="generic-name">{generic_name}</p>}
          {composition && <p className="composition"><strong>{t.composition}:</strong> {composition}</p>}
          {manufacturer && (
            <p className="composition" style={{ marginTop: '0.25rem' }}>
               {t.manufacturer}: {manufacturer}
            </p>
          )}
          <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.75rem', flexWrap: 'wrap' }}>
            {price && (
              <span style={{ color: 'var(--accent)', fontWeight: 600, fontSize: '1.1rem' }}>
                ₹{price}
              </span>
            )}
            {nppa_ceiling_price && nppa_ceiling_price !== '' && (
              <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                NPPA Ceiling: ₹{nppa_ceiling_price}
              </span>
            )}
          </div>
          {confidence > 0 && (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '0.5rem' }}>
              Confidence: {(confidence * 100).toFixed(0)}%
            </p>
          )}
        </div>
      </div>

      {/* Accordion sections */}
      <div className="accordion">
        <AccordionItem title={t.indications} icon={FiActivity} items={indications} defaultOpen />
        <AccordionItem title={t.uses} icon={FiHeart} items={uses} />
        <AccordionItem title={t.side_effects} icon={FiAlertCircle} items={side_effects} />
        <AccordionItem title={t.precautions} icon={FiShield} items={precautions} />
        <AccordionItem title={t.contraindications} icon={FiX} items={contraindications} />
        <AccordionItem title={t.interactions} icon={FiZap} items={interactions} />
        <AccordionItem title={t.warnings} icon={FiInfo} items={warnings} />
        {dosage && (
          <AccordionItem title={t.dosage} icon={FiPackage} items={[dosage]} />
        )}
        {storage && (
          <AccordionItem title={t.storage} icon={FiPackage} items={[storage]} />
        )}
      </div>

      {/* Generic alternatives */}
      {cheaper_alternatives && cheaper_alternatives.length > 0 && (
        <div style={{ marginTop: '2rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <FiDollarSign style={{ color: 'var(--accent)' }} />
            {t.cheaper_alternatives}
          </h3>
          <div className="alternatives-grid">
            {cheaper_alternatives.map((alt, i) => (
              <div className="alt-card" key={i}>
                <div className="alt-name">{alt.name}</div>
                <div className="alt-price">{alt.price}</div>
                <div className="alt-source">
                  {alt.source === 'jan_aushadhi' ? 'Jan Aushadhi' : alt.source}
                  {alt.drug_code && ` • Code: ${alt.drug_code}`}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
