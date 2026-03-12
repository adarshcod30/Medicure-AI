import { useRef, useState, useCallback } from 'react';
import { FiUploadCloud, FiX } from 'react-icons/fi';
import { locales } from '../locales';

export default function ImageUpload({ onImageSelected, disabled, lang = 'en-US' }) {
  const [preview, setPreview] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef(null);

  const t = locales[lang] || locales['en-US'];

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return;
    setPreview(URL.createObjectURL(file));
    onImageSelected(file);
  }, [onImageSelected]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [handleFile]);

  const handleChange = (e) => {
    handleFile(e.target.files[0]);
  };

  const removeImage = () => {
    setPreview(null);
    onImageSelected(null);
    if (fileRef.current) fileRef.current.value = '';
  };

  return (
    <div>
      {!preview ? (
        <div
          className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <FiUploadCloud className="icon" />
          <p>
            {t.upload_title} — <span className="highlight">browse files</span>
          </p>
          <p style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Supports JPG, PNG, WEBP — photos of medicine strips, boxes, or labels
          </p>
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            onChange={handleChange}
            style={{ display: 'none' }}
            disabled={disabled}
          />
        </div>
      ) : (
        <div className="preview-container">
          <img src={preview} alt="Medicine preview" />
          {!disabled && (
            <button className="remove-btn" onClick={removeImage} title="Remove image">
              <FiX />
            </button>
          )}
        </div>
      )}
    </div>
  );
}
