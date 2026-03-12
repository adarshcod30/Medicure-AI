const express = require('express');
const multer = require('multer');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const fs = require('fs');
const { getDb } = require('../db/schema');

const router = express.Router();

// Configure Multer storage
const uploadDir = path.join(__dirname, '..', 'uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, `${uuidv4()}${ext}`);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
  fileFilter: (req, file, cb) => {
    const allowed = ['.jpg', '.jpeg', '.png', '.webp', '.bmp'];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowed.includes(ext)) cb(null, true);
    else cb(new Error('Only image files (jpg, png, webp, bmp) are allowed'));
  }
});

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

// POST /api/scan — Upload image and get medicine analysis
router.post('/', upload.single('image'), async (req, res) => {
  const startTime = Date.now();

  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file uploaded' });
    }

    const scanId = uuidv4();
    const imagePath = req.file.filename;

    // Forward to ML microservice
    const FormData = require('form-data');
    const formData = new FormData();
    formData.append('image', fs.createReadStream(req.file.path));

    let mlResponse;
    try {
      mlResponse = await axios.post(`${ML_SERVICE_URL}/analyze`, formData, {
        headers: { ...formData.getHeaders() },
        timeout: 60000 // 60 second timeout for LLM calls
      });
    } catch (mlError) {
      console.error('ML Service error:', mlError.message);
      return res.status(503).json({
        error: 'ML service unavailable. Please ensure the Python microservice is running.',
        details: mlError.message
      });
    }

    const result = mlResponse.data;

    // Store in SQLite
    const db = getDb();
    db.prepare(`
      INSERT INTO scans (id, user_id, image_path, result_json, ocr_text, ocr_confidence, method)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).run(
      scanId,
      req.user?.id || null, // Changed from 'anonymous' to avoid FK errors
      imagePath,
      JSON.stringify(result),
      result.ocr_text || '',
      result.confidence || 0,
      result.method || 'ocr'
    );

    // Record metric
    const elapsed = Date.now() - startTime;
    db.prepare(`
      INSERT INTO system_metrics (endpoint, response_time_ms, status_code)
      VALUES (?, ?, ?)
    `).run('/api/scan', elapsed, 200);

    res.json({
      scan_id: scanId,
      image_url: `/uploads/${imagePath}`,
      ...result
    });
  } catch (err) {
    console.error('Scan error:', err);
    res.status(500).json({ error: 'Internal server error', details: err.message });
  }
});

// GET /api/scan/:id — Get a specific scan result
router.get('/:id', (req, res) => {
  try {
    const db = getDb();
    const scan = db.prepare('SELECT * FROM scans WHERE id = ?').get(req.params.id);
    if (!scan) return res.status(404).json({ error: 'Scan not found' });

    res.json({
      ...scan,
      result_json: JSON.parse(scan.result_json || '{}')
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
