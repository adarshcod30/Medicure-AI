const express = require('express');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const { getDb } = require('../db/schema');

const router = express.Router();

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

// POST /api/search — Search medicine by name (text-based lookup)
router.post('/', async (req, res) => {
  const startTime = Date.now();

  try {
    const { name, target_language } = req.body;

    if (!name || !name.trim()) {
      return res.status(400).json({ error: 'Medicine name is required' });
    }

    const scanId = uuidv4();

    // Forward to ML microservice
    let mlResponse;
    try {
      mlResponse = await axios.post(`${ML_SERVICE_URL}/search`, {
        name: name.trim(),
        target_language: target_language || 'en-US'
      }, { timeout: 60000 });
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
      req.user?.id || null,
      '',
      JSON.stringify(result.data || result),
      name.trim(),
      1.0,
      'text_search'
    );

    // Record metric
    const elapsed = Date.now() - startTime;
    db.prepare(`
      INSERT INTO system_metrics (endpoint, response_time_ms, status_code)
      VALUES (?, ?, ?)
    `).run('/api/search', elapsed, 200);

    res.json({
      scan_id: scanId,
      ...result
    });
  } catch (err) {
    console.error('Search error:', err);
    res.status(500).json({ error: 'Internal server error', details: err.message });
  }
});

module.exports = router;
