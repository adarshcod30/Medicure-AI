const express = require('express');
const axios = require('axios');
const { getDb } = require('../db/schema');

const router = express.Router();

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

// POST /api/chat — Send a message in the context of a scan
router.post('/', async (req, res) => {
  try {
    const { scan_id, message, target_language } = req.body;

    if (!scan_id || !message) {
      return res.status(400).json({ error: 'scan_id and message are required' });
    }

    const db = getDb();

    // Load medicine context from the scan
    const scan = db.prepare('SELECT result_json FROM scans WHERE id = ?').get(scan_id);
    if (!scan) {
      return res.status(404).json({ error: 'Scan not found. Please upload a medicine image first.' });
    }

    const medicineContext = JSON.parse(scan.result_json || '{}');

    // Load chat history for this scan
    const history = db.prepare(
      'SELECT role, content FROM chat_messages WHERE scan_id = ? ORDER BY created_at ASC LIMIT 20'
    ).all(scan_id);

    // Store user message
    db.prepare(
      'INSERT INTO chat_messages (scan_id, role, content) VALUES (?, ?, ?)'
    ).run(scan_id, 'user', message);

    // Forward to ML microservice
    let mlResponse;
    try {
      mlResponse = await axios.post(`${ML_SERVICE_URL}/chat`, {
        medicine_context: medicineContext,
        message: message,
        history: history,
        target_language: target_language || 'en-US'
      }, { timeout: 30000 });
    } catch (mlError) {
      console.error('ML chat error:', mlError.message);
      return res.status(503).json({
        error: 'ML service unavailable for chat',
        details: mlError.message
      });
    }

    const assistantReply = mlResponse.data.reply || mlResponse.data.response;

    // Store assistant reply
    db.prepare(
      'INSERT INTO chat_messages (scan_id, role, content) VALUES (?, ?, ?)'
    ).run(scan_id, 'assistant', assistantReply);

    res.json({
      response: assistantReply,
      scan_id: scan_id
    });
  } catch (err) {
    console.error('Chat error:', err);
    res.status(500).json({ error: 'Internal server error', details: err.message });
  }
});

// GET /api/chat/:scan_id — Get chat history for a scan
router.get('/:scan_id', (req, res) => {
  try {
    const db = getDb();
    const messages = db.prepare(
      'SELECT role, content, created_at FROM chat_messages WHERE scan_id = ? ORDER BY created_at ASC'
    ).all(req.params.scan_id);

    res.json({ messages });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
