const express = require('express');
const { getDb } = require('../db/schema');

const router = express.Router();

// GET /api/history — Get scan history (paginated)
router.get('/', (req, res) => {
  try {
    const db = getDb();
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 20;
    const offset = (page - 1) * limit;

    const userId = req.user?.id || 'anonymous';

    const scans = db.prepare(`
      SELECT id, image_path, result_json, ocr_confidence, method, created_at
      FROM scans
      WHERE user_id = ?
      ORDER BY created_at DESC
      LIMIT ? OFFSET ?
    `).all(userId, limit, offset);

    const total = db.prepare(
      'SELECT COUNT(*) as count FROM scans WHERE user_id = ?'
    ).get(userId);

    const formattedScans = scans.map(scan => {
      let result = {};
      try { result = JSON.parse(scan.result_json || '{}'); } catch {}
      return {
        id: scan.id,
        image_url: `/uploads/${scan.image_path}`,
        brand_name: result.brand_name || 'Unknown',
        generic_name: result.generic_name || '',
        confidence: scan.ocr_confidence,
        method: scan.method,
        created_at: scan.created_at
      };
    });

    res.json({
      scans: formattedScans,
      pagination: {
        page,
        limit,
        total: total.count,
        pages: Math.ceil(total.count / limit)
      }
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// DELETE /api/history/:id — Delete a scan and its chat history
router.delete('/:id', (req, res) => {
  try {
    const db = getDb();
    db.prepare('DELETE FROM chat_messages WHERE scan_id = ?').run(req.params.id);
    db.prepare('DELETE FROM scans WHERE id = ?').run(req.params.id);
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
