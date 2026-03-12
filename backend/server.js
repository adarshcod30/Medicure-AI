require('dotenv').config({ path: require('path').join(__dirname, '..', '.env') });

const express = require('express');
const cors = require('cors');
const path = require('path');
const { getDb } = require('./db/schema');
const { authMiddleware } = require('./middleware/auth');

const app = express();
const PORT = process.env.BACKEND_PORT || 3001;

// ─── Middleware ──────────────────────────────────────────
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:5173',
  credentials: true
}));
app.use(express.json({ limit: '10mb' }));

// Serve uploaded images as static files
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// ─── Request logging ────────────────────────────────────
app.use((req, res, next) => {
  const start = Date.now();
  res.on('finish', () => {
    const elapsed = Date.now() - start;
    console.log(`${req.method} ${req.originalUrl} → ${res.statusCode} (${elapsed}ms)`);
  });
  next();
});

// ─── Health check (no auth required) ────────────────────
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'MediCure Backend Gateway',
    timestamp: new Date().toISOString()
  });
});

// ─── Routes ─────────────────────────────────────────────
app.use('/api/auth', require('./routes/auth'));
app.use('/api/scan', authMiddleware, require('./routes/scan'));
app.use('/api/search', authMiddleware, require('./routes/search'));
app.use('/api/chat', authMiddleware, require('./routes/chat'));
app.use('/api/history', authMiddleware, require('./routes/history'));

// ─── Error handler ──────────────────────────────────────
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  if (err.code === 'LIMIT_FILE_SIZE') {
    return res.status(413).json({ error: 'File too large. Maximum size is 10MB.' });
  }
  res.status(500).json({ error: 'Internal server error' });
});

// ─── Start server ───────────────────────────────────────
app.listen(PORT, () => {
  getDb(); // Initialize database
  console.log(`
  ╔══════════════════════════════════════════════╗
  ║   🏥 MediCure Backend Gateway               ║
  ║   Running on http://localhost:${PORT}          ║
  ║   Environment: ${process.env.NODE_ENV || 'development'}             ║
  ╚══════════════════════════════════════════════╝
  `);
});

module.exports = app;
