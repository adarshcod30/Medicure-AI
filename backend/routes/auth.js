const express = require('express');
const jwt = require('jsonwebtoken');
const { v4: uuidv4 } = require('uuid');
const { getDb } = require('../db/schema');
const { generateToken, JWT_SECRET } = require('../middleware/auth');

const router = express.Router();

// POST /api/auth/google — Validate Google OAuth token and return JWT
router.post('/google', async (req, res) => {
  try {
    const { credential } = req.body;

    if (!credential) {
      return res.status(400).json({ error: 'Credential is required' });
    }

    const decoded = jwt.decode(credential);
    if (!decoded || !decoded.email) {
      return res.status(400).json({ error: 'Invalid Google token' });
    }

    const { email, name, picture } = decoded;

    const db = getDb();

    // Check if user exists
    let user = db.prepare('SELECT * FROM users WHERE email = ?').get(email);

    if (!user) {
      // Create new user
      const userId = uuidv4();
      db.prepare(
        'INSERT INTO users (id, email, name, avatar) VALUES (?, ?, ?, ?)'
      ).run(userId, email, name || '', picture || '');

      user = { id: userId, email, name: name || '', avatar: picture || '' };
    }

    // Generate JWT
    const token = generateToken(user);

    res.json({
      token,
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        avatar: user.avatar
      }
    });
  } catch (err) {
    console.error('Auth error:', err);
    res.status(500).json({ error: 'Authentication failed' });
  }
});

// GET /api/auth/me — Get current user (from JWT)
router.get('/me', (req, res) => {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader) {
      // Dev mode fallback
      if (process.env.NODE_ENV !== 'production') {
        return res.json({
          user: { id: 'dev-user-001', email: 'dev@medicure.local', name: 'Dev User', avatar: '' }
        });
      }
      return res.status(401).json({ error: 'Not authenticated' });
    }

    const token = authHeader.split(' ')[1];
    const decoded = jwt.verify(token, JWT_SECRET);
    res.json({ user: decoded });
  } catch {
    res.status(401).json({ error: 'Invalid token' });
  }
});


// POST /api/auth/dev-login — Development bypass
router.post('/dev-login', (req, res) => {
  if (process.env.NODE_ENV === 'production') {
    return res.status(403).json({ error: 'Dev login disabled in production' });
  }

  const db = getDb();
  const devEmail = 'dev@medicure.local';
  
  let user = db.prepare('SELECT * FROM users WHERE email = ?').get(devEmail);
  if (!user) {
    const userId = uuidv4();
    db.prepare('INSERT INTO users (id, email, name, avatar) VALUES (?, ?, ?, ?)').run(userId, devEmail, 'Dev User', '');
    user = { id: userId, email: devEmail, name: 'Dev User', avatar: '' };
  }

  const token = generateToken(user);
  res.json({ token, user: { id: user.id, email: user.email, name: user.name, avatar: user.avatar } });
});
module.exports = router;
