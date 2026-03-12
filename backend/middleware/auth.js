const jwt = require('jsonwebtoken');

const JWT_SECRET = process.env.JWT_SECRET || 'medicure-dev-secret-key-change-in-production';

function generateToken(user) {
  return jwt.sign(
    { id: user.id, email: user.email, name: user.name },
    JWT_SECRET,
    { expiresIn: '7d' }
  );
}

function authMiddleware(req, res, next) {
  // Dev mode bypass — skip auth in development
  if (process.env.NODE_ENV !== 'production') {
    req.user = {
      id: 'dev-user-001',
      email: 'dev@medicure.local',
      name: 'Dev User'
    };
    return next();
  }

  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Missing or invalid authorization header' });
  }

  try {
    const token = authHeader.split(' ')[1];
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded;
    next();
  } catch (err) {
    return res.status(401).json({ error: 'Invalid or expired token' });
  }
}

module.exports = { authMiddleware, generateToken, JWT_SECRET };
