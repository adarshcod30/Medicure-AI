const Database = require('better-sqlite3');
const path = require('path');

const DB_PATH = process.env.NODE_ENV === 'production' 
  ? path.join(__dirname, '..', 'database.sqlite')
  : '/tmp/medicure-db/database.sqlite';

let db;

function getDb() {
  if (!db) {
    db = new Database(DB_PATH);
    db.pragma('journal_mode = WAL');
    db.pragma('foreign_keys = ON');
    initTables();
  }
  return db;
}

function initTables() {
  db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      email TEXT UNIQUE NOT NULL,
      name TEXT,
      avatar TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS scans (
      id TEXT PRIMARY KEY,
      user_id TEXT,
      image_path TEXT NOT NULL,
      result_json TEXT,
      ocr_text TEXT,
      ocr_confidence REAL DEFAULT 0,
      method TEXT DEFAULT 'ocr',
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (user_id) REFERENCES users(id)
    );

    CREATE TABLE IF NOT EXISTS chat_messages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      scan_id TEXT NOT NULL,
      role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
      content TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (scan_id) REFERENCES scans(id)
    );

    CREATE TABLE IF NOT EXISTS system_metrics (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      endpoint TEXT NOT NULL,
      response_time_ms INTEGER,
      status_code INTEGER,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
  `);

  console.log('✅ SQLite tables initialized');
}

module.exports = { getDb };
