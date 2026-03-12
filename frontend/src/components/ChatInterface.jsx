import { useState, useRef, useEffect } from 'react';
import { FiSend, FiX } from 'react-icons/fi';
import { sendChatMessage } from '../services/api';
import { locales } from '../locales';

export default function ChatInterface({ scanId, targetLanguage = 'en-US' }) {
  const t = locales[targetLanguage] || locales['en-US'];

  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: t.safety_disclaimer ? `Hi! I'm your medicine assistant. ${t.ask_placeholder}` : 'Hi! I\'m your medicine assistant. Ask me anything about this medicine — side effects, dosage, interactions, or cheaper alternatives.',
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  // Update initial message when language changes
  useEffect(() => {
    setMessages([{
      role: 'assistant',
      content: targetLanguage === 'en-US' 
        ? "Hi! I'm your medicine assistant. Ask me anything about this medicine — side effects, dosage, interactions, or cheaper alternatives."
        : t.ask_placeholder + " (AI Assistant)"
    }]);
  }, [targetLanguage, t.ask_placeholder]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || loading) return;

    setMessages((prev) => [...prev, { role: 'user', content: text }]);
    setInput('');
    setLoading(true);

    try {
      const res = await sendChatMessage(scanId, text, targetLanguage);
      const reply = res.response || res.reply || res.data?.reply || 'Sorry, I couldn\'t generate a response.';
      setMessages((prev) => [...prev, { role: 'assistant', content: reply }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Something went wrong. Please try again.' },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="card chat-container">
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-bubble ${msg.role}`}>
            {msg.content}
          </div>
        ))}
        {loading && (
          <div className="typing-indicator">
            <span /><span /><span />
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-row">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder={t.ask_placeholder || "Ask about this medicine…"}
          disabled={loading}
        />
        <button
          className="btn btn-primary"
          onClick={handleSend}
          disabled={loading || !input.trim()}
          style={{ padding: '0.75rem 1rem' }}
        >
          <FiSend />
        </button>
      </div>
    </div>
  );
}
