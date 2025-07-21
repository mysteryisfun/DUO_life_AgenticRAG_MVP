// FastAPI backend base URL
const API_BASE = 'http://localhost:8000';

// 1. Generate New Session
export async function getNewSession() {
  const res = await fetch(`${API_BASE}/new_session`);
  if (!res.ok) throw new Error('Failed to create new session');
  return res.json();
}

export async function getNewSessionNgrok() {
  const res = await fetch('https://8f9578c45001.ngrok-free.app/new_session');
  if (!res.ok) throw new Error('Failed to create new session (ngrok)');
  return res.json();
}

// 2. Query the Agent (Streaming)
export async function queryAgent({ question, session_id, onLog, onToken, onFinal }) {
  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id }),
  });
  if (!res.body) throw new Error('No response body');
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary;
    while ((boundary = buffer.indexOf('\n')) !== -1) {
      const line = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 1);
      if (!line) continue;
      try {
        const obj = JSON.parse(line);
        if (obj.type === 'log' && onLog) onLog(obj.data);
        else if (obj.type === 'token' && onToken) onToken(obj.data);
        else if (obj.type === 'final_response' && onFinal) onFinal(obj.data);
      } catch (e) {
        // Ignore JSON parse errors for incomplete lines
      }
    }
  }
}

// 3. Clear Session Memory
export async function clearSessionMemory(session_id) {
  const res = await fetch(`${API_BASE}/clear_memory`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id }),
  });
  if (!res.ok) throw new Error('Failed to clear session memory');
  return res.json();
} 