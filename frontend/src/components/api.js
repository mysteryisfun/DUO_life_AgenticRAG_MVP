// FastAPI backend base URL
const API_BASE = 'http://localhost:8000';

// 1. Generate New Session
export async function getNewSession() {
  const res = await fetch(`${API_BASE}/new_session`);
  if (!res.ok) throw new Error('Failed to create new session');
  return res.json();
}

export async function getNewSessionNgrok() {
  try {
    const res = await fetch(`${API_BASE}/new_session`, {
      method: 'GET',
      headers: {
        'ngrok-skip-browser-warning': 'true',
        'Content-Type': 'application/json',
      },
      mode: 'cors'
    });
    if (!res.ok) throw new Error('Failed to create new session (ngrok)');
    const data = await res.json();
    console.log('Session ID:', data.session_id);
    return data.session_id;
  } catch (error) {
    console.warn('Ngrok session failed, using mock session:', error.message);
    // Return a mock session ID for development
    return 'mock-session-' + Date.now();
  }
}

// 2. Query the Agent (Streaming)
export async function queryAgent({ question, session_id, onLog, onToken, onFinal }) {
  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, session_id }),
    });

    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`Backend request failed with status ${res.status}: ${errorText}`);
    }

    const responseData = await res.json();

    if (onFinal) {
      // The backend sends a response like { "type": "final_response", "data": { ... } }.
      // We need to extract the nested "data" object for the component.
      if (responseData.type === 'final_response' && responseData.data) {
        onFinal(responseData.data);
      } else {
        // Handle other potential response structures if needed.
        onFinal(responseData);
      }
    }
  } catch (error) {
    console.error('Failed to query agent:', error);
    if (onFinal) {
      onFinal({
        final_answer: `An error occurred: ${error.message}`,
        sources: []
      });
    }
  }
}

// 3. Clear Session Memory
export async function clearSessionMemory(session_id) {
  try {
    const res = await fetch(`${API_BASE}/clear_memory`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'ngrok-skip-browser-warning': 'true'
      },
      body: JSON.stringify({ session_id }),
    });
    if (!res.ok) throw new Error('Failed to clear session memory');
    return res.json();
  } catch (error) {
    console.warn('Clear memory failed, continuing anyway:', error.message);
    return { success: true }; // Mock success response
  }
} 