import React, { useState, useRef, useEffect } from 'react';
import { getNewSessionNgrok as getNewSession, queryAgent, clearSessionMemory } from './api';

// Placeholder AI/brain SVG icon for LimitlessMind.ai
const avatarIcon = (
  <svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-10 h-10">
    <circle cx="20" cy="20" r="20" fill="#2563eb" />
    <path d="M28 20c0 4-3.5 7-8 7s-8-3-8-7 3.5-7 8-7 8 3 8 7z" fill="#fff" />
    <path d="M16 18c0-1.1.9-2 2-2s2 .9 2 2" stroke="#2563eb" strokeWidth="1.5" strokeLinecap="round" />
    <circle cx="17.5" cy="21.5" r="1" fill="#2563eb" />
    <circle cx="22.5" cy="21.5" r="1" fill="#2563eb" />
    <path d="M18 24c.5.5 1.5.5 2 0" stroke="#2563eb" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);

function linkify(text) {
  if (!text) return null; // Prevent crash if text is not a string
  const urlRegex = /(https?:\/\/[^\s]+)/g;
  return text.split(urlRegex).map((part, i) =>
    urlRegex.test(part) ? (
      <a key={i} href={part} className="text-blue-500 underline" target="_blank" rel="noopener noreferrer">{part}</a>
    ) : (
      part
    )
  );
}

const TYPING_SPEED = 20; // ms per character

const ChatWidget = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [show, setShow] = useState(false); // Start closed
  const [streaming, setStreaming] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [loadingSession, setLoadingSession] = useState(false);
  const [clearingChat, setClearingChat] = useState(false);
  const chatEndRef = useRef(null);

  // New: Open chat and create session on button click
  const handleOpenChat = async () => {
    setShow(true);
    setLoadingSession(true);
    try {
      const session_id = await getNewSession();
      setSessionId(session_id);
      setMessages([
        { id: 1, sender: 'bot', text: "Hi there! I'm DUO LIFE 🤖. How can I help you today?", streaming: false }
      ]);
    } finally {
      setLoadingSession(false);
    }
  };

  // Animate chat bubbles
  useEffect(() => {
    const chatBubbles = document.querySelectorAll('.chat-bubble');
    if (chatBubbles.length > 0) {
      const lastBubble = chatBubbles[chatBubbles.length - 1];
      lastBubble.classList.add('animate-fade-slide-in');
    }
  }, [messages]);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, streaming, expanded]);

  // Handle sending user message and streaming bot response
  const handleSend = async () => {
    if (!input.trim() || streaming || !sessionId) return;
    const userMsg = { id: messages.length + 1, sender: 'user', text: input, streaming: false };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setStreaming(true);
    let botMsgId = userMsg.id + 1;
    let botMsg = { id: botMsgId, sender: 'bot', text: '', streaming: true };
    setMessages(prev => [...prev, botMsg]);
    let fullAnswer = '';
    await queryAgent({
      question: userMsg.text,
      session_id: sessionId,
      onLog: (data) => {
        // Optionally show logs as status messages
        setMessages(prev => {
          // Remove previous log if present
          const filtered = prev.filter(m => m.type !== 'log');
          return [...filtered, { id: botMsgId + 0.1, sender: 'bot', text: data.message, streaming: false, type: 'log' }];
        });
      },
      onToken: (data) => {
        fullAnswer += data.chunk;
        setMessages(prev => prev.map(m =>
          m.id === botMsgId ? { ...m, text: fullAnswer } : m
        ));
      },
      onFinal: (data) => {
        // Robustly handle backend response to find the answer text
        const answer = data.final_answer || data.answer || data.text || (typeof data === 'string' ? data : JSON.stringify(data));
        const sources = data.sources || null;
        
        setMessages(prev => prev
          .filter(m => m.id !== botMsgId + 0.1) // Remove log
          .map(m => m.id === botMsgId
            ? { ...m, text: answer, streaming: false, sources }
            : m
          )
        );
        setStreaming(false);
      }
    });
  };

  const handleQuickReply = (reply) => {
    if (streaming) return;
    setInput(reply);
    setTimeout(() => handleSend(), 100);
  };

  const handleClear = async () => {
    if (!sessionId || clearingChat) return;
    
    setClearingChat(true);
    try {
      // Clear the backend session memory
      await clearSessionMemory(sessionId);
      
      // Reset the frontend chat messages
      setMessages([
        { id: 1, sender: 'bot', text: "Hi there! I'm DUO LIFE 🤖. How can I help you today?", streaming: false }
      ]);
      
      console.log('Chat history cleared for session:', sessionId);
    } catch (error) {
      console.error('Error clearing chat:', error);
      // Still clear the frontend even if backend fails
      setMessages([
        { id: 1, sender: 'bot', text: "Hi there! I'm DUO LIFE 🤖. How can I help you today?", streaming: false }
      ]);
    } finally {
      setClearingChat(false);
    }
  };

  if (!show) return (
    <button onClick={handleOpenChat} className="fixed bottom-6 right-6 bg-blue-500 hover:bg-blue-600 text-white rounded-full p-4 shadow-lg z-50">
      {/* Chat icon */}
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-7 h-7">
        <path strokeLinecap="round" strokeLinejoin="round" d="M8 10h.01M12 10h.01M16 10h.01M21 12c0 4.418-4.03 8-9 8a9.77 9.77 0 01-4-.8l-4 1 1-3.2A7.963 7.963 0 013 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
      </svg>
    </button>
  );

  const widgetSize = expanded
    ? 'w-[95vw] max-w-2xl h-[80vh] min-h-[500px]'
    : 'w-80 max-w-full';
  const chatAreaHeight = expanded ? { maxHeight: '60vh' } : { maxHeight: '350px' };

  return (
    <div className={`fixed bottom-6 right-6 bg-white rounded-3xl shadow-2xl flex flex-col z-50 border border-blue-200 overflow-hidden animate-fade-in transition-all duration-300 ${widgetSize}`} style={{ minWidth: expanded ? 350 : undefined }}>
      {/* Header */}
      <div className="flex flex-col items-center justify-center px-4 pt-4 pb-2 bg-gradient-to-r from-blue-500 to-blue-400 rounded-t-3xl relative">
        <div className="flex items-center gap-3 w-full">
          <span className="w-10 h-10 flex items-center justify-center bg-white rounded-full border-2 border-white shadow">{avatarIcon}</span>
          <div>
            <span className="text-white font-bold text-lg">DUO LIFE</span>
          </div>
          {/* Expand/Collapse Button */}
          <button
            onClick={() => setExpanded((prev) => !prev)}
            className="ml-2 text-white hover:bg-blue-600/30 rounded-full w-8 h-8 flex items-center justify-center transition"
            title={expanded ? 'Collapse' : 'Expand'}
          >
            {expanded ? (
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-5 h-5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-5 h-5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 8V6a2 2 0 012-2h2M20 16v2a2 2 0 01-2 2h-2M16 4h2a2 2 0 012 2v2M8 20H6a2 2 0 01-2-2v-2" />
              </svg>
            )}
          </button>
          <button onClick={() => setShow(false)} className="ml-auto text-white text-2xl font-bold hover:bg-blue-600/30 rounded-full w-8 h-8 flex items-center justify-center">×</button>
        </div>
        <button 
          onClick={handleClear} 
          disabled={clearingChat || streaming}
          className={`absolute right-4 top-2 text-xs text-white px-2 py-1 rounded transition ${
            clearingChat || streaming 
              ? 'bg-gray-400/40 cursor-not-allowed' 
              : 'bg-blue-400/40 hover:bg-blue-500/60'
          }`}
        >
          {clearingChat ? 'Clearing...' : 'Clear Chat'}
        </button>
      </div>
      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3 bg-blue-50" style={chatAreaHeight}>
        {loadingSession ? (
          <div className="text-center text-blue-400 py-8">Loading chat...</div>
        ) : (
          messages.map((msg, idx) => (
            <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`chat-bubble rounded-2xl px-4 py-2 max-w-[75%] shadow-sm ${msg.sender === 'user' ? 'bg-blue-500 text-white rounded-br-md' : 'bg-white text-gray-800 border border-blue-100 rounded-bl-md'} ${msg.streaming ? 'animate-pulse' : ''} transition-all duration-300`}
                style={{ animationDelay: `${idx * 60}ms` }}>
                {msg.sources && msg.sources.length > 0 ? (
                  <>
                    <div className="mt-2 text-base text-gray-800">{msg.text}</div>
                    <div className="custom-scrollbar flex flex-row gap-4 overflow-x-auto pb-2 mt-2">
                      {msg.sources.map((src, i) => (
                        <a
                          key={i}
                          href={src.links[0]}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex flex-col items-center min-w-[160px] bg-white rounded-xl shadow-lg p-3 border border-blue-100 hover:bg-blue-100 transition-transform hover:scale-105 duration-200 cursor-pointer"
                        >
                          <img
                            src="/image.png"
                            alt={src.name}
                            className="w-20 h-20 rounded-lg object-cover mb-2 border border-gray-200 shadow-sm"
                          />
                          <span className="font-bold text-blue-700 text-sm mb-1 text-center">{src.name}</span>
                          {src.category && src.category !== 'Uncategorized' && (
                            <span className="text-xs text-gray-500 mb-1 text-center">{src.category}</span>
                          )}
                          <span className="text-xs text-gray-400 mb-1 text-center">{src.type}</span>
                        </a>
                      ))}
                    </div>
                  </>
                ) : (
                  <>
                    {linkify(msg.text)}
                    {msg.streaming && (
                      <span className="typing-indicator inline-flex items-center justify-center bg-gray-100 rounded-full px-3 py-1 ml-2">
                        <span className="dot">●</span>
                        <span className="dot">●</span>
                        <span className="dot">●</span>
                      </span>
                    )}
                  </>
                )}
              </div>
            </div>
          ))
        )}
        {/* Quick Replies */}
        {/* The quick replies section is removed as per the edit hint. */}
        <div ref={chatEndRef} />
      </div>
      {/* Input Area */}
      <div className="flex items-center gap-2 p-3 border-t bg-white relative">
        <input
          type="text"
          className="flex-1 rounded-full border border-blue-200 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400 bg-blue-50"
          placeholder="Type your message..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          disabled={streaming || loadingSession}
        />
        <button
          onClick={handleSend}
          className="bg-gradient-to-br from-blue-500 to-blue-400 hover:from-blue-600 hover:to-blue-500 text-white rounded-full p-2 shadow-lg transition flex items-center justify-center disabled:opacity-50"
          disabled={streaming || loadingSession}
        >
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-6 h-6">
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 10.5l7.5 7.5 7.5-7.5M12 3v15.75" />
          </svg>
        </button>
      </div>
      <div className="text-xs text-gray-400 text-center pb-2 pt-1">Powered by <span className="font-semibold text-blue-500">LimitlessMind.ai</span></div>
      {/* Animations for chat bubbles */}
    </div>
  );
};

export default ChatWidget; 