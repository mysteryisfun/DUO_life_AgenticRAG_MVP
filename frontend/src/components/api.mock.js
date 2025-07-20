// Mocked API for demo purposes
export async function getNewSession() {
  return new Promise(resolve => setTimeout(() => resolve({ session_id: 'demo-session-1234' }), 300));
}

export async function queryAgent({ question, session_id, onLog, onToken, onFinal }) {
  // Simulate log
  if (onLog) onLog({ message: 'Searching knowledge base...' });
  await new Promise(r => setTimeout(r, 500));
  // Simulate streaming answer
  const answer = question.toLowerCase().includes('skincare')
    ? 'Yes, DUO LIFE Hydrating Serum and DUO LIFE Night Cream are great for skincare.'
    : "Thank you for your message! We'll get back to you soon.";
  for (let i = 0; i < answer.length; i += 5) {
    if (onToken) onToken({ chunk: answer.slice(i, i + 5) });
    await new Promise(r => setTimeout(r, 40));
  }
  // Simulate final response with sources for skincare
  if (onFinal) {
    onFinal({
      session_id,
      final_answer: answer,
      sources: question.toLowerCase().includes('skincare') ? [
        {
          name: 'DUO LIFE Hydrating Serum',
          links: ['https://duolife.com/product/hydrating-serum'],
          type: 'Product',
          category: 'Skincare',
          snippet: 'A hydrating serum for glowing skin.'
        },
        {
          name: 'DUO LIFE Night Cream',
          links: ['https://duolife.com/product/night-cream'],
          type: 'Product',
          category: 'Skincare',
          snippet: 'A nourishing night cream for skin repair.'
        }
      ] : []
    });
  }
}

export async function clearSessionMemory(session_id) {
  return new Promise(resolve => setTimeout(() => resolve({ message: `Conversation memory for session ${session_id} has been cleared.` }), 200));
} 