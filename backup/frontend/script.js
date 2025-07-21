document.addEventListener('DOMContentLoaded', () => {
    const chatLauncher = document.getElementById('chat-launcher');
    const chatWidget = document.getElementById('chat-widget');
    const closeBtn = document.getElementById('close-btn');
    const reloadBtn = document.getElementById('reload-btn');
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const chatMessages = document.getElementById('chat-messages');
    const typingIndicator = document.getElementById('typing-indicator');

    let sessionId = null;
    let isLoading = false;

    // --- EVENT LISTENERS ---

    // Toggle chat widget
    chatLauncher.addEventListener('click', async () => {
        chatWidget.classList.toggle('open');
        if (chatWidget.classList.contains('open') && !sessionId) {
            await createNewSession();
        }
    });

    // Close chat widget
    closeBtn.addEventListener('click', () => {
        chatWidget.classList.remove('open');
    });

    // Reload session (clear memory)
    reloadBtn.addEventListener('click', async () => {
        if (sessionId) {
            try {
                await fetch('/clear_memory', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId }),
                });
                chatMessages.innerHTML = '';
                appendBotMessage("Memory cleared. How can I help you start over?");
            } catch (error) {
                console.error('Error clearing memory:', error);
                appendBotMessage("Sorry, I couldn't clear the memory. Please try again.");
            }
        }
    });

    // Handle form submission
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const message = messageInput.value.trim();
        if (message && !isLoading) {
            sendMessage(message);
            messageInput.value = '';
        }
    });

    // --- API CALLS ---

    async function createNewSession() {
        try {
            const response = await fetch('/new_session');
            if (!response.ok) throw new Error('Failed to create session');
            const data = await response.json();
            sessionId = data.session_id;
            appendBotMessage("Hello! I'm the DuoLife Assistant. How can I help you today?");
        } catch (error) {
            console.error('Error creating session:', error);
            appendBotMessage("Sorry, I'm having trouble connecting. Please try again later.");
        }
    }

    async function sendMessage(question) {
        if (!sessionId) {
            appendBotMessage("Something went wrong. Please reload the chat.");
            return;
        }

        appendUserMessage(question);
        setLoading(true);

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, question: question }),
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            const result = await response.json();
            handleResponse(result);

        } catch (error) {
            console.error('Error during fetch:', error);
            appendBotMessage("Oops! Something went wrong while getting your answer. Please try again.");
        } finally {
            setLoading(false);
        }
    }

    // --- UI HELPER FUNCTIONS ---

    function handleResponse(response) {
        if (response && response.type === 'final_response' && response.data) {
            const { final_answer, sources } = response.data;
            appendBotMessage(final_answer);

            if (sources && sources.length > 0) {
                appendProducts(sources);
            }
        } else {
             appendBotMessage("I received an unusual response. Please try rephrasing your question.");
        }
    }

    function appendUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message user';
        messageElement.textContent = message;
        chatMessages.appendChild(messageElement);
        scrollToBottom();
    }

    function appendBotMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message bot';
        messageElement.textContent = message;
        chatMessages.appendChild(messageElement);
        scrollToBottom();
    }

    function appendProducts(sources) {
        const productsContainer = document.createElement('div');
        productsContainer.className = 'products-container';

        sources.forEach(source => {
            const productCard = document.createElement('a');
            productCard.className = 'product-card';
            productCard.href = source.links[0] || '#';
            productCard.target = '_blank';
            productCard.rel = 'noopener noreferrer';

            productCard.innerHTML = `
                <img src="/image.png" alt="${source.name}">
                <div class="name">${source.name}</div>
                <div class="type">${source.type}</div>
            `;
            productsContainer.appendChild(productCard);
        });

        chatMessages.appendChild(productsContainer);
        scrollToBottom();
    }

    function setLoading(state) {
        isLoading = state;
        typingIndicator.classList.toggle('show', state);
        messageInput.disabled = state;
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});
