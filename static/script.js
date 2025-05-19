document.addEventListener('DOMContentLoaded', function() {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const loading = document.getElementById('loading');
      // Function to add a message to the chat
    function addMessage(message, isUser = false, references = [], toolsUsed = [], isFallback = false) {
        const messageContainer = document.createElement('div');
        messageContainer.className = 'message-container';
        if (isUser) messageContainer.className += ' user-container';
        
        const messageDiv = document.createElement('div');
        messageDiv.className = isUser ? 'user-message' : 'bot-message';
        if (isFallback) messageDiv.className += ' fallback-message';
        messageDiv.textContent = message;
        
        // If it's a fallback response, add an indicator
        if (isFallback) {
            const fallbackIndicator = document.createElement('div');
            fallbackIndicator.className = 'fallback-indicator';
            fallbackIndicator.textContent = 'Direct answer from Gemini (not in Bhagavad Gita)';
            messageContainer.appendChild(fallbackIndicator);
        }
        
        messageContainer.appendChild(messageDiv);
        
        // Add "Explain more" button and references toggle for bot messages
        if (!isUser) {
            const buttonContainer = document.createElement('div');
            buttonContainer.className = 'button-container';
            
            // Add "Explain more" button
            const explainBtn = document.createElement('button');
            explainBtn.className = 'action-btn explain-more-btn';
            explainBtn.textContent = 'Explain more';
            explainBtn.onclick = function() {
                const explainQuery = `Explain more: ${message}`;
                sendQuery(explainQuery);
            };
            buttonContainer.appendChild(explainBtn);
            
            // Add "Show References" button if references exist
            if (references.length > 0 || toolsUsed.length > 0) {
                const referencesBtn = document.createElement('button');
                referencesBtn.className = 'action-btn references-btn';
                referencesBtn.textContent = 'Show References';
                referencesBtn.onclick = function() {
                    const refContent = referencesDiv.style.display === 'none' || !referencesDiv.style.display;
                    referencesDiv.style.display = refContent ? 'block' : 'none';
                    referencesBtn.textContent = refContent ? 'Hide References' : 'Show References';
                };
                buttonContainer.appendChild(referencesBtn);
                
                // Create references div (initially hidden)
                const referencesDiv = document.createElement('div');
                referencesDiv.className = 'references-container';
                referencesDiv.style.display = 'none';
                
                // Add tools used information
                if (toolsUsed.length > 0) {
                    const toolsTitle = document.createElement('h4');
                    toolsTitle.textContent = 'Tools Used:';
                    referencesDiv.appendChild(toolsTitle);
                    
                    const toolsList = document.createElement('ul');
                    toolsList.className = 'tools-list';
                    toolsUsed.forEach(tool => {
                        const toolItem = document.createElement('li');
                        toolItem.textContent = `${tool.name}${tool.input ? ': ' + tool.input : ''}`;
                        toolsList.appendChild(toolItem);
                    });
                    referencesDiv.appendChild(toolsList);
                }
                
                // Add references information
                if (references.length > 0) {
                    const refsTitle = document.createElement('h4');
                    refsTitle.textContent = 'References:';
                    referencesDiv.appendChild(refsTitle);
                    
                    const refsList = document.createElement('ul');
                    refsList.className = 'references-list';
                    references.forEach(ref => {
                        const refItem = document.createElement('li');
                        refItem.textContent = ref;
                        refsList.appendChild(refItem);
                    });
                    referencesDiv.appendChild(refsList);
                }
                
                messageContainer.appendChild(referencesDiv);
            }
            
            messageContainer.appendChild(buttonContainer);
        }
        
        chatBox.appendChild(messageContainer);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
    
    // Function to send query to backend
    async function sendQuery(query) {
        loading.style.display = 'block';
        
        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query }),
            });
            
            const data = await response.json();
            loading.style.display = 'none';
              if (response.ok) {
                // Pass references and tools data if they exist
                const references = data.references || [];
                const toolsUsed = data.tools_used || [];
                const isFallback = data.is_fallback || false;
                
                // Handle regular responses and fallback responses separately
                addMessage(data.response, false, references, toolsUsed, isFallback);
            } else {
                addMessage('Sorry, I encountered an error: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            loading.style.display = 'none';
            addMessage('Sorry, an error occurred while processing your request.');
            console.error('Error:', error);
        }
    }
    
    // Send button click event
    sendBtn.addEventListener('click', function() {
        const query = userInput.value.trim();
        if (query) {
            addMessage(query, true);
            userInput.value = '';
            sendQuery(query);
        }
    });
    
    // Enter key press event
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            const query = userInput.value.trim();
            if (query) {
                addMessage(query, true);
                userInput.value = '';
                sendQuery(query);
            }
        }
    });
});