document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-btn');
    const clearButton = document.getElementById('clear-chat');
    const statusText = document.getElementById('status-text');
    const statusBadge = document.getElementById('status-badge');
    const suggestionButtons = document.querySelectorAll('.suggestion-btn');
    const chatContainer = document.querySelector('.chat-container');
    
    // Voice input/output elements
    const voiceButton = document.getElementById('voice-btn');
    const ttsToggle = document.getElementById('tts-toggle');
    const audioPlayerContainer = document.getElementById('audio-player-container');
    const playPauseBtn = document.getElementById('play-pause-btn');
    const audioProgress = document.getElementById('audio-progress');
    
    // Stats elements
    const avgPrice = document.getElementById('avg-price');
    const avgSqft = document.getElementById('avg-sqft');
    const totalProperties = document.getElementById('total-properties');
    const priceRange = document.getElementById('price-range');
    
    // Add marked.js library for Markdown support
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/marked/4.0.2/marked.min.js';
    document.head.appendChild(script);
    
    // Add highlight.js for code syntax highlighting
    const highlightCSS = document.createElement('link');
    highlightCSS.rel = 'stylesheet';
    highlightCSS.href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.5.1/styles/github.min.css';
    document.head.appendChild(highlightCSS);
    
    const highlightJS = document.createElement('script');
    highlightJS.src = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.5.1/highlight.min.js';
    document.head.appendChild(highlightJS);
    
    // Chat history
    let chatHistory = [];
    let isProcessing = false;
    
    // Voice recognition variables
    let recognition = null;
    let isRecognizing = false;
    let audioPlayer = new Audio();
    let ttsEnabled = true;
    
    // Initialize speech recognition if available
    function initSpeechRecognition() {
        // Check if browser supports speech recognition
        if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            let silenceTimeout;
            const SILENCE_DELAY = 3000; // 3 seconds

            recognition.onstart = function() {
                isRecognizing = true;
                voiceButton.classList.add('recording');
                voiceButton.innerHTML = '<i class="fas fa-stop"></i>';
                updateStatus('Listening...', 'processing');
                
                // Start silence timer
                silenceTimeout = setTimeout(() => {
                    stopRecognition();
                }, SILENCE_DELAY);
            };

            recognition.onresult = function(event) {
                // Clear existing silence timer
                clearTimeout(silenceTimeout);
                
                let interimTranscript = '';
                let finalTranscript = '';

                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    if (event.results[i].isFinal) {
                        finalTranscript += transcript;
                    } else {
                        interimTranscript += transcript;
                    }
                }

                // Update input field with transcription
                userInput.value = finalTranscript || interimTranscript;
                userInput.dispatchEvent(new Event('input'));

                // Restart silence timer
                silenceTimeout = setTimeout(() => {
                    stopRecognition();
                }, SILENCE_DELAY);
            };

            recognition.onerror = function(event) {
                console.error('Speech recognition error:', event.error);
                stopRecognition();
                updateStatus('Online', 'online');
            };

            recognition.onend = function() {
                clearTimeout(silenceTimeout);
                stopRecognition();
                updateStatus('Online', 'online');
                sendMessage();

            };

            return true;
        }
        return false;
    }

    function toggleSpeechRecognition() {
        if (!recognition) {
            const initialized = initSpeechRecognition();
            if (!initialized) {
                alert('Speech recognition is not supported in this browser.');
                return;
            }
        }

        if (isRecognizing) {
            stopRecognition();
            voiceButton.innerHTML = '<i class="fas fa-microphone"></i>';
            voiceButton.classList.remove('recording');
        } else {
            startRecognition();
            voiceButton.innerHTML = '<i class="fas fa-stop"></i>';
            voiceButton.classList.add('recording');
        }
    }

    function startRecognition() {
        try {
            recognition.start();
        } catch (e) {
            console.error('Recognition error:', e);
            // Try to reinitialize
            recognition = null;
            initSpeechRecognition();
            try {
                recognition.start();
            } catch (e) {
                console.error('Failed to restart recognition:', e);
                alert('Failed to start speech recognition. Please try again.');
            }
        }
    }

    function stopRecognition() {
        if (recognition && isRecognizing) {
            recognition.stop();
            isRecognizing = false;
            voiceButton.classList.remove('recording');
            voiceButton.innerHTML = '<i class="fas fa-microphone"></i>';
            clearTimeout(silenceTimeout);
        }
    }
    
    // TTS functions
    function toggleTTS() {
        ttsEnabled = !ttsEnabled;
        ttsToggle.classList.toggle('active', ttsEnabled);
        
        // Stop any current playback when TTS is turned off
        if (!ttsEnabled) {
            stopAudio();
        }
    }
    
    function playTTS(text) {
        if (!ttsEnabled) return;
        
        fetch('/audio/tts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        })
        .then(response => response.json())
        .then(data => {
            if (data.audio) {
                const audioUrl = `data:audio/mp3;base64,${data.audio}`;
                audioPlayer.src = audioUrl;
                audioPlayer.load();
                audioPlayer.currentTime = 0;
                showAudioPlayer();
                audioPlayer.play().catch(e => console.error('Audio playback error:', e));
            }

        })
        .catch(error => {
            console.error('TTS error:', error);
        });
    }
    
    function showAudioPlayer() {
        audioPlayerContainer.classList.remove('hidden');
        updatePlayButton();
        
        // Set up audio events if not already done
        if (!audioPlayer.onplay) {
            setupAudioEvents();
        }
    }
    
    function setupAudioEvents() {
        audioPlayer.onplay = updatePlayButton;
        audioPlayer.onpause = updatePlayButton;
        audioPlayer.onended = function() {
            updatePlayButton();
            setTimeout(() => {
                audioPlayerContainer.classList.add('hidden');
            }, 1000);
        };
        
        // Update progress bar during playback
        audioPlayer.ontimeupdate = function() {
            const percent = (audioPlayer.currentTime / audioPlayer.duration) * 100;
            audioProgress.style.width = percent + '%';
        };
    }
    
    function updatePlayButton() {
        if (audioPlayer.paused) {
            playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
        } else {
            playPauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
        }
    }
    
    function toggleAudioPlayback() {
        if (audioPlayer.paused) {
            audioPlayer.play()
                .catch(e => console.error('Audio playback error:', e));
        } else {
            audioPlayer.pause();
        }
        updatePlayButton();
    }
    
    function stopAudio() {
        if (audioPlayer.src) {
            audioPlayer.pause();
            audioPlayer.currentTime = 0;
            updatePlayButton();
            audioPlayerContainer.classList.add('hidden');
        }
    }
    
    // Check server status and load property stats on page load
    checkStatus();
    loadPropertyStats();
    
    // Event Listeners
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Voice events
    voiceButton.addEventListener('click', toggleSpeechRecognition);
    ttsToggle.addEventListener('click', toggleTTS);
    playPauseBtn.addEventListener('click', toggleAudioPlayback);
    
    // Add input resize functionality
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        const maxHeight = 150; // Maximum height before scrolling
        this.style.height = (this.scrollHeight > maxHeight ? maxHeight : this.scrollHeight) + 'px';
    });
    
    // Focus input on page load
    userInput.focus();
    
    clearButton.addEventListener('click', clearChat);
    
    // Suggestion buttons
    suggestionButtons.forEach(button => {
        button.addEventListener('click', function() {
            const query = this.getAttribute('data-query');
            userInput.value = query;
            userInput.dispatchEvent(new Event('input')); // Trigger height adjustment
            sendMessage();
        });
    });
    
    // Functions
    function sendMessage() {
        const message = userInput.value.trim();
        
        if (message === '' || isProcessing) return;
        
        // Stop any current voice recording
        stopRecognition();
        
        // Stop any current audio playback
        stopAudio();
        
        // Get current time for message timestamp
        const now = new Date();
        const time = now.getHours() + ':' + (now.getMinutes() < 10 ? '0' : '') + now.getMinutes();
        
        // Add user message to chat
        addMessage(message, 'user', time);
        
        // Clear input and reset height
        userInput.value = '';
        userInput.style.height = 'auto';
        userInput.focus();
        
        // Show typing indicator
        showTypingIndicator();
        
        // Set processing state
        isProcessing = true;
        updateStatus('Processing...', 'processing');
        
        // Send request to backend
        fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: message,
                chat_history: chatHistory
            })
        })
        .then(response => response.json())
        .then(data => {
            // Remove typing indicator
            removeTypingIndicator();
            
            // Get current time for response timestamp
            const responseTime = new Date();
            const responseTimeStr = responseTime.getHours() + ':' + 
                (responseTime.getMinutes() < 10 ? '0' : '') + responseTime.getMinutes();
            
            // Add assistant response
            addMessage(data.answer, 'assistant', responseTimeStr);
            
            // Play TTS response if enabled
            if (ttsEnabled) {
                // Extract plain text from markdown for better TTS
                const plainText = stripMarkdown(data.answer);
                playTTS(plainText);
            }
            
            // Update chat history with both messages
            chatHistory.push(
                { role: 'user', content: message },
                { role: 'assistant', content: data.answer },
                { role: 'system', content: JSON.stringify(data.system_state) }
            );
            
            // Reset processing state
            isProcessing = false;
            updateStatus('Online', 'online');
        })
        .catch(error => {
            console.error('Error:', error);
            removeTypingIndicator();
            
            const errorTime = new Date();
            const errorTimeStr = errorTime.getHours() + ':' + 
                (errorTime.getMinutes() < 10 ? '0' : '') + errorTime.getMinutes();
                
            addMessage('Sorry, there was an error processing your request. Please try again.', 'assistant', errorTimeStr);
            isProcessing = false;
            updateStatus('Error', 'error');
        });
    }
    
    // Helper function to strip markdown for better TTS
    function stripMarkdown(text) {
        return text
            .replace(/\*\*(.*?)\*\*/g, '$1') // Bold
            .replace(/\*(.*?)\*/g, '$1')     // Italic
            .replace(/\[(.*?)\]\((.*?)\)/g, '$1') // Links
            .replace(/#{1,6}\s?(.*)/g, '$1') // Headers
            .replace(/```[a-z]*\n([\s\S]*?)```/g, 'Code block removed.') // Code blocks
            .replace(/`(.*?)`/g, '$1')       // Inline code
            .replace(/^\s*[-*+]\s+/gm, '')   // Lists
            .replace(/^\s*\d+\.\s+/gm, '')   // Numbered lists
            .replace(/\n{2,}/g, '. ')        // Multiple newlines with period for pause
            .replace(/\n/g, ' ')             // Single newlines
            .trim();
    }
    
    function addMessage(content, role, time) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', role);
        
        const avatar = document.createElement('div');
        avatar.classList.add('message-avatar');
        
        const icon = document.createElement('i');
        icon.classList.add('fas', role === 'user' ? 'fa-user' : 'fa-robot');
        
        avatar.appendChild(icon);
        
        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');
        
        // Create message header
        const messageHeader = document.createElement('div');
        messageHeader.classList.add('message-header');
        
        const nameSpan = document.createElement('span');
        nameSpan.classList.add('bot-name');
        nameSpan.textContent = role === 'user' ? 'You' : 'HomeHelper';
        
        const timeSpan = document.createElement('span');
        timeSpan.classList.add('message-time');
        timeSpan.textContent = time || 'Just now';
        
        messageHeader.appendChild(nameSpan);
        messageHeader.appendChild(timeSpan);
        
        messageContent.appendChild(messageHeader);
        
        // Parse markdown for assistant messages only
        if (role === 'assistant' && typeof marked !== 'undefined') {
            const messageBody = document.createElement('div');
            messageBody.classList.add('message-body');
            
            // Configure marked options
            marked.setOptions({
                gfm: true,
                breaks: true,
                highlight: function(code, language) {
                    if (typeof hljs !== 'undefined' && language && hljs.getLanguage(language)) {
                        try {
                            return hljs.highlight(code, { language }).value;
                        } catch (error) {
                            console.error('Highlight error:', error);
                        }
                    }
                    return code;
                }
            });
            
            // Render markdown
            messageBody.innerHTML = marked.parse(content);
            
            // Apply syntax highlighting to code blocks
            if (typeof hljs !== 'undefined') {
                messageBody.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightElement(block);
                });
            }
            
            messageContent.appendChild(messageBody);
        } else {
            // Regular text for user messages
            const paragraph = document.createElement('p');
            paragraph.textContent = content;
            messageContent.appendChild(paragraph);
        }
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
        
        // Add message with animation
        messageDiv.style.opacity = '0';
        messageDiv.style.transform = 'translateY(20px)';
        chatMessages.appendChild(messageDiv);
        
        // Trigger reflow
        void messageDiv.offsetWidth;
        
        // Apply animation
        messageDiv.style.transition = 'opacity 0.3s ease-out, transform 0.3s ease-out';
        messageDiv.style.opacity = '1';
        messageDiv.style.transform = 'translateY(0)';
        
        // Smooth scroll to bottom with animation
        smoothScrollToBottom();
    }
    
    function smoothScrollToBottom() {
        const scrollHeight = chatContainer.scrollHeight;
        const height = chatContainer.clientHeight;
        const maxScrollTop = scrollHeight - height;
        
        // Use requestAnimationFrame for smooth animation
        const startTime = performance.now();
        const startPos = chatContainer.scrollTop;
        const duration = 300; // milliseconds
        
        function step(currentTime) {
            const elapsed = currentTime - startTime;
            
            if (elapsed < duration) {
                // Easing function: easeInOutQuad
                const t = elapsed / duration;
                const factor = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
                
                chatContainer.scrollTop = startPos + factor * (maxScrollTop - startPos);
                requestAnimationFrame(step);
            } else {
                chatContainer.scrollTop = maxScrollTop;
            }
        }
        
        requestAnimationFrame(step);
    }
    
    function showTypingIndicator() {
        const now = new Date();
        const time = now.getHours() + ':' + (now.getMinutes() < 10 ? '0' : '') + now.getMinutes();
        
        const typingDiv = document.createElement('div');
        typingDiv.classList.add('message', 'assistant');
        typingDiv.id = 'typing-indicator';
        
        const avatar = document.createElement('div');
        avatar.classList.add('message-avatar');
        
        const icon = document.createElement('i');
        icon.classList.add('fas', 'fa-robot');
        
        avatar.appendChild(icon);
        
        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');
        
        // Create message header
        const messageHeader = document.createElement('div');
        messageHeader.classList.add('message-header');
        
        const nameSpan = document.createElement('span');
        nameSpan.classList.add('bot-name');
        nameSpan.textContent = 'HomeHelper';
        
        const timeSpan = document.createElement('span');
        timeSpan.classList.add('message-time');
        timeSpan.textContent = time;
        
        messageHeader.appendChild(nameSpan);
        messageHeader.appendChild(timeSpan);
        
        // Create typing indicator dots
        const typingContainer = document.createElement('div');
        typingContainer.classList.add('typing-indicator');
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.classList.add('typing-dot');
            typingContainer.appendChild(dot);
        }
        
        messageContent.appendChild(messageHeader);
        messageContent.appendChild(typingContainer);
        
        typingDiv.appendChild(avatar);
        typingDiv.appendChild(messageContent);
        
        chatMessages.appendChild(typingDiv);
        smoothScrollToBottom();
    }
    
    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    function clearChat() {
        // Stop any current audio playback
        stopAudio();
        
        // Add fade-out animation to messages
        const messages = chatMessages.querySelectorAll('.message:not(:first-child)');
        let animationCount = 0;
        const totalAnimations = messages.length;
        
        if (totalAnimations === 0) {
            return; // No messages to clear
        }
        
        messages.forEach((msg, index) => {
            // Stagger the animations
            setTimeout(() => {
                msg.style.transition = 'opacity 0.2s ease-out, transform 0.2s ease-out';
                msg.style.opacity = '0';
                msg.style.transform = 'translateY(10px)';
                
                msg.addEventListener('transitionend', function handler() {
                    msg.removeEventListener('transitionend', handler);
                    animationCount++;
                    
                    if (animationCount === totalAnimations) {
                        // All animations complete, now remove messages
                        while (chatMessages.children.length > 1) {
                            chatMessages.removeChild(chatMessages.lastChild);
                        }
                        
                        // Reset chat history
                        chatHistory = [];
                        
                        // Reset status
                        updateStatus('Online', 'online');
                        
                        // Focus on input
                        userInput.focus();
                    }
                });
            }, index * 50); // Stagger by 50ms per message
        });
    }
    
    function updateStatus(text, status) {
        statusText.textContent = text;
        
        // Remove all status classes
        statusBadge.classList.remove('online', 'offline', 'error', 'processing');
        
        // Add appropriate class
        statusBadge.classList.add(status);
    }
    
    function checkStatus() {
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'ok') {
                    updateStatus('Online', 'online');
                } else {
                    updateStatus('System Issue', 'error');
                }
            })
            .catch(error => {
                console.error('Status check error:', error);
                updateStatus('Offline', 'offline');
            });
    }
    
    function loadPropertyStats() {
        fetch('/property_summary')
            .then(response => response.json())
            .then(data => {
                // Update stats in the sidebar with animation
                if (data.avg_price) {
                    animateValue(avgPrice, 0, data.avg_price, 1500, val => `$${formatNumber(val)}`);
                }
                
                if (data.avg_sqft) {
                    animateValue(avgSqft, 0, data.avg_sqft, 1500, val => `${formatNumber(val)} sq ft`);
                }
                
                if (data.total_properties) {
                    animateValue(totalProperties, 0, data.total_properties, 1500, formatNumber);
                }
                
                if (data.price_range) {
                    priceRange.textContent = `$${formatNumber(data.price_range.min)} - $${formatNumber(data.price_range.max)}`;
                }
            })
            .catch(error => {
                console.error('Error loading property stats:', error);
                avgPrice.textContent = 'Unavailable';
                avgSqft.textContent = 'Unavailable';
                totalProperties.textContent = 'Unavailable';
                priceRange.textContent = 'Unavailable';
            });
    }
    
    // Helper function to animate number values
    function animateValue(element, start, end, duration, formatter) {
        const startTime = performance.now();
        
        function updateValue(timestamp) {
            const elapsed = timestamp - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function: easeOutQuad
            const easing = 1 - (1 - progress) * (1 - progress);
            const current = Math.floor(start + (end - start) * easing);
            
            element.textContent = formatter(current);
            
            if (progress < 1) {
                requestAnimationFrame(updateValue);
            }
        }
        
        requestAnimationFrame(updateValue);
    }
    
    // Helper function to format numbers with commas
    function formatNumber(num) {
        return new Intl.NumberFormat('en-US', { 
            maximumFractionDigits: 0 
        }).format(Math.round(num));
    }
    
    // Set up polling for status checks
    setInterval(checkStatus, 60000); // Check status every minute
});