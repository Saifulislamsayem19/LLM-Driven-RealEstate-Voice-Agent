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
    
    // Mobile sidebar drawer
    const menuButton = document.getElementById('menu-btn');
    const sidebarBackdrop = document.getElementById('sidebar-backdrop');

    // marked, DOMPurify and highlight.js are loaded (with SRI) in index.html.
    // Links in bot replies open in a new tab without giving it access to this page.
    if (typeof DOMPurify !== 'undefined') {
        DOMPurify.addHook('afterSanitizeAttributes', function(node) {
            if (node.tagName === 'A') {
                node.setAttribute('target', '_blank');
                node.setAttribute('rel', 'noopener noreferrer');
            }
        });
    }

    // Chat history
    let chatHistory = [];
    let isProcessing = false;
    
    // Voice input: record in the browser and transcribe on our server (/audio/transcribe).
    // Works in every modern browser (incl. Firefox), and audio only goes to the app's own speech provider.
    let mediaRecorder = null;
    let audioChunks = [];
    let discardRecording = false;
    let isStartingRecording = false;
    let isRecognizing = false;
    let silenceMonitor = null;
    let audioPlayer = new Audio();
    let ttsEnabled = true;
    let accumulatedTranscript = '';
    let userFocusedInput = false;
    const SILENCE_DELAY = 10000;       // stop if nothing is said for this long
    const END_OF_SPEECH_DELAY = 1500;  // stop this long after the speaker pauses
    const MAX_RECORDING_MS = 60000;
    const SPEECH_LEVEL = 0.02;         // RMS level that counts as speech

    function pickRecordingType() {
        const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus'];
        return types.find(t => MediaRecorder.isTypeSupported(t)) || '';
    }

    function setRecordingUI(active) {
        voiceButton.classList.toggle('recording', active);
        voiceButton.innerHTML = active ? '<i class="fas fa-stop"></i>' : '<i class="fas fa-microphone"></i>';
        voiceButton.setAttribute('aria-label', active ? 'Stop recording' : 'Voice input');
        if (active) updateStatus('Listening...', 'processing');
    }

    async function startRecognition() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
            alert('Voice input is not supported in this browser.');
            return;
        }
        isStartingRecording = true;
        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
        } catch (e) {
            console.error('Microphone error:', e);
            alert('Microphone access is blocked. Allow it in your browser settings to use voice input.');
            return;
        } finally {
            isStartingRecording = false;
        }

        const mimeType = pickRecordingType();
        const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
        mediaRecorder = recorder;
        audioChunks = [];
        discardRecording = false;

        recorder.ondataavailable = e => {
            if (e.data && e.data.size) audioChunks.push(e.data);
        };
        recorder.onstop = () => {
            stream.getTracks().forEach(track => track.stop());
            stopSilenceMonitor();
            isRecognizing = false;
            setRecordingUI(false);

            const blob = new Blob(audioChunks, { type: recorder.mimeType || mimeType || 'audio/webm' });
            if (discardRecording || blob.size === 0) {
                updateStatus('Online', 'online');
                userFocusedInput = false;
                return;
            }
            transcribeRecording(blob);
        };

        recorder.start();
        isRecognizing = true;
        setRecordingUI(true);
        startSilenceMonitor(stream);
    }

    // Stop automatically when the speaker pauses, nobody speaks, or the recording gets too long
    function startSilenceMonitor(stream) {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        const startedAt = Date.now();
        let lastSoundAt = startedAt;
        let heardSpeech = false;
        let ctx = null, analyser = null, samples = null;

        if (AudioCtx) {
            ctx = new AudioCtx();
            analyser = ctx.createAnalyser();
            analyser.fftSize = 1024;
            ctx.createMediaStreamSource(stream).connect(analyser);
            samples = new Uint8Array(analyser.fftSize);
        }

        const timer = setInterval(() => {
            const now = Date.now();
            if (analyser) {
                analyser.getByteTimeDomainData(samples);
                let sum = 0;
                for (const v of samples) {
                    const x = (v - 128) / 128;
                    sum += x * x;
                }
                if (Math.sqrt(sum / samples.length) > SPEECH_LEVEL) {
                    lastSoundAt = now;
                    heardSpeech = true;
                }
            }
            if ((heardSpeech && now - lastSoundAt > END_OF_SPEECH_DELAY) ||
                (!heardSpeech && now - startedAt > SILENCE_DELAY) ||
                now - startedAt > MAX_RECORDING_MS) {
                stopRecognition();
            }
        }, 100);

        silenceMonitor = { timer, ctx };
    }

    function stopSilenceMonitor() {
        if (!silenceMonitor) return;
        clearInterval(silenceMonitor.timer);
        if (silenceMonitor.ctx) silenceMonitor.ctx.close();
        silenceMonitor = null;
    }

    async function transcribeRecording(blob) {
        updateStatus('Transcribing...', 'processing');
        const ext = blob.type.includes('mp4') ? 'm4a' : blob.type.includes('ogg') ? 'ogg' : 'webm';
        const form = new FormData();
        form.append('audio_file', blob, `speech.${ext}`);

        try {
            const response = await fetch('/audio/transcribe', { method: 'POST', body: form });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'Transcription failed');

            const text = (data.text || '').trim();
            // Keep whatever was typed before recording started
            const prefix = accumulatedTranscript ? accumulatedTranscript + ' ' : '';
            userInput.value = (prefix + text).trim();
            userInput.dispatchEvent(new Event('input'));
            updateStatus('Online', 'online');

            if (text && !userFocusedInput) {  // only auto-send if the user didn't click into the box
                sendMessage();
            }
        } catch (e) {
            console.error('Transcription error:', e);
            updateStatus('Voice input failed', 'error');
        } finally {
            userFocusedInput = false;
        }
    }

    function toggleSpeechRecognition() {
        if (isStartingRecording) return;
        if (isRecognizing) {
            stopRecognition();
        } else {
            stopAudio();                                    // abort TTS before listening
            accumulatedTranscript = userInput.value.trim(); // preserve existing text
            startRecognition();
        }
    }

    // discard=true stops without transcribing (e.g. the user pressed Send meanwhile)
    function stopRecognition(discard = false) {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            discardRecording = discard;
            mediaRecorder.stop();  // triggers onstop, which resets the UI and transcribes
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

        // Stop whatever is currently playing before fetching new audio
        audioPlayer.pause();
        audioPlayer.removeAttribute('src');
        audioPlayer.load();
        audioPlayerContainer.classList.add('hidden');

        fetch('/audio/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        })
        .then(r => r.json())
        .then(data => {
            if (data.audio) {
                audioPlayer.src = 'data:' + (data.mime_type || 'audio/mpeg') + ';base64,' + data.audio;
                audioPlayer.load();
                audioPlayer.currentTime = 0;
                showAudioPlayer();
                audioPlayer.play().catch(e => console.error('Audio playback error:', e));
            }
        })
        .catch(error => console.error('TTS error:', error));
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
            }, 1500);
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
    
    // Focus input on page load (not on touch devices, where it would pop up the keyboard)
    if (!window.matchMedia('(pointer: coarse)').matches) {
        userInput.focus();
    }

    clearButton.addEventListener('click', clearChat);

    // Sidebar drawer (small screens)
    function setSidebarOpen(open) {
        document.body.classList.toggle('sidebar-open', open);
        menuButton.setAttribute('aria-expanded', String(open));
    }
    menuButton.addEventListener('click', () => setSidebarOpen(!document.body.classList.contains('sidebar-open')));
    sidebarBackdrop.addEventListener('click', () => setSidebarOpen(false));
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') setSidebarOpen(false);
    });

    // Suggestion buttons
    suggestionButtons.forEach(button => {
        button.addEventListener('click', function() {
            setSidebarOpen(false);
            const query = this.getAttribute('data-query');
            userInput.value = query;
            userInput.dispatchEvent(new Event('input')); // Trigger height adjustment
            sendMessage();
        });
    });

    userInput.addEventListener('mousedown', function() {
        if (isRecognizing) {
            userFocusedInput = true;
        }
    });

    userInput.addEventListener('touchstart', function() {
        if (isRecognizing) {
            userFocusedInput = true;
        }
    });
    
    // Functions
    function sendMessage() {
        const message = userInput.value.trim();
        
        if (message === '' || isProcessing) return;
        
        // Stop any current voice recording without transcribing it
        stopRecognition(true);
        
        // Stop any current audio playback
        stopAudio();
        accumulatedTranscript = '';   // ← reset so next session starts clean
        userFocusedInput = false;
        
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

            // Rate-limit / validation errors carry a `detail` message instead of an answer
            if (typeof data.answer !== 'string') {
                data = { answer: data.detail || 'Sorry, there was an error processing your request. Please try again.' };
            }

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
        
        // Parse markdown for assistant messages only. Never render it unsanitized: replies can
        // echo user input, so raw HTML from the model could run scripts (XSS).
        if (role === 'assistant' && typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
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
            messageBody.innerHTML = DOMPurify.sanitize(marked.parse(content));

            // Let wide tables scroll sideways inside the bubble instead of overflowing small screens
            messageBody.querySelectorAll('table').forEach(table => {
                const wrapper = document.createElement('div');
                wrapper.classList.add('table-scroll');
                table.parentNode.insertBefore(wrapper, table);
                wrapper.appendChild(table);
            });

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