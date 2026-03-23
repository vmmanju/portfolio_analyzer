const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');

// API Configuration
const API_URL = '/api/v1/assistant/query';

// Initialize Markdown
marked.setOptions({
    gfm: true,
    breaks: true,
    highlight: function(code, lang) {
        return code;
    }
});

function appendMessage(text, role, rawData = null) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    
    if (role === 'ai') {
        const contentDiv = document.createElement('div');
        contentDiv.className = 'ai-content';
        contentDiv.innerHTML = marked.parse(text);
        msgDiv.appendChild(contentDiv);
        
        // Check for charts (Price Movement)
        if (rawData && rawData.get_price_movement && rawData.get_price_movement.history) {
            const chartDiv = document.createElement('div');
            chartDiv.className = 'chart-container';
            const chartId = `chart-${Date.now()}`;
            chartDiv.id = chartId;
            msgDiv.appendChild(chartDiv);
            
            setTimeout(() => renderChart(chartId, rawData.get_price_movement), 100);
        }
    } else {
        msgDiv.textContent = text;
    }
    
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    // Remove welcome message if active
    const welcome = document.querySelector('.welcome-message');
    if (welcome) welcome.remove();
}

function renderChart(containerId, data) {
    const history = data.history;
    const x = history.map(d => d.date);
    const y = history.map(d => d.close);
    
    const trace = {
        x: x,
        y: y,
        type: 'scatter',
        mode: 'lines',
        name: data.symbol,
        line: {
            color: '#3b82f6',
            width: 2
        },
        fill: 'tozeroy',
        fillcolor: 'rgba(59, 130, 246, 0.1)'
    };
    
    const layout = {
        title: {
            text: `${data.symbol} Price History`,
            font: { color: '#f8fafc', size: 14 }
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { t: 40, b: 40, l: 40, r: 20 },
        height: 350,
        xaxis: {
            gridcolor: 'rgba(255,255,255,0.05)',
            tickfont: { color: '#94a3b8' }
        },
        yaxis: {
            gridcolor: 'rgba(255,255,255,0.05)',
            tickfont: { color: '#94a3b8' }
        }
    };
    
    Plotly.newPlot(containerId, [trace], layout, {responsive: true, displayModeBar: false});
}

async function handleSend() {
    const query = userInput.value.trim();
    if (!query) return;
    
    appendMessage(query, 'user');
    userInput.value = '';
    
    // Show Thinking indicator
    const thinkingDiv = document.createElement('div');
    thinkingDiv.className = 'message ai thinking';
    thinkingDiv.textContent = 'Analyzing...';
    chatContainer.appendChild(thinkingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_query: query })
        });
        
        chatContainer.removeChild(thinkingDiv);
        const data = await response.json();
        
        if (data.explanation) {
            appendMessage(data.explanation, 'ai', data.raw_outputs);
        } else {
            appendMessage("I'm sorry, I couldn't process that request.", 'ai');
        }
    } catch (error) {
        if (thinkingDiv.parentNode) chatContainer.removeChild(thinkingDiv);
        appendMessage("Connection error. Please ensure the backend is running.", 'ai');
        console.error(error);
    }
}

function sendSuggestion(text) {
    userInput.value = text;
    handleSend();
}

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleSend();
});

sendButton.addEventListener('click', handleSend);
