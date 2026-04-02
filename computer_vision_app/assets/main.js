const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const tempCanvas = document.createElement('canvas');
const tempCtx = tempCanvas.getContext('2d');
let ws;
let lastGestures = [];
let currentMatchedGesture = null;
let gestureContainer;
let lastPanelData = null;

function shouldUpdatePanel(newGestures, newMatched) {
    if (!lastPanelData) return true;
    
    const probThreshold = 0.15;
    const gestureThreshold = 0.25;
    
    if (newGestures.length !== lastPanelData.gestures.length) return true;
    if (newMatched !== lastPanelData.matched) return true;
    
    for (let i = 0; i < newGestures.length; i++) {
        const newG = newGestures[i];
        const lastG = lastPanelData.gestures[i];
        
        if (newG.label !== lastG.label) return true;
        if (newG.gesture !== lastG.gesture) return true;
        
        if (Math.abs(newG.probability - lastG.probability) > probThreshold) return true;
    }
    
    return false;
}

function updateGesturePanel(gestures, matched) {
    if (!gestureContainer) return;
    
    if (!shouldUpdatePanel(gestures, matched)) return;
    
    lastPanelData = {
        gestures: gestures.map(g => ({ ...g })),
        matched: matched
    };
    
    let html = '';
    
    if (gestures.length === 0) {
        html += '<div class="empty-state">No hands detected yet...</div>';
    } else {
        gestures.forEach(g => {
            html += `
                <div class="gesture-badge">
                    <span class="hand-label">${g.label}</span>
                    <span class="gesture-name">${g.gesture}</span>
                    <span class="gesture-prob">${(g.probability * 100).toFixed(0)}%</span>
                </div>
            `;
        });
    }
    
    if (matched) {
        html += `
            <div class="matched-gesture">
                <img src="/images/gestures/${matched}.png" alt="${matched}">
                <div class="matched-title">${matched}</div>
            </div>
        `;
    }
    
    gestureContainer.innerHTML = html;
}

navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
    video.onloadedmetadata = () => {
        canvas.width = tempCanvas.width = video.videoWidth || 640;
        canvas.height = tempCanvas.height = video.videoHeight || 480;
        gestureContainer = document.getElementById('gesture-container');
        initWS();
    };
});

function initWS() {
    const protocol = location.protocol === 'https:' ? 'wss:': 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);
    ws.onmessage = (e) => {
        const message = e.data;
        const data = JSON.parse(message);

        const img = new Image();
        img.onload = () => {
            lastGestures = data.gestures || [];
            currentMatchedGesture = data.matchedGesture;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
            drawLabels();
            updateGesturePanel(lastGestures, currentMatchedGesture);
            sendFrame();
        };
        img.src = data.image;
    };
    ws.onopen = sendFrame;
    ws.onclose = () => setTimeout(initWS, 1000);
}

function drawLabels() {
    ctx.font = 'bold 18px Poppins, sans-serif';
    ctx.fillStyle = '#fff';
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.lineWidth = 3;
    lastGestures.forEach((g, i) => {
        const text = `${g.label}: ${g.gesture}`;
        ctx.strokeText(text, 20, 50 + (i * 35));
        ctx.fillText(text, 20, 50 + (i * 35));
    });
}

function sendFrame() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        tempCtx.drawImage(video, 0, 0);
        ws.send(JSON.stringify({ image: tempCanvas.toDataURL('image/jpeg', 0.6) }));
    }
}
