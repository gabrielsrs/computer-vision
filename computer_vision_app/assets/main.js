const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const tempCanvas = document.createElement('canvas');
const tempCtx = tempCanvas.getContext('2d');
const gestureImg = new Image();
let ws;
let lastGestures = [];
let currentMatchedGesture = null;

navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
    video.onloadedmetadata = () => {
        canvas.width = tempCanvas.width = video.videoWidth;
        canvas.height = tempCanvas.height = video.videoHeight;
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
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            lastGestures = data.gestures || [];
            currentMatchedGesture = data.matchedGesture;

            if (currentMatchedGesture) {
                gestureImg.src = `/images/gestures/${currentMatchedGesture}.png`;
                gestureImg.onload = () => {
                    ctx.drawImage(gestureImg, 0, 0, canvas.width, canvas.height);
                    drawLabels();
                    sendFrame();
                };
            } else {
                ctx.drawImage(img, 0, 0);
                drawLabels();
                sendFrame();
            }
        };
        img.src = data.image;
    };
    ws.onopen = sendFrame;
    ws.onclose = () => setTimeout(initWS, 1000);
}

function drawLabels() {
    ctx.font = '16px sans-serif';
    ctx.fillStyle = '#00ff00';
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 3;
    lastGestures.forEach((g, i) => {
        const text = `${g.label}: ${g.gesture} (${g.probability.toFixed(2)})`;
        ctx.strokeText(text, 20, 50 + (i * 30));
        ctx.fillText(text, 20, 50 + (i * 30));
    });
}

function sendFrame() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        tempCtx.drawImage(video, 0, 0);
        ws.send(JSON.stringify({ image: tempCanvas.toDataURL('image/jpeg', 0.6) }));
    }
}
