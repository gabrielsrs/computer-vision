const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const tempCanvas = document.createElement('canvas');
const tempCtx = tempCanvas.getContext('2d');
let ws;

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
        const img = new Image();
        img.onload = () => {
            ctx.drawImage(img, 0, 0);
            sendFrame();
        };
        img.src = e.data;
    };
    ws.onopen = sendFrame;
    ws.onclose = () => setTimeout(initWS, 1000);
}

function sendFrame() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        tempCtx.drawImage(video, 0, 0);
        ws.send(JSON.stringify({ image: tempCanvas.toDataURL('image/jpeg', 0.6)}))
    }
}