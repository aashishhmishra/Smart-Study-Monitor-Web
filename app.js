// ============================================================
// Smart Study Monitor — Web Version
// Uses MediaPipe Face Landmarker + TensorFlow.js COCO-SSD
// ============================================================

import {
    FaceLandmarker,
    FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/vision_bundle.mjs";

// -------------------- DOM Elements --------------------
const video = document.getElementById("webcam");
const videoWrapper = document.getElementById("videoWrapper");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const warningEl = document.getElementById("warning");
const eyeRatioEl = document.getElementById("eyeRatioDisplay");
const placeholder = document.getElementById("placeholder");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusText = document.getElementById("statusText");
const dotFace = document.getElementById("dotFace");
const dotPhone = document.getElementById("dotPhone");

const alarmSleep = document.getElementById("alarmSleep");
const alarmFace = document.getElementById("alarmFace");
const alarmPhone = document.getElementById("alarmPhone");

// -------------------- Constants --------------------
const LEFT_EYE_TOP = 159;
const LEFT_EYE_BOTTOM = 145;
const FACE_LEFT = 130;
const FACE_RIGHT = 243;

const SLEEP_THRESHOLD_FRAMES = 15;
const COVER_THRESHOLD_FRAMES = 20;
const EYE_RATIO_THRESHOLD = 11.0;
const PHONE_CONFIDENCE = 0.5;
const PHONE_CHECK_INTERVAL = 400; // ms between phone inference runs

// -------------------- State --------------------
let faceLandmarker = null;
let phoneDetector = null;
let isRunning = false;
let animFrameId = null;

let closedFrames = 0;
let coveredFrames = 0;
let currentPlaying = null;
let lastVideoTime = 0;

let lastPhoneCheck = 0;
let isCheckingPhone = false;
let cachedPhoneBoxes = [];

// -------------------- Helpers --------------------
function distance(p1, p2, width, height) {
    const dx = (p1.x - p2.x) * width;
    const dy = (p1.y - p2.y) * height;
    return Math.sqrt(dx * dx + dy * dy);
}

function showWarning(text, type) {
    warningEl.textContent = text;
    warningEl.className = `warning ${type}`;
}

function hideWarning() {
    warningEl.className = "warning hidden";
}

function unlockAudio() {
    [alarmSleep, alarmFace, alarmPhone].forEach((audio) => {
        audio.muted = true;
        const playPromise = audio.play();
        if (playPromise !== undefined) {
            playPromise
                .then(() => {
                    audio.pause();
                    audio.currentTime = 0;
                    audio.muted = false;
                })
                .catch(() => {
                    audio.muted = false;
                });
        } else {
            audio.muted = false;
        }
    });
}

function isAudioPlaying() {
    return (
        !alarmSleep.paused ||
        !alarmFace.paused ||
        !alarmPhone.paused
    );
}

function playAlarm(audioEl, label) {
    if (!isAudioPlaying()) {
        audioEl.currentTime = 0;
        audioEl.play().catch(() => {});
        currentPlaying = label;
    }
}

// -------------------- Model Loading --------------------
async function createFaceLandmarker(vision, delegate) {
    return FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate,
        },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
    });
}

async function loadModels() {
    statusText.textContent = "Status: Loading Face Model...";
    dotFace.className = "dot loading";

    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm"
    );

    try {
        faceLandmarker = await createFaceLandmarker(vision, "GPU");
    } catch {
        faceLandmarker = await createFaceLandmarker(vision, "CPU");
    }

    dotFace.className = "dot active";
    statusText.textContent = "Status: Loading Phone Model...";
    dotPhone.className = "dot loading";

    phoneDetector = await window.cocoSsd.load({ base: "lite_mobilenet_v2" });

    dotPhone.className = "dot active";
    statusText.textContent = "Status: Models Ready ✅";
}

// -------------------- Camera --------------------
async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
    });
    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            videoWrapper.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
            resolve();
        };
    });
}

function stopCamera() {
    const stream = video.srcObject;
    if (stream) {
        stream.getTracks().forEach((t) => t.stop());
        video.srcObject = null;
    }
}

// -------------------- Phone Detection (Non-blocking) --------------------
async function checkPhoneAsync() {
    if (!phoneDetector || isCheckingPhone || video.readyState < 2) return;
    isCheckingPhone = true;
    try {
        const predictions = await phoneDetector.detect(video);
        if (!isRunning) return;

        cachedPhoneBoxes = predictions
            .filter((p) => p.class === "cell phone" && p.score > PHONE_CONFIDENCE)
            .map((p) => ({ bbox: p.bbox, score: p.score }));
    } catch (err) {
        console.warn("Phone detection error:", err);
    } finally {
        isCheckingPhone = false;
    }
}

function drawPhoneBoxes() {
    for (const item of cachedPhoneBoxes) {
        const [x, y, w, h] = item.bbox;
        // Video is mirrored via CSS scaleX(-1), so mirror X coordinate to align overlay & keep text readable
        const mirroredX = canvas.width - x - w;

        ctx.strokeStyle = "#ff00ff";
        ctx.lineWidth = 3;
        ctx.strokeRect(mirroredX, y, w, h);

        const label = `Phone ${Math.round(item.score * 100)}%`;
        ctx.font = "bold 16px Inter, sans-serif";
        const textWidth = ctx.measureText(label).width;
        const labelY = y > 26 ? y - 24 : y + h + 2;

        ctx.fillStyle = "#ff00ff";
        ctx.fillRect(mirroredX, labelY, textWidth + 12, 22);

        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, mirroredX + 6, labelY + 16);
    }
}

// -------------------- Detection Loop --------------------
function detectLoop() {
    if (!isRunning) return;

    const now = performance.now();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (video.readyState < 2) {
        animFrameId = requestAnimationFrame(detectLoop);
        return;
    }

    // ----- 1. Face Detection -----
    let isSleepy = false;
    let isFaceCovered = false;

    if (faceLandmarker) {
        const timestamp = Math.max(Math.round(now), lastVideoTime + 1);
        lastVideoTime = timestamp;
        const faceResults = faceLandmarker.detectForVideo(video, timestamp);

        if (faceResults.faceLandmarks && faceResults.faceLandmarks.length > 0) {
            coveredFrames = 0;
            const lm = faceResults.faceLandmarks[0];

            const eyeDist = distance(lm[LEFT_EYE_TOP], lm[LEFT_EYE_BOTTOM], canvas.width, canvas.height);
            const faceDist = distance(lm[FACE_LEFT], lm[FACE_RIGHT], canvas.width, canvas.height);
            const ratio = faceDist > 0 ? (eyeDist / faceDist) * 100 : 0;

            eyeRatioEl.textContent = `Eye Ratio: ${Math.round(ratio)}`;

            if (ratio < EYE_RATIO_THRESHOLD) {
                closedFrames++;
            } else {
                closedFrames = 0;
            }

            if (closedFrames >= SLEEP_THRESHOLD_FRAMES) {
                isSleepy = true;
            }
        } else {
            closedFrames = 0;
            coveredFrames++;

            if (coveredFrames >= COVER_THRESHOLD_FRAMES) {
                isFaceCovered = true;
            }

            eyeRatioEl.textContent = "Eye Ratio: --";
        }
    }

    // ----- 2. Phone Detection (throttled & non-blocking) -----
    if (phoneDetector && !isCheckingPhone && now - lastPhoneCheck >= PHONE_CHECK_INTERVAL) {
        lastPhoneCheck = now;
        checkPhoneAsync();
    }

    const phoneDetected = cachedPhoneBoxes.length > 0;
    if (phoneDetected) {
        drawPhoneBoxes();
    }

    // ----- 3. Alarm & Warning Logic -----
    const audioBusy = isAudioPlaying();

    if (!audioBusy) {
        currentPlaying = null;
    }

    if (isFaceCovered) {
        showWarning("🙈 DON'T COVER YOUR FACE!", "face");
        dotFace.className = "dot alert";
        if (!audioBusy) playAlarm(alarmFace, "face");
    } else if (isSleepy) {
        showWarning("😴 WAKE UP & STUDY!", "sleep");
        dotFace.className = "dot alert";
        if (!audioBusy) playAlarm(alarmSleep, "sleep");
    } else if (phoneDetected) {
        showWarning("📱 PUT THE PHONE AWAY!", "phone");
        dotPhone.className = "dot alert";
        if (!audioBusy) playAlarm(alarmPhone, "phone");
    } else if (audioBusy && currentPlaying) {
        // Keep showing warning while audio plays
        if (currentPlaying === "face") showWarning("🙈 DON'T COVER YOUR FACE!", "face");
        else if (currentPlaying === "sleep") showWarning("😴 WAKE UP & STUDY!", "sleep");
        else if (currentPlaying === "phone") showWarning("📱 PUT THE PHONE AWAY!", "phone");
    } else {
        hideWarning();
        dotFace.className = "dot active";
        dotPhone.className = "dot active";
    }

    animFrameId = requestAnimationFrame(detectLoop);
}

// -------------------- Start / Stop --------------------
async function startMonitoring() {
    startBtn.disabled = true;
    unlockAudio();
    statusText.textContent = "Status: Starting camera...";

    try {
        await startCamera();
        placeholder.classList.add("hidden");
        eyeRatioEl.classList.remove("hidden");

        if (!faceLandmarker || !phoneDetector) {
            await loadModels();
        }

        isRunning = true;
        statusText.textContent = "Status: Monitoring 🟢";
        stopBtn.disabled = false;

        detectLoop();
    } catch (err) {
        console.error(err);
        statusText.textContent = "Error: " + err.message;
        startBtn.disabled = false;
    }
}

function stopMonitoring() {
    isRunning = false;
    if (animFrameId) cancelAnimationFrame(animFrameId);

    stopCamera();
    hideWarning();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    eyeRatioEl.classList.add("hidden");
    placeholder.classList.remove("hidden");

    // Stop all audio
    [alarmSleep, alarmFace, alarmPhone].forEach((a) => {
        a.pause();
        a.currentTime = 0;
    });

    closedFrames = 0;
    coveredFrames = 0;
    currentPlaying = null;
    lastVideoTime = 0;
    lastPhoneCheck = 0;
    isCheckingPhone = false;
    cachedPhoneBoxes = [];

    startBtn.disabled = false;
    stopBtn.disabled = true;
    dotFace.className = "dot";
    dotPhone.className = "dot";
    statusText.textContent = "Status: Stopped";
}

// -------------------- Event Listeners --------------------
startBtn.addEventListener("click", startMonitoring);
stopBtn.addEventListener("click", stopMonitoring);
