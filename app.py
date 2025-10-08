from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import time
from playsound import playsound
import threading
from fastapi.responses import HTMLResponse
app = FastAPI(title="Drowsiness Detector")

# ---------- Config ----------
EAR_CONSEC_FRAMES = 15
MAR_THRESH = 0.7
ALERT_THRESHOLD = 0.75

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

blink_counter = 0
fatigue_score = 0.0

# ---------- Helper ----------
def smooth(prev, new, alpha=0.2):
    return alpha * new + (1 - alpha) * prev

def analyze_frame(frame):
    global blink_counter, fatigue_score
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    score = 0.0
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) < 2:
                blink_counter += 1
            else:
                blink_counter = 0

            score = 1.0 if blink_counter >= EAR_CONSEC_FRAMES else 0.0
            fatigue_score = smooth(fatigue_score, score)
            break
    else:
        fatigue_score = smooth(fatigue_score, 0.0)

    return fatigue_score, blink_counter

# ---------- Alert Sound ----------
def alert_sound():
    playsound("alert.mp3")  # put an alert.mp3 in the same folder

def stream_frames():
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        score, blink_count = analyze_frame(frame)

        color = (0, 0, 255) if score > ALERT_THRESHOLD else (0, 255, 0)
        cv2.putText(frame, f"Fatigue: {score:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"BlinkCount: {blink_count}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if score > ALERT_THRESHOLD:
            threading.Thread(target=alert_sound).start()  # play sound in background

        _, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

@app.get("/video")
def video():
    return StreamingResponse(stream_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/")
def home():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)