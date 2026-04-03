import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math

MODEL_PATH = "models/hand_landmarker.task"

# -------- MediaPipe --------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# -------- UI CONFIG --------
UI_W, UI_H = 1200, 700
CAM_X, CAM_Y = 300, 120
CAM_W, CAM_H = 600, 400

HAND_BONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

def dist(a,b):
    return math.hypot(a.x-b.x, a.y-b.y)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (CAM_W, CAM_H))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)

    result = detector.detect(mp_image)

    gesture = "NO HAND"
    confidence = 0

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # bones
        for a,b in HAND_BONES:
            p1, p2 = hand[a], hand[b]
            cv2.line(
                frame,
                (int(p1.x*CAM_W), int(p1.y*CAM_H)),
                (int(p2.x*CAM_W), int(p2.y*CAM_H)),
                (255,255,255), 2
            )

        # points
        for lm in hand:
            cv2.circle(
                frame,
                (int(lm.x*CAM_W), int(lm.y*CAM_H)),
                4, (0,0,255), -1
            )

        # simple open palm confidence
        wrist = hand[0]
        ups = sum([hand[i].y < wrist.y for i in [8,12,16,20]])
        confidence = int((ups/4)*100)
        gesture = "OPEN PALM" if confidence > 60 else "UNKNOWN"

    # -------- MAIN UI CANVAS --------
    ui = np.zeros((UI_H, UI_W, 3), dtype=np.uint8)

    # borders
    cv2.rectangle(ui, (0,0), (UI_W-1, UI_H-1), (0,255,255), 2)

    # camera window
    cv2.rectangle(ui, (CAM_X-2, CAM_Y-2),
                  (CAM_X+CAM_W+2, CAM_Y+CAM_H+2), (0,255,255), 2)

    ui[CAM_Y:CAM_Y+CAM_H, CAM_X:CAM_X+CAM_W] = frame

    # left panel
    cv2.putText(ui, "SYSTEM LOGS", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # right panel
    cv2.putText(ui, f"Gesture: {gesture}", (950, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.putText(ui, f"Confidence: {confidence}%", (950, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # confidence bar
    cv2.rectangle(ui, (950, 270), (950+confidence*2, 295), (0,255,0), -1)
    cv2.rectangle(ui, (950, 270), (1150, 295), (255,255,255), 2)

    # title
    cv2.putText(ui, "AI GESTURE CONTROL CENTER",
                (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,255), 3)

    cv2.imshow("Gesture UI", ui)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
