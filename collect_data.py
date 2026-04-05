import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os

MODEL_PATH = "models/hand_landmarker.task"
LABEL = "OPEN_PALM"  # change this manually each run

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

os.makedirs("data", exist_ok=True)
file = open("data/gestures.csv", "a", newline="")
writer = csv.writer(file)

print("Press 'c' to capture, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        wrist = hand[0]
        features = []
        for lm in hand:
            features.extend([lm.x - wrist.x, lm.y - wrist.y])

        cv2.putText(frame, f"Label: {LABEL}", (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            writer.writerow(features + [LABEL])
            print("Captured")

    cv2.imshow("Collect Data", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

file.close()
cap.release()
cv2.destroyAllWindows()
