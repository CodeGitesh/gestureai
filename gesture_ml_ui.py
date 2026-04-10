import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import numpy as np

MODEL_PATH = "models/hand_landmarker.task"
clf = joblib.load("gesture_model.pkl")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)
    result = detector.detect(mp_image)

    text = "NO HAND"

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        wrist = hand[0]

        features = []
        for lm in hand:
            features.extend([lm.x - wrist.x, lm.y - wrist.y])

        X = np.array(features).reshape(1, -1)
        probs = clf.predict_proba(X)[0]
        pred = clf.classes_[np.argmax(probs)]
        confidence = int(np.max(probs) * 100)

        text = f"{pred} ({confidence}%)"

    cv2.putText(frame, text, (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    cv2.imshow("ML Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
