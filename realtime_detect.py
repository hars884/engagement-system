import cv2
import numpy as np
import tensorflow as tf
import librosa
import sounddevice as sd
import moviepy.editor as mp
import tempfile

IMG_SIZE = 128
CLASSES = ["Low", "Mid", "High"]

model = tf.keras.models.load_model("engagement_model.h5")

cap = cv2.VideoCapture(0)

def predict_frame(frame):
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_norm = frame_resized / 255.0
    dummy_audio = np.zeros((1, 40))
    pred = model.predict([np.expand_dims(frame_norm, axis=0), dummy_audio])
    return CLASSES[np.argmax(pred)]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    pred_label = predict_frame(frame)
    cv2.putText(frame, f"Engagement: {pred_label}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Engagement Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()