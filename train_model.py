import os
import cv2
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import moviepy.editor as mp

# -------------------------------
# CONFIGURATION
# -------------------------------
DATA_DIR = "data"
CLASSES = ["low", "mid", "high"]
IMG_SIZE = 128
SAMPLES_PER_VIDEO = 5   # Number of frames extracted from each video
AUDIO_DURATION = 3      # seconds per audio clip
SR = 16000              # audio sampling rate

# -------------------------------
# FEATURE EXTRACTION FUNCTIONS
# -------------------------------

def extract_visual_features(video_path):
    """Extract a few representative frames and return as numpy array"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // SAMPLES_PER_VIDEO)
    
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0
            frames.append(frame)
        if len(frames) >= SAMPLES_PER_VIDEO:
            break
    cap.release()
    
    if len(frames) < SAMPLES_PER_VIDEO:
        # pad with zeros if not enough frames
        while len(frames) < SAMPLES_PER_VIDEO:
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))
    return np.array(frames).mean(axis=0)  # average of frames


def extract_audio_features(video_path):
    """Extract MFCC features from audio"""
    try:
        clip = mp.VideoFileClip(video_path)
        audio = clip.audio
        temp_audio_path = "temp.wav"
        audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
        
        y, sr = librosa.load(temp_audio_path, sr=SR, duration=AUDIO_DURATION)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean
    except Exception as e:
        print(f"Audio extraction error: {e}")
        return np.zeros(40)

# -------------------------------
# LOAD DATASET
# -------------------------------

X_visual, X_audio, y = [], [], []

for idx, cls in enumerate(CLASSES):
    folder = os.path.join(DATA_DIR, cls)
    for file in tqdm(os.listdir(folder), desc=f"Loading {cls}"):
        if not file.endswith(('.mp4', '.avi', '.mov')):
            continue
        path = os.path.join(folder, file)
        vis_feat = extract_visual_features(path)
        aud_feat = extract_audio_features(path)
        X_visual.append(vis_feat)
        X_audio.append(aud_feat)
        y.append(idx)

X_visual = np.array(X_visual)
X_audio = np.array(X_audio)
y = to_categorical(y, num_classes=len(CLASSES))

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
Xv_train, Xv_test, Xa_train, Xa_test, y_train, y_test = train_test_split(
    X_visual, X_audio, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODEL ARCHITECTURE
# -------------------------------

# Visual branch
input_visual = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base_cnn = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_visual)
x1 = GlobalAveragePooling2D()(base_cnn.output)

# Audio branch
input_audio = Input(shape=(40,))
x2 = Dense(64, activation='relu')(input_audio)

# Combine
combined = concatenate([x1, x2])
x = Dense(128, activation='relu')(combined)
x = Dropout(0.3)(x)
output = Dense(len(CLASSES), activation='softmax')(x)

model = Model(inputs=[input_visual, input_audio], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# -------------------------------
# TRAIN MODEL
# -------------------------------
history = model.fit(
    [Xv_train, Xa_train], y_train,
    validation_data=([Xv_test, Xa_test], y_test),
    epochs=10,
    batch_size=8
)

# -------------------------------
# SAVE MODEL
# -------------------------------
model.save("engagement_model.h5")
print("✅ Model saved as engagement_model.h5")
