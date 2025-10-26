import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import moviepy.editor as mp
import os

# Load model once
MODEL_PATH = "engagement_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = 128
CLASSES = ["Low", "Mid", "High"]

def predict_frame(frame):
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_norm = frame_resized / 255.0
    dummy_audio = np.zeros((1, 40))
    pred = model.predict([np.expand_dims(frame_norm, axis=0), dummy_audio], verbose=0)
    return CLASSES[np.argmax(pred)]

st.title("🎥 Engagement Detection")
st.write("Upload a short video to test the engagement level.")

uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        tmpfile.write(uploaded_video.read())
        video_path = tmpfile.name

    # Display uploaded video
    st.video(video_path)

    if st.button("Analyze Engagement"):
        st.write("⏳ Processing video... please wait...")

        # Use MoviePy to safely read video
        try:
            clip = mp.VideoFileClip(video_path)
            total_frames = int(clip.fps * clip.duration)
            frames_to_analyze = min(total_frames, 30)  # analyze only first 30 frames

            frame_results = []
            for i, frame in enumerate(clip.iter_frames(fps=clip.fps)):
                if i >= frames_to_analyze:
                    break
                pred_label = predict_frame(frame)
                frame_results.append(pred_label)

            # Calculate majority prediction
            if frame_results:
                final_pred = max(set(frame_results), key=frame_results.count)
                st.success(f"🧠 Engagement Level: **{final_pred}**")
            else:
                st.warning("No frames processed from the video.")
        except Exception as e:
            st.error(f"Error processing video: {e}")
        finally:
            # Close clip and safely delete file
            if 'clip' in locals():
                clip.close()
            try:
                os.remove(video_path)
            except PermissionError:
                pass  # Windows file lock — ignore safely
