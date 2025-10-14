import cv2
import os
from pathlib import Path
from deepface import DeepFace
import numpy as np

try:
    import mediapipe as mp
except Exception:
    mp = None


def detect_faces_and_emotions(frames_dir, max_frames=500):
    frames = sorted(Path(frames_dir).glob('*.jpg'))[:max_frames]
    emotions = []
    face_counts = []
    attention_scores = []
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        try:
            # DeepFace emotion detector
            res = DeepFace.analyze(img_path=str(f), actions=['emotion'], enforce_detection=False)
            # DeepFace returns dict; normalize
            if isinstance(res, list):
                res = res[0]
            em = res.get('emotion', {})
            emotions.append(em)
            # face count heuristic
            # DeepFace may also return region; using face detector separately is better
            face_counts.append(1 if res.get('region') else 0)
        except Exception as e:
            # fallback: 0
            emotions.append({})
            face_counts.append(0)

        # simple attention heuristic: detect eyes via Haar cascades or head pose later
        # placeholder 0.5
        attention_scores.append(0.5)

    # aggregate
    # compute percent positive emotions (happy/neutral as positive proxy)
    pos = 0
    total = 0
    for e in emotions:
        if not e:
            continue
        total += 1
        dominant = max(e.items(), key=lambda x: x[1])[0]
        if dominant in ('happy', 'neutral', 'surprise'):
            pos += 1
    positive_pct = (pos / total) if total else 0.0
    avg_attention = float(np.mean(attention_scores)) if attention_scores else 0.0
    avg_faces = float(np.mean(face_counts)) if face_counts else 0.0
    return {
        'positive_emotion_pct': positive_pct,
        'avg_attention': avg_attention,
        'avg_face_count': avg_faces
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', required=True)
    args = parser.parse_args()
    print(detect_faces_and_emotions(args.frames))
