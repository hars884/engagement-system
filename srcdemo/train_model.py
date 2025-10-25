import os
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sentence_transformers import SentenceTransformer

# NOTE: Full multimodal deep learning fusion is out of scope for this prototype; this script shows
# a simple approach: extract numeric features and train a regressor to predict engagement score.


def extract_text_features(transcript_segments):
    # concatenate and compute embedding
    s = ' '.join([seg.get('text','') for seg in transcript_segments])
    model = SentenceTransformer('all-MiniLM-L6-v2')
    emb = model.encode([s])[0]
    return emb


def extract_audio_features(audio_metrics):
    return np.array([audio_metrics.get('avg_energy',0), audio_metrics.get('transient_rate',0), audio_metrics.get('discipline_index',0)])


def extract_video_features(video_metrics):
    return np.array([video_metrics.get('positive_emotion_pct',0), video_metrics.get('avg_attention',0), video_metrics.get('avg_face_count',0)])


def train_from_dataset(dataset_dir, model_out='models/engagement_rf.pkl'):
    # Expect dataset_dir contains one subdir per sample with report.json and label.txt
    X = []
    y = []
    for sample in os.listdir(dataset_dir):
        sdir = os.path.join(dataset_dir, sample)
        if not os.path.isdir(sdir):
            continue
        rpath = os.path.join(sdir, 'report.json')
        lpath = os.path.join(sdir, 'label.txt')
        if not os.path.exists(rpath) or not os.path.exists(lpath):
            continue
        with open(rpath) as f:
            rep = json.load(f)
        with open(lpath) as f:
            lbl = float(f.read().strip())
        tfeat = extract_text_features(rep.get('nlp', {}).get('questions', []))
        afeat = extract_audio_features(rep.get('audio', {}))
        vfeat = extract_video_features(rep.get('video', {}))
        feat = np.concatenate([tfeat, afeat, vfeat])
        X.append(feat)
        y.append(lbl)
    X = np.vstack(X)
    y = np.array(y)
    # simple regressor
    clf = RandomForestRegressor(n_estimators=100)
    clf.fit(X, y)
    import joblib
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(clf, model_out)
    print('Saved model to', model_out)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='dataset dir with per-sample report.json and label.txt')
    parser.add_argument('--out', default='models/engagement_rf.pkl')
    args = parser.parse_args()
    train_from_dataset(args.data, model_out=args.out)
