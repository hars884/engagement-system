from typing import List, Dict
import re
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# For simple sentiment we can use a small transformer or rule-based using word lists
try:
    from transformers import pipeline
    _sentiment = pipeline('sentiment-analysis')
except Exception:
    _sentiment = None

# simple HOT-question heuristics
HOT_KEYWORDS = [
    'why', 'how', 'explain', 'compare', 'contrast', 'justify', 'evaluate', 'analyse', 'analyze', 'propose'
]
LOT_KEYWORDS = [
    'what is', 'when', 'where', 'who', 'define', 'name', 'list', 'state', 'spell'
]


def is_question(sentence: str) -> bool:
    return '?' in sentence or sentence.strip().lower().startswith(tuple(HOT_KEYWORDS + LOT_KEYWORDS + ['what', 'why', 'how', 'when', 'where', 'who']))


def classify_hot_lot(sentence: str) -> str:
    s = sentence.lower()
    # basic heuristics
    if any(s.startswith(k) for k in HOT_KEYWORDS) or any(k + ' ' in s for k in HOT_KEYWORDS):
        return 'HOT'
    if any(s.startswith(k) for k in LOT_KEYWORDS) or any(k + ' ' in s for k in LOT_KEYWORDS):
        return 'LOT'
    # fallback: length heuristic
    return 'HOT' if len(s.split()) > 7 else 'LOT'


def sentiment_of_text(text: str) -> Dict:
    if _sentiment:
        out = _sentiment(text[:512])
        return out[0]
    # fallback: naive polarity
    pos_words = ['good', 'great', 'well', 'nice', 'excellent', 'happy', 'understand']
    neg_words = ['bad', 'boring', 'confused', 'don\'t', "don't", 'no', 'not']
    score = sum(text.lower().count(w) for w in pos_words) - sum(text.lower().count(w) for w in neg_words)
    label = 'POSITIVE' if score > 0 else ('NEGATIVE' if score < 0 else 'NEUTRAL')
    return {'label': label, 'score': float(score)}


def analyze_transcript_segments(segments: List[Dict]) -> Dict:
    """Given Whisper segments (with text, start, end), produce:
    - total_questions, HOT_count, LOT_count
    - participation_estimate: number of unique speakers (heuristic)
    - sentiments by time
    """
    questions = []
    hot = 0
    lot = 0
    total = 0
    sentiments = []
    for seg in segments:
        text = seg.get('text', '').strip()
        if not text:
            continue
        # split sentences
        sents = re.split(r'[\n\.\?]+', text)
        for s in sents:
            if not s.strip():
                continue
            if is_question(s):
                total += 1
                lab = classify_hot_lot(s)
                if lab == 'HOT':
                    hot += 1
                else:
                    lot += 1
                questions.append({'text': s.strip(), 'type': lab, 'start': seg.get('start'), 'end': seg.get('end')})
        sentiments.append({'start': seg.get('start'), 'end': seg.get('end'), 'sentiment': sentiment_of_text(text)})

    # participation estimate (simple heuristic): count distinct short phrases like "I think" "yes" "no" "um" as markers.
    # Better: speaker diarization (not implemented here)
    # We'll use number of long utterances as proxy for distinct participants
    long_utt_count = sum(1 for seg in segments if len(seg.get('text','').split()) > 6)
    participation_diversity = min(1.0, long_utt_count / 10.0)  # scaled 0-1

    return {
        'questions_total': total,
        'hot': hot,
        'lot': lot,
        'questions': questions,
        'sentiments': sentiments,
        'participation_diversity': participation_diversity
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--segments', required=True, help='path to transcript_segments.json')
    args = parser.parse_args()
    import json
    with open(args.segments) as f:
        segs = json.load(f)
    out = analyze_transcript_segments(segs)
    print(json.dumps(out, indent=2))