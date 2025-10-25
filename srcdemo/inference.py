import json
from pathlib import Path
from src.preprocess import extract_audio, extract_frames, transcribe_audio_whisper
from src.nlp_analysis import analyze_transcript_segments
from src.audio_analysis import compute_noise_metrics, detect_overlapping_speech
from src.video_analysis import detect_faces_and_emotions
from src.scoring import compute_engagement_score, classify_engagement


def run_pipeline(video_path, tmp_out='./data/tmp', whisper_model='small'):
    tmp_out = Path(tmp_out)
    tmp_out.mkdir(parents=True, exist_ok=True)
    frames_dir = tmp_out / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    audio_path = tmp_out / 'audio.wav'

    print('Extracting frames...')
    extract_frames(video_path, frames_dir, fps=1)
    print('Extracting audio...')
    extract_audio(video_path, audio_path)
    print('Transcribing (Whisper)...')
    try:
        segments = transcribe_audio_whisper(audio_path, model_name=whisper_model)
    except Exception as e:
        print('Transcription failed:', e)
        segments = []

    print('Running NLP analysis...')
    nlp_out = analyze_transcript_segments(segments)

    print('Audio analysis...')
    audio_out = compute_noise_metrics(audio_path)
    overlap = detect_overlapping_speech(audio_path)

    print('Video analysis...')
    video_out = detect_faces_and_emotions(frames_dir)

    # compute derived features
    hot_ratio = (nlp_out['hot'] / nlp_out['questions_total']) if nlp_out['questions_total'] else 0.0
    participation_diversity = nlp_out.get('participation_diversity', 0.0)
    positive_emotion_pct = video_out.get('positive_emotion_pct', 0.0)
    discipline_index = audio_out.get('discipline_index', 0.0)
    # teacher-student talk balance: heuristic: ratio of teacher words to student words
    # TODO: speaker diarization to compute actual value. For now assume 0.5
    teacher_student_balance = 0.5

    score = compute_engagement_score(hot_ratio, participation_diversity, positive_emotion_pct, discipline_index, teacher_student_balance)
    cls = classify_engagement(score)

    report = {
        'score': score,
        'class': cls,
        'hot_ratio': hot_ratio,
        'participation_diversity': participation_diversity,
        'positive_emotion_pct': positive_emotion_pct,
        'discipline_index': discipline_index,
        'nlp': nlp_out,
        'audio': audio_out,
        'audio_overlap': overlap,
        'video': video_out
    }
    out_file = tmp_out / 'report.json'
    with open(out_file, 'w') as f:
        json.dump(report, f, indent=2)
    print('Report written to', out_file)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--out', default='./data/tmp')
    parser.add_argument('--whisper_model', default='small')
    args = parser.parse_args()
    run_pipeline(args.video, tmp_out=args.out, whisper_model=args.whisper_model)
