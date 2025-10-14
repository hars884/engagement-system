import os
import subprocess
from moviepy import VideoFileClip
import cv2
import numpy as np
from pathlib import Path

# Whisper import (openai/whisper)
try:
    import whisper
except Exception:
    whisper = None


def extract_audio(video_path, out_audio_path, sr=16000):
    """Extracts audio using moviepy and writes wav file"""
    video = VideoFileClip(str(video_path))
    audio = video.audio
    audio.write_audiofile(str(out_audio_path), fps=sr)
    audio.close()
    video.close()
    return out_audio_path


def extract_frames(video_path, out_dir, fps=1):
    """Extract frames at the given fps (frames per second) and save to out_dir"""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = int(round(video_fps / fps))
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            fname = os.path.join(out_dir, f"frame_{saved:06d}.jpg")
            cv2.imwrite(fname, frame)
            saved += 1
        count += 1
    cap.release()
    return out_dir


def transcribe_audio_whisper(audio_path, model_name='small'):
    """Run Whisper and return list of segments with (start, end, text)
    Requires whisper to be installed and models available."""
    if whisper is None:
        raise RuntimeError("whisper not installed; please pip install openai-whisper or whisper")
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path))
    # result contains 'segments' with start/end/text
    segments = result.get('segments', [])
    return segments


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--out', default='./data/tmp')
    parser.add_argument('--fps', type=int, default=1)
    parser.add_argument('--whisper_model', default='small')
    args = parser.parse_args()

    out_dir = Path(args.out)
    frames_dir = out_dir / 'frames'
    audio_path = out_dir / 'audio.wav'
    frames_dir.mkdir(parents=True, exist_ok=True)
    extract_frames(args.video, frames_dir, fps=args.fps)
    extract_audio(args.video, audio_path)
    try:
        segments = transcribe_audio_whisper(audio_path, model_name=args.whisper_model)
        # save segments
        import json
        with open(out_dir / 'transcript_segments.json', 'w') as f:
            json.dump(segments, f, indent=2)
        print('Transcription saved to', out_dir / 'transcript_segments.json')
    except Exception as e:
        print('Whisper transcription failed:', e)
        print('You can still use the frames and audio for downstream processing.')
