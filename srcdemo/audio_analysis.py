import librosa
import numpy as np
from scipy.signal import find_peaks


def compute_noise_metrics(wav_path, sr=16000):
    y, sr = librosa.load(wav_path, sr=sr)
    # short-time energy
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    energy = np.array([np.sum(np.abs(y[i:i+frame_len])**2) for i in range(0, len(y)-frame_len, hop_len)])
    avg_energy = float(np.mean(energy))
    peaks, _ = find_peaks(energy, height=np.percentile(energy, 90))
    transient_rate = len(peaks) / max(1, len(energy))
    # discipline index: lower noise and fewer transients == higher discipline
    discipline_index = max(0.0, 1.0 - (avg_energy / (np.percentile(energy, 99) + 1e-9)) - transient_rate)
    discipline_index = float(np.clip(discipline_index, 0.0, 1.0))
    return {
        'avg_energy': avg_energy,
        'transient_rate': transient_rate,
        'discipline_index': discipline_index
    }


def detect_overlapping_speech(wav_path, sr=16000, threshold=0.02):
    # naive approach: count frames where short-time energy is high in multiple frequency bands
    y, sr = librosa.load(wav_path, sr=sr)
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    band_energy = S.sum(axis=0)
    high_frames = band_energy > (np.percentile(band_energy, 75))
    overlap_ratio = float(np.sum(high_frames) / len(high_frames))
    return {'overlap_ratio': overlap_ratio}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav', required=True)
    args = parser.parse_args()
    print(compute_noise_metrics(args.wav))
    print(detect_overlapping_speech(args.wav))

