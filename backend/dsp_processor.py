"""4th-Order Butterworth Bandpass Filter (100Hz - 7.8kHz) & Log-Mel Spectrogram Extractor."""
import librosa
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_and_mel(y, sr=16000):
    nyq = 0.5 * sr
    b, a = butter(4, [100.0 / nyq, 7800.0 / nyq], btype="band")
    y_filt = filtfilt(b, a, y)
    mel = librosa.feature.melspectrogram(y=y_filt, sr=sr, n_fft=1024, hop_length=512, n_mels=128, fmin=100, fmax=7800)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
