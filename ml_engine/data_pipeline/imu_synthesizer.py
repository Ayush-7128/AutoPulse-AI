"""Physics-Locked 100Hz 3-Axis IMU (ax, ay, az) Synthesizer & Feature Extractor."""
import numpy as np
from scipy.stats import kurtosis

def extract_imu_physics_vector(ax, ay, az, sr=100.0):
    rms_x, rms_y, rms_z = np.sqrt(np.mean(ax**2)), np.sqrt(np.mean(ay**2)), np.sqrt(np.mean(az**2))
    kurt_z = float(kurtosis(az, fisher=False))
    crest_z = float(np.max(np.abs(az)) / (rms_z + 1e-6))
    fft_z = np.abs(np.fft.rfft(az - np.mean(az)))
    freqs = np.fft.rfftfreq(len(az), d=1.0 / sr)
    dom_freq = float(freqs[np.argmax(fft_z)])
    psd = (fft_z**2) / (np.sum(fft_z**2) + 1e-9)
    spec_ent = float(-np.sum(psd * np.log2(psd + 1e-12)))
    return [rms_x, rms_y, rms_z, kurt_z, crest_z, dom_freq, spec_ent, float(np.mean(ax))]
