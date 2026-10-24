"""SNR-Controlled Indian Street & Dual-Tone Horn Audio Augmentation Module."""
import numpy as np

def mix_indian_traffic_noise(clean_sig, horn_sig, snr_db=10.0):
    p_sig = np.mean(clean_sig ** 2) + 1e-9
    p_noise = np.mean(horn_sig ** 2) + 1e-9
    alpha = np.sqrt(p_sig / (p_noise * (10.0 ** (snr_db / 10.0))))
    mixed = clean_sig + alpha * horn_sig
    return 0.95 * (mixed / (np.max(np.abs(mixed)) + 1e-6))
