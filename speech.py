import numpy as np
from scipy.signal import hamming, lfilter
from scipy.fftpack import fft2
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import os

def extract_glottal_waveform(signal, lpc_order=12):
    from scipy.linalg import toeplitz, solve

    autocorr = np.correlate(signal, signal, mode='full')[len(signal) - 1:]

    R = toeplitz(autocorr[:lpc_order])
    r = autocorr[1:lpc_order + 1]
    lpc_coeffs = solve(R, r)
    lpc_coeffs = np.insert(-lpc_coeffs, 0, 1)

    return lfilter(lpc_coeffs, [1.0], signal)

def compute_bispectrum(signal, frame_size=256):

    step = frame_size // 2
    frames = [signal[i:i + frame_size] * hamming(frame_size) 
              for i in range(0, len(signal) - frame_size + 1, step)]

    bispectrum = sum(np.abs(fft2(frame.reshape((16, 16)))) ** 2 for frame in frames)
    return bispectrum / len(frames)

def compute_bispectrum_features(bispectrum):

    total_amplitude = np.sum(bispectrum)

    f1_indices = np.arange(bispectrum.shape[0])
    f2_indices = np.arange(bispectrum.shape[1])
    f1_center = np.sum(f1_indices[:, None] * bispectrum) / total_amplitude
    f2_center = np.sum(f2_indices[None, :] * bispectrum) / total_amplitude

    probabilities = bispectrum / total_amplitude
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))

    mean_amplitude = np.mean(bispectrum)

    return f1_center, f2_center, entropy, mean_amplitude

def plot_bispectrum_contour_combined(speech_bispectrum, glottal_bispectrum):

    f1 = np.linspace(-0.5, 0.5, speech_bispectrum.shape[0])
    f2 = np.linspace(-0.5, 0.5, speech_bispectrum.shape[1])
    F1, F2 = np.meshgrid(f1, f2)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.contourf(F1, F2, 20 * np.log10(speech_bispectrum + 1e-9), levels=50, cmap='jet')
    plt.colorbar(label="Amplitude (dB)")
    plt.title("a) Speech Signal Bispectrum Contour",loc='left')
    plt.xlabel("Frequency (f1)")
    plt.ylabel("Frequency (f2)")

    plt.subplot(1, 2, 2)
    plt.contourf(F1, F2, 20 * np.log10(glottal_bispectrum + 1e-9), levels=50, cmap='jet')
    plt.colorbar(label="Amplitude (dB)")
    plt.title("b) Glottal Waveform Bispectrum Contour",loc='left')
    plt.xlabel("Frequency (f1)")
    plt.ylabel("Frequency (f2)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "/Users/liu/Downloads/new/sad.wav"

    try:
        signal, fs = librosa.load(file_path, sr=None)

        glottal_waveform = extract_glottal_waveform(signal)

        speech_bispectrum = compute_bispectrum(signal)

        glottal_bispectrum = compute_bispectrum(glottal_waveform)

        speech_features = compute_bispectrum_features(speech_bispectrum)
        print(f"Speech Bispectrum Features:\n  f1 Center: {speech_features[0]}\n  f2 Center: {speech_features[1]}\n  Entropy: {speech_features[2]}\n  Mean Amplitude: {speech_features[3]}")

        glottal_features = compute_bispectrum_features(glottal_bispectrum)
        print(f"Glottal Bispectrum Features:\n  f1 Center: {glottal_features[0]}\n  f2 Center: {glottal_features[1]}\n  Entropy: {glottal_features[2]}\n  Mean Amplitude: {glottal_features[3]}")

        plot_bispectrum_contour_combined(speech_bispectrum, glottal_bispectrum)

    except Exception as e:
        print(f"error")
