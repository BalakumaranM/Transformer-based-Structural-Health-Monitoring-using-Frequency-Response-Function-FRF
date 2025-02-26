import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def moving_average_smoothing(array, kernel_size=5):
    if kernel_size < 2:
        return array
    pad_size = kernel_size // 2
    padded = np.pad(array, (pad_size, pad_size), mode='edge')
    smoothed = np.convolve(padded, np.ones(kernel_size)/kernel_size, mode='valid')
    return smoothed

def process_file(file_path, downsample_factor=6, unwrap_phase=True, smoothing=False, smoothing_kernel=5):
    try:
        df = pd.read_csv(file_path, sep="\t", engine="python")
    except Exception:
        df = pd.read_csv(file_path, sep=r'\s+', engine="python")
    
    freq = df["Frequency"].values.astype(np.float32)
    mag_db = df["Magnitude"].values.astype(np.float32)
    phase_deg = df["Phase"].values.astype(np.float32)
    
    if unwrap_phase:
        phase_rad = np.deg2rad(phase_deg)
        phase_rad_unwrapped = np.unwrap(phase_rad)
        phase_deg = np.rad2deg(phase_rad_unwrapped)
    
    if downsample_factor > 1:
        freq = freq[::downsample_factor]
        mag_db = mag_db[::downsample_factor]
        phase_deg = phase_deg[::downsample_factor]
    
    if smoothing and len(freq) >= smoothing_kernel:
        mag_db = moving_average_smoothing(mag_db, kernel_size=smoothing_kernel)
        phase_deg = moving_average_smoothing(phase_deg, kernel_size=smoothing_kernel)
    
    return freq, mag_db, phase_deg

def plot_raw_vs_processed(file_path, freq_max=2000.0, downsample_factor=6, unwrap_phase=True, smoothing=True, smoothing_kernel=5):
    try:
        df_raw = pd.read_csv(file_path, sep="\t", engine="python")
    except Exception:
        df_raw = pd.read_csv(file_path, sep="\s+", engine="python")
    freq_raw = df_raw["Frequency"].values.astype(np.float32)
    mag_raw = df_raw["Magnitude"].values.astype(np.float32)
    phase_raw = df_raw["Phase"].values.astype(np.float32)
    
    freq_proc, mag_proc, phase_proc = process_file(file_path,
                                                    downsample_factor=downsample_factor,
                                                    unwrap_phase=unwrap_phase,
                                                    smoothing=smoothing,
                                                    smoothing_kernel=smoothing_kernel)
    
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    plt.plot(freq_raw, mag_raw, label="Raw Magnitude", alpha=0.7)
    plt.plot(freq_proc, mag_proc, label="Processed Magnitude", alpha=0.7)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Magnitude: Raw vs Processed")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(freq_raw, phase_raw, label="Raw Phase", alpha=0.7)
    plt.plot(freq_proc, phase_proc, label="Processed Phase", alpha=0.7)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (degrees)")
    plt.title("Phase: Raw vs Processed")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "./Data/Beam_damage_2.96/P-1-1.txt"
    plot_raw_vs_processed(file_path, freq_max=2000.0, downsample_factor=6, unwrap_phase=True, smoothing=True, smoothing_kernel=5)
