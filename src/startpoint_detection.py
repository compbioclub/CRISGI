import numpy as np
from scipy.signal import medfilt
from sklearn.preprocessing import normalize

def extract_energy(signal, frame_size, hop_size):
    energy = np.array([np.sum(np.abs(signal[i:i+frame_size]**2)) for i in range(0, len(signal) - frame_size + 1, hop_size)])
    return energy

def short_term_energy(signal, frame_size, hop_size, window_function=np.hamming):
    num_samples = len(signal)
    num_frames = (num_samples - frame_size) // hop_size + 1
    energy = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frame = signal[start:end]
        windowed_frame = frame * window_function(frame_size)
        energy[i] = np.sum(windowed_frame ** 2)

    return energy

def detect_start_point(signal_matrix, sample_rate, frame_size_ms=25, hop_size_ms=10, energy_function=extract_energy):
    """
    Parameters:
    - energy_function: Function to use for energy calculation (default is extract_energy).
    
    """
    
    num_signals, num_samples = signal_matrix.shape
    frame_size = int(frame_size_ms * sample_rate / 1000)
    hop_size = int(hop_size_ms * sample_rate / 1000)

    # Check if hop_size is zero or if num_samples is too small
    if hop_size == 0 or (num_samples - frame_size) < 0:
        raise ValueError("Invalid parameters: hop_size is zero or num_samples is too small compared to frame_size.")
    
    num_frames = (num_samples - frame_size) // hop_size + 1
    
    # Extract energy features for each signal
    energies = np.zeros((num_signals, num_frames))
    for i in range(num_signals):
        energies[i, :] = energy_function(signal_matrix[i, :], frame_size, hop_size)
    
    # Normalize energies
    energies = normalize(energies, axis=1)

    # Sum the energies across signals
    summed_energy = np.sum(energies, axis=0)

    # Apply median filtering to smooth the energy
    smoothed_energy = medfilt(summed_energy, kernel_size=5)

    # Detect start point using thresholding
    threshold = np.mean(smoothed_energy) + 2 * np.std(smoothed_energy)
    start_point = np.where(smoothed_energy > threshold)[0][0]

    # Convert the index back to the original sample index
    start_sample = start_point * hop_size

    return start_sample