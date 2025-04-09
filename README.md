# UPHY-404-CTP-Project
Project Title: Audio Signal Analysis and Visualization



**This code generates, visualizes, analyzes (via FFT), and plays a high-frequency sound wave along with its harmonics, simulating vibrating string behavior.**

import numpy as np
import matplotlib.pyplot as plt

# Define sound wave parameters
frequency = 200000
duration = 0.01  
sample_rate = 44100

# Time
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Generate sine wave
wave = np.sin(2 * np.pi * frequency * t)

# Plot the wave
plt.figure(figsize=(8,4))
plt.plot(t, wave)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"Pure Sine Wave - {frequency} Hz")
plt.grid()
plt.show()


# Simulate harmonics on a string
L = 1  
x = np.linspace(0, L, 1000)

harmonics = [1, 2, 3]  # First three harmonics

plt.figure(figsize=(8,5))
for n in harmonics:
    y = np.sin(n * np.pi * x / L)  
    plt.plot(x, y, label=f"Harmonic {n}")

plt.xlabel("Position on String")
plt.ylabel("Amplitude")
plt.title("Vibrating String Harmonics")
plt.legend()
plt.grid()
plt.show()


from scipy.fft import fft, fftfreq

# Compute Fourier Transform
N = len(t)  
yf = fft(wave)  
xf = fftfreq(N, 1/sample_rate)  # Frequency axis

# Plot FFT magnitude spectrum
plt.figure(figsize=(8,4))
plt.plot(xf[:N//2], np.abs(yf[:N//2]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Fourier Transform of Sound Wave")
plt.grid()
plt.show()


import sounddevice as sd

def generate_note(frequency, duration=1, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)  
    for n in range(2, 6):  # Adding harmonics
        wave += (0.5/n) * np.sin(2 * np.pi * n * frequency * t)
    return wave

note = generate_note(frequency, duration=2)
sd.play(note, 44100)  



**This code records audio from a microphone, analyzes it using the Fast Fourier Transform (FFT), and detects the dominant frequency (pitch) of the sound, then displays its spectrum.**

import numpy as np
import sounddevice as sd
import scipy.fftpack as fft
import matplotlib.pyplot as plt

# Parameters
fs = 44100  
duration = 3
threshold = 0.01  # Minimum signal level to detect voice

def record_audio(duration, fs):
    """Record audio from the microphone."""
    print("Recording... Speak into the microphone.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    return audio

def detect_frequency(audio, fs):
    """Detect dominant frequency using FFT."""
    n = len(audio)
    audio_fft = np.abs(fft.fft(audio))
    freqs = np.fft.fftfreq(n, 1/fs)
    
    # Consider only positive frequencies
    positive_freqs = freqs[:n//2]
    positive_fft = audio_fft[:n//2]
    
    positive_fft[positive_fft < threshold] = 0
    
    peak_idx = np.argmax(positive_fft)
    peak_freq = positive_freqs[peak_idx]
    
    return peak_freq, positive_freqs, positive_fft

def plot_spectrum(freqs, fft_vals, peak_freq):
    """Plot frequency spectrum."""
    plt.plot(freqs, fft_vals)
    plt.title(f"Detected Frequency: {peak_freq:.2f} Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

audio_data = record_audio(duration, fs)
peak_freq, freqs, fft_vals = detect_frequency(audio_data, fs)

# Check 
if peak_freq > 20 and peak_freq < 3000:
    print(f"Detected frequency: {peak_freq:.2f} Hz")
    plot_spectrum(freqs, fft_vals, peak_freq)
else:
    print("No significant speech detected. Try speaking louder or closer to the microphone.")

