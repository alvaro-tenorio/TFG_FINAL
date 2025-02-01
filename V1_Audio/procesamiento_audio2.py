import numpy as np
import librosa
import matplotlib.pyplot as plt


def build_melspectrogram(audio_route, n_fft, hop_length, n_mels):
    audio_ndarray, sr = librosa.load(path=audio_route, sr=None)
    #Get mel spectogram
    melspectrogram = librosa.feature.melspectrogram(
        y = audio_ndarray, 
        n_fft = n_fft, 
        hop_length = hop_length, 
        n_mels = n_mels
    )
    # Express in decibel units
    melspectrogram = librosa.power_to_db(melspectrogram, ref=1e-12)   #  ref=np.max 
    # Flatten the spectrogram
    #melspectrogram = melspectrogram.reshape(1, -1)
    return melspectrogram

"""
FFT_SIZE = 4096
HOP_LENGTH = int(FFT_SIZE * 0.5)
N_MELS = 50
melspectrogram = build_melspectrogram(audio_route='./audio/anomalias_nomuyanomalo_60s.wav', n_fft=FFT_SIZE, hop_length=HOP_LENGTH, n_mels=N_MELS)

plt.figure(figsize=(10, 4))
librosa.display.specshow(
    melspectrogram, 
    sr=44100, 
    hop_length=HOP_LENGTH, 
    x_axis='time', 
    y_axis='mel', 
    cmap='viridis'
)
plt.colorbar(format='%+2.0f dB')
plt.title('log-Mel Spectrogram')
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Frequency (Hz)", fontsize=12)
plt.tight_layout()
plt.savefig("audio_anomalo_melspectrogram")
plt.close()
"""
import matplotlib.pyplot as plt
import numpy as np
import wave

# Ruta al archivo .wav
wav_file_path = "./audio/nueva_base_para_ML.wav"  

# Leer el archivo .wav
with wave.open(wav_file_path, "r") as wav_file:
    # Obtener parámetros básicos
    n_frames = wav_file.getnframes()
    sampling_rate = wav_file.getframerate()
    n_channels = wav_file.getnchannels()

    # Leer datos de la señal
    frames = wav_file.readframes(n_frames)
    signal = np.frombuffer(frames, dtype=np.int16)

# Crear un vector de tiempo para la señal
time = np.linspace(0, len(signal) / sampling_rate, num=len(signal))

# Graficar la amplitud de la señal
plt.figure(figsize=(14, 6))
plt.plot(time, signal, label=f"Sampling Rate: {sampling_rate} Hz", color="blue", linewidth=0.6)
plt.title("Audio Signal", fontsize=14)
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.grid(alpha=0.4)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("audio_signal")

