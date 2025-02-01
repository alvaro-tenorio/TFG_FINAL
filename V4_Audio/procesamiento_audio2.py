import numpy as np
import librosa



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