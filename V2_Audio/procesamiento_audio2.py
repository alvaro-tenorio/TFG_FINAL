import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split

def build_melspectrogram(audio_ndarray, sample_rate, n_fft, hop_length, n_mels):
    #Get mel spectogram
    melspectrogram = librosa.feature.melspectrogram(
        y = audio_ndarray, 
        sr= sample_rate,
        n_fft = n_fft, 
        hop_length = hop_length, 
        n_mels = n_mels
    )
    # Express in decibel units
    melspectrogram = librosa.power_to_db(melspectrogram, ref=1e-12)   #  ref=np.max 
    # Flatten the spectrogram
    #melspectrogram = melspectrogram.reshape(1, -1)
    return melspectrogram

def build_melspectrogram_from_route(audio_route, n_fft, hop_length, n_mels):
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

FFT_SIZE = 4096
HOP_LENGTH = int(FFT_SIZE * 0.5)
N_MELS = 50
normal_dataset = build_melspectrogram_from_route(audio_route='/home/alvaro/TFG/V3_Audio/audio/nueva_base_para_ML.wav', n_fft=FFT_SIZE, hop_length=HOP_LENGTH, n_mels=N_MELS).T
anomalous_dataset = build_melspectrogram_from_route(audio_route='/home/alvaro/TFG/V3_Audio/audio/anomalias_nomuyanomalo_30s.wav', n_fft=FFT_SIZE, hop_length=HOP_LENGTH, n_mels=N_MELS).T
normal_dataset = np.hstack((normal_dataset, np.full((normal_dataset.shape[0],1), 1))) #anadimos una columna al final que es la etiqueta de sample "normal"
anomalous_dataset = np.hstack((anomalous_dataset, np.full((anomalous_dataset.shape[0],1), 0))) #lo mismo pero cn 0 para anomalia
data_labeled = np.vstack((normal_dataset, anomalous_dataset)) #tenemos todas las muestras labeled
labels = data_labeled[:, -1]
data = data_labeled[:, 0:-1]
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=21)
MIN_VAL = tf.reduce_min(train_data)
MAX_VAL = tf.reduce_max(train_data)
