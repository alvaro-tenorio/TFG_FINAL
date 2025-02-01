import numpy as np
from matplotlib import pyplot as plt
import os
import matplotlib.pyplot as plt
import librosa

def build_melspectrogram(y,sr, n_fft, hop_length, n_mels):

    #Get mel spectogram
    melspectrogram = librosa.feature.melspectrogram(
        y = y, 
        sr= sr,
        n_fft = n_fft, 
        hop_length = hop_length, 
        n_mels = n_mels
    )
    # Express in decibel units
    melspectrogram = librosa.power_to_db(melspectrogram, ref=1e-12)   #  ref=np.max 
    # Flatten the spectrogram
    #melspectrogram = melspectrogram.reshape(1, -1)
    return melspectrogram


###### EXTRACCION RAW DATA #################
data = np.genfromtxt('/home/alvaro/TFG/V1_acc/medidas/anomalo_encima_3mins.txt', delimiter='\t', skip_header=1)
    # en data la primera columna es nan en su totalidad
clean_data = data[:, 1:] #limpiamos data 

sr = 10 #Accelerometer documentation default output rate in Hz
###### TRATAMIENTO DATA ###############
acc_x = clean_data[:, 0]
acc_y = clean_data[:, 1]
acc_z = clean_data[:, 2]

# Combine axes to get magnitude
acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

# Remove mean to isolate dynamic components (remove gravity)
media = np.mean(acc_magnitude)
acc_magnitude -= np.mean(acc_magnitude)

# Compute melspectrogram
n_fft = 32
hop_length = int(n_fft/2)
n_mels = 10
melspectrogram = build_melspectrogram(y=acc_magnitude, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels= n_mels)

plt.figure(figsize=(10, 4))
librosa.display.specshow(
    melspectrogram, 
    sr=sr, 
    hop_length=hop_length, 
    x_axis='time', 
    y_axis='mel', 
    cmap='viridis'
)
plt.colorbar(format='%+2.0f dB')
plt.title('log-Mel Spectrogram')
plt.tight_layout()
plt.savefig("acc_melspectrogram")
