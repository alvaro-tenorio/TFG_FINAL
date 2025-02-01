import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
import os
import scipy
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
import IPython.display as ipd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pickle 

from V1_Audio.antiguo.procesamiento_audio import frame_audio, get_filter_points, get_filters, dct

###### MODELO GMM########
# Lo recuperamos del pickle
file_path = 'modelo_normal_largo.pkl'
with open(file_path, 'rb') as file:
    gm = pickle.load(file)

##########################

###### AUDIO EN EL QUE ENCONTRAR ANOMALIAS #########
sample_rate, audio = wavfile.read('/home/alvaro/TFG/V1_Audio/anormal_al_final2.wav')
######## PARAMETERS ########
ep = -100 #variar para encontrar optima 
FFT_size = 4096
overlapping = 0.5
freq_min = 0
freq_high = sample_rate / 2
mel_filter_num = 40
dct_filter_num = 20
anomalia = False

#PROCESAMIENTO AUDIO PARA PROBAR ANOMALIAS 
audio_framed = frame_audio(audio, FFT_size=FFT_size, overlapping=overlapping, sample_rate=sample_rate)
window = get_window("hann", FFT_size, fftbins=True)
audio_win = audio_framed * window
audio_winT = np.transpose(audio_win)
audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')
for n in range(audio_fft.shape[1]):
    audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]
audio_fft = np.transpose(audio_fft)
audio_power = np.square(np.abs(audio_fft))
filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=44100)
filters = get_filters(filter_points, FFT_size)
enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
filters *= enorm[:, np.newaxis]
audio_filtered = np.dot(filters, np.transpose(audio_power))
audio_log = 10.0 * np.log10(audio_filtered)
dct_filters = dct(dct_filter_num, mel_filter_num)
cepstral_coefficents = np.dot(dct_filters, audio_log)

###################################################################

######## EVALUATE MODEL ###################
acc_readings= cepstral_coefficents.T #queremos array de la forma (n_samples, n_features)
anomaliasbi= np.zeros_like(acc_readings)  
p = gm.score_samples(acc_readings)
count = 0
for i in p:

    if i < ep:
        print(count)
        print("ANOMALIAAAA")
        anomalia = True
        anomaliasbi[count]=1
    elif i > ep:
        print("no anomalia")
    count=count+1
#############################################

####### SHOW RESULTS #######
plt.figure()
plt.plot(anomaliasbi)
plt.savefig("anomalias.png")