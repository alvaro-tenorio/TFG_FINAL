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
# MODELO ESTADISTICO

#funciones 
"""
def multivariate_gaussian(dataset,mu,sigma):
    p = multivariate_normal(allow_singular=True, mean=mu, cov=sigma)
    return p.logpdf(dataset)

def estimate_gaussian(dataset):
    mu =np.mean(dataset, axis =0)
    sigma = np.cov(dataset.T)

    return mu, sigma
"""

# FUNCIONES AUDIO

sample_rate, audio = wavfile.read('/home/alvaro/TFG/V1_Audio/normal_largo.wav')
print("Sample rate: {0}Hz".format(sample_rate))
print("Audio duration: {0}s".format(len(audio) / sample_rate))

def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio

"""
audio = normalize_audio(audio)
plt.figure(figsize=(15,4))
plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(audio)), audio)
"""
#plt.grid(True)
#plt.savefig("prueba_.png")

def frame_audio(audio, FFT_size=4096, overlapping=0.5, sample_rate=44100):
    # hop_size in s
    hop_size=(FFT_size/sample_rate)*overlapping
    audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
    frame_len = np.round(sample_rate * hop_size).astype(int)
    frame_num = int((len(audio) - FFT_size) / frame_len) #+ 1
    frames = np.zeros((frame_num,FFT_size))
    
    for n in range(frame_num):
        frames[n] = audio[n*frame_len:n*frame_len+FFT_size]
    
    return frames


FFT_size = 4096
overlapping= 0.5

audio_framed = frame_audio(audio, FFT_size=FFT_size, overlapping=overlapping, sample_rate=sample_rate)
print("Framed audio shape: {0}".format(audio_framed.shape))

window = get_window("hann", FFT_size, fftbins=True)
audio_win = audio_framed * window

audio_winT = np.transpose(audio_win)

audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')

for n in range(audio_fft.shape[1]):
    audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]

audio_fft = np.transpose(audio_fft)

audio_winT = np.transpose(audio_win)

audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')

for n in range(audio_fft.shape[1]):
    audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]

audio_fft = np.transpose(audio_fft)

audio_power = np.square(np.abs(audio_fft))

freq_min = 0
freq_high = sample_rate / 2
mel_filter_num = 40
def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def met_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)

def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    
    #print("MEL min: {0}".format(fmin_mel))
    #print("MEL max: {0}".format(fmax_mel))
    
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = met_to_freq(mels)
    
    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs

filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=44100)

def get_filters(filter_points, FFT_size):
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    
    return filters

filters = get_filters(filter_points, FFT_size)

enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
filters *= enorm[:, np.newaxis]

audio_filtered = np.dot(filters, np.transpose(audio_power))
audio_log = 10.0 * np.log10(audio_filtered)

def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num,filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
        
    return basis

dct_filter_num = 20

dct_filters = dct(dct_filter_num, mel_filter_num)

cepstral_coefficents = np.dot(dct_filters, audio_log)


#GENERACION MODELO 
gm = GaussianMixture(n_components= dct_filter_num).fit(cepstral_coefficents.T) #se crea el modelo 
"fit(X) donde X es un array con shape (n_samples, n_features) por ello cepstral_coefficients.Transpuesta"
ep = -100 #variar para encontrar optima 

anomalia = False
#GENERACION AUDIO PARA PROBAR ANOMALIAS 
sr, audio_p = wavfile.read('/home/alvaro/TFG/V1_Audio/anormal_al_final2.wav')
#audioN = normalize_audio(audio_p) NO NORMALIZAMOS YA QUE ES CONTRAPRODUCENTE PARA ENCONTRAR ANOMALIAS 
audio_framed_p = frame_audio(audio_p, FFT_size=FFT_size, overlapping=overlapping, sample_rate=sr)
audio_win_p = audio_framed_p * window
audio_winT_p= np.transpose(audio_win_p)
audio_fft_p = np.empty((int(1 + FFT_size // 2), audio_winT_p.shape[1]), dtype=np.complex64, order='F')
for n in range(audio_fft_p.shape[1]):
    audio_fft_p[:, n] = fft.fft(audio_winT_p[:, n], axis=0)[:audio_fft_p.shape[0]]
audio_fft_p = np.transpose(audio_fft_p)
audio_winT_p = np.transpose(audio_win_p)
audio_fft_p = np.empty((int(1 + FFT_size // 2), audio_winT_p.shape[1]), dtype=np.complex64, order='F')
for n in range(audio_fft_p.shape[1]):
    audio_fft_p[:, n] = fft.fft(audio_winT_p[:, n], axis=0)[:audio_fft_p.shape[0]]
audio_fft_p = np.transpose(audio_fft_p)
audio_power_p = np.square(np.abs(audio_fft_p))
audio_filtered_p = np.dot(filters, np.transpose(audio_power_p))
audio_log_p = 10.0 * np.log10(audio_filtered_p)

cepstral_coefficents_p = np.dot(dct_filters, audio_log_p)

acc_readings= cepstral_coefficents_p.T #queremos array de la forma (n_samples, n_features)
anomaliasbi= np.zeros_like(acc_readings)  
p = gm.score_samples(acc_readings)
#print(acc_readings, p)
#capture = [acc_readings, p]
#recordings.append(capture)
#time.sleep(0.5)
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

plt.figure()
plt.plot(anomaliasbi)
plt.savefig("anomalias.png")






    


        

