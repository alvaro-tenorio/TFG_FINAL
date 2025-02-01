import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
from sklearn.mixture import GaussianMixture
from procesamiento_audio2 import build_melspectrogram

from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle 

############HYPERPARAMETROS###################
SR=44100 #Hz
FFT_SIZE = 4096
HOP_LENGTH = int(FFT_SIZE * 0.5)
N_MELS = 50

#########EXTRACCION DE DATOS#############
normal_dataset = build_melspectrogram(audio_route='./audio/nueva_base_para_ML.wav', n_fft=FFT_SIZE, hop_length=HOP_LENGTH, n_mels=N_MELS).T

fig, ax = plt.subplots(figsize=(6, 3.84))
fig2, ay = plt.subplots(figsize=(6, 3.84))

n_components = range(1, 40)
covariance_types = ['spherical', 'tied', 'full'] #quitamos spherical y diagonal porque siempre daban peores resultadoscdd
#covariance_types = ['tied', 'full']
results= []
for covariance_type in covariance_types:
    valores_bic = []
    valores_aic = []
    for i in n_components:
        modelo = GaussianMixture(n_components=i, covariance_type=covariance_type)
        modelo = modelo.fit(normal_dataset)
        valores_bic.append(modelo.bic(normal_dataset))
        valores_aic.append(modelo.aic(normal_dataset))
    
    min_bic = min(valores_bic)
    n_comp = valores_bic.index(min_bic)
    results.append({
        "covariance_type":covariance_type,
        "best_ncomp": n_comp,
        "min_bic":min_bic
    })
    ax.plot(n_components, valores_bic, label=covariance_type)
    ay.plot(n_components, valores_aic, label=covariance_type)
        
ax.set_title("Valores BIC")
ax.set_xlabel("Número componentes")
ax.legend()
fig.savefig("./graficas_gmm/modelTunning")

ay.set_title("Valores AIC")
ay.set_xlabel("Número componentes")
ay.legend()
fig2.savefig("./graficas_gmm/modelTunning2")

best_result = min(results, key=lambda x: x["min_bic"])
print(f"Best parameters: n_compononents={best_result['best_ncomp']}, covariance_type={best_result['covariance_type']}")

######### GUARDAR EL MODELO EN UN PICKLE #############################################################
gm = GaussianMixture(n_components=best_result['best_ncomp'], covariance_type=best_result['covariance_type'])
gm = gm.fit(normal_dataset)

file_path = 'modelo_normal_largo.pkl'

with open(file_path, 'wb') as file:
    pickle.dump(gm, file)

print(f"Object saved to {file_path}")
