import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
from sklearn.mixture import GaussianMixture
from procesamiento_aceleracion import build_melspectrogram

from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle 

############HYPERPARAMETROS###################
SR=10 #Hz
N_FFT=32
HOP_LENGTH= int(N_FFT/2)
N_MELS = 16

#########EXTRACCION DE DATOS#############
data = np.genfromtxt('/home/alvaro/TFG/V1_acc/medidas/ventilador_encima_normal.txt', delimiter='\t', skip_header=1)
    # en data la primera columna es nan en su totalidad
clean_data = data[:, 1:] #limpiamos data 
## procesado informacion para el modelo ####
acc_x = clean_data[:, 0]
acc_y = clean_data[:, 1]
acc_z = clean_data[:, 2]
# Combine axes to get magnitude
acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
normal_dataset = build_melspectrogram(y=acc_magnitude, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels= N_MELS).T #queremos (samples, features)

# Tunning del modelo GMM
# ==============================================================================
fig, ax = plt.subplots(figsize=(6, 3.84))
fig2, ay = plt.subplots(figsize=(6, 3.84))

n_components = range(1, 50)
covariance_types = ['spherical', 'tied', 'diag', 'full']
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
fig.savefig("/home/alvaro/TFG/V1_acc/graficas_gmm/modelTunning")

ay.set_title("Valores AIC")
ay.set_xlabel("Número componentes")
ay.legend()
fig2.savefig("/home/alvaro/TFG/V1_acc/graficas_gmm/modelTunning2")

best_result = min(results, key=lambda x: x["min_bic"])
print(f"Best parameters: n_compononents={best_result['best_ncomp']}, covariance_type={best_result['covariance_type']}")

######### GUARDAR EL MODELO EN UN PICKLE #############################################################
gm = GaussianMixture(n_components=best_result['best_ncomp'], covariance_type=best_result['covariance_type'])
gm = gm.fit(normal_dataset)
file_path = 'modelo_normal_largo.pkl'

with open(file_path, 'wb') as file:
    pickle.dump(gm, file)

print(f"Object saved to {file_path}")
