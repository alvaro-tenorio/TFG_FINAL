import numpy as np
from sklearn.mixture import GaussianMixture
import os
import matplotlib.pyplot as plt
import pickle 

#SACAMOS LOS DATOS DE LA GRABACION "NORMAL" DEL ACELEROMETRO
data = np.genfromtxt('/home/alvaro/TFG/V2_acc/medidas/normal3.txt', delimiter='\t', skip_header=1)
    # en data la primera columna es nan en su totalidad
clean_data = data[:, 1:] #limpiamos data 

########### GENERACION MODELO CON GMM ###############################################################
gm = GaussianMixture(n_components= 6).fit(clean_data) #se crea el modelo 
"fit(X) donde X es un array con shape (n_samples, n_features)"

p= gm.score_samples(clean_data)
plt.figure()
plt.hist(p)
plt.savefig("histo.png")
plt.close()
######################################################################################################
######### GUARDAR EL MODELO EN UN PICKLE #############################################################

file_path = 'modelo_normal_largo.pkl'

with open(file_path, 'wb') as file:
    pickle.dump(gm, file)

print(f"Object saved to {file_path}")