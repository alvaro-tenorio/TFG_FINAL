import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt

# MODELO ESTADISTICO

#funciones 
def multivariate_gaussian(dataset,mu,sigma):
    p = multivariate_normal(mean=mu, cov=sigma)
    return p.logpdf(dataset)

def estimate_gaussian(dataset):
    mu =np.mean(dataset, axis =0)
    sigma = np.cov(dataset.T)

    return mu, sigma

# prueba de generacion de datos pseudoaleatorios
# datos = np.random.normal(0, 0.08, 300)
# mu, sigma = estimate_gaussian(datos)

data = np.genfromtxt('/home/alvaro/TFG/V1_acc/ventilador_normal.txt', delimiter='\t', skip_header=1)
# en data la primera columna es nan en su totalidad

clean_data = data[:, 1:] #limpiamos data 

mu, sigma = estimate_gaussian(clean_data)


#IMPORTS
import os
import glob
import time

anomalia = False
#while True:
data_prueba = np.genfromtxt('/home/alvaro/TFG/V1_acc/ventilador_normal.txt', delimiter='\t', skip_header=1)

# en data la primera columna es nan en su totalidad
clean_data_2 = data_prueba[:, 1:] #limpiamos data 

acc_readings = clean_data_2  
anomaliasbi= np.zeros_like(acc_readings)  
p = multivariate_gaussian(acc_readings, mu, sigma)

# Buscamos el threshold
print(np.mean(p))
plt.hist(p)
plt.xlabel("score on normal dataset")
plt.ylabel("No of examples")
plt.show()
plt.savefig("./graficas/score hist")
plt.close()

ep = -6 #umbral de anomal'ia
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
plt.savefig("./graficas/anomalias.png")
plt.close()






    


        

