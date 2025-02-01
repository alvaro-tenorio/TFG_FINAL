import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt

# EXTRAER DATOS DEL .TXT


# MODELO ESTADISTICO

#funciones 
def multivariate_gaussian(dataset,mu,sigma):
    p = multivariate_normal(mean=mu, cov=sigma)

    return p.pdf(dataset)

def estimate_gaussian(dataset):
    mu =np.mean(dataset, axis =0)
    sigma = np.cov(dataset.T)

    return mu, sigma

ep = 10

if __name__ == '__main__':
    data = np.genfromtxt('/home/alvaro/TFG/V1/normal3.txt', delimiter='\t', skip_header=1)
    # en data la primera columna es nan en su totalidad
    clean_data = data[:, 1:] #limpiamos data 
    #print(clean_data)
    

    #sacamos mu y sigma de nuestra recogida de datos "normales"
    mu, sigma = estimate_gaussian(clean_data)

    p = multivariate_gaussian(clean_data, mu, sigma)

    #outliers = np.asarray(np.where(p<ep))
    """
    plt.figure()
    plt.xlabel("accZ")
    plt.ylabel("wZ")
    plt.plot(clean_data[:,2], clean_data[:,5], "bx")
    plt.plot(clean_data[outliers, 2], clean_data[outliers, 5], "ro")
    plt.savefig("prueba.png")
    """

    #plt.hist(p)
    #plt.savefig("mi-histograma3.png")


