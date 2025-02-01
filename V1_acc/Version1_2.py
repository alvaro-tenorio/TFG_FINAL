import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.mixture import GaussianMixture
from procesamiento_aceleracion import build_melspectrogram
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
############HYPERPARAMETROS###################
SR=10 #Hz
N_FFT=32
HOP_LENGTH= int(N_FFT/2)
N_MELS = 16

#########EXTRACCION DE DATOS#############
data = np.genfromtxt('./medidas/ventilador_encima_normal.txt', delimiter='\t', skip_header=1)
    # en data la primera columna es nan en su totalidad
clean_data = data[:, 1:] #limpiamos data 
## procesado informacion para el modelo ####
acc_x = clean_data[:, 0]
acc_y = clean_data[:, 1]
acc_z = clean_data[:, 2]
# Combine axes to get magnitude
acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)


normal_dataset = build_melspectrogram(y=acc_magnitude, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels= N_MELS).T #queremos (samples, features)

######## testeo del modelo ########
data_prueba = np.genfromtxt('./medidas/anomalo_encima_3mins.txt', delimiter='\t', skip_header=1)
clean_data_2 = data_prueba[:, 1:] #limpiamos data 
## procesado informacion para el modelo ####
acc_x = clean_data_2[:, 0]
acc_y = clean_data_2[:, 1]
acc_z = clean_data_2[:, 2]
# Combine axes to get magnitude
acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
anomalous_dataset = build_melspectrogram(y=acc_magnitude, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels= N_MELS).T

normal_dataset = np.hstack((normal_dataset, np.full((normal_dataset.shape[0],1), 1))) #anadimos una columna al final que es la etiqueta de sample "normal"
anomalous_dataset = np.hstack((anomalous_dataset, np.full((anomalous_dataset.shape[0],1), 0))) #lo mismo pero cn 0 para anomalia

data_labeled = np.vstack((normal_dataset, anomalous_dataset)) #tenemos todas las muestras labeled
labels = data_labeled[:, -1]
data = data_labeled[:, 0:-1]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=21)

# Separate "normal" data
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]


#creacion modelo
file_path = './modelo_normal_largo.pkl'
with open(file_path, 'rb') as file:
    gm = pickle.load(file)

train_score = gm.score_samples(normal_train_data)
threshold=np.percentile(train_score, 1)
print("threshold:{}".format(threshold))
###############################

normal_scores = gm.score_samples(normal_train_data)
anomalous_scores = gm.score_samples(anomalous_test_data)

# Plot reconstruction losses on normal samples 
plt.close()
plt.hist(normal_scores, bins=50)
plt.xlabel("Train scores")
plt.ylabel("No of examples")
plt.show()
plt.savefig("./graficas_gmm/train scores")
plt.close()

plt.hist(anomalous_scores, bins=50)
plt.xlabel("Test scores")
plt.ylabel("No of examples")
plt.show()
plt.savefig("./graficas_gmm/test score")
plt.close()

# DETECT ANOMALIES ON THE TEST SET
def predict(data, threshold):
    score = gm.score_samples(data)
    predictions = []
    for sample_score in score:
        if(sample_score<threshold):
            predictions.append(False)
        else:
            predictions.append(True)
    return np.array(predictions), score

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))

# PREDICT ANOMALIES
predictions, loss = predict(test_data, threshold)
print_stats(predictions, test_labels)

# Calculate false positives and false negatives
false_positives = np.sum((predictions == False) & (test_labels == True))  # Normal classified as anomaly
false_negatives = np.sum((predictions == True) & (test_labels == False))  # Anomaly classified as normal

# Calculate total normal and anomalous examples
total_normals = np.sum(test_labels == True)
total_anomalies = np.sum(test_labels == False)

# Calculate rates
false_alarm_rate = false_positives / total_normals if total_normals > 0 else 0
miss_error_rate = false_negatives / total_anomalies if total_anomalies > 0 else 0

print(f"False Alarms (False Positives): {false_positives}")
print(f"Miss Errors (False Negatives): {false_negatives}")
print(f"False Alarm Rate: {false_alarm_rate:.2%}")
print(f"Miss Error Rate: {miss_error_rate:.2%}")

#Confusion Matrix
cm = confusion_matrix(test_labels.astype(int), predictions.astype(int))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomalous", "Normal"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("./graficas_gmm/confusion_matrix.png")
plt.close()
print("Confusion Matrix:")
print(cm)


"""
anomaliasbi= np.zeros_like(acc_readings)  
p = gm.score_samples(acc_readings)
plt.figure()
plt.hist(p)
plt.savefig("./graficas_gmm/score_normal.png")
plt.close()
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
plt.savefig("./graficas_gmm/anomalias.png")

"""