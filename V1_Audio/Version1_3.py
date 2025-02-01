import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from procesamiento_audio2 import build_melspectrogram 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.mixture import GaussianMixture
import pickle

# PARAMETERS
FFT_SIZE = 4096
HOP_LENGTH = int(FFT_SIZE * 0.5)
N_MELS = 50

# Load audio files
#sample_rate, normal_audio = wavfile.read('/home/alvaro/TFG/V3_Audio3/audio/nueva_base_para_ML.wav') #este audio dura unos 600 s de audio "normal"
#sr, abnormal_audio = wavfile.read('/home/alvaro/TFG/V3_Audio3/audio/anomalias_nomuyanomalo_30s.wav') #este son 10 segundos de audio "anormal"

# Ensure audio sample rates match
#if sample_rate != sr:
#    raise ValueError("Sample rates of the normal and anomalous audio files must match!")

normal_dataset = build_melspectrogram(audio_route='./audio/nueva_base_para_ML.wav', n_fft=FFT_SIZE, hop_length=HOP_LENGTH, n_mels=N_MELS).T
anomalous_dataset = build_melspectrogram(audio_route='./audio/anomalias_nomuyanomalo_60s.wav', n_fft=FFT_SIZE, hop_length=HOP_LENGTH, n_mels=N_MELS).T

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

############## CARGA DEL MODELO ENTRENADO ########################
file_path = './modelo_normal_largo.pkl'
with open(file_path, 'rb') as file:
    gm = pickle.load(file)

train_score = gm.score_samples(normal_test_data)
threshold=np.percentile(train_score, 1)
print("threshold:{}".format(threshold))

normal_scores = train_score
anomalous_scores = gm.score_samples(anomalous_test_data)

# Plot reconstruction losses on normal samples 
plt.hist(normal_scores, bins=200)
plt.xlabel("Train scores")
plt.ylabel("No of examples")
plt.show()
plt.savefig("./graficas_gmm/train scores")
plt.close()

plt.hist(anomalous_scores, bins=50)
plt.xlabel("Test scores")
plt.ylabel("No of examples")
plt.show()
plt.savefig("./graficas_gmm/test scores")
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
