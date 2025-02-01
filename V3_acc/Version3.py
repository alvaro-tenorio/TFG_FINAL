import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve, auc
from tensorflow import keras
from keras import layers, losses
#from tensorflow.python.keras.datasets import fashion_mnist
from keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from procesamiento_aceleracion import build_melspectrogram_from_acc

###### GENERACION DE ANOMALIAS SINTETICAS####
# Synthetic anomaly generation functions
def add_gaussian_noise(data, mean=0, std=0.05):
    noise = np.random.normal(mean, std, data.shape)
    return data + noise

def distort_signal(data, scale_factor=1.5, shift=0.2):
    scaled = data * scale_factor
    shifted = scaled + shift
    return shifted

############HYPERPARAMETROS###################
SR=10 #Hz
N_FFT=32
HOP_LENGTH= int(N_FFT/2)
N_MELS = 16
# best parameters found in Busqueda_parametros
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

normal_dataset = build_melspectrogram_from_acc(acc_magnitude=acc_magnitude, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels= N_MELS).T #queremos (samples, features)

######## datos anomalos para testeo del modelo ########
data_prueba = np.genfromtxt('./medidas/anomalo_encima_3mins.txt', delimiter='\t', skip_header=1)
clean_data_2 = data_prueba[:, 1:] #limpiamos data 
## procesado informacion para el modelo ####
acc_x = clean_data_2[:, 0]
acc_y = clean_data_2[:, 1]
acc_z = clean_data_2[:, 2]
# Combine axes to get magnitude
acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

anomalous_dataset = build_melspectrogram_from_acc(acc_magnitude=acc_magnitude, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels= N_MELS).T

normal_dataset = np.hstack((normal_dataset, np.full((normal_dataset.shape[0],1), 1))) #anadimos una columna al final que es la etiqueta de sample "normal"
anomalous_dataset = np.hstack((anomalous_dataset, np.full((anomalous_dataset.shape[0],1), 0))) #lo mismo pero cn 0 para anomalia

data_labeled = np.vstack((normal_dataset, anomalous_dataset)) #tenemos todas las muestras labeled
labels = data_labeled[:, -1]
data = data_labeled[:, 0:-1]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=21)

#NORMALIZATION OF DATASET
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)

# Separate "normal" data
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]


#plot normal and anomalus samples
plt.grid()
plt.plot(np.arange(N_MELS), normal_train_data[0])
plt.title("A Normal sample")
plt.show()
plt.savefig("./graficas/normal_sample")
plt.close()

plt.grid()
plt.plot(np.arange(N_MELS), anomalous_train_data[0])
plt.title("An Anomalous Sample")
plt.show()
plt.savefig("./graficas/anomalous_sample")
plt.close()

#Create model
class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(16, activation="tanh"),
      layers.Dense(8, activation="tanh"),
      layers.Dense(4, activation="tanh")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(8, activation="tanh"),
      layers.Dense(16, activation="tanh"),
      layers.Dense(N_MELS, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

autoencoder.compile(optimizer=optimizer, loss='mae')
#autoencoder.compile(optimizer='adam', loss='mean_squared_error')
#Train model (only with normal data)
history = autoencoder.fit(normal_train_data, normal_train_data, epochs=100, batch_size=512,validation_data=(test_data, test_data), shuffle=True)
plt.figure(figsize=(12,6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch number")
plt.ylabel("Loss")
plt.legend()
plt.savefig("./graficas/history")
plt.close()

encoded_data = autoencoder.encoder(normal_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

#PLOT RECONSTRUCTION ERROR ON A NORMAL SAMPLE
plt.plot(normal_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(N_MELS), decoded_data[0], normal_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.xlabel("Mel frequency Band")
plt.ylabel("Normalized Power")
plt.savefig("./graficas/reconstruction_error_normal")
plt.close()

#PLOT RECONSTRUCTION ERROR ON AN ANOMALOUS SAMPLE
encoded_data = autoencoder.encoder(anomalous_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(anomalous_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(N_MELS), decoded_data[0], anomalous_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.xlabel("Mel frequency Band")
plt.savefig("./graficas/reconstruction_error_anomalous")
plt.close()


##### FIND ANOMALIES ########
# PLOT RECONSTRUCTION ERROR ON NORMAL SAMPLES IN ORDER TO FIND THE THRESHOLD
reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)

plt.hist(train_loss[None,:], bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()
plt.savefig("./graficas/train loss")
plt.close()

# SET THE THRESHOLD
#threshold = np.mean(train_loss) + np.std(train_loss)
threshold = np.percentile(train_loss, 99) # situamos un threshold el cual es mayor que el 99% de los train losses
print("Threshold: ", threshold)

#PLOT RECONSTRUCTION ERROR ON ANOMALOUS SAMPLES 
reconstructions = autoencoder.predict(anomalous_test_data)
test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)

plt.hist(test_loss[None, :], bins=50)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.show()
plt.savefig("./graficas/test loss")
plt.close()

# USE THE THRESHOLD TO FIND ANOMALIES 
def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  predictions = tf.math.less(loss, threshold)
  return predictions.numpy(), loss.numpy()

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))

#PREDICT ANOMALIES
predictions, loss = predict(autoencoder, test_data, threshold)
print_stats(predictions, test_labels)


# Calculate false positives and false negatives
false_positives = np.sum((predictions == 0) & (test_labels == 1))  # Normal classified as anomaly
false_negatives = np.sum((predictions == 1) & (test_labels == 0))  # Anomaly classified as normal

# Calculate total normal and anomalous examples
total_normals = np.sum(test_labels == 1)
total_anomalies = np.sum(test_labels == 0)

# Calculate rates
false_alarm_rate = false_positives / total_normals if total_normals > 0 else 0
miss_error_rate = false_negatives / total_anomalies if total_anomalies > 0 else 0

print(f"False Alarms (False Positives): {false_positives}")
print(f"Miss Errors (False Negatives): {false_negatives}")
print(f"False Alarm Rate: {false_alarm_rate:.2%}")
print(f"Miss Error Rate: {miss_error_rate:.2%}")

# Visualize reconstruction errors
plt.hist(loss[test_labels], bins=50, alpha=0.6, label='Normal Samples')
plt.hist(loss[~test_labels], bins=50, alpha=0.6, label='Anomalous Samples')
plt.axvline(x=threshold, color='red', linestyle='--', label='Threshold')
plt.xlabel("Reconstruction Loss")
plt.ylabel("Number of Samples")
plt.legend()
plt.title("Reconstruction Loss Distribution")
plt.savefig("./graficas/reconstructionErrorDistribution")
plt.close()

#Confusion Matrix
cm = confusion_matrix(test_labels.astype(int), predictions.astype(int))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomalous", "Normal"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("./graficas/confusion_matrix.png")
plt.close()
print("Confusion Matrix:")
print(cm)

# Calculate F1-score
f1 = f1_score(test_labels, predictions)
print(f"F1-Score: {f1}")

#test_labels = np.vstack((np.ones([100,1]), np.zeros([100,1])))
#loss = np.vstack((np.random.normal(loc=1, scale=0.5, size=[100,1]), np.random.normal(loc=0, scale=0.5, size=[100,1])))
                 
precision, recall, thresholds = precision_recall_curve(test_labels, loss)
auc_pr = auc(recall, precision)
plt.plot(recall)
plt.plot(precision)
plt.savefig("./graficas/recall")
plt.close()

plt.plot(recall, precision, label=f'AUC-PR: {auc_pr:.2f}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig("./graficas/precision-recall")

#######PREDICT ON SYNTETIC DATA 
print("SYNTHETIC DATAAAAAAAA")
# Generate synthetic anomalies for evaluation
synthetic_anomalies = distort_signal(test_data, scale_factor=1.01, shift=0.1)
test_data_with_synthetic = np.vstack((test_data, synthetic_anomalies))
test_labels_with_synthetic = np.hstack((test_labels, np.zeros(len(synthetic_anomalies)).astype(bool)))
predictions, loss = predict(autoencoder, test_data_with_synthetic, threshold)
print_stats(predictions, test_labels_with_synthetic)

# Calculate false positives and false negatives
false_positives = np.sum((predictions == 0) & (test_labels_with_synthetic == 1))  # Normal classified as anomaly
false_negatives = np.sum((predictions == 1) & (test_labels_with_synthetic == 0))  # Anomaly classified as normal

# Calculate total normal and anomalous examples
total_normals = np.sum(test_labels_with_synthetic == 1)
total_anomalies = np.sum(test_labels_with_synthetic == 0)

# Calculate rates
false_alarm_rate = false_positives / total_normals if total_normals > 0 else 0
miss_error_rate = false_negatives / total_anomalies if total_anomalies > 0 else 0

print(f"False Alarms (False Positives): {false_positives}")
print(f"Miss Errors (False Negatives): {false_negatives}")
print(f"False Alarm Rate: {false_alarm_rate:.2%}")
print(f"Miss Error Rate: {miss_error_rate:.2%}")

# Visualize reconstruction errors
plt.hist(loss[test_labels_with_synthetic], bins=50, alpha=0.6, label='Normal Samples')
plt.hist(loss[~test_labels_with_synthetic], bins=50, alpha=0.6, label='Anomalous Samples')
plt.axvline(x=threshold, color='red', linestyle='--', label='Threshold')
plt.xlabel("Reconstruction Loss")
plt.ylabel("Number of Samples")
plt.legend()
plt.title("Reconstruction Loss Distribution")
plt.savefig("./graficas_sin/reconstructionErrorDistribution")
plt.close()

#Confusion Matrix
cm = confusion_matrix(test_labels_with_synthetic.astype(int), predictions.astype(int))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomalous", "Normal"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("./graficas_sin/confusion_matrix.png")
plt.close()
print("Confusion Matrix:")
print(cm)

# Calculate F1-score
f1 = f1_score(test_labels_with_synthetic, predictions)
print(f"F1-Score: {f1}")

precision, recall, thresholds = precision_recall_curve(test_labels_with_synthetic, loss)
auc_pr = auc(recall, precision)
plt.plot(recall)
plt.plot(precision)
plt.savefig("./graficas_sin/recall")
plt.close()

plt.plot(recall, precision, label=f'AUC-PR: {auc_pr:.2f}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig("./graficas_sin/precision-recall")
