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
import tensorflow_model_optimization as tfmot
from keras.models import Model

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
# Remove mean to isolate dynamic components (remove gravity)
#acc_magnitude -= np.mean(acc_magnitude)
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
# Remove mean to isolate dynamic components (remove gravity)
#acc_magnitude -= np.mean(acc_magnitude)
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

def base_model():
  model = keras.Sequential([
    keras.layers.Dense(16, activation="tanh"),
    keras.layers.Dense(8, activation="tanh"),
    keras.layers.Dense(4, activation="tanh"),
    keras.layers.Dense(8, activation="tanh"),
    keras.layers.Dense(16, activation="tanh"),
    keras.layers.Dense(N_MELS, activation="sigmoid")
  ])
  return model
base_autoencoder = base_model()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
base_autoencoder.compile(optimizer=optimizer, loss='mae')
#Train model (only with normal data)
base_autoencoder.fit(normal_train_data, normal_train_data, epochs=100, batch_size=512,validation_data=(test_data, test_data), shuffle=True)

#****QUANTIZATION*****
quant_aware_autoencoder = tfmot.quantization.keras.quantize_model(base_autoencoder)
quant_aware_autoencoder.compile(optimizer='adam', loss='mae')
quant_aware_autoencoder.summary()

quant_aware_autoencoder.fit(normal_train_data, normal_train_data, epochs=50, batch_size=512,validation_data=(test_data, test_data), shuffle=True)

##### FIND ANOMALIES ########
# PLOT RECONSTRUCTION ERROR ON NORMAL SAMPLES IN ORDER TO FIND THE THRESHOLD
reconstructions = quant_aware_autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)

plt.hist(train_loss[None,:], bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()
plt.savefig("./graficas_q/train loss")
plt.close()

# SET THE THRESHOLD
threshold = np.percentile(train_loss, 99)
print("Threshold: ", threshold)

#PLOT RECONSTRUCTION ERROR ON ANOMALOUS SAMPLES 
reconstructions = quant_aware_autoencoder.predict(anomalous_test_data)
test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)

plt.hist(test_loss[None, :], bins=50)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.show()
plt.savefig("./graficas_q/test loss")
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
predictions, loss = predict(quant_aware_autoencoder, test_data, threshold)
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
plt.savefig("./graficas_q/reconstructionErrorDistribution")
plt.close()

#Confusion Matrix
cm = confusion_matrix(test_labels.astype(int), predictions.astype(int))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomalous", "Normal"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("./graficas_q/confusion_matrix.png")
plt.close()
print("Confusion Matrix:")
print(cm)

# Calculate F1-score
f1 = f1_score(test_labels, predictions)
print(f"F1-Score: {f1}")

# SAVE THE MODEL TO THEN CONVERT IT TO TFLITE

#CONVERSION TO TFLITE as a quantized aware model
converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_autoencoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()
#After this, you have an actually quantized model with int8 weights and uint8 activations.
# now save it to a.tflite model so we can use it un the edge tpu
with open('model_quant.tflite', 'wb') as f:
  f.write(quantized_tflite_model)
  
print("Quantized Autoencoder model converted to tflite")