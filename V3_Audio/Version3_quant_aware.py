import os
import numpy as np
import tensorflow as tf
import tempfile

from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve, auc
#from tensorflow import keras

from tensorflow_model_optimization.python.core.keras.compat import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from procesamiento_audio2 import build_melspectrogram
import tensorflow_model_optimization as tfmot
from keras.models import Model

#Download the raw audio data

sample_rate, normal_audio = wavfile.read('/home/alvaro/TFG/V3_Audio/audio/nueva_base_para_ML.wav') #este audio dura unos 600 s de audio "normal"
sr, abnormal_audio = wavfile.read('/home/alvaro/TFG/V3_Audio/audio/anomalias_nomuyanomalo_60s.wav') #este son 10 segundos de audio "anormal"

#PARAMETERS
FFT_size = 4096
overlapping = 0.5
hop_length = int(FFT_size*overlapping)
n_mels = 50

dct_filter_num= n_mels
mel_filter_num = dct_filter_num*2

#CREATION OF DATASET
normal_dataset = build_melspectrogram(audio_route='/home/alvaro/TFG/V3_Audio/audio/nueva_base_para_ML.wav', n_fft=FFT_size, hop_length=hop_length, n_mels=n_mels).T
anomalous_dataset = build_melspectrogram(audio_route='/home/alvaro/TFG/V3_Audio/audio/anomalias_nomuyanomalo_60s.wav', n_fft=FFT_size, hop_length=hop_length, n_mels=n_mels).T

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
plt.plot(np.arange(dct_filter_num), normal_train_data[0])
plt.title("A Normal sample")
plt.show()
plt.savefig("./graficas_q/normal_sample")
plt.close()

plt.grid()
plt.plot(np.arange(dct_filter_num), anomalous_train_data[0])
plt.title("An Anomalous Sample")
plt.show()
plt.savefig("./graficas_q/anomalous_sample")
plt.close()


def base_model():
  model = keras.Sequential([
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(dct_filter_num, activation="sigmoid")
  ])
  return model
base_autoencoder = base_model()

base_autoencoder.compile(optimizer='adam', loss='mae')
#autoencoder.compile(optimizer='adam', loss='mean_squared_error')
#Train model (only with normal data)
base_autoencoder.fit(normal_train_data, normal_train_data, epochs=20, batch_size=512,validation_data=(test_data, test_data), shuffle=True)

#****QUANTIZATION*****
quant_aware_autoencoder = tfmot.quantization.keras.quantize_model(base_autoencoder)
quant_aware_autoencoder.compile(optimizer='adam', loss='mae')
quant_aware_autoencoder.summary()

quant_aware_autoencoder.fit(normal_train_data, normal_train_data, epochs=20, batch_size=512,validation_data=(test_data, test_data), shuffle=True)


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
#threshold = np.mean(train_loss) + np.std(train_loss)
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

#test_labels = np.vstack((np.ones([100,1]), np.zeros([100,1])))
#loss = np.vstack((np.random.normal(loc=1, scale=0.5, size=[100,1]), np.random.normal(loc=0, scale=0.5, size=[100,1])))
                 
precision, recall, thresholds = precision_recall_curve(test_labels, loss)
auc_pr = auc(recall, precision)
plt.plot(recall)
plt.plot(precision)
plt.savefig("./graficas_q/recall")
plt.close()

plt.plot(recall, precision, label=f'AUC-PR: {auc_pr:.2f}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig("./graficas_q/precision-recall")
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