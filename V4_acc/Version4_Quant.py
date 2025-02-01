import numpy as np
import tflite_runtime.interpreter as tflite
import tensorflow as tf
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from procesamiento_aceleracion import build_melspectrogram_from_acc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

###########HYPERPARAMETROS###################
SR=10 #Hz
N_FFT= 32
HOP_LENGTH= int(N_FFT/2)
N_MELS= 16
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

# Load TFLite model with Coral support
model_path = "./model_quant.tflite"
interpreter = tflite.Interpreter(model_path=model_path,
  experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()

print("edge tpu")
# Get input/output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Function to run inference on a single sample
def run_autoencoder(model_interpreter, sample):
    # Reshape input to match model's expected dimensions
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    input_data = tf.reshape(tensor=sample, shape=input_shape)
    input_data = tf.cast(input_data,input_dtype)

    # Set input tensor
    model_interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    model_interpreter.invoke()
    
    # Get reconstructed output
    output_data = model_interpreter.get_tensor(output_details[0]['index'])
    if 'quantization' not in output_details:
        return output_data
    scale, zero_point = output_details['quantization']
    if scale == 0:
        return output_data - zero_point
    return scale * (output_data - zero_point)

# Calculate reconstruction loss for all samples
def calculate_reconstruction_loss(data, interpreter):
    losses = []
    for sample in data:
        reconstructed= run_autoencoder(interpreter, sample)
        #sample = tf.reshape(tensor=sample, shape=input_details[0]['shape']) #aplicamos el mismo casting que aplicara el modelo para calcular el output
        #sample = tf.cast(sample, input_details[0]['dtype'])
        loss = np.mean(np.abs(sample - reconstructed))  # MAE loss 
        losses.append(loss)
    return np.array(losses)

#Plot Reconstruction error on a normal sample 
decoded_normal = run_autoencoder(interpreter,normal_test_data[0])
fig, ax1 = plt.subplots()
ax1.plot(normal_test_data[0], 'b')
ax2 = ax1.twinx()
ax2.plot(decoded_normal[0], 'r')
#plt.plot(decoded_normal[0], 'r')
#ax2.fill_between(np.arange(N_MELS), decoded_normal[0], normal_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.savefig("./graficas_q/reconstruction_error_normal")
plt.close()

#Plot Reconstruction error on a normal sample 
decoded_anomalous = run_autoencoder(interpreter,anomalous_test_data[0])
fig, ax1 = plt.subplots()
ax1.plot(anomalous_test_data[0], 'b')
ax2 = ax1.twinx()
ax2.plot(decoded_anomalous[0], 'r')
#plt.plot(decoded_normal[0], 'r')
#ax2.fill_between(np.arange(N_MELS), decoded_anomalous[0], anomalous_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.savefig("./graficas_q/reconstruction_error_anomalous")
plt.close()

# Evaluate on normal and anomalous data
normal_losses = calculate_reconstruction_loss(normal_train_data, interpreter)
anomalous_losses = calculate_reconstruction_loss(anomalous_test_data, interpreter)

# Plot reconstruction losses on normal samples 
plt.hist(normal_losses, bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()
plt.savefig("./graficas_q/train loss")
plt.close()

# SET THE THRESHOLD
#threshold = np.mean(normal_losses) + np.std(normal_losses)
threshold = np.percentile(normal_losses, 99)
print("Threshold: ", threshold)

plt.hist(anomalous_losses, bins=50)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.show()
plt.savefig("./graficas_q/test loss")
plt.close()

# DETECT ANOMALIES ON THE TEST SET
def predict(data, interpreter, threshold):
    loss = calculate_reconstruction_loss(data,interpreter)
    predictions = []
    for sample_loss in loss:
        if(sample_loss>threshold):
            predictions.append(False)
        else:
            predictions.append(True)
    return np.array(predictions), loss

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))

# PREDICT ANOMALIES
predictions, loss = predict(test_data, interpreter, threshold)
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