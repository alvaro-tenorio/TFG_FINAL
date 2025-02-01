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
from procesamiento_audio2 import build_melspectrogram  
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# PARAMETERS
FFT_SIZE = 4096
HOP_LENGTH = int(FFT_SIZE * 0.5)
N_MELS = 50

# Load audio files
sample_rate, normal_audio = wavfile.read('/home/alvaro/TFG/V3_Audio3/audio/nueva_base_para_ML.wav') #este audio dura unos 600 s de audio "normal"
sr, abnormal_audio = wavfile.read('/home/alvaro/TFG/V3_Audio3/audio/anomalias_nomuyanomalo_30s.wav') #este son 10 segundos de audio "anormal"

# Ensure audio sample rates match
if sample_rate != sr:
    raise ValueError("Sample rates of the normal and anomalous audio files must match!")

# Preprocess audio files
normal_dataset = build_melspectrogram(audio_route='/home/alvaro/TFG/V3_Audio3/audio/nueva_base_para_ML.wav', n_fft=FFT_SIZE, hop_length=HOP_LENGTH, n_mels=N_MELS).T
anomalous_dataset = build_melspectrogram(audio_route='/home/alvaro/TFG/V3_Audio3/audio/anomalias_nomuyanomalo_30s.wav', n_fft=FFT_SIZE, hop_length=HOP_LENGTH, n_mels=N_MELS).T

# Normalize data (same normalization as during training)
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
model_path = "model.tflite"
interpreter = tflite.Interpreter(model_path,
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
    input_data = tf.reshape(sample, input_shape)

    
    # Set input tensor
    model_interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    model_interpreter.invoke()
    
    # Get reconstructed output
    output_data = model_interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Calculate reconstruction loss for all samples
def calculate_reconstruction_loss(data, interpreter):
    losses = []
    for sample in data:
        reconstructed = run_autoencoder(interpreter, sample)
        loss = np.mean(np.abs(sample - reconstructed))  # MAE loss
        losses.append(loss)
    return np.array(losses)

#Plot Reconstruction error on a normal sample 
decoded_normal = run_autoencoder(interpreter,normal_test_data[0])
plt.plot(normal_test_data[0], 'b')
plt.plot(decoded_normal[0], 'r')
plt.fill_between(np.arange(N_MELS), decoded_normal[0], normal_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.savefig("./graficas/reconstruction_error_normal")
plt.close()

#Plot Reconstruction error on a normal sample 
decoded_anomalous = run_autoencoder(interpreter,anomalous_test_data[0])
plt.plot(anomalous_test_data[0], 'b')
plt.plot(decoded_anomalous[0], 'r')
plt.fill_between(np.arange(N_MELS), decoded_anomalous[0], anomalous_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.savefig("./graficas/reconstruction_error_anomalous")
plt.close()


# Evaluate on normal and anomalous data
normal_losses = calculate_reconstruction_loss(normal_train_data, interpreter)
anomalous_losses = calculate_reconstruction_loss(anomalous_test_data, interpreter)

# Plot reconstruction losses on normal samples 
plt.hist(normal_losses, bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()
plt.savefig("./graficas/train loss")
plt.close()


# SET THE THRESHOLD
#threshold = np.mean(normal_losses) + np.std(normal_losses)
threshold = 0.02
print("Threshold: ", threshold)

plt.hist(anomalous_losses, bins=50)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.show()
plt.savefig("./graficas/test loss")
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

#Confusion Matrix
cm = confusion_matrix(test_labels.astype(int), predictions.astype(int))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anomalous", "Normal"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("./graficas/confusion_matrix.png")
plt.close()
print("Confusion Matrix:")
print(cm)