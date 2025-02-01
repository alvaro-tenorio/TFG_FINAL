import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers, losses
from keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from procesamiento_aceleracion import build_melspectrogram_from_acc

# Auxiliar functions
def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

############HYPERPARAMETROS###################
SR=10 #Hz
N_FFT=8
HOP_LENGTH= int(N_FFT/2)
N_MELS =16 

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
acc_magnitude -= np.mean(acc_magnitude)
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
acc_magnitude -= np.mean(acc_magnitude)
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


class AnomalyDetector(Model):
    def __init__(self, encoder_layers, decoder_layers, activation, n_mels):
        super(AnomalyDetector, self).__init__()
        
        # Build encoder
        self.encoder = tf.keras.Sequential([
            layers.Dense(units, activation=activation) for units in encoder_layers
        ])
        
        # Build decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(units, activation=activation) for units in decoder_layers[:-1]
        ] + [layers.Dense(decoder_layers[-1], activation="sigmoid")])  # Output layer
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

# Define grid search ranges
encoder_configs = [[64, 32, 16], [32, 16, 8], [16, 8, 4]]
decoder_configs = [[8, 16, N_MELS], [16, 32, N_MELS], [32, 64, N_MELS]]
activations = ["relu", "tanh"]
learning_rates = [0.001, 0.0005, 0.0001]
batch_sizes = [128, 256, 512]

# Store results
results = []

for encoder_layers in encoder_configs:
    for decoder_layers in decoder_configs:
        for activation in activations:
            for learning_rate in learning_rates:
                for batch_size in batch_sizes:
                    print(f"Testing: Encoder={encoder_layers}, Decoder={decoder_layers}, Activation={activation}, LR={learning_rate}, Batch={batch_size}")

                    # Create model
                    autoencoder = AnomalyDetector(encoder_layers, decoder_layers, activation, N_MELS)
                    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                    autoencoder.compile(optimizer=optimizer, loss='mae')

                    # Train the model
                    history = autoencoder.fit(
                        normal_train_data, normal_train_data,
                        epochs=10,  # Use a smaller number of epochs for testing
                        batch_size=batch_size,
                        validation_data=(test_data, test_data),
                        verbose=0
                    )

                    # Evaluate performance
                    reconstructions = autoencoder.predict(normal_train_data)
                    train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)
                    threshold = np.percentile(train_loss, 99)
                    reconstructions = autoencoder.predict(anomalous_test_data)
                    test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)
                    
                    
                    preds = predict(autoencoder, test_data, threshold)
                    accuracy = accuracy_score(test_labels, preds)
                    precision = precision_score(test_labels, preds)
                    recall = recall_score(test_labels, preds)
                    f1 = f1_score(test_labels, preds)

                    # Store results
                    results.append({
                        "encoder_layers": encoder_layers,
                        "decoder_layers": decoder_layers,
                        "activation": activation,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1-score": f1
                    })
                    print(f"Results: Accuracy={accuracy}, Precision={precision}, Recall={recall}, f1-score={f1}")

# Find the best result
best_result = max(results, key=lambda x: x["f1-score"])
print(f"Best Configuration: {best_result}")

#Best Configuration: {'encoder_layers': [32, 16, 8], 'decoder_layers': [16, 32, 20], 'activation': 'tanh', 'learning_rate': 0.0001, 'batch_size': 128, 'accuracy': 0.9917355371900827, 'precision': 0.9916666666666667, 'recall': 1.0, 'f1-score': 0.99581589958159}
#Best Configuration: {'encoder_layers': [64, 32, 16], 'decoder_layers': [8, 16, 20], 'activation': 'relu', 'learning_rate': 0.0005, 'batch_size': 128, 'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0}
#{'encoder_layers': [64, 32, 16], 'decoder_layers': [8, 16, 16], 'activation': 'relu', 'learning_rate': 0.0005, 'batch_size': 128, 'accuracy': 0.996542783059637, 'precision': 1.0, 'recall': 0.9961501443695862, 'f1-score': 0.9980713596914176}
