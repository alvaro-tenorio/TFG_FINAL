import os
import numpy as np
import tensorflow as tf

from scipy.io import wavfile
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers, losses
#from tensorflow.python.keras.datasets import fashion_mnist
from keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from procesamiento_audio2 import build_melspectrogram

# Auxiliar functions
def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

#Preproccesing parameters
fft_size = 4096
n_mels= 50

hop_length = int(fft_size * 0.5)  # 50% overlap
normal_dataset = build_melspectrogram(
    audio_route='/home/alvaro/TFG/V3_Audio/audio/nueva_base_para_ML.wav',
    n_fft=fft_size,
    hop_length=hop_length,
    n_mels=n_mels
        ).T
anomalous_dataset = build_melspectrogram(
    audio_route='/home/alvaro/TFG/V3_Audio/audio/anomalias_nomuyanomalo_60s.wav',
    n_fft=fft_size,
    hop_length=hop_length,
    n_mels=n_mels
    ).T

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
encoder_configs = [[128, 64, 32], [256, 128, 64], [64, 32]]
decoder_configs = [[64, 128, n_mels], [32, 64, n_mels]]
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
                    autoencoder = AnomalyDetector(encoder_layers, decoder_layers, activation, n_mels)
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
                    #threshold = np.mean(train_loss) + np.std(train_loss)
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
                    print(f"Results: Accuracy={accuracy}, Precision={precision}, Recall={recall}, f1-score:{f1}")

# Find the best result
best_result = max(results, key=lambda x: x["f1-score"])
print(f"Best Configuration: {best_result}")
#{'encoder_layers': [128, 64, 32], 'decoder_layers': [64, 128, 50], 'activation': 'relu', 'learning_rate': 0.0001, 'batch_size': 128}

