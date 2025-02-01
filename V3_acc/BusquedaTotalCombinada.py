import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers, losses
from keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score
from procesamiento_aceleracion import build_melspectrogram_from_acc

# Auxiliary functions
def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold)

# Data extraction
data = np.genfromtxt('./medidas/ventilador_normal_larguisimo.txt', delimiter='\t', skip_header=1)
clean_data = data[:, 1:]  # Remove NaNs from the first column
acc_x = clean_data[:, 0]
acc_y = clean_data[:, 1]
acc_z = clean_data[:, 2]
acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
acc_magnitude -= np.mean(acc_magnitude)

data_prueba = np.genfromtxt('./medidas/anomaloMucho_largo.txt', delimiter='\t', skip_header=1)
clean_data_2 = data_prueba[:, 1:]
acc_x_anomalo = clean_data_2[:, 0]
acc_y_anomalo = clean_data_2[:, 1]
acc_z_anomalo = clean_data_2[:, 2]
acc_magnitude_anomalo = np.sqrt(acc_x_anomalo**2 + acc_y_anomalo**2 + acc_z_anomalo**2)
acc_magnitude_anomalo -= np.mean(acc_magnitude_anomalo)

# Parameter ranges
fft_sizes = [8, 16, 32, 64, 128]
n_mels_list = [2, 4, 8, 10, 16, 20, 25, 32, 40]
encoder_configs = [[128, 64, 32], [256, 128, 64], [64, 32]]
decoder_configs = [[64, 128, 1], [32, 64, 1]]
activations = ["relu", "tanh"]
learning_rates = [0.001, 0.0005, 0.0001]
batch_sizes = [128, 256, 512]

# Store results
results = []

for fft_size in fft_sizes:
    for n_mels in n_mels_list:
        print(f"Testing Preprocessing: FFT_size={fft_size}, n_mels={n_mels}")

        hop_length = int(fft_size * 0.5)
        normal_dataset = build_melspectrogram_from_acc(
            acc_magnitude=acc_magnitude, sr=10, n_fft=fft_size, hop_length=hop_length, n_mels=n_mels
        ).T
        anomalous_dataset = build_melspectrogram_from_acc(
            acc_magnitude=acc_magnitude_anomalo, sr=10, n_fft=fft_size, hop_length=hop_length, n_mels=n_mels
        ).T

        # Label datasets
        normal_dataset = np.hstack((normal_dataset, np.full((normal_dataset.shape[0], 1), 1)))
        anomalous_dataset = np.hstack((anomalous_dataset, np.full((anomalous_dataset.shape[0], 1), 0)))
        data_labeled = np.vstack((normal_dataset, anomalous_dataset))
        labels = data_labeled[:, -1]
        data = data_labeled[:, 0:-1]

        # Split data
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=21)

        # Normalize data
        min_val = tf.reduce_min(train_data)
        max_val = tf.reduce_max(train_data)
        train_data = (train_data - min_val) / (max_val - min_val)
        test_data = (test_data - min_val) / (max_val - min_val)

        # Separate "normal" data
        train_labels = train_labels.astype(bool)
        test_labels = test_labels.astype(bool)
        normal_train_data = train_data[train_labels]
        normal_test_data = test_data[test_labels]
        anomalous_test_data = test_data[~test_labels]

        for encoder_layers in encoder_configs:
            for decoder_layers in decoder_configs:
                for activation in activations:
                    for learning_rate in learning_rates:
                        for batch_size in batch_sizes:
                            decoder_layers[-1]=n_mels
                            print(f"Testing Model: Encoder={encoder_layers}, Decoder={decoder_layers}, Activation={activation}, LR={learning_rate}, Batch={batch_size}")
                            
                            # Create model
                            class AnomalyDetector(Model):
                                def __init__(self):
                                    super(AnomalyDetector, self).__init__()
                                    self.encoder = tf.keras.Sequential([
                                        layers.Dense(units, activation=activation) for units in encoder_layers
                                    ])
                                    self.decoder = tf.keras.Sequential([
                                        layers.Dense(units, activation=activation) for units in decoder_layers[:-1]
                                    ] + [layers.Dense(decoder_layers[-1], activation="sigmoid")])

                                def call(self, x):
                                    encoded = self.encoder(x)
                                    decoded = self.decoder(encoded)
                                    return decoded

                            autoencoder = AnomalyDetector()
                            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                            autoencoder.compile(optimizer=optimizer, loss='mae')

                            # Train model
                            autoencoder.fit(normal_train_data, normal_train_data, epochs=10, batch_size=batch_size, validation_data=(test_data, test_data), verbose=0)

                            # Evaluate model
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
                                "FFT_size": fft_size,
                                "n_mels": n_mels,
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

# Find best result
print("Final del Loop")
best_result_accuracy = max(results, key=lambda x: x["accuracy"])
best_result_recall = max(results, key=lambda x: x["recall"])
best_result_f1 = max(results, key=lambda x: x["f1-score"])
print(f"Best Accuracy Configuration: {best_result_accuracy}")
print(f"Best Recall Configuration: {best_result_recall}")
print(f"Best Recall Configuration: {best_result_recall}")