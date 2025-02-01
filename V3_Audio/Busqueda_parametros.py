import os
import numpy as np
import tensorflow as tf

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


# Define parameter ranges
fft_sizes = [256, 512, 1024, 2048, 4096]
n_mels_list = [20, 32, 40, 50, 64, 70, 80, 90, 128]

# Store results
results = []

for fft_size in fft_sizes:
    for n_mels in n_mels_list:
        print(f"Testing FFT_size={fft_size}, n_mels={n_mels}")

        # Update preprocessing parameters
        hop_length = int(fft_size * 0.5)  # 50% overlap
        normal_dataset = build_melspectrogram(
            audio_route='./audio/nueva_base_para_ML.wav',
            n_fft=fft_size,
            hop_length=hop_length,
            n_mels=n_mels
        ).T
        anomalous_dataset = build_melspectrogram(
            audio_route='./audio/anomalias_nomuyanomalo_60s.wav',
            n_fft=fft_size,
            hop_length=hop_length,
            n_mels=n_mels
        ).T

        # Add labels
        normal_dataset = np.hstack((normal_dataset, np.full((normal_dataset.shape[0], 1), 1)))
        anomalous_dataset = np.hstack((anomalous_dataset, np.full((anomalous_dataset.shape[0], 1), 0)))

        # Combine datasets and split
        data_labeled = np.vstack((normal_dataset, anomalous_dataset))
        labels = data_labeled[:, -1]
        data = data_labeled[:, 0:-1]
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=0.2, random_state=21
        )

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

        anomalous_train_data = train_data[~train_labels]
        anomalous_test_data = test_data[~test_labels]

        #Create model
        class AnomalyDetector(Model):
            def __init__(self):
                super(AnomalyDetector, self).__init__()
                self.encoder = tf.keras.Sequential([
                    layers.Dense(128, activation="relu"),
                    layers.Dense(64, activation="relu"),
                    layers.Dense(32, activation="relu")])

                self.decoder = tf.keras.Sequential([
                    layers.Dense(64, activation="relu"),
                    layers.Dense(128, activation="relu"),
                    layers.Dense(n_mels, activation="sigmoid")])

            def call(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded


        # Train autoencoder
        autoencoder = AnomalyDetector()
        autoencoder.compile(optimizer='adam', loss='mae')
        history = autoencoder.fit(normal_train_data, normal_train_data, epochs=20, batch_size=512,validation_data=(test_data, test_data), shuffle=True)

        # Find threshold
        reconstructions = autoencoder.predict(normal_train_data)
        train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)
        #threshold = np.mean(train_loss) + np.std(train_loss)
        threshold = np.percentile(train_loss, 99)

        # Evaluate model
        reconstructions = autoencoder.predict(anomalous_test_data)
        test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)
        preds = predict(autoencoder, test_data, threshold)
        accuracy = accuracy_score(test_labels, preds)
        precision = precision_score(test_labels, preds)
        recall = recall_score(test_labels, preds)
        f1 = f1_score(test_labels, preds)

        # Save results
        results.append({
            "FFT_size": fft_size,
            "n_mels": n_mels,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "F1-score":f1
        })
        print(f"FFT_size={fft_size}, n_mels={n_mels} -> Accuracy={accuracy}, Precision={precision}, Recall={recall}")

# Find best parameters
best_result = max(results, key=lambda x: x["F1-score"])
print(f"Best parameters: {best_result}")

#Best parameters:{'FFT_size': 4096, 'n_mels': 50}
