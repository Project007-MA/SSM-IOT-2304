

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.signal import butter, lfilter
from pywt import wavedec
from sklearn.cross_decomposition import CCA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cdist
import mne  # For EEG .edf file handling

# 1. EEG Preprocessing (Bandpass Filter)
def butter_bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=160.0, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

# Feature Extraction using WT + CCA (Simplified)
def extract_features(data):
    wt_features = [wavedec(signal, 'db4', level=4)[0] for signal in data]
    cca = CCA(n_components=1)
    X = np.array(wt_features)
    Y = np.roll(X, shift=1, axis=0)
    cca.fit(X, Y)
    X_c, _ = cca.transform(X, Y)
    return X_c

# 3. CNN Model for Authentication
def create_auth_model(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# EEG Dataset Loading from .edf file
raw = mne.io.read_raw_edf("your_file.edf", preload=True)
raw.filter(0.5, 50.0)
eeg_data = raw.copy().pick_types(eeg=True).get_data().T
eeg_data = eeg_data[:1000]
labels = np.random.randint(0, 2, eeg_data.shape[0])

filtered_data = np.array([butter_bandpass_filter(sig, 0.5, 50) for sig in eeg_data])
features = extract_features(filtered_data)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = create_auth_model(X_train.shape[1:])
model.fit(X_train, y_train, epochs=10, batch_size=32)
predictions = (model.predict(X_test) > 0.5).astype("int32")

# Performance Metrics
acc = accuracy_score(y_test, predictions)
prec = precision_score(y_test, predictions)
rec = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
print("\nEEG Authentication Metrics:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
cmd = ConfusionMatrixDisplay(cm, display_labels=["Not Authenticated", "Authenticated"])
cmd.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - EEG Authentication")
plt.show()

# Phase II: Sensor Trust Evaluation & Security Enhancements

def compute_direct_trust(sent, ack):
    return ack / sent if sent > 0 else 0

def compute_total_trust(direct_trust, indirect_trust, w=0.7):
    return w * direct_trust + (1 - w) * indirect_trust

sensor_data = np.random.randint(1, 10, (10, 2))
trust_scores = [compute_direct_trust(s, a) for s, a in sensor_data]
total_trust = [compute_total_trust(t, np.mean(trust_scores)) for t in trust_scores]
trustworthy = np.array(total_trust) > 0.5
print("Trustworthy Sensors:", np.where(trustworthy)[0])

# DI-SC Clustering

def dunn_index(points, labels):
    clusters = [points[labels == i] for i in np.unique(labels)]
    inter_cluster = np.min([cdist(c1, c2).min() for i, c1 in enumerate(clusters) for c2 in clusters[i+1:]])
    intra_cluster = np.max([cdist(c, c).max() for c in clusters])
    return inter_cluster / intra_cluster if intra_cluster != 0 else 0

sensor_coords = np.random.rand(sum(trustworthy), 2)
best_score, best_k, best_labels = 0, 2, None
for k in range(2, 5):
    sc = SpectralClustering(n_clusters=k, affinity='nearest_neighbors')
    labels = sc.fit_predict(sensor_coords)
    score = dunn_index(sensor_coords, labels)
    if score > best_score:
        best_score, best_k, best_labels = score, k, labels
print("Best Clusters (DI-SC):", best_k)

# R-SAO

def r_sao(clusters):
    energy = np.random.rand(len(clusters))
    return np.argmax(energy)

cluster_heads = [r_sao(sensor_coords[best_labels == i]) for i in range(best_k)]
print("Cluster Heads:", cluster_heads)

# O-MENN Anomaly Detection

def create_omen_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Simulated Network Data for Anomaly Detection
net_data = np.random.rand(1000, 20)
net_labels = np.random.randint(0, 2, 1000)
Xn_train, Xn_test, yn_train, yn_test = train_test_split(net_data, net_labels, test_size=0.2, random_state=42)
omen_model = create_omen_model(Xn_train.shape[1])
omen_model.fit(Xn_train, yn_train, epochs=10, batch_size=32)

net_predictions = (omen_model.predict(Xn_test) > 0.5).astype("int32")
print("Anomaly Detection Accuracy:", accuracy_score(yn_test, net_predictions))