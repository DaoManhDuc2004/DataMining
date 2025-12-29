import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

# --- BƯỚC 1: CHỌN LỰA DỮ LIỆU (QUÉT SÂU) ---
DATA_PATH = r"D:\thuyloiuniversity\Mon hoc tren lop\KhaiPhaDuLieu\BTL\NV3\Data"

full_file_paths = []
file_names = []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith('.wav'):
            full_file_paths.append(os.path.join(root, file))
            file_names.append(file)

if len(full_file_paths) == 0:
    print("KHÔNG TÌM THẤY FILE .WAV!")
    exit()

# --- BƯỚC 2 & 3: TIỀN XỬ LÝ & CHUYỂN DẠNG ---
def get_advanced_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, duration=5)
        y = librosa.effects.preemphasis(y)
        y, _ = librosa.effects.trim(y)
        y = librosa.util.normalize(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        return np.hstack([np.mean(mfcc.T, axis=0), np.mean(delta.T, axis=0)])
    except:
        return None

print("Đang trích xuất đặc trưng cho Phân cụm phân cấp...")
features = []
valid_names = []
for f_path, f_name in zip(full_file_paths, file_names):
    feat = get_advanced_features(f_path)
    if feat is not None:
        features.append(feat)
        valid_names.append(f_name)

X = np.array(features)
X_scaled = StandardScaler().fit_transform(X)

# --- BƯỚC 4: KHAI PHÁ DỮ LIỆU (HIERARCHICAL) ---
Z = linkage(X_scaled, method='ward')
n_clusters = 46 
hc_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels = hc_model.fit_predict(X_scaled)

# --- BƯỚC 5: TRÌNH DIỄN VÀ ĐÁNH GIÁ ---
plt.figure(figsize=(20, 10))
dendrogram(Z, labels=valid_names, leaf_rotation=90, leaf_font_size=6)
plt.title("Sơ đồ cây phân cấp (Dendrogram) - Voice Clustering")
plt.tight_layout()
plt.show()

sil = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {sil:.4f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(12, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='nipy_spectral', s=60, edgecolors='k')
plt.title(f"Kết quả Hierarchical Clustering (K=46)")
plt.show()