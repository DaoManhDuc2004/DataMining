import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# ==========================================
# BƯỚC 1: CHỌN LỰA DỮ LIỆU (DATA SELECTION - QUÉT SÂU)
# ==========================================
DATA_PATH = r"D:\thuyloiuniversity\Mon hoc tren lop\KhaiPhaDuLieu\BTL\NV3\Data"

if not os.path.exists(DATA_PATH):
    print(f"LỖI: Thư mục {DATA_PATH} không tồn tại!")
    exit()

# Sử dụng os.walk để tìm tất cả file .wav trong các thư mục con
full_file_paths = []
file_names = []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith('.wav'):
            full_file_paths.append(os.path.join(root, file))
            file_names.append(file)

print(f"--- Bắt đầu quy trình KDD với {len(full_file_paths)} file âm thanh tìm thấy ---")

# Kiểm tra nếu không tìm thấy file nào thì dừng lại để tránh lỗi
if len(full_file_paths) == 0:
    print("KHÔNG TÌM THẤY FILE .WAV NÀO TRONG THƯ MỤC DATA VÀ CÁC THƯ MỤC CON!")
    exit()

# ==========================================
# BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU (PRE-PROCESSING)
# ==========================================
def preprocess_audio(y):
    y_filt = librosa.effects.preemphasis(y)
    y_trimmed, _ = librosa.effects.trim(y_filt)
    y_norm = librosa.util.normalize(y_trimmed)
    return y_norm

# ==========================================
# BƯỚC 3: CHUYỂN DẠNG DỮ LIỆU (DATA TRANSFORMATION)
# ==========================================
def extract_robust_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, duration=5)
        y = preprocess_audio(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        combined = np.hstack([
            np.mean(mfcc.T, axis=0), 
            np.mean(delta_mfcc.T, axis=0),
            np.mean(delta2_mfcc.T, axis=0)
        ])
        return combined
    except Exception as e:
        return None

print("Đang trích xuất đặc trưng (có thể mất vài phút)...")
features_list = []
valid_files = []

for f_path, f_name in zip(full_file_paths, file_names):
    feat = extract_robust_features(f_path)
    if feat is not None:
        features_list.append(feat)
        valid_files.append(f_name)

X = np.array(features_list)
X_scaled = StandardScaler().fit_transform(X)

# ==========================================
# BƯỚC 4: KHAI PHÁ DỮ LIỆU (DATA MINING - K-MEANS)
# ==========================================
n_clusters = 46 # Bạn có 46 người nói
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# ==========================================
# BƯỚC 5: TRÌNH DIỄN VÀ ĐÁNH GIÁ
# ==========================================
sil_score = silhouette_score(X_scaled, labels)
db_score = davies_bouldin_score(X_scaled, labels)
print(f"\n--- KẾT QUẢ: Silhouette: {sil_score:.4f} | DB Index: {db_score:.4f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(12, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='gist_ncar', s=80, legend=False)
plt.title(f"K-Means Clustering - 46 Speakers (Silhouette: {sil_score:.2f})")
plt.show()

pd.DataFrame({'File': valid_files, 'Cluster': labels}).to_csv("KMeans_Results.csv", index=False)