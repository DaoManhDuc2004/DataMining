from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from typing import Optional

# =========================
# CONFIG
# =========================
DATA_PATH = Path(r"D:\thuyloiuniversity\Mon hoc tren lop\KhaiPhaDuLieu\BTL\NV3\Data")

TARGET_SR = 16000
DURATION_SEC = 5.0
N_MFCC = 13
RANDOM_STATE = 42

TARGET_K = 46

# Thử rộng hơn để có cơ hội ra nhiều cụm
PCA_LIST = [10, 15, 20, 25, 30]
MIN_SAMPLES_LIST = [2, 3, 4]                 # mỗi speaker 10 file -> 2-4 hợp lý
EPS_LIST = np.arange(0.04, 0.26, 0.01)       # cosine eps thường nhỏ (0.04-0.25)

# Ràng buộc để tìm “gần 46 cụm”
CLUSTER_RANGE = (35, 60)
MAX_NOISE = 0.40

# =========================
# LOAD WAV
# =========================
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Không thấy DATA_PATH: {DATA_PATH}")

wav_paths = sorted([p for p in DATA_PATH.rglob("*.wav")])
print(f"--- Tìm thấy {len(wav_paths)} file wav ---")
if not wav_paths:
    raise RuntimeError("Không có file wav nào!")

def preprocess_audio(y: np.ndarray) -> np.ndarray:
    y = librosa.effects.preemphasis(y)
    y, _ = librosa.effects.trim(y, top_db=30)
    y = librosa.util.normalize(y)
    return y

def stats(mat: np.ndarray) -> np.ndarray:
    return np.hstack([np.mean(mat, axis=1), np.std(mat, axis=1)])

def extract_features(file_path: Path) -> Optional[np.ndarray]:
    try:
        y, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
        y = preprocess_audio(y)
        target_len = int(TARGET_SR * DURATION_SEC)
        y = librosa.util.fix_length(y, size=target_len)

        mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=N_MFCC)
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)

        feat = np.hstack([stats(mfcc), stats(d1), stats(d2)])  # 78 dims
        return feat.astype(np.float32)
    except Exception as e:
        print(f"[SKIP] {file_path}: {e}")
        return None

print("Đang trích xuất đặc trưng...")
X_list, file_rel, true_speaker = [], [], []
for p in wav_paths:
    feat = extract_features(p)
    if feat is None:
        continue
    X_list.append(feat)
    file_rel.append(str(p.relative_to(DATA_PATH)))
    true_speaker.append(p.parent.name)

X = np.vstack(X_list)
true_ids = pd.factorize(np.array(true_speaker))[0]
print(f"--- Mẫu hợp lệ: {X.shape[0]} | Feature dim: {X.shape[1]} ---")

# Scale trước PCA
X_scaled = StandardScaler().fit_transform(X)

def get_stats(labels: np.ndarray):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_rate = float(np.mean(labels == -1))
    ari = adjusted_rand_score(true_ids, labels)
    nmi = normalized_mutual_info_score(true_ids, labels)
    return n_clusters, noise_rate, ari, nmi

best_quality = None      # ưu tiên đúng speaker
best_quality_row = None

best_target = None       # ưu tiên gần 46 cụm
best_target_row = None

rows = []

for pca_dim in PCA_LIST:
    pca = PCA(n_components=pca_dim, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    # cosine cần L2 normalize
    X_db = normalize(X_pca, norm="l2")

    for ms in MIN_SAMPLES_LIST:
        for eps in EPS_LIST:
            labels = DBSCAN(eps=float(eps), min_samples=int(ms), metric="cosine", algorithm="brute").fit_predict(X_db)
            n_clusters, noise_rate, ari, nmi = get_stats(labels)

            rows.append((pca_dim, ms, float(eps), n_clusters, noise_rate, ari, nmi))

            # 1) Best theo chất lượng (ARI ưu tiên mạnh)
            score_q = (ari, nmi, -noise_rate)
            if best_quality is None or score_q > best_quality:
                best_quality = score_q
                best_quality_row = (pca_dim, ms, float(eps), n_clusters, noise_rate, ari, nmi)

            # 2) Best gần 46 cụm: chỉ xét nếu clusters nằm trong range và noise không quá cao
            if CLUSTER_RANGE[0] <= n_clusters <= CLUSTER_RANGE[1] and noise_rate <= MAX_NOISE:
                score_t = (-abs(n_clusters - TARGET_K), ari, nmi, -noise_rate)
                if best_target is None or score_t > best_target:
                    best_target = score_t
                    best_target_row = (pca_dim, ms, float(eps), n_clusters, noise_rate, ari, nmi)

df = pd.DataFrame(rows, columns=["PCA", "min_samples", "eps", "clusters", "noise_rate", "ARI", "NMI"])

print("\n=== BEST theo chất lượng (ARI/NMI) ===")
print("PCA, min_samples, eps, clusters, noise_rate, ARI, NMI =")
print(best_quality_row)

print("\n=== BEST gần 46 cụm (ràng buộc clusters & noise) ===")
print("PCA, min_samples, eps, clusters, noise_rate, ARI, NMI =")
print(best_target_row)



# Nếu có best_target_row thì fit final để xuất DBSCAN_Results.csv
final_row = best_target_row if best_target_row is not None else best_quality_row
pca_dim, ms, eps, n_clusters, noise_rate, ari, nmi = final_row

pca = PCA(n_components=pca_dim, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
X_db = normalize(X_pca, norm="l2")

labels = DBSCAN(eps=float(eps), min_samples=int(ms), metric="cosine", algorithm="brute").fit_predict(X_db)
n_clusters, noise_rate, ari, nmi = get_stats(labels)

print(f"\n--- FINAL --- PCA={pca_dim}, min_samples={ms}, eps={eps:.2f}")
print(f"clusters={n_clusters}, noise_rate={noise_rate:.3f}, ARI={ari:.4f}, NMI={nmi:.4f}")

out = pd.DataFrame({"File": file_rel, "TrueSpeakerFolder": true_speaker, "Cluster": labels})


# Thêm vào cuối file để Trình diễn (Presentation)
plt.figure(figsize=(10, 7))
# Dùng PCA 2D để vẽ
pca_plot = PCA(n_components=2).fit_transform(X_scaled)
plt.scatter(pca_plot[:, 0], pca_plot[:, 1], c=labels, cmap='nipy_spectral', s=10, alpha=0.6)
plt.title(f"DBSCAN Clustering Visualization (Clusters: {n_clusters})")
plt.colorbar(label="Cluster ID")
plt.show()
