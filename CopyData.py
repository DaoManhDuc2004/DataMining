import os
import shutil

# ===============================
# CẤU HÌNH ĐƯỜNG DẪN
# ===============================
SRC_PATH = r"C:\Users\ACER\Downloads\DataKhaiPha\vivos\train\waves"
DST_PATH = r"D:\thuyloiuniversity\Mon hoc tren lop\KhaiPhaDuLieu\BTL\NV3\Data"

MAX_FILES_PER_SPEAKER = 10

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(DST_PATH, exist_ok=True)

# Lấy danh sách thư mục người nói
speaker_folders = sorted([
    d for d in os.listdir(SRC_PATH)
    if os.path.isdir(os.path.join(SRC_PATH, d))
])

print(f"Phát hiện {len(speaker_folders)} người nói")

# ===============================
# COPY FILE
# ===============================
for speaker in speaker_folders:
    src_speaker_path = os.path.join(SRC_PATH, speaker)
    dst_speaker_path = os.path.join(DST_PATH, speaker)

    # Tạo thư mục người nói bên đích
    os.makedirs(dst_speaker_path, exist_ok=True)

    # Lấy danh sách file wav, sắp xếp theo STT
    wav_files = sorted([
        f for f in os.listdir(src_speaker_path)
        if f.lower().endswith(".wav")
    ])

    # Chỉ lấy tối đa 10 file
    selected_files = wav_files[:MAX_FILES_PER_SPEAKER]

    print(f"\n{speaker}: copy {len(selected_files)} file")

    for file_name in selected_files:
        src_file = os.path.join(src_speaker_path, file_name)
        dst_file = os.path.join(dst_speaker_path, file_name)

        shutil.copy2(src_file, dst_file)

    print(f"  ✔ Hoàn thành {speaker}")

print("\n🎉 COPY DATASET HOÀN TẤT!")
