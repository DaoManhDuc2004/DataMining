import os

# ===============================
# CẤU HÌNH ĐƯỜNG DẪN GỐC
# ===============================
BASE_PATH = r"C:\Users\ACER\Downloads\DataKhaiPha\vivos\train\waves"

# Lấy danh sách thư mục người nói (VIVOSSPKxx)
speaker_folders = sorted([
    d for d in os.listdir(BASE_PATH)
    if os.path.isdir(os.path.join(BASE_PATH, d))
])

print(f"Phát hiện {len(speaker_folders)} thư mục người nói")

# ===============================
# ĐỔI TÊN THƯ MỤC + FILE
# ===============================
for idx, old_speaker in enumerate(speaker_folders, start=1):
    old_speaker_path = os.path.join(BASE_PATH, old_speaker)

    # Tên người nói mới
    new_speaker = f"nguoiNoi{idx:02d}"
    new_speaker_path = os.path.join(BASE_PATH, new_speaker)

    print(f"\nĐang xử lý: {old_speaker} → {new_speaker}")

    # ---- ĐỔI TÊN FILE TRƯỚC ----
    wav_files = sorted([
        f for f in os.listdir(old_speaker_path)
        if f.lower().endswith(".wav")
    ])

    for file_idx, old_file in enumerate(wav_files, start=1):
        old_file_path = os.path.join(old_speaker_path, old_file)

        new_file_name = f"{new_speaker}_STT{file_idx:02d}.wav"
        new_file_path = os.path.join(old_speaker_path, new_file_name)

        os.rename(old_file_path, new_file_path)

    print(f"  ✔ Đã đổi {len(wav_files)} file")

    # ---- ĐỔI TÊN THƯ MỤC SAU ----
    os.rename(old_speaker_path, new_speaker_path)
    print(f"  ✔ Đã đổi tên thư mục")

print("\n🎉 HOÀN TẤT ĐỔI TÊN DATASET!")
