"""
Trích landmark từ video thô, lấy mẫu lại về 45 bước cố định.

Chạy: python src/extract_landmarks.py [--source raw|raw_external] [--two-hands]

Đầu vào : raw/{person}/{code}_{person}_{block}_{seq}.mp4   (chỉ đọc, không sửa)
Đầu ra  : landmarks/raw/{person}/{code}_{person}_{block}_{seq}.npy
          shape = (45, 63)  nếu 1 tay
          shape = (45, 126) nếu 2 tay (nối [Left, Right], tay thiếu = 0)

Ghi chú kỹ thuật quan trọng
----------------------------
1. Dùng MediaPipe Tasks API (HandLandmarker), KHÔNG dùng mp.solutions.hands.
   API cũ (solutions) đã bị deprecate và một số bản cài mediapipe >=0.10
   không còn kèm theo nữa — chỉ còn API Tasks. Script tự tải model
   hand_landmarker.task về lần chạy đầu (cần mạng một lần duy nhất).

2. Lấy mẫu lại theo THỜI GIAN, không cắt cứng 45 frame đầu: máy quay có FPS
   khác nhau (có máy tụt xuống 18-20 khi thiếu sáng), nên số frame thô trong
   1.5 giây không giống nhau giữa các máy. Nội suy tuyến tính theo mốc thời
   gian chuẩn hóa [0,1] đưa mọi clip về đúng 45 bước, bất kể quay được bao
   nhiêu frame gốc.
"""
import argparse
import sys
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from classes import RESAMPLE_STEPS

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def ensure_model():
    if MODEL_PATH.exists():
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Dang tai model ve {MODEL_PATH} (chi can 1 lan)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Da tai xong.")


def make_landmarker():
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_raw_landmarks(video_path: Path):
    """Trả về list các frame, mỗi frame là dict {'Left': arr(21,3) hoặc None, 'Right': ...}
    Tạo landmarker MỚI cho mỗi video: RunningMode.VIDEO yêu cầu timestamp tăng dần liên tục
    trên CÙNG một landmarker — dùng chung 1 landmarker cho nhiều video (mỗi video lại bắt đầu
    từ 0ms) gây lỗi 'Input timestamp must be monotonically increasing.' ngay khi sang video kế."""
    import mediapipe as mp

    landmarker = make_landmarker()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    n_no_hand = 0
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * (1000.0 / fps))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            entry = {"Left": None, "Right": None}
            if result.hand_landmarks:
                for lm_list, handedness in zip(result.hand_landmarks, result.handedness):
                    label = handedness[0].category_name  # "Left" / "Right"
                    pts = np.array([[p.x, p.y, p.z] for p in lm_list], dtype=np.float32)
                    entry[label] = pts
            if entry["Left"] is None and entry["Right"] is None:
                n_no_hand += 1
            frames.append(entry)
            frame_idx += 1
    finally:
        cap.release()
        landmarker.close()
    return frames, n_no_hand


def to_feature_sequence(frames, two_hands: bool):
    """Chuyển list frame -> mảng (T, D). Tay vắng mặt = vector 0 (không nội suy hộ)."""
    dim = 126 if two_hands else 63
    if not frames:
        return np.zeros((0, dim), dtype=np.float32)
    seq = []
    for entry in frames:
        left = entry["Left"] if entry["Left"] is not None else np.zeros((21, 3), dtype=np.float32)
        vec = left.flatten()
        if two_hands:
            right = entry["Right"] if entry["Right"] is not None else np.zeros((21, 3), dtype=np.float32)
            vec = np.concatenate([vec, right.flatten()])
        seq.append(vec)
    return np.stack(seq)


def resample_time(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Nội suy tuyến tính theo trục thời gian về đúng target_len bước."""
    n = seq.shape[0]
    dim = seq.shape[1] if seq.ndim > 1 and n > 0 else 63
    if n == 0:
        return np.zeros((target_len, dim), dtype=np.float32)
    if n == 1:
        return np.repeat(seq, target_len, axis=0)
    src_t = np.linspace(0, 1, n)
    dst_t = np.linspace(0, 1, target_len)
    out = np.empty((target_len, seq.shape[1]), dtype=np.float32)
    for d in range(seq.shape[1]):
        out[:, d] = np.interp(dst_t, src_t, seq[:, d])
    return out


def process_folder(src_dir: Path, dst_dir: Path, two_hands: bool):
    dst_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(src_dir.glob("*.mp4"))
    if not videos:
        print(f"Khong tim thay video nao trong {src_dir}")
        return

    bad_files = []
    for i, vp in enumerate(videos, 1):
        out_path = dst_dir / (vp.stem + ".npy")
        if out_path.exists():
            continue  # đã trích rồi, bỏ qua — cho phép chạy lại giữa chừng nếu bị ngắt
        frames, n_no_hand = extract_raw_landmarks(vp)
        if len(frames) == 0:
            print(f"[{i}/{len(videos)}] LOI: {vp.name} khong doc duoc frame nao")
            bad_files.append(vp.name)
            continue
        miss_ratio = n_no_hand / len(frames)
        seq = to_feature_sequence(frames, two_hands)
        seq = resample_time(seq, RESAMPLE_STEPS)
        np.save(out_path, seq)
        flag = "  << CANH BAO thieu tay >20%, nen quay bu" if miss_ratio > 0.2 else ""
        print(f"[{i}/{len(videos)}] {vp.name} -> {out_path.name}  (mat tay {miss_ratio*100:.0f}%){flag}")
        if miss_ratio > 0.2:
            bad_files.append(vp.name)

    if bad_files:
        print("\n=== Danh sach can quay bu (mat tay qua 20% hoac loi doc) ===")
        for f in bad_files:
            print(" -", f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--two-hands", action="store_true",
                     help="Bat che do 2 tay (126 chieu). Mac dinh 1 tay (63 chieu).")
    args = ap.parse_args()

    ensure_model()

    src_root = ROOT / "raw"
    dst_root = ROOT / "landmarks" / "raw"
    people_dirs = [p for p in src_root.iterdir() if p.is_dir()] if src_root.exists() else []
    if not people_dirs:
        print(f"Khong co thu muc nguoi nao trong {src_root}")
        return

    for person_dir in sorted(people_dirs):
        print(f"\n=== Nguoi: {person_dir.name} ===")
        process_folder(person_dir, dst_root / person_dir.name, args.two_hands)


if __name__ == "__main__":
    main()
