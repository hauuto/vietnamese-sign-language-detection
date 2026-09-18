"""
Trực quan hoá landmark: overlay lên video gốc, và cắt ảnh mẫu ngẫu nhiên cho báo cáo.

Yêu cầu: các thư viện đã có sẵn trong requirements.txt (cv2, mediapipe, numpy)

CÁCH DÙNG (chạy bằng nút Run của IDE, không cần gõ lệnh dòng lệnh):
    Sửa 2 biến MODE và VIDEO_PATH ở phần "CẤU HÌNH" bên dưới, rồi bấm Run.

    MODE = "video"   -> xuất video overlay cho ĐÚNG 1 video, đường dẫn khai báo ở VIDEO_PATH
    MODE = "all"     -> xuất video overlay cho TOÀN BỘ video trong raw/ (mất nhiều thời gian)
    MODE = "report"  -> cắt N_PER_PERSON ảnh mẫu ngẫu nhiên mỗi người, làm ảnh minh hoạ báo cáo
"""
import random
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from extract_landmarks import ensure_model, make_landmarker  # dùng lại đúng cấu hình landmarker của pipeline chính

# ============================== CẤU HÌNH — SỬA Ở ĐÂY ==============================
MODE = "report"                       # "video" | "all" | "report"
VIDEO_PATH = "raw/hau/a_hau_A_1.mp4"  # chỉ dùng khi MODE = "video"
N_PER_PERSON = 4                      # chỉ dùng khi MODE = "report"
SEED = 42                             # chỉ dùng khi MODE = "report"
# ====================================================================================

ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = ROOT / "raw"
OUT_ROOT = ROOT / "visualize_output"
PEOPLE = ["hau", "khoi", "tai", "vy"]

# Danh sách 21 cặp điểm nối khớp ngón tay theo đúng thứ tự MediaPipe Hands quy định
# (0 = cổ tay, 4/8/12/16/20 = đầu 5 ngón). Hardcode ở đây vì bản mediapipe dùng API Tasks
# không còn kèm module mp.solutions (API cũ) để lấy HAND_CONNECTIONS tự động.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),        # ngón trỏ
    (0, 9), (9, 10), (10, 11), (11, 12),   # ngón giữa
    (0, 13), (13, 14), (14, 15), (15, 16), # ngón áp út
    (0, 17), (17, 18), (18, 19), (19, 20), # ngón út
    (5, 9), (9, 13), (13, 17),             # nối giữa các gốc ngón (lòng bàn tay)
]


def draw_landmarks(frame, hand_landmarks_list):
    """Vẽ các điểm + đường nối khớp lên 1 khung hình (frame đã ở dạng BGR, sẽ bị sửa trực tiếp)."""
    h, w = frame.shape[:2]
    for lm_list in hand_landmarks_list:
        pts = [(int(p.x * w), int(p.y * h)) for p in lm_list]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    return frame


def overlay_video(video_path: Path, out_dir: Path = OUT_ROOT / "overlay"):
    """Chạy lại MediaPipe trên video gốc, vẽ landmark đè lên từng khung hình, xuất video mới."""
    ensure_model()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (video_path.stem + "_overlay.mp4")

    landmarker = make_landmarker()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

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
            if result.hand_landmarks:
                draw_landmarks(frame, result.hand_landmarks)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        landmarker.close()

    print(f"Đã lưu video overlay: {out_path}")
    return out_path


def overlay_all_videos(out_dir: Path = OUT_ROOT / "overlay"):
    """Chạy overlay_video() cho TOÀN BỘ video trong raw/{person}/, không bỏ qua ai."""
    ensure_model()
    all_videos = sorted(RAW_ROOT.glob("*/*.mp4"))
    if not all_videos:
        print(f"Không tìm thấy video nào trong {RAW_ROOT}")
        return

    print(f"Tổng số video cần xử lý: {len(all_videos)}")
    for i, vp in enumerate(all_videos, 1):
        out_path = out_dir / (vp.stem + "_overlay.mp4")
        if out_path.exists():
            print(f"[{i}/{len(all_videos)}] {vp.name} -> đã có sẵn, bỏ qua")
            continue
        overlay_video(vp, out_dir=out_dir)
        print(f"[{i}/{len(all_videos)}] {vp.name} -> xong")

    print(f"\nĐã xử lý xong. Video overlay nằm trong: {out_dir.resolve()}")


def sample_report_frames(n_per_person: int = 4, out_dir: Path = OUT_ROOT / "report", seed: int = 42):
    """Với mỗi người, chọn ngẫu nhiên n_per_person video, mỗi video lấy 1 khung hình có landmark
    (ưu tiên khung hình CÓ phát hiện tay, để ảnh minh hoạ không bị trống), lưu thành ảnh .png riêng.
    """
    ensure_model()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    for person in PEOPLE:
        person_dir = RAW_ROOT / person
        videos = sorted(person_dir.glob("*.mp4"))
        if not videos:
            print(f"Không tìm thấy video nào cho {person}, bỏ qua.")
            continue

        chosen = rng.sample(videos, k=min(n_per_person, len(videos)))
        for vp in chosen:
            landmarker = make_landmarker()
            cap = cv2.VideoCapture(str(vp))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            frames_with_hand = []  # (frame_idx, frame_bgr, hand_landmarks)
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
                    if result.hand_landmarks:
                        frames_with_hand.append((frame_idx, frame.copy(), result.hand_landmarks))
                    frame_idx += 1
            finally:
                cap.release()
                landmarker.close()

            if not frames_with_hand:
                print(f"  {vp.name}: không khung hình nào phát hiện được tay, bỏ qua.")
                continue

            # lấy khung hình ở giữa danh sách các khung CÓ tay, cho tư thế ký hiệu rõ ràng nhất
            idx, frame, hand_landmarks = frames_with_hand[len(frames_with_hand) // 2]
            draw_landmarks(frame, hand_landmarks)

            out_name = f"{person}_{vp.stem}_frame{idx}.png"
            cv2.imwrite(str(out_dir / out_name), frame)
            print(f"  {vp.name} -> {out_name} (khung {idx}/{frame_idx})")

    print(f"\nĐã lưu ảnh minh hoạ báo cáo vào: {out_dir.resolve()}")


if __name__ == "__main__":
    if MODE == "video":
        overlay_video(Path(VIDEO_PATH))
    elif MODE == "all":
        overlay_all_videos()
    elif MODE == "report":
        sample_report_frames(n_per_person=N_PER_PERSON, seed=SEED)
    else:
        print(f"MODE không hợp lệ: {MODE!r}. Chỉ nhận 'video', 'all', hoặc 'report'.")