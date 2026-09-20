"""
demo.py — Live demo so sánh 4 mô hình E0/E1/E2/E3 cùng lúc.

Yêu cầu cấu trúc thư mục (đã có theo ảnh bạn gửi):
  models/
    hand_landmarker.task
    E0/E0_fold{0,1,2,3}.pt
    E1/E1_fold{0,1,2,3}.pt   (khi Tài train xong)
    E2/E2_fold{0,1,2,3}.pt   (khi Vỹ train xong)
    E3/E3_fold{0,1,2,3}.pt   (khi Khôi train xong)

Model nào chưa có checkpoint sẽ tự động bị bỏ qua khi demo (không lỗi),
để bạn có thể chạy demo ngay bây giờ chỉ với E0, rồi bổ sung dần khi
Tài/Vỹ/Khôi gửi checkpoint.
"""

import re
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Cấu hình — sửa ở đây nếu cần
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
LANDMARKER_TASK = MODELS_DIR / "hand_landmarker.task"

# Thư mục chứa dữ liệu landmark gốc — CHỈ dùng để suy ra đúng thứ tự 34 lớp
# (ALL_CLASSES) giống hệt lúc train, vì lúc train dùng sorted(set(...)) trên
# toàn bộ dataset. Nếu không có ở máy demo, xem phần "ALL_CLASSES thủ công" bên dưới.
LANDMARK_DIR_FOR_CLASSES = ROOT / "landmarks" / "raw"

FRAME_INDEX = 22
SEQ_LEN = 45
DEMO_FOLD = 0  # fold dùng để demo (0=hau,1=khoi,2=tai,3=vy bị giữ lại lúc train fold đó)
CAM_INDEX = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Suy ra ALL_CLASSES đúng thứ tự như lúc train
# ---------------------------------------------------------------------------

def infer_all_classes() -> list[str]:
    fname_re = re.compile(r"^([a-z_]+)_([a-z]+)_([AB])_(\d+)\.npy$")
    if LANDMARK_DIR_FOR_CLASSES.exists():
        codes = set()
        for f in LANDMARK_DIR_FOR_CLASSES.glob("*/*.npy"):
            m = fname_re.match(f.name)
            if m:
                codes.add(m.group(1))
        if codes:
            return sorted(codes)
    raise FileNotFoundError(
        f"Không tìm thấy {LANDMARK_DIR_FOR_CLASSES} để suy ra danh sách lớp. "
        "Nếu chạy demo trên máy không có dữ liệu gốc, thay ALL_CLASSES bên dưới "
        "bằng danh sách 34 mã lớp cố định (copy từ log 'Số lớp: 34' lúc train)."
    )


ALL_CLASSES = infer_all_classes()
IDX_TO_CODE = {i: c for i, c in enumerate(ALL_CLASSES)}
NUM_CLASSES = len(ALL_CLASSES)
print(f"Số lớp: {NUM_CLASSES}")


# ---------------------------------------------------------------------------
# Chuẩn hóa dùng cho E1 — PHẢI giống hệt hàm normalize_frame trong train_E1.ipynb
# ---------------------------------------------------------------------------

def normalize_frame(frame_63: np.ndarray) -> np.ndarray:
    pts = frame_63.reshape(21, 3)
    wrist = pts[0].copy()
    pts = pts - wrist
    scale = np.linalg.norm(pts[9]) + 1e-8
    pts = pts / scale
    return pts.reshape(63)


# ---------------------------------------------------------------------------
# Kiến trúc mô hình — PHẢI giống hệt lúc train để load_state_dict không lỗi
# ---------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class LSTMClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_size: int = 128, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=1,
                             batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(0.3)
        out_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(self.dropout(last))


# ---------------------------------------------------------------------------
# Wrapper thống nhất cho từng thực nghiệm: load checkpoint + tiền xử lý + predict
# ---------------------------------------------------------------------------

class ExperimentModel:
    """Bọc 1 thực nghiệm (E0/E1/E2/E3): tự load checkpoint nếu có, trả None nếu chưa có."""

    def __init__(self, name: str, build_fn, prepare_fn):
        self.name = name
        self.model = None
        ckpt_path = MODELS_DIR / name / f"{name}_fold{DEMO_FOLD}.pt"
        if ckpt_path.exists():
            model = build_fn()
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            model.to(DEVICE).eval()
            self.model = model
            print(f"[{name}] Đã load checkpoint: {ckpt_path}")
        else:
            print(f"[{name}] CHƯA có checkpoint ({ckpt_path}) — sẽ bỏ qua khi demo.")
        self.prepare_fn = prepare_fn

    def predict(self, seq_45x63: np.ndarray):
        """seq_45x63: (45, 63) chuỗi landmark thô của 1 lượt ký hiệu.
        Trả về (mã_lớp, độ tin cậy, latency_ms) hoặc None nếu model chưa sẵn sàng."""
        if self.model is None:
            return None
        x = self.prepare_fn(seq_45x63)  # (63,) hoặc (45,63) tuỳ thực nghiệm
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(x_t)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)
        latency_ms = (time.perf_counter() - t0) * 1000

        return IDX_TO_CODE[int(idx)], float(conf), latency_ms


EXPERIMENTS = {
    "E0": ExperimentModel(
        "E0",
        build_fn=lambda: MLPClassifier(63, NUM_CLASSES),
        prepare_fn=lambda seq: seq[FRAME_INDEX, :],
    ),
    "E1": ExperimentModel(
        "E1",
        build_fn=lambda: MLPClassifier(63, NUM_CLASSES),
        prepare_fn=lambda seq: normalize_frame(seq[FRAME_INDEX, :]),
    ),
    "E2": ExperimentModel(
        "E2",
        build_fn=lambda: LSTMClassifier(63, NUM_CLASSES, bidirectional=False),
        prepare_fn=lambda seq: seq,  # (45, 63) nguyên chuỗi
    ),
    "E3": ExperimentModel(
        "E3",
        build_fn=lambda: LSTMClassifier(63, NUM_CLASSES, bidirectional=True),
        prepare_fn=lambda seq: seq,
    ),
}


# ---------------------------------------------------------------------------
# Trích xuất landmark trực tiếp từ webcam bằng MediaPipe Tasks API
# ---------------------------------------------------------------------------

base_options = mp_python.BaseOptions(model_asset_path=str(LANDMARKER_TASK))
landmarker_options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
)
landmarker = mp_vision.HandLandmarker.create_from_options(landmarker_options)


def extract_landmark(frame_bgr, timestamp_ms: int):
    """Trả về vector (63,) toạ độ x,y,z của 21 điểm, hoặc None nếu không thấy tay."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if not result.hand_landmarks:
        return None
    pts = result.hand_landmarks[0]
    return np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32).reshape(63)


# ---------------------------------------------------------------------------
# Vòng lặp demo chính — buffer trượt 45 khung, chạy đồng thời các mô hình có sẵn
# ---------------------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    buffer = deque(maxlen=SEQ_LEN)
    frame_idx = 0

    print("\nNhấn 'q' để thoát. Đưa tay vào khung hình để bắt đầu nhận diện.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        landmark = extract_landmark(frame, frame_idx * 33)  # xấp xỉ ~30fps
        frame_idx += 1

        if landmark is not None:
            buffer.append(landmark)
        else:
            buffer.clear()  # mất tay giữa chừng -> reset cửa sổ, tránh trộn 2 lượt ký hiệu khác nhau

        y_text = 30
        if len(buffer) == SEQ_LEN:
            seq = np.stack(buffer)  # (45, 63)
            for name, exp_model in EXPERIMENTS.items():
                pred = exp_model.predict(seq)
                if pred is None:
                    text = f"{name}: (chưa có checkpoint)"
                else:
                    code, conf, latency_ms = pred
                    text = f"{name}: {code}  ({conf*100:.0f}%, {latency_ms:.2f}ms)"
                cv2.putText(frame, text, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_text += 30
        else:
            cv2.putText(frame, f"Đang gom khung: {len(buffer)}/{SEQ_LEN}", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow("Team 9 - Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()