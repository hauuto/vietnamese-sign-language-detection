"""Realtime inference cho checkpoint E3 DomainInvariantMotionBiLSTM.

Ví dụ:
  python demo.py --self-test
  python demo.py --camera 1
  python demo.py --camera-url http://192.168.1.10:8080/video
"""

import argparse
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "models" / "E3.pt"
DEFAULT_LANDMARKER = ROOT / "models" / "hand_landmarker.task"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
SEQ_LEN = 45
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_console():
    """Tránh UnicodeEncodeError trên một số terminal Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


class E3DomainInvariantMotionBiLSTM(nn.Module):
    """Kiến trúc inference khớp chính xác checkpoint E3."""

    PARENT = (0, 0, 1, 2, 3, 0, 5, 6, 7, 5, 9, 10, 11, 9, 13, 14, 15, 13, 17, 18, 19)

    def __init__(self, input_dim, proj_dim, hidden_dim, num_layers,
                 num_classes, num_domains, dropout):
        super().__init__()
        if input_dim % 63 != 0:
            raise ValueError(f"input_dim phải chia hết cho 63, nhận được {input_dim}")
        self.n_hands = input_dim // 63
        self.proj = nn.Sequential(
            nn.Linear(input_dim * 4, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.30),
        )
        self.dw3 = nn.Conv1d(proj_dim, proj_dim, 3, padding=1, groups=proj_dim)
        self.dw5_d2 = nn.Conv1d(
            proj_dim, proj_dim, 5, padding=4, dilation=2, groups=proj_dim
        )
        self.pw = nn.Conv1d(proj_dim, proj_dim, 1)
        self.temporal_norm = nn.LayerNorm(proj_dim)
        self.temporal_drop = nn.Dropout(dropout * 0.25)
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        out_dim = hidden_dim * 2
        self.out_norm = nn.LayerNorm(out_dim)
        self.attn = nn.Sequential(
            nn.Linear(out_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.embed = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.class_head = nn.Linear(out_dim, num_classes)
        domain_hidden = max(32, hidden_dim // 2)
        self.domain_head = nn.Sequential(
            nn.Linear(out_dim, domain_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(domain_hidden, num_domains),
        )

    def _unit_bone(self, x):
        batch, steps, dims = x.shape
        points = x.reshape(batch, steps, self.n_hands, 21, 3)
        parent = torch.as_tensor(self.PARENT, dtype=torch.long, device=x.device)
        bone = points - points[:, :, :, parent, :]
        length = torch.linalg.vector_norm(bone, dim=-1, keepdim=True).clamp_min(1e-6)
        unit = bone / length
        unit[:, :, :, 0, :] = 0.0
        return unit.reshape(batch, steps, dims)

    def forward(self, x):
        velocity = torch.cat((torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]), dim=1)
        acceleration = torch.cat(
            (torch.zeros_like(velocity[:, :1]), velocity[:, 1:] - velocity[:, :-1]), dim=1
        )
        z = self.proj(torch.cat((x, velocity, acceleration, self._unit_bone(x)), dim=-1))
        channels = z.transpose(1, 2)
        local = torch.nn.functional.gelu(self.dw3(channels))
        broader = torch.nn.functional.gelu(self.dw5_d2(channels))
        convolution = self.pw(0.5 * (local + broader)).transpose(1, 2)
        z = self.temporal_norm(z + self.temporal_drop(convolution))
        output, _ = self.lstm(z)
        output = self.out_norm(output)
        attention = torch.softmax(self.attn(output), dim=1)
        attention_pool = (output * attention).sum(dim=1)
        mean_pool = output.mean(dim=1)
        embedding = self.embed(torch.cat((attention_pool, mean_pool), dim=1))
        return self.class_head(embedding)


def normalize_hand_sequence_keep_motion(X: np.ndarray) -> np.ndarray:
    """Sequence normalization giống hệt pipeline train E3."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or X.shape[1] != SEQ_LEN or X.shape[2] not in (63, 126):
        raise ValueError(f"Cần input (N, {SEQ_LEN}, 63/126), nhận được {X.shape}")
    samples, steps, dims = X.shape
    hands = dims // 63
    points = X.reshape(samples, steps, hands, 21, 3).copy()
    result = np.zeros_like(points, dtype=np.float32)
    palm_ids = np.array([5, 9, 13, 17], dtype=np.int64)

    for hand_index in range(hands):
        hand = points[:, :, hand_index]
        wrist = hand[:, :, 0, :]
        local = hand - wrist[:, :, None, :]
        palm_radius = np.linalg.norm(local[:, :, palm_ids, :], axis=-1)
        frame_scale = np.median(palm_radius, axis=-1)
        valid = frame_scale > 1e-4
        for sample_index in range(samples):
            valid_indices = np.flatnonzero(valid[sample_index])
            if len(valid_indices) == 0:
                continue
            anchor = wrist[sample_index, valid_indices[0]].copy()
            scale = max(float(np.median(frame_scale[sample_index, valid_indices])), 1e-6)
            result[sample_index, valid_indices, hand_index, 0, :] = (
                wrist[sample_index, valid_indices] - anchor
            ) / scale
            result[sample_index, valid_indices, hand_index, 1:, :] = (
                local[sample_index, valid_indices, 1:, :] / scale
            )
    return result.reshape(samples, steps, dims).astype(np.float32)


class E3Predictor:
    def __init__(self, checkpoint_path: Path):
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy checkpoint E3: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        required = {
            "classes", "input_dim", "proj_dim", "hidden_dim", "num_layers",
            "dropout", "domain_map", "model_state_dicts",
        }
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(f"Checkpoint thiếu các field: {sorted(missing)}")
        if checkpoint.get("model_name") != "DomainInvariantMotionBiLSTM":
            raise ValueError(f"Sai loại model: {checkpoint.get('model_name')!r}")

        self.checkpoint = checkpoint
        self.classes = np.asarray(checkpoint["classes"])
        self.input_dim = int(checkpoint["input_dim"])
        self.use_normalization = bool(checkpoint.get("use_hand_normalization", True))
        self.model = E3DomainInvariantMotionBiLSTM(
            input_dim=self.input_dim,
            proj_dim=int(checkpoint["proj_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_layers=int(checkpoint["num_layers"]),
            num_classes=len(self.classes),
            num_domains=len(checkpoint["domain_map"]),
            dropout=float(checkpoint["dropout"]),
        )
        self.model.load_state_dict(checkpoint["model_state_dicts"][0], strict=True)
        self.model.to(DEVICE).eval()

    def predict(self, sequence: np.ndarray):
        batch = np.asarray(sequence, dtype=np.float32)[None, ...]
        if self.use_normalization:
            batch = normalize_hand_sequence_keep_motion(batch)
        tensor = torch.from_numpy(batch).to(DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        class_index = int(probabilities.argmax())
        return self.classes[class_index], float(probabilities[class_index]), latency_ms


def ensure_landmarker(path: Path) -> Path:
    """Tải model MediaPipe chính thức nếu máy chưa có."""
    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    print(f"Đang tải MediaPipe Hand Landmarker về {path} ...")
    try:
        urllib.request.urlretrieve(LANDMARKER_URL, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print("Đã tải hand_landmarker.task.")
    return path


def create_landmarker(model_path: Path, input_dim: int):
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=input_dim // 63,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def extract_landmark(landmarker, frame_bgr, timestamp_ms: int, input_dim: int):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection = landmarker.detect_for_video(image, timestamp_ms)
    if not detection.hand_landmarks:
        return None

    if input_dim == 63:
        points = detection.hand_landmarks[0]
        return np.asarray([[p.x, p.y, p.z] for p in points], dtype=np.float32).reshape(63)

    hands = {"Left": np.zeros((21, 3), dtype=np.float32),
             "Right": np.zeros((21, 3), dtype=np.float32)}
    for points, handedness in zip(detection.hand_landmarks, detection.handedness):
        label = handedness[0].category_name
        hands[label] = np.asarray([[p.x, p.y, p.z] for p in points], dtype=np.float32)
    return np.concatenate((hands["Left"].reshape(-1), hands["Right"].reshape(-1)))


def open_camera(camera_index: int, camera_url: str | None):
    if camera_url:
        capture = cv2.VideoCapture(camera_url)
        source_name = camera_url
    else:
        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(camera_index)
        source_name = f"camera index {camera_index}"
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise RuntimeError(f"Không mở được {source_name}")
    return capture


def run_self_test(predictor: E3Predictor, sample_path: Path | None):
    if sample_path is None:
        person = predictor.checkpoint.get("outer_test_person", "vy")
        candidates = sorted((ROOT / "landmarks" / "raw" / person).glob("*.npy"))
        if not candidates:
            candidates = sorted((ROOT / "landmarks" / "raw").glob("*/*.npy"))
        if not candidates:
            raise FileNotFoundError("Không có landmark .npy để self-test")
        sample_path = candidates[0]
    sequence = np.load(sample_path).astype(np.float32)
    label, confidence, latency_ms = predictor.predict(sequence)
    print(f"Sample     : {sample_path}")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence * 100:.2f}%")
    print(f"Latency    : {latency_ms:.2f} ms ({DEVICE})")


def run_camera(predictor: E3Predictor, args):
    landmarker_path = ensure_landmarker(args.landmarker)
    landmarker = create_landmarker(landmarker_path, predictor.input_dim)
    capture = open_camera(args.camera, args.camera_url)
    buffer = deque(maxlen=SEQ_LEN)
    prediction = None
    last_timestamp = -1
    print("Nhấn Q để thoát, R để xóa buffer 45 frame.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Không đọc được frame từ camera.")
                break
            timestamp = max(last_timestamp + 1, int(time.monotonic() * 1000))
            last_timestamp = timestamp
            landmark = extract_landmark(landmarker, frame, timestamp, predictor.input_dim)
            if landmark is None:
                buffer.clear()
                prediction = None
            else:
                buffer.append(landmark)
                if len(buffer) == SEQ_LEN:
                    prediction = predictor.predict(np.stack(buffer))

            cv2.putText(
                frame, f"Frames: {len(buffer)}/{SEQ_LEN}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 190, 255), 2,
            )
            if prediction is not None:
                label, confidence, latency_ms = prediction
                text = f"E3: {label}  {confidence * 100:.1f}%  {latency_ms:.2f}ms"
                color = (0, 255, 0) if confidence >= args.confidence else (0, 165, 255)
                cv2.putText(
                    frame, text, (12, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2
                )
            cv2.imshow("Vietnamese Sign Language - E3", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                buffer.clear()
                prediction = None
    finally:
        capture.release()
        landmarker.close()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Demo realtime checkpoint E3")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--landmarker", type=Path, default=DEFAULT_LANDMARKER)
    parser.add_argument("--camera", type=int, default=0,
                        help="Index webcam Windows; camera điện thoại thường là 1 hoặc 2")
    parser.add_argument("--camera-url", help="URL MJPEG/RTSP nếu dùng app IP camera")
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--self-test", action="store_true",
                        help="Inference một file .npy, không mở camera")
    parser.add_argument("--sample", type=Path, help="File .npy dùng với --self-test")
    parser.add_argument("--download-landmarker", action="store_true",
                        help="Chỉ tải hand_landmarker.task rồi thoát")
    return parser.parse_args()


def main():
    configure_console()
    args = parse_args()
    if args.download_landmarker:
        ensure_landmarker(args.landmarker)
        return
    predictor = E3Predictor(args.checkpoint)
    checkpoint = predictor.checkpoint
    print(
        f"Đã load E3: {len(predictor.classes)} lớp, input={predictor.input_dim}, "
        f"outer test={checkpoint.get('outer_test_person')}, device={DEVICE}"
    )
    if args.self_test:
        run_self_test(predictor, args.sample)
    else:
        run_camera(predictor, args)


if __name__ == "__main__":
    main()
