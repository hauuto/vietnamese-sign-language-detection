"""Realtime demo so sánh song song 4 thực nghiệm E0, E1, E2, E3 trên cùng một luồng camera.

Mỗi thực nghiệm có một ô kết quả riêng trên màn hình. Mỗi thực nghiệm có thể nạp 1 hoặc
nhiều checkpoint (ví dụ 4 fold) và lấy trung bình dự đoán.

Loại checkpoint được tự nhận dạng:
  - MLP (E0/E1): dict có "model_state_dict". E1 nếu "experiment" == "E1" -> chuẩn hóa cổ tay + tỉ lệ.
    Dùng khung index 22 của cửa sổ 45 frame (giống lúc train).
  - AttentionLSTM (E2/E3): "model_name" == "AttentionLSTM"; một chiều hay hai chiều được
    suy ra từ state_dict (có trọng số *_reverse -> Bi-LSTM).
  - E3+: "model_name" == "DomainInvariantMotionBiLSTM".

Tính năng chung:
  1. Hai chế độ: models/baseline/ và models/extended/ (nhấn M để đổi khi chạy).
     Thực nghiệm nào không có bản mở rộng thì chế độ mở rộng tự dùng bản baseline.
  2. TTA lật ngang (trung bình softmax bản gốc + bản lật) — tắt bằng --no-tta.
  3. Khóa bàn tay: chỉ nhận dạng tay được khóa; muốn đổi tay thì rút hết tay,
     chỉ chừa tay muốn dùng (khung chỉ còn đúng 1 tay trong vài frame).
  4. Từ chối dự đoán: ngưỡng xác suất + khoảng cách top1-top2 (+ ngưỡng energy tùy chọn),
     kèm ổn định theo thời gian (nhãn chỉ hiện khi K lần dự đoán liên tiếp giống nhau).

Ví dụ:
  python demo.py --self-test
  python demo.py --camera 1 --signer-seen yes
  python demo.py --mode extended
  python demo.py --camera-url http://192.168.1.10:8080/video

Lưu ý đánh giá: số accuracy trong báo cáo lấy từ LOSO (mỗi fold chỉ đánh giá trên người
không có trong train của fold đó). KHÔNG dùng ensemble này để chấm điểm lại trên chính
bộ dữ liệu đã train — mỗi mẫu đã nằm trong train của 3/4 model nên điểm sẽ cao giả tạo.
"""

import argparse
import sys
import time
import urllib.request
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ("E0", "E1", "E2", "E3")
DEFAULT_LANDMARKER = ROOT / "models" / "hand_landmarker.task"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
SEQ_LEN = 45
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Màu BGR cho OpenCV
GREEN, GRAY, ORANGE, RED, WHITE = (0, 200, 0), (150, 150, 150), (0, 165, 255), (0, 0, 255), (255, 255, 255)


def configure_console():
    """Tránh UnicodeEncodeError trên một số terminal Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Mô hình (giữ nguyên kiến trúc để khớp checkpoint)
# ---------------------------------------------------------------------------
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


class MLPClassifier(nn.Module):
    """E0/E1/E0+: input -> 128 -> 64 -> num_classes (ReLU, Dropout 0.3).
    E0+ có thêm z-score từng chiều (buffer mean/std) ở đầu vào."""

    def __init__(self, input_dim, num_classes, use_z=False):
        super().__init__()
        self.use_z = use_z
        if use_z:
            self.register_buffer("mean", torch.zeros(input_dim))
            self.register_buffer("std", torch.ones(input_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        if self.use_z:
            x = (x - self.mean) / self.std
        return self.net(x)


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        return (lstm_out * weights).sum(dim=1)


class AttentionLSTM(nn.Module):
    """E2 (bidirectional=False) và E3 (bidirectional=True) — khớp notebook E2/E3."""

    def __init__(self, input_dim, proj_dim, hidden_dim, num_layers, num_classes,
                 dropout=0.35, bidirectional=False):
        super().__init__()
        feat_dim = input_dim * 3
        self.proj = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, proj_dim), nn.GELU(), nn.Dropout(dropout * 0.30)
        )
        self.lstm = nn.LSTM(
            input_size=proj_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional,
        )
        seq_dim = hidden_dim * (2 if bidirectional else 1)
        self.out_norm = nn.LayerNorm(seq_dim)
        self.attn_pool = AttentionPool(seq_dim)
        out_dim = seq_dim * 2
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(out_dim // 2, num_classes)
        )

    def forward(self, x):
        dx = torch.cat((torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]), dim=1)
        ddx = torch.cat((torch.zeros_like(dx[:, :1]), dx[:, 1:] - dx[:, :-1]), dim=1)
        out, _ = self.lstm(self.proj(torch.cat((x, dx, ddx), dim=-1)))
        out = self.out_norm(out)
        return self.head(torch.cat((self.attn_pool(out), out.mean(dim=1)), dim=-1))


def normalize_frame_e1(X: np.ndarray) -> np.ndarray:
    """E1: (N, D) -> dời gốc về cổ tay, chia khoảng cách 3D lớn nhất từ cổ tay (giống notebook E1)."""
    hands = X.reshape(X.shape[0], -1, 21, 3)
    centered = hands - hands[:, :, 0:1, :]
    scales = np.max(np.linalg.norm(centered, axis=-1), axis=-1, keepdims=True)[..., None]
    return (centered / np.maximum(scales, 1e-6)).reshape(X.shape[0], -1).astype(np.float32)


def flip_raw_frame(X: np.ndarray) -> np.ndarray:
    """E0: lật trên tọa độ ảnh thô x' = 1 - x; tay mất landmark giữ 0; 2 tay thì đổi slot."""
    hands = X.reshape(X.shape[0], -1, 21, 3).copy()
    valid = np.abs(hands).sum(axis=(-1, -2)) > 1e-7
    hands[..., 0] = np.where(valid[..., None], 1.0 - hands[..., 0], hands[..., 0])
    if hands.shape[1] == 2:
        hands = hands[:, ::-1].copy()
    return hands.reshape(X.shape).astype(np.float32)


def flip_normalized_frame(X: np.ndarray) -> np.ndarray:
    """E1: dữ liệu đã chuẩn hóa -> x' = -x."""
    hands = X.reshape(X.shape[0], -1, 21, 3).copy()
    hands[..., 0] = -hands[..., 0]
    if hands.shape[1] == 2:
        hands = hands[:, ::-1].copy()
    return hands.reshape(X.shape).astype(np.float32)


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


def flip_normalized(X: np.ndarray) -> np.ndarray:
    """Lật ngang dữ liệu ĐÃ chuẩn hóa: x' = -x (giống horizontal_flip_sequence khi normalized=True).
    Khung mất tay đang là 0 nên vẫn giữ 0. Với 2 tay thì đổi slot trái/phải."""
    samples, steps, dims = X.shape
    hands = X.reshape(samples, steps, dims // 63, 21, 3).copy()
    hands[..., 0] = -hands[..., 0]
    if hands.shape[2] == 2:
        hands = hands[:, :, ::-1].copy()
    return hands.reshape(samples, steps, dims)


# ---------------------------------------------------------------------------
# Predictor: ensemble + TTA + từ chối dự đoán
# ---------------------------------------------------------------------------
class ExperimentPredictor:
    """Một thực nghiệm (E0/E1/E2/E3) gồm 1 hoặc nhiều checkpoint cùng loại (vd 4 fold).
    Nhận cửa sổ thô (45, D) và tự làm đúng tiền xử lý của thực nghiệm đó."""

    def __init__(self, name, checkpoint_paths, use_tta=True):
        self.name, self.use_tta = name, use_tta
        self.models, self.checkpoints = [], []
        self.classes = self.kind = None
        for path in checkpoint_paths:
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(f"[{name}] Không tìm thấy checkpoint: {path}")
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            kind, models = self._build(ckpt, path)
            classes = np.asarray(ckpt["classes"])
            if self.kind is None:
                self.kind, self.classes = kind, classes
                self.input_dim = int(ckpt.get("input_dim") or models[0].net[0].in_features)
                self.frame_index = int(ckpt.get("frame_index", 22))
                self.use_normalization = bool(ckpt.get("use_hand_normalization", True))
            elif kind != self.kind or not np.array_equal(classes, self.classes):
                raise ValueError(f"[{name}] {path.name}: khác loại model hoặc danh sách lớp")
            self.models.extend(m.to(DEVICE).eval() for m in models)
            self.checkpoints.append((path, ckpt))
        if not self.models:
            raise FileNotFoundError(f"[{name}] Không có checkpoint")

    @staticmethod
    def _build(ckpt, path):
        name = ckpt.get("model_name")
        if name == "DomainInvariantMotionBiLSTM":
            models = []
            for state in ckpt["model_state_dicts"]:
                m = E3DomainInvariantMotionBiLSTM(
                    int(ckpt["input_dim"]), int(ckpt["proj_dim"]), int(ckpt["hidden_dim"]),
                    int(ckpt["num_layers"]), len(ckpt["classes"]), len(ckpt["domain_map"]),
                    float(ckpt["dropout"]))
                m.load_state_dict(state, strict=True)
                models.append(m)
            return "E3+", models
        if name == "AttentionLSTM":
            models = []
            for state in ckpt["model_state_dicts"]:
                bidir = any("reverse" in k for k in state)
                m = AttentionLSTM(int(ckpt["input_dim"]), int(ckpt["proj_dim"]), int(ckpt["hidden_dim"]),
                                  int(ckpt["num_layers"]), len(ckpt["classes"]), float(ckpt["dropout"]), bidir)
                m.load_state_dict(state, strict=True)
                models.append(m)
            return ("BiLSTM" if bidir else "LSTM"), models
        if "state_dicts" in ckpt:
            # E0+ (notebook ablation): danh sách state_dict MLP, có thể kèm z-score, tọa độ thô.
            models = []
            for state in ckpt["state_dicts"]:
                use_z = "mean" in state
                m = MLPClassifier(state["net.0.weight"].shape[1], len(ckpt["classes"]), use_z)
                m.load_state_dict(state, strict=True)
                models.append(m)
            return "MLP-raw", models
        if "model_state_dict" in ckpt:
            # E1 lưu với tên "network.*", E0 lưu "net.*" -> đổi về "net.*".
            state = {k.replace("network.", "net.", 1): v for k, v in ckpt["model_state_dict"].items()}
            m = MLPClassifier(int(ckpt["input_dim"]), len(ckpt["classes"]))
            m.load_state_dict(state, strict=True)
            normalized = ckpt.get("experiment") == "E1" or bool(ckpt.get("coordinate_normalization", False))
            return ("MLP-norm" if normalized else "MLP-raw"), [m]
        raise ValueError(f"{path.name}: không nhận dạng được loại checkpoint")

    @property
    def label(self):
        return f"{self.name} ({self.kind}, {len(self.models)} model)"


    def _views(self, window):
        """Trả về batch các view (gốc + lật nếu TTA) theo đúng tiền xử lý của thực nghiệm."""
        if self.kind in ("MLP-raw", "MLP-norm"):
            frame = window[self.frame_index][None].astype(np.float32)
            if self.kind == "MLP-norm":
                frame = normalize_frame_e1(frame)
                flipped = flip_normalized_frame(frame)
            else:
                flipped = flip_raw_frame(frame)
            return np.concatenate([frame, flipped]) if self.use_tta else frame
        seq = np.asarray(window, dtype=np.float32)[None]
        if self.use_normalization:
            seq = normalize_hand_sequence_keep_motion(seq)
        return np.concatenate([seq, flip_normalized(seq)]) if self.use_tta else seq

    def predict(self, window: np.ndarray):
        tensor = torch.from_numpy(self._views(window)).to(DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            logits = torch.stack([m(tensor) for m in self.models])      # (M, V, C)
            probs = torch.softmax(logits, dim=-1).mean(dim=(0, 1))
            energy = float(-torch.logsumexp(logits.mean(dim=(0, 1)), dim=0))
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        top = torch.topk(probs, k=min(2, probs.numel()))
        p1 = float(top.values[0])
        return {"label": str(self.classes[int(top.indices[0])]), "prob": p1,
                "margin": p1 - (float(top.values[1]) if probs.numel() > 1 else 0.0),
                "energy": energy, "latency_ms": latency_ms}


def load_mode(models_dir: Path, mode: str, use_tta: bool, baseline=None):
    """Nạp các thực nghiệm cho một chế độ.
    baseline: dict tên -> predictor của chế độ baseline, dùng làm dự phòng cho chế độ mở rộng.
    Trả về list (tên hiển thị, predictor)."""
    panels = []
    for name in EXPERIMENTS:
        paths = sorted((models_dir / mode).glob(f"{name}*.pt"))
        if paths:
            display = name if mode == "baseline" else f"{name}+"
            panels.append((display, ExperimentPredictor(display, paths, use_tta)))
        elif baseline and name in baseline:
            panels.append((f"{name} (base)", baseline[name]))      # không có bản mở rộng -> dùng baseline
        elif mode == "baseline":
            print(f"[{name}] không có checkpoint baseline -> bỏ qua")
    return panels


class Decision:
    """Từ chối dự đoán + ổn định theo thời gian.

    Một dự đoán được CHẤP NHẬN khi: prob >= min_prob, margin >= min_margin,
    và (nếu bật) energy <= max_energy. Nhãn chỉ được HIỂN THỊ khi `stable_k`
    dự đoán chấp nhận liên tiếp cho cùng một nhãn.
    """

    def __init__(self, min_prob, min_margin, max_energy, stable_k):
        self.min_prob, self.min_margin, self.max_energy = min_prob, min_margin, max_energy
        self.history = deque(maxlen=stable_k)
        self.stable_k = stable_k
        self.shown = None

    def reset(self):
        self.history.clear()
        self.shown = None

    def accept(self, result):
        if result["prob"] < self.min_prob:
            return False, "prob thấp"
        if result["margin"] < self.min_margin:
            return False, "margin thấp"
        if self.max_energy is not None and result["energy"] > self.max_energy:
            return False, "energy cao"
        return True, ""

    def update(self, result):
        ok, reason = self.accept(result)
        self.history.append(result["label"] if ok else None)
        if len(self.history) == self.stable_k and self.history[0] is not None \
                and all(h == self.history[0] for h in self.history):
            self.shown = self.history[0]
        elif not ok:
            self.shown = None
        return ok, reason


# ---------------------------------------------------------------------------
# Khóa bàn tay
# ---------------------------------------------------------------------------
class HandLock:
    """Máy trạng thái IDLE / LOCKED.

    - IDLE -> LOCKED: khung hình có ĐÚNG 1 tay liên tục `acquire_frames` frame.
    - LOCKED: mỗi frame chọn tay có cổ tay gần vị trí cũ nhất (trong bán kính `max_jump`,
      đã hiệu chỉnh tỉ lệ khung hình). Các tay khác bị bỏ qua.
    - LOCKED -> IDLE: tay đã khóa biến mất quá `grace_frames` frame.
    Hệ quả: muốn đổi tay, rút hết tay ra (hoặc rút tay đang khóa), chỉ chừa tay muốn dùng.
    """

    def __init__(self, acquire_frames=8, grace_frames=10, max_jump=0.15, aspect=16 / 9):
        self.acquire_frames, self.grace_frames = acquire_frames, grace_frames
        self.max_jump, self.aspect = max_jump, aspect
        self.reset()

    def reset(self):
        self.locked_wrist = None
        self.locked_label = None
        self.missing = 0
        self.single_count = 0

    @property
    def is_locked(self):
        return self.locked_wrist is not None

    def _dist(self, a, b):
        return float(np.hypot((a[0] - b[0]) * self.aspect, a[1] - b[1]))

    def update(self, hands):
        """hands: list of (points (21,3) chuẩn hóa ảnh, handedness label).
        Trả về (index tay được chọn hoặc None, sự kiện: 'locked' | 'unlocked' | None)."""
        event = None
        if not self.is_locked:
            if len(hands) == 1:
                self.single_count += 1
                if self.single_count >= self.acquire_frames:
                    self.locked_wrist = hands[0][0][0, :2].copy()
                    self.locked_label = hands[0][1]
                    self.missing = 0
                    return 0, "locked"
            else:
                self.single_count = 0
            return None, event

        best, best_d = None, None
        for i, (points, label) in enumerate(hands):
            d = self._dist(points[0, :2], self.locked_wrist)
            # Ưu tiên cùng nhãn trái/phải; tay khác nhãn phải gần hơn hẳn mới được nhận.
            if label != self.locked_label:
                d *= 1.5
            if d <= self.max_jump and (best_d is None or d < best_d):
                best, best_d = i, d
        if best is None:
            self.missing += 1
            if self.missing > self.grace_frames:
                self.reset()
                event = "unlocked"
            return None, event
        self.locked_wrist = hands[best][0][0, :2].copy()
        self.missing = 0
        return best, event


# ---------------------------------------------------------------------------
# MediaPipe + camera
# ---------------------------------------------------------------------------
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


def create_landmarker(model_path: Path, max_hands: int):
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=max_hands,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def detect_hands(landmarker, frame_bgr, timestamp_ms: int):
    """Trả về list (points (21,3) float32, handedness label) cho mọi tay phát hiện được."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection = landmarker.detect_for_video(image, timestamp_ms)
    hands = []
    for points, handedness in zip(detection.hand_landmarks, detection.handedness):
        arr = np.asarray([[p.x, p.y, p.z] for p in points], dtype=np.float32)
        hands.append((arr, handedness[0].category_name))
    return hands


def to_feature(hands, selected, input_dim):
    """Chuyển tay được chọn thành vector đặc trưng đúng định dạng lúc train."""
    if input_dim == 63:
        return hands[selected][0].reshape(63)
    slots = {"Left": np.zeros((21, 3), np.float32), "Right": np.zeros((21, 3), np.float32)}
    for points, label in hands:
        slots[label] = points
    return np.concatenate((slots["Left"].reshape(-1), slots["Right"].reshape(-1)))


def open_camera(camera_index: int, camera_url):
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


def draw_hand_box(frame, points, color, text=None):
    h, w = frame.shape[:2]
    xs, ys = points[:, 0] * w, points[:, 1] * h
    x1, y1, x2, y2 = int(xs.min()) - 10, int(ys.min()) - 10, int(xs.max()) + 10, int(ys.max()) + 10
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if text:
        cv2.putText(frame, text, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def put(frame, text, y, color=WHITE, scale=0.7):
    # Viền đen để chữ dễ đọc trên mọi nền. OpenCV không vẽ được dấu tiếng Việt nên chữ trên hình dùng ASCII.
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


# ---------------------------------------------------------------------------
# Chế độ chạy
# ---------------------------------------------------------------------------
def run_self_test(modes, sample_path):
    if sample_path is None:
        candidates = sorted((ROOT / "landmarks" / "raw").glob("*/*.npy"))
        if not candidates:
            raise FileNotFoundError("Không có landmark .npy để self-test")
        sample_path = candidates[0]
    window = np.load(sample_path).astype(np.float32)
    print(f"Sample: {sample_path}")
    for mode, (panels, decisions) in modes.items():
        print(f"== Chế độ {mode.upper()} ==")
        for (display, panel), decision in zip(panels, decisions):
            r = panel.predict(window)
            ok, reason = decision.accept(r)
            print(f"  {display:<10} {panel.kind:<9} {len(panel.models)} model -> {r['label']:<8} "
                  f"p={r['prob'] * 100:5.1f}% m={r['margin'] * 100:5.1f}% E={r['energy']:7.3f} "
                  f"{r['latency_ms']:6.2f}ms {'OK' if ok else 'TỪ CHỐI: ' + reason}")
    print("Lưu ý: file .npy trong dữ liệu train không phải phép đánh giá hợp lệ.")


def run_camera(modes, args):
    mode = args.mode if args.mode in modes else "baseline"
    panels, decisions = modes[mode]
    input_dims = {p.input_dim for ps, _ in modes.values() for _, p in ps}
    if len(input_dims) != 1:
        raise ValueError(f"Các thực nghiệm phải cùng input_dim, nhận được {input_dims}")
    input_dim = input_dims.pop()
    landmarker = create_landmarker(ensure_landmarker(args.landmarker), args.max_hands)
    capture = open_camera(args.camera, args.camera_url)
    ok, frame = capture.read()
    aspect = frame.shape[1] / frame.shape[0] if ok else 16 / 9
    lock = HandLock(args.acquire_frames, args.grace_frames, args.max_jump, aspect)
    buffer = deque(maxlen=SEQ_LEN)
    frames_since_pred = 0
    results = [None] * 8
    last_timestamp = -1
    seen_text = {"yes": "Signer IS in training data",
                 "no": "Signer NOT in training data",
                 "unknown": ""}[args.signer_seen]
    print("Q: thoát | R: xóa buffer + mở khóa tay | M: đổi baseline <-> mở rộng")

    def reset_all():
        nonlocal frames_since_pred
        buffer.clear()
        frames_since_pred = 0
        for _, ds in modes.values():
            for d in ds:
                d.reset()
        for i in range(len(results)):
            results[i] = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Không đọc được frame từ camera.")
                break
            timestamp = max(last_timestamp + 1, int(time.monotonic() * 1000))
            last_timestamp = timestamp
            hands = detect_hands(landmarker, frame, timestamp)

            if input_dim == 63:
                selected, event = lock.update(hands)
                if event in ("locked", "unlocked"):
                    reset_all()          # đổi tay -> bắt đầu cửa sổ mới
            else:
                selected = 0 if hands else None

            for i, (points, label) in enumerate(hands):
                if i == selected:
                    draw_hand_box(frame, points, GREEN, f"LOCKED ({label})")
                else:
                    draw_hand_box(frame, points, GRAY, "ignored")

            if selected is not None:
                buffer.append(to_feature(hands, selected, input_dim))
                frames_since_pred += 1
                if len(buffer) == SEQ_LEN and frames_since_pred >= args.stride:
                    frames_since_pred = 0
                    window = np.stack(buffer)
                    for i, ((_, panel), decision) in enumerate(zip(panels, decisions)):
                        results[i] = panel.predict(window)
                        decision.update(results[i])
            # Tay khóa tạm mất (trong grace) -> giữ buffer, không thêm frame.

            # ----- Hiển thị -----
            if input_dim == 63:
                state = "LOCKED" if lock.is_locked else (
                    "No hand" if not hands else
                    f"Locking... {lock.single_count}/{args.acquire_frames}" if len(hands) == 1
                    else f"{len(hands)} hands: keep only ONE hand to lock")
                put(frame, f"Hand: {state}", 28, GREEN if lock.is_locked else ORANGE)
            put(frame, f"Frames: {len(buffer)}/{SEQ_LEN}   Mode: {mode.upper()} (M to switch)", 56, ORANGE, 0.6)

            # Bảng 4 thực nghiệm: nền tối bán trong suốt để chữ dễ đọc
            top = 72
            height = 34 * len(panels) + 10
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, top), (min(frame.shape[1] - 5, 620), top + height), (0, 0, 0), -1)
            frame[:] = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
            for i, ((display, panel), decision, r) in enumerate(zip(panels, decisions, results)):
                y = top + 28 + 34 * i
                if decision.shown is not None:
                    text, color = f"{display}: {decision.shown}", GREEN
                elif r is not None:
                    text, color = f"{display}: ?", RED
                else:
                    text, color = f"{display}: ...", GRAY
                if r is not None:
                    text += f"   ({r['label']} {r['prob'] * 100:.0f}%, {r['latency_ms']:.1f}ms)"
                put(frame, text, y, color, 0.7)
            if seen_text:
                put(frame, seen_text, frame.shape[0] - 15, WHITE, 0.55)

            cv2.imshow("Vietnamese Sign Language - E0 | E1 | E2 | E3", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                lock.reset()
                reset_all()
            if key == ord("m") and len(modes) > 1:
                mode = "extended" if mode == "baseline" else "baseline"
                panels, decisions = modes[mode]
                for d in decisions:
                    d.reset()
                for i in range(len(results)):
                    results[i] = None
                frames_since_pred = args.stride     # dự đoán ngay ở frame kế tiếp, giữ buffer
    finally:
        capture.release()
        landmarker.close()
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Demo realtime so sánh E0/E1/E2/E3 (khóa tay, từ chối dự đoán)")
    p.add_argument("--models-dir", type=Path, default=ROOT / "models",
                   help="Thư mục chứa baseline/ và extended/")
    p.add_argument("--mode", choices=("baseline", "extended"), default="baseline",
                   help="Chế độ khi khởi động; nhấn M để đổi trong lúc chạy")
    p.add_argument("--landmarker", type=Path, default=DEFAULT_LANDMARKER)
    p.add_argument("--camera", type=int, default=0,
                   help="Index webcam; camera điện thoại thường là 1 hoặc 2")
    p.add_argument("--camera-url", help="URL MJPEG/RTSP nếu dùng app IP camera")
    p.add_argument("--no-tta", action="store_true", help="Tắt TTA lật ngang")
    # Từ chối dự đoán
    p.add_argument("--min-prob", type=float, default=0.60, help="Xác suất tối thiểu của lớp cao nhất")
    p.add_argument("--min-margin", type=float, default=0.15, help="Chênh lệch tối thiểu top1 - top2")
    p.add_argument("--max-energy", type=float, default=None,
                   help="Ngưỡng energy (tùy chọn); xem giá trị E in ra khi chạy --self-test")
    p.add_argument("--stable-k", type=int, default=3, help="Số dự đoán liên tiếp giống nhau để hiện nhãn")
    p.add_argument("--stride", type=int, default=3, help="Dự đoán mỗi N frame")
    # Khóa tay
    p.add_argument("--max-hands", type=int, default=4, help="Số tay tối đa MediaPipe phát hiện")
    p.add_argument("--acquire-frames", type=int, default=8, help="Số frame chỉ có 1 tay để khóa")
    p.add_argument("--grace-frames", type=int, default=10, help="Số frame mất tay trước khi mở khóa")
    p.add_argument("--max-jump", type=float, default=0.15, help="Bán kính bám tay giữa 2 frame")
    # Khác
    p.add_argument("--signer-seen", choices=("yes", "no", "unknown"), default="unknown",
                   help="Hiện trên màn hình người demo có trong dữ liệu train hay không")
    p.add_argument("--self-test", action="store_true", help="Inference một file .npy, không mở camera")
    p.add_argument("--sample", type=Path, help="File .npy dùng với --self-test")
    p.add_argument("--download-landmarker", action="store_true", help="Chỉ tải hand_landmarker.task rồi thoát")
    return p.parse_args()


def main():
    configure_console()
    args = parse_args()
    if args.download_landmarker:
        ensure_landmarker(args.landmarker)
        return
    use_tta = not args.no_tta
    baseline_panels = load_mode(args.models_dir, "baseline", use_tta)
    if not baseline_panels:
        raise SystemExit(f"Không có checkpoint nào trong {args.models_dir / 'baseline'}")
    baseline_map = {d: p for d, p in baseline_panels}
    extended_panels = load_mode(args.models_dir, "extended", use_tta, baseline_map)
    has_extended = any(not d.endswith("(base)") for d, _ in extended_panels)

    all_panels = [p for _, p in baseline_panels + extended_panels]
    classes = all_panels[0].classes
    for p in all_panels[1:]:
        if not np.array_equal(p.classes, classes):
            raise SystemExit(f"{p.name}: danh sách lớp khác {all_panels[0].name}")

    def make(panels):
        return panels, [Decision(args.min_prob, args.min_margin, args.max_energy, args.stable_k) for _ in panels]

    modes = {"baseline": make(baseline_panels)}
    if has_extended:
        modes["extended"] = make(extended_panels)
    else:
        print("Không có checkpoint trong extended/ -> chỉ có chế độ baseline")

    for mode, (panels, _) in modes.items():
        print(f"== {mode.upper()} ==")
        for display, p in panels:
            folds = [ck.get("outer_test_person", ck.get("test_person", "?")) for _, ck in p.checkpoints]
            print(f"  {display:<10} {p.kind:<9} {len(p.models)} model, test fold={folds}")
    print(f"TTA={'on' if use_tta else 'off'}, device={DEVICE}")
    if args.self_test:
        run_self_test(modes, args.sample)
    else:
        run_camera(modes, args)


if __name__ == "__main__":
    main()