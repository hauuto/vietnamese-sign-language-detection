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
  4. Từ chối dự đoán: ngưỡng xác suất + khoảng cách top1-top2 + đồng thuận giữa các fold/view
     (+ energy tùy chọn), kèm ổn định theo thời gian (K lần dự đoán liên tiếp giống nhau).
     Ngưỡng chỉnh trực tiếp khi chạy bằng phím 1/2, 3/4, 5/6.
  6. Ổn định landmark: ngưỡng tin cậy MediaPipe cao hơn (0.7/0.7/0.6), tối đa 2 tay,
     lọc One Euro (--smooth display|all|off, phím S).
  5. Giao diện 3 tầng: (1) kết quả chính của E3 (ký hiệu, độ tin cậy, chấp nhận/từ chối)
     và chuỗi đã nhận; (2) so sánh nhanh 4 thực nghiệm; (3) chi tiết kỹ thuật (phím D).
     Chữ có dấu cần Pillow (pip install pillow).

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
        top = torch.topk(probs, k=min(3, probs.numel()))
        idx = [int(i) for i in top.indices]
        vals = [float(v) for v in top.values]
        # Đồng thuận: tỉ lệ cặp (fold, view TTA) có argmax trùng lớp top-1 của ensemble.
        votes = logits.argmax(dim=-1).flatten()
        agree = float((votes == idx[0]).float().mean())
        return {"label": str(self.classes[idx[0]]), "prob": vals[0],
                "margin": vals[0] - (vals[1] if len(vals) > 1 else 0.0),
                "agree": agree, "n_votes": int(votes.numel()),
                "top": [(str(self.classes[i]), v) for i, v in zip(idx, vals)],
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

    Một dự đoán được CHẤP NHẬN khi thỏa đồng thời:
      - prob   >= min_prob   : lớp top-1 đủ chắc;
      - margin >= min_margin : top-1 tách xa top-2 (loại động tác "na ná" giữa 2 chữ);
      - agree  >= min_agree  : đa số cặp (fold, view TTA) cùng bỏ phiếu cho top-1;
      - energy <= max_energy : tùy chọn.
    Nhãn chỉ HIỂN THỊ khi `stable_k` dự đoán được chấp nhận liên tiếp trùng nhau.
    Mô hình không được huấn luyện với lớp "không phải ký hiệu", nên đây là heuristic
    giảm nhận nhầm chứ không loại bỏ hoàn toàn.
    """

    def __init__(self, thresholds, stable_k):
        self.t = thresholds          # dict dùng chung, chỉnh được lúc chạy
        self.history = deque(maxlen=stable_k)
        self.stable_k = stable_k
        self.shown = None
        self.last_ok, self.last_reason = False, ""

    def reset(self):
        self.history.clear()
        self.shown = None
        self.last_ok, self.last_reason = False, ""

    def accept(self, result):
        t = self.t
        if result["prob"] < t["min_prob"]:
            return False, "prob"
        if result["margin"] < t["min_margin"]:
            return False, "margin"
        if result.get("agree", 1.0) < t["min_agree"]:
            return False, "agree"
        if t.get("max_energy") is not None and result["energy"] > t["max_energy"]:
            return False, "energy"
        return True, ""

    def update(self, result):
        """Cập nhật; trả về nhãn vừa được chốt (lần đầu hiển thị) hoặc None."""
        ok, reason = self.accept(result)
        self.last_ok, self.last_reason = ok, reason
        self.history.append(result["label"] if ok else None)
        before = self.shown
        if len(self.history) == self.stable_k and self.history[0] is not None \
                and all(h == self.history[0] for h in self.history):
            self.shown = self.history[0]
        elif not ok:
            self.shown = None
        return self.shown if (self.shown is not None and self.shown != before) else None


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


def create_landmarker(model_path: Path, max_hands: int, det_conf=0.5, presence_conf=0.5, track_conf=0.5):
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=det_conf,
        min_hand_presence_confidence=presence_conf,
        min_tracking_confidence=track_conf,
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


class LandmarkSmoother:
    """Bộ lọc One Euro (Casiez và cộng sự, 2012) cho 21x3 landmark.
    Tay đứng yên -> lọc mạnh (bớt rung); tay di chuyển nhanh -> lọc nhẹ (bớt trễ)."""

    def __init__(self, min_cutoff=1.0, beta=10.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = self.dx_prev = self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, np.float32)
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        a = self._alpha(self.min_cutoff + self.beta * np.abs(dx_hat), dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat.astype(np.float32)


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


# ---------------------------------------------------------------------------
# Giao diện
#   Tầng 1 (người xem): ký hiệu hiện tại + độ tin cậy + chấp nhận/từ chối, chuỗi đã nhận.
#   Tầng 2 (người vận hành): so sánh nhanh 4 thực nghiệm.
#   Tầng 3 (kỹ thuật): ngưỡng, margin, phiếu, latency — ẩn, bật bằng phím D.
# ---------------------------------------------------------------------------
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:          # không có Pillow -> chữ ASCII bằng OpenCV
    HAS_PIL = False

# Màu ngữ nghĩa (BGR). Màu chỉ mang nghĩa trạng thái, không dùng để phân biệt model.
C_BG = (24, 22, 20)
C_PANEL = (36, 33, 30)
C_ROW_HI = (52, 47, 43)
C_LINE = (62, 57, 53)
C_TEXT = (240, 238, 235)
C_MUTED = (160, 150, 145)
C_OK = (94, 197, 34)          # chấp nhận
C_WAIT = (11, 158, 245)       # chưa chắc / đang xác nhận
C_REJECT = (68, 68, 239)      # từ chối
C_ACCENT = (246, 130, 59)     # hệ thống / trung tính
C_TRACK = (70, 64, 60)

PRETTY = {"aa": "Â", "aw": "Ă", "dd": "Đ", "ee": "Ê", "oo": "Ô", "ow": "Ơ", "uw": "Ư",
          "tone_f": "huyền", "tone_j": "nặng", "tone_r": "hỏi", "tone_s": "sắc", "tone_x": "ngã"}
KIND_NAME = {"MLP-raw": "MLP · tọa độ thô", "MLP-norm": "MLP · chuẩn hóa", "LSTM": "LSTM",
             "BiLSTM": "Bi-LSTM", "E3+": "Bi-LSTM mở rộng"}
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def pretty(label):
    return PRETTY.get(label, label.upper())


def _ascii(text):
    import unicodedata
    text = text.replace("Đ", "D").replace("đ", "d")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


class TextLayer:
    """Gom chữ trong một frame rồi vẽ một lượt bằng Pillow (OpenCV không vẽ được dấu tiếng Việt)."""
    FONT_CANDIDATES = {
        False: ["segoeui.ttf", "arial.ttf", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
        True: ["segoeuib.ttf", "arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
               "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    }

    def __init__(self):
        self.items = []
        self.cache = {}
        self.enabled = HAS_PIL and self._font(14, False) is not None

    def _font(self, size, bold):
        key = (size, bold)
        if key not in self.cache:
            self.cache[key] = None
            for name in self.FONT_CANDIDATES[bold]:
                try:
                    self.cache[key] = ImageFont.truetype(name, size)
                    break
                except OSError:
                    continue
        return self.cache[key]

    def add(self, text, xy, size=16, color=C_TEXT, bold=False, anchor="la"):
        self.items.append((str(text), (int(xy[0]), int(xy[1])), size, color, bold, anchor))

    def width(self, text, size=16, bold=False):
        if self.enabled:
            return self._font(size, bold).getlength(str(text))
        return cv2.getTextSize(_ascii(str(text)), cv2.FONT_HERSHEY_SIMPLEX, size / 30, 1)[0][0]

    def flush(self, img):
        if not self.items:
            return
        if self.enabled:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            for text, xy, size, color, bold, anchor in self.items:
                draw.text(xy, text, font=self._font(size, bold), fill=color[::-1], anchor=anchor)
            img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        else:
            for text, (x, y), size, color, bold, anchor in self.items:
                text = _ascii(text)
                scale = size / 30
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2 if bold else 1)
                if anchor[0] == "m":
                    x -= tw // 2
                elif anchor[0] == "r":
                    x -= tw
                y += th if anchor[1] in "at" else (th // 2 if anchor[1] == "m" else 0)
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if bold else 1,
                            cv2.LINE_AA)
        self.items.clear()


def rounded_rect(img, p1, p2, color, radius=10):
    x1, y1 = p1
    x2, y2 = p2
    r = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def progress_bar(img, x, y, w, h, frac, color):
    rounded_rect(img, (x, y), (x + w, y + h), C_TRACK, h // 2)
    fill = int(w * max(0.0, min(1.0, frac)))
    if fill >= h:
        rounded_rect(img, (x, y), (x + fill, y + h), color, h // 2)


def panel_state(decision, result, stable_k):
    """Trạng thái hiển thị của một thực nghiệm: (nhãn hoặc None, màu, trạng thái ngắn)."""
    if result is None:
        return None, C_MUTED, "wait"
    if decision.shown is not None:
        return decision.shown, C_OK, "ok"
    if decision.last_ok:
        return result["label"], C_WAIT, "confirm"
    return result["label"], C_REJECT, "reject"


class DemoUI:
    CAM_H = 540
    SIDE_W = 430
    SEQ_H = 104
    FOOT_H = 40

    def __init__(self, thresholds, stable_k, signer_text=""):
        self.t = thresholds
        self.stable_k = stable_k
        self.signer_text = signer_text
        self.show_details = False
        self.text = TextLayer()
        if not self.text.enabled:
            print("Cảnh báo: không có Pillow/font -> chữ trên màn hình sẽ không dấu (pip install pillow)")

    # ---- camera -------------------------------------------------------------
    def _camera(self, frame, hands, selected, cam_state, fps, latency_ms):
        T = self.text
        h0, w0 = frame.shape[:2]
        H = self.CAM_H
        W = int(round(w0 * H / h0))
        cam = cv2.resize(frame, (W, H))
        if not hands:                               # khung gợi ý đặt tay, rất mờ
            gw, gh = int(W * 0.32), int(H * 0.5)
            gx, gy = (W - gw) // 2, (H - gh) // 2
            overlay = cam.copy()
            for k in range(0, gw, 18):
                cv2.line(overlay, (gx + k, gy), (gx + min(k + 9, gw), gy), C_TEXT, 2)
                cv2.line(overlay, (gx + k, gy + gh), (gx + min(k + 9, gw), gy + gh), C_TEXT, 2)
            for k in range(0, gh, 18):
                cv2.line(overlay, (gx, gy + k), (gx, gy + min(k + 9, gh)), C_TEXT, 2)
                cv2.line(overlay, (gx + gw, gy + k), (gx + gw, gy + min(k + 9, gh)), C_TEXT, 2)
            cam[:] = cv2.addWeighted(overlay, 0.35, cam, 0.65, 0)
            T.add("Đặt một bàn tay vào đây", (W // 2, gy + gh // 2), 16, C_MUTED, anchor="mm")
        for i, (points, label) in enumerate(hands):
            xs, ys = points[:, 0] * W, points[:, 1] * H
            x1, y1 = int(xs.min()) - 16, int(ys.min()) - 16
            x2, y2 = int(xs.max()) + 16, int(ys.max()) + 16
            if i == selected:
                for a, b in HAND_EDGES:
                    cv2.line(cam, (int(xs[a]), int(ys[a])), (int(xs[b]), int(ys[b])), C_OK, 2, cv2.LINE_AA)
                for px, py in zip(xs, ys):
                    cv2.circle(cam, (int(px), int(py)), 3, C_TEXT, -1, cv2.LINE_AA)
                cv2.rectangle(cam, (x1, y1), (x2, y2), C_OK, 1, cv2.LINE_AA)
                T.add("tay trái" if label == "Left" else "tay phải", (x1, y1 - 6), 13, C_OK, anchor="lb")
            else:
                cv2.rectangle(cam, (x1, y1), (x2, y2), C_MUTED, 1, cv2.LINE_AA)
                T.add("bỏ qua", (x1, y1 - 6), 13, C_MUTED, anchor="lb")

        # trạng thái camera: luôn cùng một vị trí
        title, detail, color = cam_state
        w_chip = int(max(T.width(title, 15, True), T.width(detail, 12))) + 40
        overlay = cam.copy()
        rounded_rect(overlay, (14, 14), (14 + w_chip, 62), (0, 0, 0), 12)
        cam[:] = cv2.addWeighted(overlay, 0.65, cam, 0.35, 0)
        cv2.circle(cam, (31, 30), 6, color, -1, cv2.LINE_AA)
        T.add(title, (44, 30), 15, C_TEXT, True, "lm")
        T.add(detail, (44, 50), 12, C_MUTED, anchor="lm")

        info = f"● LIVE  {fps:4.1f} FPS" + (f"  ·  {latency_ms:.1f} ms" if latency_ms is not None else "")
        w_info = int(T.width(info, 13)) + 20
        overlay = cam.copy()
        rounded_rect(overlay, (W - 14 - w_info, 14), (W - 14, 40), (0, 0, 0), 10)
        cam[:] = cv2.addWeighted(overlay, 0.65, cam, 0.35, 0)
        T.add(info, (W - 24, 27), 13, C_TEXT, anchor="rm")
        return cam

    # ---- cột phải -----------------------------------------------------------
    def _reason(self, decision, r):
        t = self.t
        code = decision.last_reason
        if code == "prob":
            return f"Độ tin cậy {r['prob'] * 100:.0f}% thấp hơn ngưỡng {t['min_prob'] * 100:.0f}%"
        if code == "margin":
            l2, p2 = r["top"][1]
            return (f"Lẫn với {pretty(l2)} ({p2 * 100:.0f}%) · cách biệt {r['margin'] * 100:.0f}%"
                    f" < {t['min_margin'] * 100:.0f}%")
        if code == "agree":
            n = r.get("n_votes", 1)
            return f"Các fold bất đồng · {round(r['agree'] * n)}/{n} phiếu (cần ≥ {t['min_agree'] * n:.0f})"
        if code == "energy":
            return "Mẫu khác xa dữ liệu huấn luyện (energy cao)"
        return ""

    def _main_result(self, img, x, y, w, display, panel, decision, r):
        """Accept và reject dùng CÙNG một cấu trúc: ký hiệu -> dòng phụ -> trạng thái -> 5 dòng tiêu chí.
        Ở trạng thái từ chối, % chỉ là độ tin cậy của ỨNG VIÊN, không phải của quyết định."""
        T, t = self.text, self.t
        T.add(f"Kết quả chính · {display} ({KIND_NAME.get(panel.kind, panel.kind)})", (x, y), 13, C_MUTED)
        label, color, state = panel_state(decision, r, self.stable_k)
        cx, cy = x + w // 2, y + 66
        if r is None:
            T.add("…", (cx, cy), 60, C_MUTED, True, "mm")
            T.add("Chờ đủ 45 frame của tay đã khóa", (cx, cy + 62), 14, C_MUTED, anchor="mm")
            return

        cand = f"{pretty(r['label'])} · {r['prob'] * 100:.0f}%"
        if state == "reject":
            T.add("?", (cx, cy), 60, C_REJECT, True, "mm")
            T.add("KHÔNG XÁC ĐỊNH", (cx, cy + 50), 18, C_REJECT, True, "mm")
            T.add(f"ứng viên gần nhất  {cand}", (cx, cy + 76), 14, C_MUTED, anchor="mm")
            headline, h_color = "✕ Không nhận vào chuỗi", C_REJECT
        else:
            big = pretty(r["label"])
            T.add(big, (cx, cy), 80 if len(big) <= 2 else 46, color, True, "mm")
            T.add(f"{r['prob'] * 100:.0f}%", (cx, cy + 56), 26, color, True, "mm")
            if state == "ok":
                headline, h_color = "✓ Chấp nhận · đã thêm vào chuỗi", C_OK
            else:
                n = sum(1 for v in decision.history if v == r["label"])
                headline, h_color = f"… Đang xác nhận {n}/{self.stable_k} · giữ nguyên ký hiệu", C_WAIT
        hy = cy + 104
        T.add(headline, (cx, hy), 16, h_color, True, "mm")

        # 5 tiêu chí, luôn cùng vị trí; tiêu chí không đạt tô đỏ
        n = r.get("n_votes", 1)
        second = r["top"][1] if len(r["top"]) > 1 else ("—", 0.0)
        need_votes = int(np.ceil(t["min_agree"] * n - 1e-9))
        rows = [
            ("Ứng viên", cand, None),
            ("Gần nhất", f"{pretty(second[0])} · {second[1] * 100:.0f}%", None),
            ("Độ tin cậy", f"{r['prob'] * 100:.0f}%  /  cần ≥ {t['min_prob'] * 100:.0f}%", r["prob"] >= t["min_prob"]),
            ("Margin", f"{r['margin'] * 100:.0f}%  /  cần ≥ {t['min_margin'] * 100:.0f}%", r["margin"] >= t["min_margin"]),
            ("Đồng thuận fold", f"{round(r['agree'] * n)}/{n}  /  cần ≥ {need_votes}", r["agree"] >= t["min_agree"]),
        ]
        ry = hy + 22
        for k, v, passed in rows:
            T.add(k, (x, ry), 13, C_MUTED)
            vcolor = C_TEXT if passed is None else (C_OK if passed else C_REJECT)
            mark = "" if passed is None else ("  ✓" if passed else "  ✕")
            T.add(v + mark, (x + w, ry), 13, vcolor, passed is not None, "ra")
            ry += 18

    def _comparison(self, img, x, y, w, panels, decisions, results, primary_idx):
        T = self.text
        cv2.line(img, (x, y), (x + w, y), C_LINE, 1)
        T.add("SO SÁNH 4 THỰC NGHIỆM", (x, y + 12), 12, C_MUTED, True)
        pr = results[primary_idx] if primary_idx < len(results) else None
        if pr is not None:
            same = sum(1 for r in results[:len(panels)] if r is not None and r["label"] == pr["label"])
            T.add(f"{same}/{len(panels)} cùng dự đoán {pretty(pr['label'])}", (x + w, y + 12), 12, C_MUTED,
                  anchor="ra")
        row_h = 38
        for i, ((display, panel), decision, r) in enumerate(zip(panels, decisions, results)):
            ry = y + 36 + i * row_h
            if i == primary_idx:
                rounded_rect(img, (x - 8, ry - 4), (x + w + 8, ry + row_h - 6), C_ROW_HI, 8)
            T.add(display, (x, ry + 4), 15, C_TEXT, True)
            T.add(KIND_NAME.get(panel.kind, panel.kind), (x, ry + 22), 11, C_MUTED)
            label, color, state = panel_state(decision, r, self.stable_k)
            lx = x + 150
            T.add("—" if label is None else pretty(label), (lx, ry + 14), 16, color, True, "lm")
            if r is not None:
                bx = x + 210
                progress_bar(img, bx, ry + 10, w - 210 - 44, 7, r["prob"], color)
                T.add(f"{r['prob'] * 100:.0f}%", (x + w, ry + 14), 13, C_TEXT, anchor="rm")

    def _key_chip(self, img, x_right, y, key):
        rounded_rect(img, (x_right - 22, y - 1), (x_right, y + 17), C_TRACK, 4)
        self.text.add(key, (x_right - 11, y + 8), 11, C_TEXT, True, "mm")

    def _details(self, img, x, y, w, panels, decisions, results):
        T, t = self.text, self.t
        cv2.line(img, (x, y), (x + w, y), C_LINE, 1)
        T.add("CHI TIẾT KỸ THUẬT", (x, y + 12), 12, C_MUTED, True)
        self._key_chip(img, x + w, y + 10, "D")
        T.add("NGƯỠNG", (x, y + 36), 11, C_MUTED, True)
        items = [("Độ tin cậy", f"{t['min_prob'] * 100:.0f}%", "1/2"), ("Margin", f"{t['min_margin'] * 100:.0f}%", "3/4"),
                 ("Đồng thuận", f"{t['min_agree'] * 100:.0f}%", "5/6"), ("Giữ K lần", f"{self.stable_k}", "")]
        col = w // 4
        for i, (k, v, keys) in enumerate(items):
            cx = x + i * col
            T.add(k, (cx, y + 52), 11, C_MUTED)
            T.add(v, (cx, y + 67), 14, C_TEXT, True)
            if keys:
                T.add(keys, (cx + T.width(v, 14, True) + 6, y + 70), 10, C_MUTED)
        hy = y + 90
        T.add("THEO MÔ HÌNH", (x, hy), 11, C_MUTED, True)
        cols = [(0, ""), (62, "Độ tin cậy"), (140, "Margin"), (205, "Phiếu"), (w, "Độ trễ")]
        for dx, name in cols[1:]:
            T.add(name, (x + dx, hy + 16), 11, C_MUTED, anchor="ra" if dx == w else "la")
        for i, ((display, panel), r) in enumerate(zip(panels, results)):
            ry = hy + 30 + i * 16
            T.add(display, (x, ry), 12, C_TEXT, True)
            if r is None:
                continue
            n = r.get("n_votes", 1)
            vals = [f"{r['prob'] * 100:.0f}%", f"{r['margin'] * 100:.0f}%", f"{round(r['agree'] * n)}/{n}",
                    f"{r['latency_ms']:.1f} ms"]
            for (dx, _), v in zip(cols[1:], vals):
                T.add(v, (x + dx, ry), 12, C_TEXT, anchor="ra" if dx == w else "la")

    # ---- chuỗi + chân trang -------------------------------------------------
    def _sequence(self, img, y, width, transcript, primary_name, seq_state):
        T = self.text
        x = 20
        rounded_rect(img, (12, y + 8), (width - 12, y + self.SEQ_H - 4), C_PANEL, 12)
        T.add("CHUỖI NHẬN DẠNG", (x + 8, y + 22), 12, C_MUTED, True)
        tx = x + 8 + int(T.width("CHUỖI NHẬN DẠNG", 12, True)) + 10
        tw = int(T.width(primary_name, 11, True)) + 14
        rounded_rect(img, (tx, y + 20), (tx + tw, y + 38), C_TRACK, 9)
        T.add(primary_name, (tx + tw // 2, y + 29), 11, C_TEXT, True, "mm")
        text, color = seq_state
        cv2.circle(img, (width - 28 - int(T.width(text, 13)) - 12, y + 29), 5, color, -1, cv2.LINE_AA)
        T.add(text, (width - 28, y + 29), 13, C_TEXT, anchor="rm")
        items = [pretty(v) for v in transcript]
        cx, max_x = x + 8, width - 40
        visible = []
        for item in reversed(items):              # hiện các ký hiệu mới nhất vừa khung
            w_item = T.width(item, 40, True) + 26
            if cx + w_item > max_x:
                break
            visible.insert(0, item)
            cx += w_item
        if not visible:
            T.add("Chưa có ký hiệu nào được chấp nhận", (x + 8, y + 70), 16, C_MUTED, anchor="lm")
        cx = x + 8
        for j, item in enumerate(visible):
            last = j == len(visible) - 1
            T.add(item, (cx, y + 70), 40, C_OK if last else C_TEXT, True, "lm")
            cx += T.width(item, 40, True) + 26

    def _footer(self, img, y, width, has_extended, mode):
        T = self.text
        keys = [("Q", "Thoát"), ("R", "Reset"), ("C", "Xóa chuỗi")]
        if has_extended:
            keys.append(("M", "Bản mở rộng" if mode == "baseline" else "Bản baseline"))
        keys.append(("D", "Chi tiết kỹ thuật"))
        cx = 20
        for k, v in keys:
            rounded_rect(img, (cx, y + 10), (cx + 24, y + 32), C_TRACK, 5)
            T.add(k, (cx + 12, y + 21), 12, C_TEXT, True, "mm")
            T.add(v, (cx + 32, y + 21), 13, C_MUTED, anchor="lm")
            cx += 32 + int(T.width(v, 13)) + 26
        if self.signer_text:
            T.add(self.signer_text, (width - 20, y + 21), 12, C_MUTED, anchor="rm")

    # ---- tổng -----------------------------------------------------------------
    def render(self, frame, hands, selected, cam_state, seq_state, mode, has_extended,
               panels, decisions, results, transcript, primary_idx, fps):
        T = self.text
        pr = results[primary_idx] if panels and primary_idx < len(results) else None
        cam = self._camera(frame, hands, selected, cam_state, fps, pr["latency_ms"] if pr else None)
        H, W = cam.shape[:2]
        width = W + self.SIDE_W
        canvas = np.full((H + self.SEQ_H + self.FOOT_H, width, 3), C_BG, np.uint8)
        canvas[:H, :W] = cam

        x, pad = W + 24, 24
        inner = self.SIDE_W - 2 * pad
        T.add("NHẬN DẠNG VSL", (x, 18), 24, C_TEXT, True)
        chip = "CHẾ ĐỘ · " + ("MỞ RỘNG" if mode == "extended" else "BASELINE")
        cw = int(T.width(chip, 11, True)) + 18
        rounded_rect(canvas, (width - pad - cw, 22), (width - pad, 42), C_ACCENT if mode == "extended" else C_TRACK, 10)
        T.add(chip, (width - pad - cw // 2, 32), 11, C_TEXT, True, "mm")

        if panels:
            display, panel = panels[primary_idx]
            self._main_result(canvas, x, 58, inner, display, panel, decisions[primary_idx], pr)
            if self.show_details:
                self._details(canvas, x, 352, inner, panels, decisions, results)
            else:
                self._comparison(canvas, x, 352, inner, panels, decisions, results, primary_idx)

        self._sequence(canvas, H, width, transcript, panels[primary_idx][0] if panels else "-", seq_state)
        self._footer(canvas, H + self.SEQ_H, width, has_extended, mode)
        T.flush(canvas)
        return canvas


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
                  f"p={r['prob'] * 100:5.1f}% m={r['margin'] * 100:5.1f}% agree={r['agree'] * 100:5.1f}% E={r['energy']:7.3f} "
                  f"{r['latency_ms']:6.2f}ms {'OK' if ok else 'TỪ CHỐI: ' + reason}")
    print("Lưu ý: file .npy trong dữ liệu train không phải phép đánh giá hợp lệ.")


def screen_size():
    """Kích thước màn hình chính (Windows qua WinAPI; hệ khác mặc định 1366x768)."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1366, 768


def fit_to_window(canvas, window_name):
    """Co giãn giao diện theo kích thước cửa sổ hiện tại, GIỮ tỉ lệ (thêm viền nền nếu dư).
    OpenCV mặc định kéo giãn ảnh cho đầy cửa sổ nên chữ và khung bị méo khi đổi cỡ cửa sổ."""
    try:
        _, _, ww, wh = cv2.getWindowImageRect(window_name)
    except cv2.error:
        return canvas
    if ww <= 0 or wh <= 0:
        return canvas
    ch, cw = canvas.shape[:2]
    scale = min(ww / cw, wh / ch)
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(canvas, (nw, nh), interpolation=interp)
    out = np.full((wh, ww, 3), C_BG, np.uint8)
    x0, y0 = (ww - nw) // 2, (wh - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def run_camera(modes, args, thresholds):
    mode = args.mode if args.mode in modes else "baseline"
    panels, decisions = modes[mode]
    input_dims = {p.input_dim for ps, _ in modes.values() for _, p in ps}
    if len(input_dims) != 1:
        raise ValueError(f"Các thực nghiệm phải cùng input_dim, nhận được {input_dims}")
    input_dim = input_dims.pop()
    landmarker = create_landmarker(ensure_landmarker(args.landmarker), args.max_hands,
                                   args.det_conf, args.presence_conf, args.track_conf)
    smoother = LandmarkSmoother(args.smooth_min_cutoff, args.smooth_beta)
    print(f"Làm mượt landmark: {args.smooth} (phím S để đổi off/display/all)")
    capture = open_camera(args.camera, args.camera_url)
    ok, frame = capture.read()
    aspect = frame.shape[1] / frame.shape[0] if ok else 16 / 9
    lock = HandLock(args.acquire_frames, args.grace_frames, args.max_jump, aspect)
    buffer = deque(maxlen=SEQ_LEN)
    frames_since_pred = 0
    results = [None] * 8
    transcript = []
    last_timestamp = -1
    signer_text = {"yes": "người demo CÓ trong dữ liệu train",
                   "no": "người demo KHÔNG có trong dữ liệu train", "unknown": ""}[args.signer_seen]
    ui = DemoUI(thresholds, args.stable_k, signer_text)
    window_name = "VSL demo - Nhom 9"
    fps, last_frame_t = 0.0, time.perf_counter()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    fitted = False
    print("Q: thoát | R: reset | C: xóa chuỗi | M: đổi chế độ | D: chi tiết kỹ thuật | 1/2, 3/4, 5/6: chỉnh ngưỡng")

    def primary_index():
        # Thẻ dùng để ghép chuỗi: E3 nếu có, không thì thẻ cuối.
        names = [d for d, _ in panels]
        for i, d in enumerate(names):
            if d.startswith("E3"):
                return i
        return len(names) - 1

    def reset_all():
        nonlocal frames_since_pred
        buffer.clear()
        frames_since_pred = 0
        for _, ds in modes.values():
            for d in ds:
                d.reset()
        for i in range(len(results)):
            results[i] = None

    def nudge(key, delta, lo=0.0, hi=1.0):
        thresholds[key] = round(min(hi, max(lo, thresholds[key] + delta)), 2)
        print(f"{key} = {thresholds[key]:.2f}")

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
                    smoother.reset()
            else:
                selected = 0 if hands else None

            raw_hands = hands
            if selected is not None and args.smooth != "off":
                points, label = hands[selected]
                hands = list(hands)
                hands[selected] = (smoother(points, time.perf_counter()), label)
            elif selected is None and not lock.is_locked:
                smoother.reset()

            if selected is not None:
                # "display": chỉ làm mượt phần vẽ; mô hình vẫn nhận landmark gốc giống lúc train.
                feat_hands = raw_hands if args.smooth == "display" else hands
                buffer.append(to_feature(feat_hands, selected, input_dim))
                frames_since_pred += 1
                if len(buffer) == SEQ_LEN and frames_since_pred >= args.stride:
                    frames_since_pred = 0
                    window = np.stack(buffer)
                    p_idx = primary_index()
                    for i, ((_, panel), decision) in enumerate(zip(panels, decisions)):
                        results[i] = panel.predict(window)
                        committed = decision.update(results[i])
                        if committed and i == p_idx:
                            transcript.append(committed)
            # Tay khóa tạm mất (trong grace) -> giữ buffer, không thêm frame.

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last_frame_t, 1e-3) if fps else 1.0 / max(now - last_frame_t, 1e-3)
            last_frame_t = now

            n_hands = len(hands)
            if input_dim == 63 and not lock.is_locked:
                if n_hands == 0:
                    cam_state = ("CHỜ BÀN TAY", "Đưa một bàn tay vào khung hình", C_MUTED)
                elif n_hands == 1:
                    cam_state = ("ĐANG KHÓA TAY", f"Giữ yên tay {lock.single_count}/{args.acquire_frames}", C_WAIT)
                else:
                    cam_state = (f"PHÁT HIỆN {n_hands} TAY", "Chỉ giữ một tay để khóa", C_REJECT)
            elif selected is None:
                cam_state = ("MẤT DẤU TAY", "Đưa tay đã khóa trở lại khung hình", C_WAIT)
            elif len(buffer) < SEQ_LEN:
                cam_state = ("ĐANG THU CỬA SỔ", f"{len(buffer)}/{SEQ_LEN} frame", C_ACCENT)
            else:
                cam_state = ("ĐANG NHẬN DẠNG", "Tay đã khóa", C_OK)

            p_idx = primary_index()
            pd = decisions[p_idx] if panels else None
            if selected is None:
                seq_state = ("Chờ bàn tay", C_MUTED)
            elif pd is not None and pd.shown is not None:
                seq_state = ("Đã nhận · đổi sang ký hiệu tiếp theo", C_OK)
            else:
                seq_state = ("Đang nhận dạng…", C_WAIT)

            canvas = ui.render(frame, hands, selected, cam_state, seq_state, mode, len(modes) > 1,
                               panels, decisions, results, transcript, p_idx, fps)
            if not fitted:                      # lần đầu: đặt cửa sổ vừa màn hình, giữ tỉ lệ
                sw, sh = screen_size()
                scale = min(1.0, 0.92 * sw / canvas.shape[1], 0.85 * sh / canvas.shape[0])
                cv2.resizeWindow(window_name, int(canvas.shape[1] * scale), int(canvas.shape[0] * scale))
                fitted = True
            cv2.imshow(window_name, fit_to_window(canvas, window_name))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                args.smooth = {"off": "display", "display": "all", "all": "off"}[args.smooth]
                smoother.reset()
                print(f"Làm mượt landmark: {args.smooth}")
            if key == ord("q"):
                break
            elif key == ord("r"):
                lock.reset()
                reset_all()
            elif key == ord("c"):
                transcript.clear()
            elif key == ord("d"):
                ui.show_details = not ui.show_details
            elif key == ord("1"):
                nudge("min_prob", -0.05)
            elif key == ord("2"):
                nudge("min_prob", +0.05)
            elif key == ord("3"):
                nudge("min_margin", -0.05)
            elif key == ord("4"):
                nudge("min_margin", +0.05)
            elif key == ord("5"):
                nudge("min_agree", -0.125)
            elif key == ord("6"):
                nudge("min_agree", +0.125)
            elif key == ord("m") and len(modes) > 1:
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
    p.add_argument("--min-prob", type=float, default=0.75, help="Xác suất tối thiểu của lớp cao nhất")
    p.add_argument("--min-margin", type=float, default=0.35, help="Chênh lệch tối thiểu top1 - top2")
    p.add_argument("--min-agree", type=float, default=0.75,
                   help="Tỉ lệ tối thiểu cặp (fold, view TTA) cùng dự đoán top-1")
    p.add_argument("--max-energy", type=float, default=None,
                   help="Ngưỡng energy (tùy chọn); xem giá trị E in ra khi chạy --self-test")
    p.add_argument("--stable-k", type=int, default=4, help="Số dự đoán liên tiếp giống nhau để hiện nhãn")
    p.add_argument("--stride", type=int, default=3, help="Dự đoán mỗi N frame")
    # Khóa tay
    p.add_argument("--max-hands", type=int, default=2, help="Số tay tối đa MediaPipe phát hiện")
    # Ổn định landmark
    p.add_argument("--det-conf", type=float, default=0.7, help="MediaPipe min_hand_detection_confidence")
    p.add_argument("--presence-conf", type=float, default=0.7, help="MediaPipe min_hand_presence_confidence")
    p.add_argument("--track-conf", type=float, default=0.6, help="MediaPipe min_tracking_confidence")
    p.add_argument("--smooth", choices=("off", "display", "all"), default="display",
                   help="Làm mượt One Euro: off | display (chỉ phần vẽ) | all (cả đầu vào mô hình)")
    p.add_argument("--smooth-min-cutoff", type=float, default=1.0,
                   help="One Euro min_cutoff: nhỏ hơn -> mượt hơn khi tay đứng yên nhưng trễ hơn")
    p.add_argument("--smooth-beta", type=float, default=10.0,
                   help="One Euro beta: lớn hơn -> bớt trễ khi tay di chuyển nhanh")
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

    thresholds = {"min_prob": args.min_prob, "min_margin": args.min_margin,
                  "min_agree": args.min_agree, "max_energy": args.max_energy}

    def make(panels):
        return panels, [Decision(thresholds, args.stable_k) for _ in panels]

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
        run_camera(modes, args, thresholds)


if __name__ == "__main__":
    main()