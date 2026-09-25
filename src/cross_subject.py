"""
Cấu hình + khung đánh giá dùng chung cho cả 4 thực nghiệm (E0, E1, E2, E3).
Copy nguyên file này vào đầu notebook Kaggle — không sửa gì trong này, chỉ viết thêm
1 class Strategy riêng cho mô hình của bạn ở cuối, theo đúng mẫu trong comment.
"""
import copy
import gc
import os
import random
import re
import time
from pathlib import Path
from collections import defaultdict
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==================== CẤU HÌNH DATASET ====================
# Dataset: https://www.kaggle.com/datasets/hauuto/vietnamese-sign-language-alphabet
LANDMARK_DIR = Path("/kaggle/input/vietnamese-sign-language-alphabet/landmarks/landmarks/raw")
PEOPLE = ("hau", "khoi", "tai", "vy")
FNAME_RE = re.compile(r"^([a-z_]+)_([a-z]+)_([AB])_(\d+)\.npy$")


def load_all_landmarks():
    """Trả về list các dict {code, person, block, seq, arr}, arr shape (45, 63) hoặc (45, 126)."""
    records = []
    for f in sorted(LANDMARK_DIR.glob("*/*.npy")):
        m = FNAME_RE.match(f.name)
        if not m:
            print(f"CẢNH BÁO: tên file không đúng quy ước: {f.name}")
            continue
        code, person, block, seq = m.groups()
        arr = np.load(f)
        records.append({
            "code": code, "person": person, "block": block,
            "seq": int(seq), "arr": arr,
        })
    return records


# ==================== GIAO DIỆN STRATEGY — MỖI MÔ HÌNH TỰ IMPLEMENT ====================
class ModelStrategy(Protocol):
    def prepare_input(self, X: np.ndarray) -> np.ndarray:
        """(N, 45, D) -> định dạng phù hợp mô hình của bạn.
        E0/E1: return X[:, 22, :]   (khung cố định thứ 23/45)
        E2/E3: return X             (giữ nguyên chuỗi 45 bước)
        """
        ...

    def train(self, X_train, y_train):
        """Trả về model_state (đối tượng bất kỳ bạn cần để predict sau này)."""
        ...

    def predict(self, model_state, X_test) -> np.ndarray:
        """Trả về mảng nhãn dự đoán, cùng độ dài với X_test."""
        ...

    def measure_latency(self, model_state, X_sample) -> float:
        """Đo thời gian suy luận cho 1 mẫu, đơn vị mili-giây."""
        ...


# ==================== KHUNG ĐÁNH GIÁ CROSS-SUBJECT — DÙNG CHUNG, KHÔNG SỬA ====================
def run_cross_subject(records, strategy: ModelStrategy, people=PEOPLE):
    """Train trên 3 người, test trên người còn lại, xoay vòng qua cả 4 người.
    Chạy 1 lần notebook là tự động lặp đủ 4 lượt, không cần tách notebook riêng.
    """
    results = []
    for test_person in people:
        train_records = [r for r in records if r["person"] != test_person]
        test_records = [r for r in records if r["person"] == test_person]

        # Một số strategy (E3/DANN) cần biết subject của từng mẫu train để
        # tạo domain label. Hook này là optional nên E1/E2 vẫn giữ nguyên API.
        set_fold_context = getattr(strategy, "set_fold_context", None)
        if set_fold_context is not None:
            set_fold_context(train_records, test_records, test_person)

        X_train_raw = np.stack([r["arr"] for r in train_records])
        y_train = np.array([r["code"] for r in train_records])
        X_test_raw = np.stack([r["arr"] for r in test_records])
        y_test = np.array([r["code"] for r in test_records])

        X_train = strategy.prepare_input(X_train_raw)
        X_test = strategy.prepare_input(X_test_raw)

        model_state = strategy.train(X_train, y_train)
        y_pred = strategy.predict(model_state, X_test)
        accuracy = float(np.mean(y_pred == y_test))
        latency_ms = strategy.measure_latency(model_state, X_test[:1])

        results.append({"test_person": test_person, "accuracy": accuracy, "latency_ms": latency_ms})
        print(f"Test trên {test_person}: accuracy={accuracy:.3f}, latency={latency_ms:.2f}ms")

    accs = [r["accuracy"] for r in results]
    lats = [r["latency_ms"] for r in results]
    print(f"\nTrung bình: accuracy={np.mean(accs):.3f} (±{np.std(accs):.3f}), "
          f"latency={np.mean(lats):.2f}ms (±{np.std(lats):.2f}ms)")
    return results

class AttentionPool(nn.Module):
    """Soft attention pooling qua trục thời gian."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lstm_out):
        # lstm_out: (B, T, H)
        scores  = self.attn(lstm_out)               # (B, T, 1)
        weights = torch.softmax(scores, dim=1)      # (B, T, 1)
        context = (lstm_out * weights).sum(dim=1)   # (B, H)
        return context


class AttentionLSTM(nn.Module):
    """
    E2 — LSTM một chiều với attention pooling.

    Input: (B, T, D)  D = 63 hoặc 126 (sau normalize)

    Pipeline:
      velocity + acceleration → concat → LayerNorm → Linear(proj)
      → LSTM(uni, layers=2) → AttentionPool + MeanPool → concat
      → FC(out_dim → out_dim//2, GELU) → Dropout → FC(num_classes)

    Không có: GRL, domain head, EMA, TTA.
    """
    def __init__(self, input_dim: int, proj_dim: int, hidden_dim: int,
                 num_layers: int, num_classes: int, dropout: float = 0.35):
        super().__init__()
        feat_dim = input_dim * 3   # pose, velocity (dx), acceleration (ddx)

        self.proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.30),
        )

        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,   # E2: một chiều
        )

        self.out_norm  = nn.LayerNorm(hidden_dim)
        self.attn_pool = AttentionPool(hidden_dim)

        out_dim = hidden_dim * 2   # attn_pool + mean_pool
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, num_classes),
        )

    def encode(self, x):
        dx  = torch.cat((torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]), dim=1)
        ddx = torch.cat((torch.zeros_like(dx[:, :1]), dx[:, 1:] - dx[:, :-1]), dim=1)
        z   = torch.cat((x, dx, ddx), dim=-1)   # (B, T, D*3)
        z   = self.proj(z)                       # (B, T, proj_dim)

        out, _ = self.lstm(z)                    # (B, T, hidden_dim)
        out    = self.out_norm(out)

        attn_vec = self.attn_pool(out)           # (B, hidden_dim)
        mean_vec = out.mean(dim=1)               # (B, hidden_dim)
        emb      = torch.cat((attn_vec, mean_vec), dim=-1)  # (B, hidden_dim*2)
        return emb

    def forward(self, x, **kwargs):
        # **kwargs để tương thích signature gọi từ outer loop
        return self.head(self.encode(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import re, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib

PROJ_DIM    = 96
HIDDEN_DIM  = 96
NUM_LAYERS  = 2       # 2 layer để có depth, bù cho không có BiLSTM
DROPOUT     = 0.35
LR          = 8e-4
INNER_MAX_EPOCHS = 100
BATCH_SIZE  = 32

SEQ_LEN = 45
VALID_FEATURE_DIMS = (63, 126) 

SEED            = 42
INNER_PATIENCE  = 18
NUM_WORKERS     = 2
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.05

USE_HAND_NORMALIZATION = True
USE_TRAIN_AUGMENT      = True
AUG_ROTATE_DEG = 8.0
AUG_JITTER_STD = 0.007
AUG_TIME_WARP  = 0.10
AUG_Z_SCALE    = 0.10

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
ARTIFACT_DIR = Path("E2_artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Normalization={USE_HAND_NORMALIZATION} | Augment={USE_TRAIN_AUGMENT}")
print(f"Model: AttentionLSTM (unidirectional) | PROJ={PROJ_DIM} HIDDEN={HIDDEN_DIM} LAYERS={NUM_LAYERS}")


def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_hand_sequence_keep_motion(X: np.ndarray) -> np.ndarray:
    """Sequence-level wrist/scale normalization, giữ wrist trajectory."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or X.shape[1] != SEQ_LEN or X.shape[2] not in VALID_FEATURE_DIMS:
        raise ValueError(f"Expected (N, {SEQ_LEN}, 63/126), got {X.shape}")
    N, T, D = X.shape
    n_hands = D // 63
    pts = X.reshape(N, T, n_hands, 21, 3).copy()
    out = np.zeros_like(pts, dtype=np.float32)
    eps = 1e-6
    palm_ids = np.array([5, 9, 13, 17], dtype=np.int64)
    for h in range(n_hands):
        hand = pts[:, :, h]
        wrist = hand[:, :, 0, :]
        local = hand - wrist[:, :, None, :]
        palm_radius = np.linalg.norm(local[:, :, palm_ids, :], axis=-1)
        frame_scale = np.median(palm_radius, axis=-1)
        valid = frame_scale > 1e-4
        for n in range(N):
            valid_idx = np.flatnonzero(valid[n])
            if len(valid_idx) == 0:
                continue
            anchor = wrist[n, valid_idx[0]].copy()
            seq_scale = max(float(np.median(frame_scale[n, valid_idx])), eps)
            out[n, valid_idx, h, 0, :] = (wrist[n, valid_idx] - anchor) / seq_scale
            out[n, valid_idx, h, 1:, :] = local[n, valid_idx, 1:, :] / seq_scale
    return out.reshape(N, T, D).astype(np.float32)


def _time_warp_tensor(x: torch.Tensor, gamma: float) -> torch.Tensor:
    T = x.shape[-2]
    u = torch.linspace(0.0, 1.0, T, dtype=x.dtype, device=x.device)
    pos = (u.clamp_min(1e-6).pow(float(gamma))) * (T - 1)
    pos[0] = 0.0
    pos[-1] = float(T - 1)
    left  = torch.floor(pos).long()
    right = torch.clamp(left + 1, max=T - 1)
    w     = (pos - left.to(pos.dtype))
    if x.ndim == 2:
        return x[left] * (1.0 - w[:, None]) + x[right] * w[:, None]
    if x.ndim == 3:
        return x[:, left] * (1.0 - w[None, :, None]) + x[:, right] * w[None, :, None]
    raise ValueError(f"Unsupported x.ndim={x.ndim}")


def augment_sequence(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    T, D = x.shape
    n_hands = D // 63
    if torch.rand(()) < 0.70:
        angle = (torch.rand(()) * 2.0 - 1.0) * np.deg2rad(AUG_ROTATE_DEG)
        c, s  = torch.cos(angle), torch.sin(angle)
        pts   = x.reshape(T, n_hands, 21, 3)
        xx    = pts[..., 0].clone()
        yy    = pts[..., 1].clone()
        pts[..., 0] = c * xx - s * yy
        pts[..., 1] = s * xx + c * yy
        x = pts.reshape(T, D)
    if torch.rand(()) < 0.80:
        frame_valid = (x.abs().sum(dim=1, keepdim=True) > 1e-7).to(x.dtype)
        x = x + torch.randn_like(x) * AUG_JITTER_STD * frame_valid
    if torch.rand(()) < 0.50:
        pts     = x.reshape(T, n_hands, 21, 3)
        z_scale = 1.0 + (torch.rand(()) * 2.0 - 1.0) * AUG_Z_SCALE
        pts[..., 2] = pts[..., 2] * z_scale
        x = pts.reshape(T, D)
    if torch.rand(()) < 0.60:
        gamma = 1.0 + float((torch.rand(()) * 2.0 - 1.0) * AUG_TIME_WARP)
        x = _time_warp_tensor(x, gamma)

        # 5. Mirror ngang (lật trái↔phải) — xác suất 50%
    # Đảo dấu tọa độ x của tất cả landmark
    # Hoạt động đúng cho cả 1 tay (D=63) lẫn 2 tay (D=126)
    if torch.rand(()) < 0.50:
        pts = x.reshape(T, n_hands, 21, 3)
        # Lật x quanh tâm x trung bình của toàn bộ frame hợp lệ
        frame_valid = (pts.abs().sum(dim=(-1, -2, -3)) > 1e-7)  # (T,)
        if frame_valid.any():
            x_coords = pts[frame_valid, :, :, 0]  # chỉ lấy frame hợp lệ
            x_center = x_coords.mean()
            pts[:, :, :, 0] = 2.0 * x_center - pts[:, :, :, 0]
        x = pts.reshape(T, D)
        
    return x


class LandmarkDomainDataset(Dataset):
    """Giữ nguyên tên để tương thích _make_loader. Trường d không dùng ở E2."""
    def __init__(self, X, y, d=None, augment=False):
        self.X       = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.y       = torch.tensor(np.asarray(y), dtype=torch.long)
        self.d       = torch.full_like(self.y, -1) if d is None else torch.tensor(np.asarray(d), dtype=torch.long)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = augment_sequence(x)
        return x, self.y[idx], self.d[idx]


def _select_epoch_from_inner(inner):
    """Chọn epoch từ mean val acc của 3 inner folds, làm trơn 3 điểm."""
    common_len = min(len(x["history"]["val_acc"]) for x in inner)
    acc_mat    = np.stack([np.asarray(x["history"]["val_acc"][:common_len],  dtype=float) for x in inner])
    loss_mat   = np.stack([np.asarray(x["history"]["val_loss"][:common_len], dtype=float) for x in inner])
    mean_acc   = acc_mat.mean(axis=0)
    mean_loss  = loss_mat.mean(axis=0)
    smooth     = mean_acc.copy()
    if common_len >= 3:
        smooth[1:-1] = (mean_acc[:-2] + mean_acc[1:-1] + mean_acc[2:]) / 3.0
    best_score = smooth.max()
    candidates = np.flatnonzero(np.isclose(smooth, best_score, atol=1e-12))
    best_idx   = candidates[np.argmin(mean_loss[candidates])] if len(candidates) > 1 else int(candidates[0])
    return int(best_idx + 1), {
        "common_len":           int(common_len),
        "mean_val_acc":         mean_acc.tolist(),
        "smooth_mean_val_acc":  smooth.tolist(),
        "mean_val_loss":        mean_loss.tolist(),
    }


# ============================================================
# E2LSTMStrategy — interface tương thích với outer LOSO loop
# ============================================================

class E2LSTMStrategy:
    def prepare_input(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return normalize_hand_sequence_keep_motion(X) if USE_HAND_NORMALIZATION else X

    def _make_loader(self, X, y, d, shuffle, seed, augment=False):
        ds        = LandmarkDomainDataset(X, y, d=d, augment=augment)
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=shuffle,
            num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
            generator=generator if shuffle else None,
            persistent_workers=(NUM_WORKERS > 0),
        )

    def _new_model(self, input_dim, num_classes, num_domains=None):
        # num_domains giữ trong signature để tương thích call-site, E2 không dùng
        return AttentionLSTM(
            input_dim=input_dim, proj_dim=PROJ_DIM, hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS, num_classes=num_classes, dropout=DROPOUT,
        ).to(DEVICE)

    @staticmethod
    def _class_criterion():
        return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    def _run_train_epoch(self, model, loader, criterion, optimizer):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for xb, yb, _ in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += loss.item() * xb.size(0)
            correct  += (logits.argmax(1) == yb).sum().item()
            total    += xb.size(0)
        return {
            "loss":        loss_sum / max(total, 1),
            "class_loss":  loss_sum / max(total, 1),
            "domain_loss": 0.0,
            "acc":         correct  / max(total, 1),
            "domain_acc":  0.0,
        }

    def _run_eval(self, model, loader, criterion):
        model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb, _ in loader:
                xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                logits  = model(xb)
                loss_sum += criterion(logits, yb).item() * xb.size(0)
                correct  += (logits.argmax(1) == yb).sum().item()
                total    += xb.size(0)
        return loss_sum / max(total, 1), correct / max(total, 1)

    def _select_epoch_one_person(self, X, y_enc, groups, val_person,
                                  input_dim, num_classes, seed):
        idx_val   = np.flatnonzero(groups == val_person)
        idx_train = np.flatnonzero(groups != val_person)
        train_people = sorted(np.unique(groups[idx_train]).tolist())

        train_loader = self._make_loader(
            X[idx_train], y_enc[idx_train], None,
            shuffle=True, seed=seed, augment=USE_TRAIN_AUGMENT,
        )
        val_loader = self._make_loader(
            X[idx_val], y_enc[idx_val], None,
            shuffle=False, seed=seed, augment=False,
        )

        set_seed(seed)
        model     = self._new_model(input_dim, num_classes)
        criterion = self._class_criterion()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

        history = {
            "train_loss": [], "train_class_loss": [], "train_domain_loss": [],
            "train_acc":  [], "train_domain_acc": [],
            "val_loss":   [], "val_acc": [], "lr": [],
        }
        best_epoch = 1
        best_acc   = -1.0
        best_loss_at_best_acc = float("inf")
        patience   = 0

        print(f"    inner val={val_person} | train={train_people} | "
              f"n_train={len(idx_train)} n_val={len(idx_val)}")

        for epoch in range(1, INNER_MAX_EPOCHS + 1):
            tr      = self._run_train_epoch(model, train_loader, criterion, optimizer)
            va_loss, va_acc = self._run_eval(model, val_loader, criterion)
            lr_now  = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(tr["loss"])
            history["train_class_loss"].append(tr["class_loss"])
            history["train_domain_loss"].append(tr["domain_loss"])
            history["train_acc"].append(tr["acc"])
            history["train_domain_acc"].append(tr["domain_acc"])
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)
            history["lr"].append(lr_now)

            improved = (va_acc > best_acc + 1e-12) or (
                abs(va_acc - best_acc) <= 1e-12 and va_loss < best_loss_at_best_acc
            )
            if improved:
                best_acc              = float(va_acc)
                best_loss_at_best_acc = float(va_loss)
                best_epoch            = epoch
                patience              = 0
            else:
                patience += 1

            scheduler.step()

            if epoch == 1 or epoch % 10 == 0 or patience >= INNER_PATIENCE:
                print(f"      epoch {epoch:3d}/{INNER_MAX_EPOCHS} | "
                      f"train_acc={tr['acc']:.3f} | "
                      f"val_acc={va_acc:.3f} loss={va_loss:.4f} | "
                      f"best={best_epoch} ({best_acc:.3f})")
            if patience >= INNER_PATIENCE:
                break

        model.to("cpu")
        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        return {
            "val_person":                val_person,
            "train_people":              train_people,
            "best_epoch":                int(best_epoch),
            "best_val_acc":              float(best_acc),
            "best_val_loss_at_best_acc": float(best_loss_at_best_acc),
            "history":                   history,
        }

    def train(self, X_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray,
              fold_name: str, seed: int = SEED, checkpoint_path=None):
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train)
        groups  = np.asarray(groups)

        le          = LabelEncoder()
        y_enc       = le.fit_transform(y_train)
        num_classes = len(le.classes_)
        input_dim   = X_train.shape[2]
        people      = sorted(np.unique(groups).tolist())

        if len(people) != 3:
            raise ValueError(f"{fold_name}: expected 3 outer-train people, got {people}")

        probe = self._new_model(input_dim, num_classes)
        print(f"[{fold_name}] model=AttentionLSTM | params={count_parameters(probe):,}")
        del probe
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # 1) Inner subject-CV để chọn epoch
        inner = []
        for i, val_person in enumerate(people):
            inner.append(self._select_epoch_one_person(
                X_train, y_enc, groups, val_person, input_dim, num_classes,
                seed=seed + 100 * (i + 1),
            ))

        median_epoch              = max(1, int(round(float(np.median([x["best_epoch"] for x in inner])))))
        final_epochs, epoch_selection = _select_epoch_from_inner(inner)
        print(f"[{fold_name}] inner best epochs={[x['best_epoch'] for x in inner]} | "
              f"median={median_epoch} | mean-curve selected={final_epochs}")
        print(f"[{fold_name}] inner best accs={[round(x['best_val_acc'], 4) for x in inner]}")

        # 2) Final fit trên đủ 3 người
        set_seed(seed + 1000)
        loader    = self._make_loader(X_train, y_enc, None,
                                      shuffle=True, seed=seed + 1000,
                                      augment=USE_TRAIN_AUGMENT)
        model     = self._new_model(input_dim, num_classes)
        criterion = self._class_criterion()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

        hist = {
            "train_loss": [], "train_class_loss": [], "train_domain_loss": [],
            "train_acc":  [], "train_domain_acc": [], "lr": [],
        }
        print(f"[{fold_name}] FINAL fit | epochs={final_epochs}")
        for epoch in range(1, final_epochs + 1):
            tr = self._run_train_epoch(model, loader, criterion, optimizer)
            hist["train_loss"].append(tr["loss"])
            hist["train_class_loss"].append(tr["class_loss"])
            hist["train_domain_loss"].append(tr["domain_loss"])
            hist["train_acc"].append(tr["acc"])
            hist["train_domain_acc"].append(tr["domain_acc"])
            hist["lr"].append(optimizer.param_groups[0]["lr"])
            scheduler.step()
            if epoch == 1 or epoch % 10 == 0 or epoch == final_epochs:
                print(f"  epoch {epoch:3d}/{final_epochs} | "
                      f"loss={tr['loss']:.4f} acc={tr['acc']:.3f}")

        model.eval()

        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dicts":      [copy.deepcopy(model.state_dict())],
                "classes":                le.classes_.tolist(),
                "input_dim":              int(input_dim),
                "proj_dim":               PROJ_DIM,
                "hidden_dim":             HIDDEN_DIM,
                "num_layers":             NUM_LAYERS,
                "dropout":                DROPOUT,
                "model_name":             "AttentionLSTM",
                "inner_results":          [{k: v for k, v in x.items() if k != "history"} for x in inner],
                "epoch_selection":        epoch_selection,
                "median_inner_epoch":     int(median_epoch),
                "final_epochs":           int(final_epochs),
                "ensemble_size":          1,
                "ensemble_seeds":         [seed + 1000],
                "outer_train_people":     people,
                "use_hand_normalization": bool(USE_HAND_NORMALIZATION),
                "use_train_augment":      bool(USE_TRAIN_AUGMENT),
                "label_smoothing":        LABEL_SMOOTHING,
                "weight_decay":           WEIGHT_DECAY,
            }, checkpoint_path)

        del optimizer, scheduler, loader
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        return {
            "models":             [model],
            "le":                 le,
            "inner":              inner,
            "epoch_selection":    epoch_selection,
            "median_inner_epoch": median_epoch,
            "final_histories":    [hist],
            "input_dim":          input_dim,
            "final_epochs":       final_epochs,
            "ensemble_size":      1,
            "ensemble_seeds":     [seed + 1000],
            "final_train_size":   len(X_train),
            "outer_train_people": people,
        }

    def predict(self, model_state, X_test: np.ndarray) -> np.ndarray:
        models = model_state["models"]
        le     = model_state["le"]
        for m in models:
            m.eval()

        X_test = np.asarray(X_test, dtype=np.float32)
        pred_ids = []

        with torch.no_grad():
            for start in range(0, len(X_test), BATCH_SIZE):
                xb = torch.tensor(
                    X_test[start:start + BATCH_SIZE],
                    dtype=torch.float32, device=DEVICE
                )  # (B, T, D)

                # --- TTA mirror: tạo bản lật ngang của cả batch ---
                T, D = xb.shape[1], xb.shape[2]
                n_hands = D // 63
                pts = xb.reshape(xb.size(0), T, n_hands, 21, 3).clone()

                # Tính x_center từ frame hợp lệ của từng sample
                frame_valid = (pts.abs().sum(dim=(-1, -2, -3)) > 1e-7)  # (B, T)
                xb_mirror = pts.clone()
                for b in range(xb.size(0)):
                    valid_frames = frame_valid[b]
                    if valid_frames.any():
                        x_coords = pts[b, valid_frames, :, :, 0]
                        x_center = x_coords.mean()
                        xb_mirror[b, :, :, :, 0] = 2.0 * x_center - pts[b, :, :, :, 0]
                xb_mirror = xb_mirror.reshape(xb.size(0), T, D)

                # --- Forward cả 2 bản, trung bình xác suất ---
                logits_orig   = models[0](xb)           # (B, num_classes)
                logits_mirror = models[0](xb_mirror)    # (B, num_classes)

                probs_orig   = torch.softmax(logits_orig,   dim=1)
                probs_mirror = torch.softmax(logits_mirror, dim=1)
                probs_avg    = (probs_orig + probs_mirror) / 2.0  # (B, num_classes)

                pred_ids.append(probs_avg.argmax(dim=1).cpu().numpy())

        return le.inverse_transform(np.concatenate(pred_ids))

    def measure_latency(self, model_state, X_sample: np.ndarray) -> float:
        model = model_state["models"][0]
        model.eval()

        x = torch.tensor(X_sample[:1], dtype=torch.float32, device=DEVICE)

        # Tạo bản mirror mẫu đo
        T, D = x.shape[1], x.shape[2]
        n_hands = D // 63
        pts = x.reshape(1, T, n_hands, 21, 3).clone()
        frame_valid = (pts.abs().sum(dim=(-1, -2, -3)) > 1e-7)[0]
        if frame_valid.any():
            x_center = pts[0, frame_valid, :, :, 0].mean()
            pts[0, :, :, :, 0] = 2.0 * x_center - pts[0, :, :, :, 0]
        x_mirror = pts.reshape(1, T, D)

        with torch.no_grad():
            for _ in range(5):  # warmup
                model(x)
                model(x_mirror)

        N = 30
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(N):
                p1 = torch.softmax(model(x), dim=1)
                p2 = torch.softmax(model(x_mirror), dim=1)
                _ = (p1 + p2) / 2.0
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / N

# ==================== VÍ DỤ CẮM STRATEGY — MỖI BẠN TỰ VIẾT PHẦN NÀY ====================
# class MyStrategy:
#     def prepare_input(self, X):
#         return X[:, 22, :]           # đổi tuỳ E0/E1 (vector) hay E2/E3 (giữ nguyên chuỗi)
#
#     def train(self, X_train, y_train):
#         ...
#
#     def predict(self, model_state, X_test):
#         ...
#
#     def measure_latency(self, model_state, X_sample):
#         ...
#
# records = load_all_landmarks()
# print(f"Tổng số mẫu đọc được: {len(records)}")
# run_cross_subject(records, MyStrategy())


# ==================== E1 — MLP + WRIST-CENTERING + SCALE-NORMALIZATION ====================
# E1 giữ nguyên khung cố định thứ 23/45 và kiến trúc MLP của E0. Biến độc lập duy
# nhất so với E0 là bước chuẩn hóa hình học trong prepare_input().
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FRAME_INDEX = 22
E1_EPOCHS = 100
E1_LEARNING_RATE = 1e-3
E1_BATCH_SIZE = 32
E1_VAL_RATIO = 0.15
E1_SEED = 42
E1_EPS = 1e-6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
E1_OUTPUT_DIR = Path("/kaggle/working/e1_outputs")
E1_PLOT_DIR = E1_OUTPUT_DIR / "plots"


def resolve_e1_landmark_dir() -> Path:
    """Tìm thư mục landmark .npy dù Kaggle đổi slug hoặc mức thư mục mount."""
    if LANDMARK_DIR.exists() and any(LANDMARK_DIR.glob("*/*.npy")):
        return LANDMARK_DIR

    candidates = [
        folder
        for folder in Path("/kaggle/input").rglob("raw")
        if folder.is_dir() and any(folder.glob("*/*.npy"))
    ]
    if not candidates:
        raise FileNotFoundError(
            "Không tìm thấy landmark .npy trong /kaggle/input. "
            "Dataset cần có cấu trúc raw/{person}/*.npy."
        )
    return sorted(candidates)[0]


def plot_e1_history(history, fold_index: int, test_person: str):
    """Xuất biểu đồ train loss và validation accuracy của một fold E1."""
    E1_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    figure, loss_axis = plt.subplots(figsize=(7, 4.5))
    loss_axis.plot(
        history["epoch"],
        history["train_loss"],
        color="#2a78d6",
        linewidth=2,
        label="Train loss",
    )
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Train loss", color="#2a78d6")
    loss_axis.tick_params(axis="y", labelcolor="#2a78d6")

    accuracy_axis = loss_axis.twinx()
    accuracy_axis.plot(
        history["epoch"],
        history["val_accuracy"],
        color="#eb6834",
        linewidth=2,
        label="Val accuracy",
    )
    accuracy_axis.set_ylabel("Val accuracy", color="#eb6834")
    accuracy_axis.tick_params(axis="y", labelcolor="#eb6834")
    accuracy_axis.set_ylim(0.0, 1.0)

    loss_lines, loss_labels = loss_axis.get_legend_handles_labels()
    accuracy_lines, accuracy_labels = accuracy_axis.get_legend_handles_labels()
    loss_axis.legend(
        loss_lines + accuracy_lines,
        loss_labels + accuracy_labels,
        loc="center right",
    )
    figure.suptitle(f"E1 — Train loss & Val accuracy (fold test={test_person})")
    figure.tight_layout()

    plot_path = E1_PLOT_DIR / f"E1_fold{fold_index}_{test_person}.png"
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")

    # savefig() chỉ ghi PNG xuống /kaggle/working. display() giúp Kaggle render
    # luôn từng biểu đồ ngay dưới cell, giống hình mẫu của E0.
    try:
        from IPython.display import display as notebook_display
        notebook_display(figure)
    except ImportError:
        pass  # ngoài notebook vẫn có file PNG đã lưu ở plot_path

    plt.close(figure)
    print(f"  Đã lưu biểu đồ: {plot_path}")
    return plot_path


class E1MLP(nn.Module):
    """MLP dùng cùng kiến trúc với E0 để phép so sánh E0/E1 công bằng."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class E1Strategy:
    """E1: MLP trên một khung đã chuẩn hóa vị trí và kích thước bàn tay."""

    def __init__(self):
        self.fold_index = 0
        E1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        E1_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    def prepare_input(self, X: np.ndarray) -> np.ndarray:
        """Lấy frame 22, wrist-center rồi scale-normalize cho từng tay.

        Scale của một tay là khoảng cách Euclid lớn nhất từ cổ tay đến một
        landmark trong tay đó. Mẫu/tay bị mất (toàn số 0) vẫn giữ nguyên số 0.
        Hàm hỗ trợ cả dữ liệu một tay (63 chiều) và hai tay (126 chiều).
        """
        X = np.asarray(X)
        if X.ndim != 3 or X.shape[1] <= FRAME_INDEX:
            raise ValueError(
                f"E1 cần X có shape (N, T, D), T > {FRAME_INDEX}; nhận được {X.shape}."
            )
        if X.shape[2] % 63 != 0:
            raise ValueError(
                f"Số chiều landmark phải là bội của 63; nhận được D={X.shape[2]}."
            )

        selected = X[:, FRAME_INDEX, :].astype(np.float32, copy=True)
        hands = selected.reshape(selected.shape[0], -1, 21, 3)

        # MediaPipe đánh số landmark 0 là cổ tay. Trừ landmark này giúp loại bỏ
        # sai khác do vị trí bàn tay/người ký trong khung hình.
        centered = hands - hands[:, :, 0:1, :]

        # Chia cho bán kính bàn tay giúp loại bỏ sai khác do khoảng cách tới camera
        # và kích thước bàn tay. maximum(..., EPS) tránh chia cho 0 khi mất tay.
        distances = np.linalg.norm(centered, axis=-1)
        scales = np.max(distances, axis=-1, keepdims=True)[..., np.newaxis]
        normalized = centered / np.maximum(scales, E1_EPS)

        return normalized.reshape(selected.shape[0], -1).astype(np.float32)

    @staticmethod
    def _stratified_train_val_indices(y: np.ndarray, seed: int):
        """Tách validation theo từng lớp, chỉ từ ba người thuộc tập train."""
        rng = np.random.default_rng(seed)
        train_indices, val_indices = [], []

        for label in np.unique(y):
            indices = np.flatnonzero(y == label)
            rng.shuffle(indices)
            if len(indices) < 2:
                raise ValueError(f"Lớp {label!r} có ít hơn 2 mẫu trong tập train.")
            number_val = max(1, int(len(indices) * E1_VAL_RATIO))
            number_val = min(number_val, len(indices) - 1)
            val_indices.extend(indices[:number_val])
            train_indices.extend(indices[number_val:])

        return np.asarray(train_indices), np.asarray(val_indices)

    def train(self, X_train, y_train):
        fold_seed = E1_SEED + self.fold_index
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)

        labels = np.asarray(sorted(np.unique(y_train).tolist()))
        label_to_index = {label: index for index, label in enumerate(labels)}
        y_encoded = np.asarray([label_to_index[label] for label in y_train], dtype=np.int64)

        train_idx, val_idx = self._stratified_train_val_indices(y_encoded, fold_seed)
        X_fit, y_fit = X_train[train_idx], y_encoded[train_idx]
        X_val, y_val = X_train[val_idx], y_encoded[val_idx]

        model = E1MLP(X_fit.shape[1], len(labels)).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=E1_LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=15, min_lr=1e-5
        )
        criterion = nn.CrossEntropyLoss()

        generator = torch.Generator().manual_seed(fold_seed)
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_fit).float(),
                torch.from_numpy(y_fit).long(),
            ),
            batch_size=E1_BATCH_SIZE,
            shuffle=True,
            generator=generator,
        )
        X_val_tensor = torch.from_numpy(X_val).float().to(DEVICE)
        y_val_tensor = torch.from_numpy(y_val).long().to(DEVICE)

        best_accuracy = -1.0
        best_epoch = 0
        best_state = None
        history = {"epoch": [], "train_loss": [], "val_accuracy": []}

        for epoch in range(E1_EPOCHS):
            model.train()
            total_loss = 0.0
            for features, targets in train_loader:
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(features), targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * features.size(0)

            model.eval()
            with torch.inference_mode():
                val_predictions = model(X_val_tensor).argmax(dim=1)
                val_accuracy = float((val_predictions == y_val_tensor).float().mean().item())
            scheduler.step(val_accuracy)

            mean_loss = total_loss / len(X_fit)
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(mean_loss)
            history["val_accuracy"].append(val_accuracy)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

            if (epoch + 1) % 20 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  [E1 fold {self.fold_index + 1}/4] epoch {epoch + 1}/{E1_EPOCHS} "
                    f"loss={mean_loss:.4f} val_acc={val_accuracy:.3f} lr={current_lr:.2e}"
                )

        model.load_state_dict(best_state)
        model.eval()

        checkpoint_path = E1_OUTPUT_DIR / f"E1_fold{self.fold_index}.pt"
        torch.save(
            {
                "experiment": "E1",
                "frame_index": FRAME_INDEX,
                "model_state_dict": model.state_dict(),
                "labels": labels.tolist(),
                "input_dim": int(X_fit.shape[1]),
                "best_epoch": best_epoch,
                "best_val_accuracy": best_accuracy,
            },
            checkpoint_path,
        )
        print(
            f"  Chọn epoch {best_epoch}, val_acc={best_accuracy:.3f}; "
            f"đã lưu {checkpoint_path}"
        )

        # run_cross_subject xoay PEOPLE theo đúng thứ tự, nên fold_index cũng xác
        # định được người đang được giữ lại làm test mà không sửa khung dùng chung.
        test_person = PEOPLE[self.fold_index]
        plot_e1_history(history, self.fold_index, test_person)

        state = {"model": model, "labels": labels}
        self.fold_index += 1
        return state

    def predict(self, model_state, X_test) -> np.ndarray:
        model = model_state["model"]
        model.eval()
        with torch.inference_mode():
            features = torch.from_numpy(np.asarray(X_test)).float().to(DEVICE)
            predicted_indices = model(features).argmax(dim=1).cpu().numpy()
        return model_state["labels"][predicted_indices]

    def measure_latency(self, model_state, X_sample) -> float:
        """Đo latency MLP trên CPU sau 10 lượt warm-up, lấy trung bình 100 lượt."""
        model = model_state["model"]
        original_device = next(model.parameters()).device
        model.to("cpu").eval()
        sample = torch.from_numpy(np.asarray(X_sample)).float()

        with torch.inference_mode():
            for _ in range(10):
                model(sample)
            elapsed_ms = []
            for _ in range(100):
                start = time.perf_counter()
                model(sample)
                elapsed_ms.append((time.perf_counter() - start) * 1000.0)

        model.to(original_device)
        return float(np.mean(elapsed_ms))


# ==================== E3 — DOMAIN-INVARIANT MOTION BI-LSTM ====================
# Triển khai theo notebook 4-fold: pose + velocity + acceleration + unit-bone,
# temporal convolution, BiLSTM, attention pooling, DANN/GRL, EMA và subject-CV.
E3_SEQ_LEN = 45
E3_VALID_FEATURE_DIMS = (63, 126)
E3_PROJ_DIM = 96
E3_HIDDEN_DIM = 96
E3_NUM_LAYERS = 1
E3_DROPOUT = 0.35
E3_LEARNING_RATE = 8e-4
E3_INNER_MAX_EPOCHS = 100
E3_BATCH_SIZE = 32
E3_SEED = 42
E3_INNER_PATIENCE = 18
E3_NUM_WORKERS = 2
E3_WEIGHT_DECAY = 1e-4
E3_LABEL_SMOOTHING = 0.05
E3_DOMAIN_LAMBDA_MAX = 0.15
E3_EMA_DECAY = 0.995
E3_USE_HAND_NORMALIZATION = True
E3_USE_TRAIN_AUGMENT = True
E3_AUG_ROTATE_DEG = 8.0
E3_AUG_JITTER_STD = 0.007
E3_AUG_TIME_WARP = 0.10
E3_AUG_Z_SCALE = 0.10
E3_TTA_GAMMAS = (1.0,)
E3_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
E3_OUTPUT_DIR = Path("/kaggle/working/e3_outputs")


class _E3GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def _e3_grad_reverse(x, lambd=1.0):
    return _E3GradReverse.apply(x, lambd)


class E3DomainInvariantMotionBiLSTM(nn.Module):
    """Motion-aware BiLSTM với domain-adversarial head dùng lúc train."""

    PARENT = (0, 0, 1, 2, 3, 0, 5, 6, 7, 5, 9, 10, 11, 9, 13, 14, 15, 13, 17, 18, 19)

    def __init__(self, input_dim, proj_dim, hidden_dim, num_layers,
                 num_classes, num_domains, dropout=0.35):
        super().__init__()
        if input_dim % 63 != 0:
            raise ValueError(f"input_dim phải chia hết cho 63, got {input_dim}")

        self.input_dim = int(input_dim)
        self.n_hands = input_dim // 63
        self.proj = nn.Sequential(
            nn.Linear(input_dim * 4, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.30),
        )
        self.dw3 = nn.Conv1d(proj_dim, proj_dim, kernel_size=3, padding=1, groups=proj_dim)
        self.dw5_d2 = nn.Conv1d(
            proj_dim, proj_dim, kernel_size=5, padding=4, dilation=2, groups=proj_dim
        )
        self.pw = nn.Conv1d(proj_dim, proj_dim, kernel_size=1)
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
        norm = torch.linalg.vector_norm(bone, dim=-1, keepdim=True).clamp_min(1e-6)
        unit = bone / norm
        unit[:, :, :, 0, :] = 0.0
        return unit.reshape(batch, steps, dims)

    def encode(self, x):
        velocity = torch.cat((torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]), dim=1)
        acceleration = torch.cat(
            (torch.zeros_like(velocity[:, :1]), velocity[:, 1:] - velocity[:, :-1]), dim=1
        )
        z = self.proj(torch.cat((x, velocity, acceleration, self._unit_bone(x)), dim=-1))

        channels = z.transpose(1, 2)
        local = torch.nn.functional.gelu(self.dw3(channels))
        broader = torch.nn.functional.gelu(self.dw5_d2(channels))
        conv = self.pw(0.5 * (local + broader)).transpose(1, 2)
        z = self.temporal_norm(z + self.temporal_drop(conv))

        out, _ = self.lstm(z)
        out = self.out_norm(out)
        attention = torch.softmax(self.attn(out), dim=1)
        attention_pool = (out * attention).sum(dim=1)
        mean_pool = out.mean(dim=1)
        return self.embed(torch.cat((attention_pool, mean_pool), dim=1))

    def forward(self, x, grl_lambda=0.0, return_domain=False):
        embedding = self.encode(x)
        class_logits = self.class_head(embedding)
        if return_domain:
            domain_logits = self.domain_head(_e3_grad_reverse(embedding, grl_lambda))
            return class_logits, domain_logits
        return class_logits


def normalize_e3_hand_sequence(X: np.ndarray) -> np.ndarray:
    """Chuẩn hóa theo cả sequence, đồng thời giữ trajectory của wrist."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or X.shape[1] != E3_SEQ_LEN or X.shape[2] not in E3_VALID_FEATURE_DIMS:
        raise ValueError(f"Expected (N, {E3_SEQ_LEN}, 63/126), got {X.shape}")

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
            sequence_scale = max(float(np.median(frame_scale[sample_index, valid_indices])), 1e-6)
            result[sample_index, valid_indices, hand_index, 0, :] = (
                wrist[sample_index, valid_indices] - anchor
            ) / sequence_scale
            result[sample_index, valid_indices, hand_index, 1:, :] = (
                local[sample_index, valid_indices, 1:, :] / sequence_scale
            )

    return result.reshape(samples, steps, dims).astype(np.float32)


def _e3_time_warp(x: torch.Tensor, gamma: float) -> torch.Tensor:
    steps = x.shape[-2]
    u = torch.linspace(0.0, 1.0, steps, dtype=x.dtype, device=x.device)
    position = u.clamp_min(1e-6).pow(float(gamma)) * (steps - 1)
    position[0] = 0.0
    position[-1] = float(steps - 1)
    left = torch.floor(position).long()
    right = torch.clamp(left + 1, max=steps - 1)
    weight = position - left.to(position.dtype)
    if x.ndim == 2:
        return x[left] * (1.0 - weight[:, None]) + x[right] * weight[:, None]
    if x.ndim == 3:
        return x[:, left] * (1.0 - weight[None, :, None]) + x[:, right] * weight[None, :, None]
    raise ValueError(f"Unsupported x.ndim={x.ndim}")


def _e3_augment_sequence(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    steps, dims = x.shape
    hands = dims // 63

    if torch.rand(()) < 0.70:
        angle = (torch.rand(()) * 2.0 - 1.0) * np.deg2rad(E3_AUG_ROTATE_DEG)
        cosine, sine = torch.cos(angle), torch.sin(angle)
        points = x.reshape(steps, hands, 21, 3)
        xx, yy = points[..., 0].clone(), points[..., 1].clone()
        points[..., 0] = cosine * xx - sine * yy
        points[..., 1] = sine * xx + cosine * yy
        x = points.reshape(steps, dims)

    if torch.rand(()) < 0.80:
        valid = (x.abs().sum(dim=1, keepdim=True) > 1e-7).to(x.dtype)
        x = x + torch.randn_like(x) * E3_AUG_JITTER_STD * valid

    if torch.rand(()) < 0.50:
        points = x.reshape(steps, hands, 21, 3)
        z_scale = 1.0 + (torch.rand(()) * 2.0 - 1.0) * E3_AUG_Z_SCALE
        points[..., 2] *= z_scale
        x = points.reshape(steps, dims)

    if torch.rand(()) < 0.60:
        gamma = 1.0 + float((torch.rand(()) * 2.0 - 1.0) * E3_AUG_TIME_WARP)
        x = _e3_time_warp(x, gamma)
    return x


class _E3LandmarkDomainDataset(Dataset):
    def __init__(self, X, y, domains=None, augment=False):
        self.X = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y), dtype=torch.long)
        self.domains = (
            torch.full_like(self.y, -1)
            if domains is None
            else torch.tensor(np.asarray(domains), dtype=torch.long)
        )
        self.augment = bool(augment)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.X[index]
        if self.augment:
            x = _e3_augment_sequence(x)
        return x, self.y[index], self.domains[index]


def _e3_set_seed(seed=E3_SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def _e3_update_ema(ema_model, model, decay, step):
    effective_decay = min(float(decay), (1.0 + step) / (10.0 + step))
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.mul_(effective_decay).add_(parameter, alpha=1.0 - effective_decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def _e3_grl_schedule(progress):
    progress = float(np.clip(progress, 0.0, 1.0))
    return float(E3_DOMAIN_LAMBDA_MAX * (2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0))


def _e3_select_epoch(inner_results):
    common_length = min(len(item["history"]["val_acc"]) for item in inner_results)
    accuracies = np.stack([
        np.asarray(item["history"]["val_acc"][:common_length], dtype=float)
        for item in inner_results
    ])
    losses = np.stack([
        np.asarray(item["history"]["val_loss"][:common_length], dtype=float)
        for item in inner_results
    ])
    mean_accuracy = accuracies.mean(axis=0)
    mean_loss = losses.mean(axis=0)
    smoothed = mean_accuracy.copy()
    if common_length >= 3:
        smoothed[1:-1] = (
            mean_accuracy[:-2] + mean_accuracy[1:-1] + mean_accuracy[2:]
        ) / 3.0
    candidates = np.flatnonzero(np.isclose(smoothed, smoothed.max(), atol=1e-12))
    best_index = int(candidates[np.argmin(mean_loss[candidates])])
    return best_index + 1, {
        "common_len": common_length,
        "mean_val_acc": mean_accuracy.tolist(),
        "smooth_mean_val_acc": smoothed.tolist(),
        "mean_val_loss": mean_loss.tolist(),
    }


class E3Strategy:
    """
    E3 — Domain-invariant Motion BiLSTM trên toàn bộ chuỗi 45 frame.

    ``run_cross_subject`` gọi ``set_fold_context`` trước mỗi fold để strategy
    nhận subject labels phục vụ inner subject-CV và domain-adversarial training.
    """

    def __init__(self):
        self._train_groups = None
        self._test_person = None
        self.fold_histories = []

    def set_fold_context(self, train_records, test_records, test_person):
        self._train_groups = np.asarray([record["person"] for record in train_records])
        self._test_person = str(test_person)

    def prepare_input(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if E3_USE_HAND_NORMALIZATION:
            return normalize_e3_hand_sequence(X)
        return X

    def _make_loader(self, X, y, domains, shuffle, seed, augment=False):
        dataset = _E3LandmarkDomainDataset(X, y, domains=domains, augment=augment)
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=E3_BATCH_SIZE,
            shuffle=shuffle,
            num_workers=E3_NUM_WORKERS,
            pin_memory=(E3_DEVICE.type == "cuda"),
            generator=generator if shuffle else None,
            persistent_workers=(E3_NUM_WORKERS > 0),
        )

    @staticmethod
    def _new_model(input_dim, num_classes, num_domains):
        return E3DomainInvariantMotionBiLSTM(
            input_dim=input_dim,
            proj_dim=E3_PROJ_DIM,
            hidden_dim=E3_HIDDEN_DIM,
            num_layers=E3_NUM_LAYERS,
            num_classes=num_classes,
            num_domains=num_domains,
            dropout=E3_DROPOUT,
        ).to(E3_DEVICE)

    @staticmethod
    def _class_criterion():
        return nn.CrossEntropyLoss(label_smoothing=E3_LABEL_SMOOTHING)

    def _train_epoch(self, model, ema_model, loader, optimizer, epoch, max_epochs, global_step):
        model.train()
        class_criterion = self._class_criterion()
        domain_criterion = nn.CrossEntropyLoss()
        totals = {"loss": 0.0, "class_loss": 0.0, "domain_loss": 0.0,
                  "correct": 0, "domain_correct": 0, "samples": 0}

        for batch_index, (xb, yb, db) in enumerate(loader):
            xb = xb.to(E3_DEVICE, non_blocking=True)
            yb = yb.to(E3_DEVICE, non_blocking=True)
            db = db.to(E3_DEVICE, non_blocking=True)
            progress = ((epoch - 1) + (batch_index + 1) / max(len(loader), 1)) / max(max_epochs, 1)

            optimizer.zero_grad(set_to_none=True)
            logits, domain_logits = model(
                xb, grl_lambda=_e3_grl_schedule(progress), return_domain=True
            )
            class_loss = class_criterion(logits, yb)
            domain_loss = domain_criterion(domain_logits, db)
            loss = class_loss + domain_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1
            _e3_update_ema(ema_model, model, E3_EMA_DECAY, global_step)

            batch_size = xb.size(0)
            totals["loss"] += loss.item() * batch_size
            totals["class_loss"] += class_loss.item() * batch_size
            totals["domain_loss"] += domain_loss.item() * batch_size
            totals["correct"] += (logits.argmax(1) == yb).sum().item()
            totals["domain_correct"] += (domain_logits.argmax(1) == db).sum().item()
            totals["samples"] += batch_size

        count = max(totals["samples"], 1)
        return {
            "loss": totals["loss"] / count,
            "class_loss": totals["class_loss"] / count,
            "domain_loss": totals["domain_loss"] / count,
            "acc": totals["correct"] / count,
            "domain_acc": totals["domain_correct"] / count,
            "global_step": global_step,
        }

    def _evaluate(self, model, loader):
        model.eval()
        criterion = self._class_criterion()
        loss_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb, _ in loader:
                xb = xb.to(E3_DEVICE, non_blocking=True)
                yb = yb.to(E3_DEVICE, non_blocking=True)
                logits = model(xb)
                loss_sum += criterion(logits, yb).item() * xb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
                total += xb.size(0)
        return loss_sum / max(total, 1), correct / max(total, 1)

    def _run_inner_fold(self, X, y, groups, val_person, input_dim, num_classes, seed):
        validation_indices = np.flatnonzero(groups == val_person)
        train_indices = np.flatnonzero(groups != val_person)
        train_people = sorted(np.unique(groups[train_indices]).tolist())
        domain_map = {person: index for index, person in enumerate(train_people)}
        train_domains = np.asarray(
            [domain_map[person] for person in groups[train_indices]], dtype=np.int64
        )
        train_loader = self._make_loader(
            X[train_indices], y[train_indices], train_domains, True, seed, E3_USE_TRAIN_AUGMENT
        )
        validation_loader = self._make_loader(
            X[validation_indices], y[validation_indices], None, False, seed, False
        )

        _e3_set_seed(seed)
        model = self._new_model(input_dim, num_classes, len(train_people))
        ema_model = copy.deepcopy(model).eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=E3_LEARNING_RATE, weight_decay=E3_WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
        history = {"train_loss": [], "train_acc": [], "train_domain_acc": [],
                   "val_loss": [], "val_acc": [], "lr": []}
        best_epoch, best_accuracy, best_loss = 1, -1.0, float("inf")
        patience = 0
        global_step = 0

        for epoch in range(1, E3_INNER_MAX_EPOCHS + 1):
            train_metrics = self._train_epoch(
                model, ema_model, train_loader, optimizer,
                epoch, E3_INNER_MAX_EPOCHS, global_step
            )
            global_step = train_metrics["global_step"]
            validation_loss, validation_accuracy = self._evaluate(ema_model, validation_loader)
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["acc"])
            history["train_domain_acc"].append(train_metrics["domain_acc"])
            history["val_loss"].append(validation_loss)
            history["val_acc"].append(validation_accuracy)
            history["lr"].append(optimizer.param_groups[0]["lr"])

            improved = validation_accuracy > best_accuracy + 1e-12 or (
                abs(validation_accuracy - best_accuracy) <= 1e-12
                and validation_loss < best_loss
            )
            if improved:
                best_epoch, best_accuracy, best_loss = epoch, validation_accuracy, validation_loss
                patience = 0
            else:
                patience += 1
            scheduler.step()
            if epoch == 1 or epoch % 10 == 0 or patience >= E3_INNER_PATIENCE:
                print(
                    f"    inner val={val_person} epoch={epoch}/{E3_INNER_MAX_EPOCHS} "
                    f"train_acc={train_metrics['acc']:.3f} val_acc={validation_accuracy:.3f}"
                )
            if patience >= E3_INNER_PATIENCE:
                break

        model.to("cpu")
        ema_model.to("cpu")
        del model, ema_model, optimizer, scheduler, train_loader, validation_loader
        gc.collect()
        if E3_DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "val_person": val_person,
            "train_people": train_people,
            "best_epoch": int(best_epoch),
            "best_val_acc": float(best_accuracy),
            "best_val_loss_at_best_acc": float(best_loss),
            "history": history,
        }

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        if self._train_groups is None or len(self._train_groups) != len(X_train):
            raise RuntimeError(
                "E3Strategy cần subject labels. Hãy gọi qua run_cross_subject() "
                "hoặc set_fold_context() trước train()."
            )
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train)
        groups = self._train_groups
        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(y_train)
        people = sorted(np.unique(groups).tolist())
        if len(people) < 2:
            raise ValueError(f"E3 cần ít nhất 2 train subjects, got {people}")

        input_dim = X_train.shape[2]
        num_classes = len(encoder.classes_)
        inner_results = []
        for index, validation_person in enumerate(people):
            inner_results.append(self._run_inner_fold(
                X_train, encoded_labels, groups, validation_person,
                input_dim, num_classes, E3_SEED + 100 * (index + 1)
            ))
        final_epochs, epoch_selection = _e3_select_epoch(inner_results)
        print(
            f"[E3 test={self._test_person}] inner best epochs="
            f"{[item['best_epoch'] for item in inner_results]}, selected={final_epochs}"
        )

        domain_map = {person: index for index, person in enumerate(people)}
        domains = np.asarray([domain_map[person] for person in groups], dtype=np.int64)
        final_seed = E3_SEED + 1000
        _e3_set_seed(final_seed)
        loader = self._make_loader(
            X_train, encoded_labels, domains, True, final_seed, E3_USE_TRAIN_AUGMENT
        )
        model = self._new_model(input_dim, num_classes, len(people))
        ema_model = copy.deepcopy(model).eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=E3_LEARNING_RATE, weight_decay=E3_WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
        final_history = {"train_loss": [], "train_class_loss": [],
                         "train_domain_loss": [], "train_acc": [],
                         "train_domain_acc": [], "lr": []}
        global_step = 0
        for epoch in range(1, final_epochs + 1):
            metrics = self._train_epoch(
                model, ema_model, loader, optimizer, epoch, final_epochs, global_step
            )
            global_step = metrics["global_step"]
            for key in ("loss", "class_loss", "domain_loss", "acc", "domain_acc"):
                final_history[f"train_{key}"].append(metrics[key])
            final_history["lr"].append(optimizer.param_groups[0]["lr"])
            scheduler.step()
            if epoch == 1 or epoch % 10 == 0 or epoch == final_epochs:
                print(
                    f"[E3 test={self._test_person}] epoch={epoch}/{final_epochs} "
                    f"loss={metrics['loss']:.4f} acc={metrics['acc']:.3f}"
                )

        model.to("cpu")
        del model, optimizer, scheduler, loader
        gc.collect()
        ema_model.to(E3_DEVICE).eval()
        state = {
            "models": [ema_model],
            "le": encoder,
            "inner": inner_results,
            "epoch_selection": epoch_selection,
            "final_history": final_history,
            "final_epochs": final_epochs,
            "input_dim": input_dim,
            "outer_train_people": people,
        }
        # Chỉ giữ metadata/history; không giữ model của các fold cũ trên GPU.
        self.fold_histories.append({
            "test_person": self._test_person,
            "inner": inner_results,
            "epoch_selection": epoch_selection,
            "final_history": final_history,
            "final_epochs": final_epochs,
            "outer_train_people": people,
        })
        return state

    @staticmethod
    def _ensemble_logits(models, inputs):
        logits = []
        for gamma in E3_TTA_GAMMAS:
            view = inputs if abs(gamma - 1.0) < 1e-12 else _e3_time_warp(inputs, gamma)
            logits.extend(model(view) for model in models)
        return torch.stack(logits, dim=0).mean(dim=0)

    def predict(self, model_state, X_test: np.ndarray) -> np.ndarray:
        models = model_state["models"]
        for model in models:
            model.eval()
        prediction_ids = []
        X_test = np.asarray(X_test, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(X_test), E3_BATCH_SIZE):
                inputs = torch.tensor(
                    X_test[start:start + E3_BATCH_SIZE], dtype=torch.float32, device=E3_DEVICE
                )
                prediction_ids.append(
                    self._ensemble_logits(models, inputs).argmax(dim=1).cpu().numpy()
                )
        return model_state["le"].inverse_transform(np.concatenate(prediction_ids))

    def measure_latency(self, model_state, X_sample: np.ndarray) -> float:
        models = model_state["models"]
        for model in models:
            model.eval()
        sample = torch.tensor(X_sample[:1], dtype=torch.float32, device=E3_DEVICE)
        with torch.no_grad():
            for _ in range(5):
                self._ensemble_logits(models, sample)
        repeats = 30
        if E3_DEVICE.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(repeats):
                self._ensemble_logits(models, sample)
        if E3_DEVICE.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0 / repeats


if __name__ == "__main__":
    # Ghi đè đường dẫn cấu hình bằng kết quả tự dò trong vùng E1; không cần sửa
    # phần khung dùng chung khi Kaggle mount dataset với dấu '_' thay vì '-'.
    LANDMARK_DIR = resolve_e1_landmark_dir()
    print(f"E1 dùng LANDMARK_DIR: {LANDMARK_DIR}")
    print(f"E1 đang huấn luyện trên: {DEVICE}")
    records = load_all_landmarks()
    if not records:
        raise FileNotFoundError(
            f"Không tìm thấy file .npy trong {LANDMARK_DIR}. "
            "Hãy Add Data dataset vietnamese-sign-language-alphabet trên Kaggle."
        )

    print(f"Tổng số mẫu đọc được: {len(records)}")
    e1_results = run_cross_subject(records, E1Strategy())
    accuracies = [result["accuracy"] for result in e1_results]
    latencies = [result["latency_ms"] for result in e1_results]
    summary = {
        "experiment": "E1",
        "description": (
            "MLP tren landmark frame index=22, wrist-centering va scale-normalization"
        ),
        "results": e1_results,
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_std_ms": float(np.std(latencies)),
    }
    result_path = E1_OUTPUT_DIR / "E1_results.json"
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu kết quả E1 tại: {result_path}")
