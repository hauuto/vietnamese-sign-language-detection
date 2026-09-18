"""
Cấu hình + khung đánh giá dùng chung cho cả 4 thực nghiệm (E0, E1, E2, E3).
Copy nguyên file này vào đầu notebook Kaggle — không sửa gì trong này, chỉ viết thêm
1 class Strategy riêng cho mô hình của bạn ở cuối, theo đúng mẫu trong comment.
"""
import re
from pathlib import Path
from collections import defaultdict
from typing import Protocol

import numpy as np

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

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,           # input shape: (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x):
        # x: (batch, 45, D)
        out, _ = self.lstm(x)           # out: (batch, 45, hidden_dim)
        last    = out[:, -1, :]         # lấy bước cuối: (batch, hidden_dim)
        last    = self.dropout(last)
        return self.fc(last)            # (batch, num_classes)

import re, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib

# Siêu tham số — chỉnh ở đây nếu muốn thử nghiệm
HIDDEN_DIM  = 128
NUM_LAYERS  = 3
DROPOUT     = 0.5
LR          = 1e-4
EPOCHS      = 1000
BATCH_SIZE  = 64
PATIENCE    = 10  
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
 
print(f"Device: {DEVICE}")
 

def normalize_landmarks(X):
    """
    Chuẩn hóa landmark: dời về wrist + chia theo khoảng cách wrist→MCP ngón giữa.

    X: (N, T, D) với D = 63 (1 tay) hoặc 126 (2 tay)
       Mỗi tay gồm 21 điểm × 3 tọa độ (x, y, z) xen kẽ.

    Trả về: (N, T, D) float32 đã chuẩn hóa.
    """
    X_norm = X.copy().astype(np.float32)
    D = X_norm.shape[-1]
    num_coords_per_hand = 63  # 21 landmarks × 3

    # Xử lý từng bàn tay
    for hand_offset in range(0, D, num_coords_per_hand):
        hand = X_norm[..., hand_offset:hand_offset + num_coords_per_hand]
        # hand shape: (N, T, 63)

        # Tọa độ wrist (landmark 0): indices 0, 1, 2
        wrist_x = hand[..., 0:1]  # (N, T, 1)
        wrist_y = hand[..., 1:2]
        wrist_z = hand[..., 2:3]

        # Dời gốc về wrist
        hand[..., 0::3] -= wrist_x
        hand[..., 1::3] -= wrist_y
        hand[..., 2::3] -= wrist_z

        # Tọa độ middle finger MCP (landmark 9): indices 27, 28, 29
        mcp_x = hand[..., 27:28]
        mcp_y = hand[..., 28:29]
        mcp_z = hand[..., 29:30]

        # Khoảng cách wrist → MCP ngón giữa (sau khi đã dời, wrist = 0)
        dist = np.sqrt(mcp_x**2 + mcp_y**2 + mcp_z**2)
        dist = np.maximum(dist, 1e-6)  # tránh chia cho 0

        # Chia toàn bộ tọa độ cho dist → scale invariant
        hand[..., 0::3] /= dist
        hand[..., 1::3] /= dist
        hand[..., 2::3] /= dist

        X_norm[..., hand_offset:hand_offset + num_coords_per_hand] = hand

    return X_norm

def augment(
    X,
    noise_std=0.5,
    shift_range=0.05,
    zoom_range=(0.8, 1.1),
    rotation_range=(-15, 15)
):
    """
    Augmentation cho landmark bàn tay.

    X:
        (N, T, 63) hoặc (N, T, 126)

    Gồm:
        1. Gaussian Noise
        2. Shifting
        3. Zooming
        4. Spatial Rotation
    """

    X_aug = X.copy().astype(np.float32)

    # =========================
    # 1. GAUSSIAN NOISE
    # =========================
    noise = np.random.normal(
        0,
        noise_std,
        X_aug.shape
    ).astype(np.float32)

    X_aug += noise

    # =========================
    # 2. SHIFTING
    # =========================
    shift_x = np.random.uniform(
        -shift_range,
        shift_range,
        size=(X_aug.shape[0], 1, 1)
    ).astype(np.float32)

    shift_y = np.random.uniform(
        -shift_range,
        shift_range,
        size=(X_aug.shape[0], 1, 1)
    ).astype(np.float32)

    X_aug[..., 0::3] += shift_x
    X_aug[..., 1::3] += shift_y

    # =========================
    # 3. ZOOMING
    # =========================
    zoom = np.random.uniform(
        zoom_range[0],
        zoom_range[1],
        size=(X_aug.shape[0], 1, 1)
    ).astype(np.float32)

    x = X_aug[..., 0::3]
    y = X_aug[..., 1::3]

    x_center = np.mean(x, axis=-1, keepdims=True)
    y_center = np.mean(y, axis=-1, keepdims=True)

    X_aug[..., 0::3] = (
        (x - x_center) * zoom + x_center
    )

    X_aug[..., 1::3] = (
        (y - y_center) * zoom + y_center
    )

    # =========================
    # 4. SPATIAL ROTATION
    # =========================
    angles = np.random.uniform(
        rotation_range[0],
        rotation_range[1],
        size=X_aug.shape[0]
    )

    angles = np.deg2rad(angles)

    cos_a = np.cos(angles)[:, None, None]
    sin_a = np.sin(angles)[:, None, None]

    x = X_aug[..., 0::3]
    y = X_aug[..., 1::3]

    x_center = np.mean(x, axis=-1, keepdims=True)
    y_center = np.mean(y, axis=-1, keepdims=True)

    x = x - x_center
    y = y - y_center

    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a

    X_aug[..., 0::3] = x_rot + x_center
    X_aug[..., 1::3] = y_rot + y_center

    return X_aug

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, augment_fn=None):
        self.X = X          # (N, 45, D) float32
        self.y = y          # (N,) long
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]     # (45, D)
        if self.augment_fn is not None:
            # augment nhận (1, 45, D), trả (1, 45, D)
            x = self.augment_fn(x[None])[0]
        return torch.tensor(x, dtype=torch.float32), self.y[idx]
 
class E2Strategy:
    """
    E2 — LSTM một chiều trên toàn bộ chuỗi 45 bước (Vỹ).
 
    prepare_input: giữ nguyên chuỗi (N, 45, D) — KHÔNG lấy 1 khung như E0/E1.
    train        : LabelEncoder → TensorDataset → train loop.
    predict      : argmax trên logits.
    measure_latency: đo thời gian suy luận 1 mẫu, lặp 100 lần để ổn định.
    """
 
    def prepare_input(self, X: np.ndarray) -> np.ndarray:
        # E2/E3: giữ nguyên toàn bộ chuỗi (N, 45, D)
        X = X.astype(np.float32)
        # Chuẩn hóa landmark trước khi đưa vào model
        X = normalize_landmarks(X)
        return X.astype(np.float32)
 
    # ------------------------------------------------------------------
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trả về dict chứa model, label_encoder — dùng lại ở predict/latency.
        """
        le = LabelEncoder()
        y_enc = le.fit_transform(y_train)           # string label → int
 
        num_classes = len(le.classes_)
        input_dim   = X_train.shape[2]              # 63 hoặc 126
 
        model = LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_classes=num_classes,
            dropout=DROPOUT,
        ).to(DEVICE)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_enc, test_size=0.15, stratify=y_enc, random_state=42
        )

        train_dataset = AugmentedDataset(X_tr, torch.tensor(y_tr, dtype=torch.long), augment_fn=augment)
        val_dataset   = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                       torch.tensor(y_val, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

        # Checkpoint: lưu model có train loss tốt nhất
        best_val_loss = float("inf")
        best_weights = None
        no_improve = 0
 
        # Thêm vào hàm train(), sau khi có y_enc
        from collections import Counter
        counts = Counter(y_tr)
        weights = torch.tensor(
            [1.0 / counts[i] for i in range(num_classes)], dtype=torch.float32
        ).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # cosine annealing để lr giảm dần, tránh dao động cuối
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
 
        history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        # Lưu history của từng fold
        if not hasattr(self, "all_histories"):
            self.all_histories = []

        model.train()
        for epoch in range(1, EPOCHS + 1):
            running_loss, correct, total = 0.0, 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                logits = model(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
 
                running_loss += loss.item() * xb.size(0)
                correct      += (logits.argmax(1) == yb).sum().item()
                total        += xb.size(0)
 
            scheduler.step()
            epoch_loss = running_loss / total
            epoch_acc  = correct / total
            history["loss"].append(epoch_loss)
            history["acc"].append(epoch_acc)

            model.eval()

            val_loss = 0.0
            val_correct = 0

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                    logits = model(xb)
                    loss = criterion(logits, yb)

                    val_loss += loss.item() * xb.size(0)

                    preds = logits.argmax(dim=1)
                    val_correct += (preds == yb).sum().item()

            val_loss /= len(X_val)
            val_acc = val_correct / len(y_val)

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            model.train()

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_weights = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                torch.save(best_weights, "E2_best_checkpoint.pt")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"Early stopping at epoch {epoch} (val_loss={val_loss:.4f})")
                    break
 
            if epoch % 20 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={epoch_loss:.4f}  acc={epoch_acc:.3f}  val_loss={val_loss:.4f}")
        
        # Load lại checkpoint tốt nhất
        model.load_state_dict(best_weights)
        model.to(DEVICE)

        print(f"Loaded best checkpoint — val_loss={best_val_loss:.4f}")

        # Lưu history của fold hiện tại
        self.all_histories.append(history)
        
        return {"model": model, "le": le, "history": history, "input_dim": input_dim}
 
    # ------------------------------------------------------------------
    def predict(self, model_state, X_test: np.ndarray) -> np.ndarray:
        model = model_state["model"]
        le    = model_state["le"]
        model.eval()
        with torch.no_grad():
            X_t    = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
            logits = model(X_t)
            preds  = logits.argmax(dim=1).cpu().numpy()
        return le.inverse_transform(preds)             # trả về string label
 
    # ------------------------------------------------------------------
    def measure_latency(self, model_state, X_sample: np.ndarray) -> float:
        """Đo latency trung bình cho 1 mẫu (ms), lặp 100 lần để ổn định."""
        model = model_state["model"]
        model.eval()
        x = torch.tensor(X_sample, dtype=torch.float32).to(DEVICE)
 
        # warmup
        with torch.no_grad():
            for _ in range(10):
                model(x)
 
        # đo chính thức
        N = 100
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(N):
                model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000 / N
        return elapsed_ms