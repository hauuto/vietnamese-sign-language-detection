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
