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