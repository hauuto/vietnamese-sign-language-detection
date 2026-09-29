# Nhận dạng ngôn ngữ ký hiệu tiếng Việt (VSL) — Nhóm 9

Nhận dạng 34 lớp ký hiệu bảng chữ cái VSL (22 chữ cái tĩnh, 7 chữ có dấu phụ, 5 thanh điệu) từ webcam.
Landmark bàn tay được trích bằng MediaPipe HandLandmarker, sau đó phân loại bằng MLP, LSTM hoặc Bi-LSTM.
Mô hình được đánh giá theo giao thức Leave-One-Subject-Out (LOSO) trên 4 người thực hiện.

| Thực nghiệm | Mô hình | Accuracy LOSO |
|---|---|---|
| E0 | MLP, landmark thô, 1 khung | 40,8 ± 6,5% |
| E1 | MLP, landmark chuẩn hóa, 1 khung | 53,1 ± 8,8% |
| E2 | AttentionLSTM một chiều, 45 khung | 64,7 ± 12,2% |
| E3 | Bi-LSTM (cấu hình E2) | 65,2 ± 10,9% |
| E3+ | Bi-LSTM mở rộng (vector xương, tích chập thời gian, GRL, EMA) | 78,3 ± 8,2% |

Dữ liệu: https://www.kaggle.com/datasets/hauuto/vietnamese-sign-language-alphabet

---

## 1. Cài đặt môi trường

Cần **Python 3.10 hoặc 3.11** (mediapipe chưa hỗ trợ ổn định Python 3.12 trở lên).

```bash
python --version

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` cần có tối thiểu: `numpy`, `opencv-python`, `mediapipe`, `torch`, `matplotlib`, `pillow`.
Pillow dùng để hiển thị chữ có dấu trên giao diện demo; thiếu Pillow thì demo vẫn chạy nhưng chữ không dấu.

Kiểm tra cài đặt:

```bash
python -c "import cv2, mediapipe, numpy, torch; print('OK')"
python src/classes.py
```

> Trên Windows dùng `python`; trên macOS/Linux có thể cần `python3`.

---

## 2. Cấu trúc thư mục

```
vsl-project/
├── src/
│   ├── classes.py              # NGUỒN DUY NHẤT cho danh sách 34 lớp — chỉ sửa ở đây
│   ├── record.py               # quay video dữ liệu
│   ├── extract_landmarks.py    # trích landmark -> .npy (45, 63)
│   └── check_landmarks.py      # kiểm tra chất lượng landmark
├── scripts/
│   └── eda.py                  # phân tích khám phá dữ liệu, vẽ hình cho báo cáo
├── notebooks/                  # notebook huấn luyện E0, E0+, E1, E2, E3, E3+
├── demo.py                     # demo nhận dạng trực tiếp từ webcam
├── raw/{person}/               # video gốc                      — KHÔNG đưa vào git
├── landmarks/raw/{lớp}/        # file .npy đã trích              — KHÔNG đưa vào git
├── models/
│   ├── hand_landmarker.task    # model MediaPipe, tự tải lần đầu — KHÔNG đưa vào git
│   ├── baseline/               # checkpoint E0, E1, E2, E3
│   └── extended/               # checkpoint E0+, E3+
├── outputs/                    # hình EDA, ảnh chụp demo
└── requirements.txt
```

Tên file landmark theo quy ước `{lớp}_{người}_{block}_{số thứ tự}.npy`, ví dụ `aw_hau_A_001.npy`.

---

## 3. Chạy nhanh demo (không cần quay lại dữ liệu)

1. Đặt checkpoint vào thư mục `models/`:
   - `models/baseline/`: `E0_*.pt`, `E1_*.pt`, `E2_*.pt`, `E3_*.pt` (mỗi thực nghiệm 1 hoặc 4 file fold).
   - `models/extended/` (không bắt buộc): `E0_*.pt` cho E0+, `E3_*.pt` cho E3+.
   - Tên file phải **bắt đầu bằng mã thực nghiệm**, ví dụ `E3_fold1.pt`.
2. Kiểm tra checkpoint mà không mở camera:
   ```bash
   python demo.py --self-test
   ```
3. Chạy demo:
   ```bash
   python demo.py                         # webcam mặc định, chế độ baseline
   python demo.py --camera 1              # chọn webcam khác
   python demo.py --mode extended         # khởi động ở chế độ mở rộng (E3+)
   python demo.py --signer-seen yes       # ghi chú người demo CÓ trong dữ liệu huấn luyện
   python demo.py --camera-url http://192.168.1.10:8080/video   # camera điện thoại qua IP
   ```
   Lần chạy đầu sẽ tự tải `hand_landmarker.task` (khoảng 10 MB, cần mạng một lần).

| Phím | Tác dụng |
|---|---|
| `Q` | Thoát |
| `R` | Đặt lại (mở khóa tay, xóa bộ đệm) |
| `C` | Xóa chuỗi ký hiệu đã nhận |
| `P` | Chụp giao diện, lưu vào `outputs/screenshots/` |
| `M` | Đổi chế độ baseline ↔ mở rộng |
| `D` | Bật/tắt chi tiết kỹ thuật (ngưỡng, phiếu, độ trễ) |
| `S` | Đổi chế độ làm mượt landmark: off → display → all |
| `1` / `2` | Giảm / tăng ngưỡng xác suất top-1 |
| `3` / `4` | Giảm / tăng ngưỡng margin |
| `5` / `6` | Giảm / tăng ngưỡng đồng thuận |

Cách dùng:
- Đưa **đúng một bàn tay** vào khung và giữ yên khoảng 8 khung để hệ thống khóa tay.
  Muốn đổi tay thì rút hết tay ra khỏi khung, rồi chỉ đưa tay muốn dùng vào.
- Mỗi ký hiệu nên giữ khoảng 2 giây, để đủ cửa sổ 45 khung và 4 lần dự đoán liên tiếp giống nhau.
- Một ký hiệu chỉ được nhận khi xác suất top-1 ≥ 75%, margin ≥ 35% và đồng thuận ≥ 75% (6/8 phiếu).
  Nếu không đạt, giao diện hiện `?` và ghi rõ tiêu chí không đạt.

Tùy chọn khác: `python demo.py --help`.

> Accuracy trong báo cáo lấy từ đánh giá LOSO. Không dùng demo để chấm điểm lại trên dữ liệu đã huấn luyện:
> mỗi mẫu đã nằm trong dữ liệu huấn luyện của 3/4 mô hình fold nên kết quả sẽ cao giả tạo.

---

## 4. Tái tạo toàn bộ quy trình

### 4.1. Quay dữ liệu

```bash
python src/record.py
```

Nhập tên người (`hau` / `khoi` / `tai` / `vy`) và block (`A` / `B`) khi được hỏi.
Mỗi người quay 4 mẫu cho mỗi lớp tĩnh và 6 mẫu cho mỗi lớp động (tổng 160 mẫu/người), mỗi video dài 3 giây.

| Phím | Tác dụng |
|---|---|
| `SPACE` | Đếm ngược rồi ghi mẫu hiện tại |
| `N` | Bỏ qua, đẩy mẫu xuống cuối hàng đợi để quay lại sau |
| `L` | Chế độ quay dài (clip đánh vần) |
| `Q` | Lưu tiến độ và thoát; chạy lại `record.py` sẽ tiếp tục đúng chỗ dừng |

File `progress_{person}_{block}.json` ở thư mục gốc ghi lại mẫu nào đã quay xong.
Xóa file này nếu muốn quay lại từ đầu.

Sau khi quay xong, nên đặt `raw/` về chỉ đọc và sao lưu sang nơi khác:

```bash
chmod -R a-w raw/      # macOS / Linux
```

### 4.2. Trích landmark

```bash
python src/extract_landmarks.py --source raw
```

Mỗi video được lấy mẫu lại về 45 khung và lưu thành một file `.npy` kích thước `(45, 63)`.
Khung không phát hiện được bàn tay được gán toàn 0.
Script bỏ qua các file `.npy` đã có, nên có thể chạy lại nếu bị ngắt giữa chừng.

### 4.3. Kiểm tra chất lượng và EDA

```bash
python src/check_landmarks.py     # liệt kê mẫu có hơn 20% số khung mất tay (chỉ để kiểm tra, không loại mẫu)
python scripts/eda.py             # hình cho báo cáo (300 dpi), lưu vào outputs/eda/
python scripts/eda.py --titles    # bản có tiêu đề trong hình, dùng cho slide
```

`outputs/eda/` gồm 4 hình và các file `eda_summary.txt`, `so_mau_nguoi_lop.csv`, `chi_tiet_mau.csv`
chứa toàn bộ số liệu dùng trong báo cáo.

### 4.4. Huấn luyện và đánh giá

Các notebook trong `notebooks/` huấn luyện và đánh giá E0, E0+, E1, E2, E3, E3+ theo giao thức LOSO 4 fold
(huấn luyện trên 3 người, kiểm tra trên người còn lại).

- E0, E0+, E1: chọn checkpoint trên 15% validation tách theo mẫu từ 3 người huấn luyện.
- E2, E3, E3+: chọn số epoch bằng inner subject-CV trên 3 người huấn luyện.

Mỗi notebook xuất checkpoint 4 fold và file kết quả JSON (accuracy, Macro-F1, độ trễ).
Chép checkpoint sang `models/baseline/` hoặc `models/extended/` theo mục 3 để chạy demo.

---

## 5. Xử lý lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `Không mở được camera index 0` | Thử `--camera 1` hoặc `--camera 2`; đóng ứng dụng khác đang dùng webcam |
| `[E3] không có checkpoint baseline -> bỏ qua` | Kiểm tra tên file trong `models/baseline/` có bắt đầu bằng `E3` không |
| `Các thực nghiệm phải cùng input_dim` | Không trộn checkpoint một tay (63) và hai tay (126) |
| Chữ trên giao diện không có dấu | `pip install pillow` |
| Landmark rung nhiều | Nhấn `S` để đổi chế độ làm mượt; tăng ánh sáng và dùng nền trơn |
| FPS thấp | Tắt TTA bằng `--no-tta`; giảm tần suất dự đoán bằng `--stride 5` |
| `UnicodeEncodeError` trên terminal Windows | Chạy `chcp 65001` trước khi chạy script |

---

## 6. Nhóm thực hiện

Nhóm 9 — Học phần Thị giác máy tính và ứng dụng, Trường Đại học Công nghiệp TP. Hồ Chí Minh.

Tô Thanh Hậu (nhóm trưởng) · Võ Tấn Tài · Đặng Thế Vỹ · Nguyễn Thanh Khôi