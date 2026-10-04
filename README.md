# Nhận dạng ngôn ngữ ký hiệu tiếng Việt (VSL) — Nhóm 9

Hệ thống nhận dạng 34 lớp ký hiệu bảng chữ cái ngón tay VSL trực tiếp từ webcam.

- **23 lớp tĩnh:** 22 chữ cái cơ bản và chữ Đ (Đ là ký hiệu tĩnh theo Thông tư 17/2020/TT-BGDĐT).
- **11 lớp động:** 6 chữ có dấu phụ (Ă, Â, Ê, Ô, Ơ, Ư) và 5 thanh điệu (sắc, huyền, hỏi, ngã, nặng).

MediaPipe HandLandmarker trích 21 điểm mốc bàn tay ở mỗi khung hình. Chuỗi điểm mốc sau đó được phân loại bằng MLP, LSTM hoặc Bi-LSTM.
Mọi mô hình được đánh giá theo giao thức Leave-One-Subject-Out (LOSO) 4 fold: huấn luyện trên 3 người, kiểm tra trên người còn lại.

| Thực nghiệm | Mô hình | Accuracy (%) | Macro-F1 (%) |
|---|---|---|---|
| E0 | MLP, landmark thô, 1 khung | 40,8 ± 6,5 | 38,4 ± 6,9 |
| E1 | MLP, landmark chuẩn hóa, 1 khung | 53,1 ± 8,8 | 52,5 ± 10,3 |
| E2 | AttentionLSTM một chiều, 45 khung | 64,7 ± 12,2 | 61,9 ± 12,5 |
| E3 | Bi-LSTM (giữ nguyên cấu hình E2) | 65,2 ± 10,9 | 63,4 ± 12,0 |
| E0+ (R2) | Nhóm mở rộng: E0 + z-score, 300 epoch | 51,4 ± 6,7 | 50,1 ± 7,6 |
| E3+ | Nhóm mở rộng: Bi-LSTM + vector xương, tích chập thời gian, GRL, EMA | 78,3 ± 8,2 | 78,1 ± 7,6 |

Kết quả là trung bình ± độ lệch chuẩn mẫu trên 4 người kiểm tra.
E0–E3 dùng để kiểm định giả thuyết. E0+ và E3+ thay đổi nhiều thành phần cùng lúc, chỉ dùng để xem mức accuracy có thể đạt được.

- Mã nguồn: https://github.com/hauuto/vietnamese-sign-language-detection
- Dữ liệu (640 mẫu, 4 người): https://www.kaggle.com/datasets/hauuto/vietnamese-sign-language-alphabet

---

## 1. Cài đặt môi trường

Cần **Python 3.10 hoặc 3.11**. MediaPipe chưa hỗ trợ ổn định Python 3.12 trở lên.

```bash
python --version

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` cần tối thiểu: `numpy`, `opencv-python`, `mediapipe`, `torch`, `matplotlib`, `pillow`.
Pillow dùng để vẽ chữ có dấu trên giao diện demo. Thiếu Pillow, demo vẫn chạy nhưng chữ hiện không dấu.

Kiểm tra cài đặt:

```bash
python -c "import cv2, mediapipe, numpy, torch; print('OK')"
python src/classes.py
```

> Trên Windows dùng `python`. Trên macOS/Linux có thể cần `python3`.

---

## 2. Cấu trúc thư mục

```
vsl-project/
├── src/
│   ├── classes.py              # nguồn duy nhất cho danh sách 34 lớp — chỉ sửa ở đây
│   ├── record.py               # quay video dữ liệu
│   ├── extract_landmarks.py    # trích landmark -> .npy (45, 63)
│   └── check_landmarks.py      # kiểm tra chất lượng landmark
├── scripts/
│   └── eda.py                  # phân tích khám phá dữ liệu, vẽ hình cho báo cáo
├── notebooks/                  # notebook huấn luyện E0, E0+, E1, E2, E3, E3+
├── demo.py                     # demo nhận dạng trực tiếp từ webcam
├── raw/{người}/                # video gốc                       — không đưa vào git
├── landmarks/raw/{lớp}/        # file .npy đã trích               — không đưa vào git
├── models/
│   ├── hand_landmarker.task    # model MediaPipe, tự tải lần đầu  — không đưa vào git
│   ├── baseline/               # checkpoint E0, E1, E2, E3
│   └── extended/               # checkpoint E0+, E3+
├── outputs/                    # hình EDA, ảnh chụp demo
└── requirements.txt
```

Tên file landmark theo quy ước `{lớp}_{người}_{block}_{số thứ tự}.npy`, ví dụ `aw_hau_A_001.npy`.

| Mã lớp | Ký hiệu | Mã lớp | Ký hiệu |
|---|---|---|---|
| `aw` | Ă | `ow` | Ơ |
| `aa` | Â | `uw` | Ư |
| `dd` | Đ | `tone_s` | dấu sắc |
| `ee` | Ê | `tone_f` | dấu huyền |
| `oo` | Ô | `tone_r` | dấu hỏi |
| | | `tone_x` | dấu ngã |
| | | `tone_j` | dấu nặng |

Các chữ cái cơ bản dùng chính chữ thường làm mã lớp (`a`, `b`, `c`, …).

---

## 3. Chạy nhanh demo (không cần quay lại dữ liệu)

**Bước 1. Đặt checkpoint vào thư mục `models/`.**

- `models/baseline/`: `E0_*.pt`, `E1_*.pt`, `E2_*.pt`, `E3_*.pt`. Mỗi thực nghiệm có 1 hoặc 4 file fold.
- `models/extended/` (không bắt buộc): `E0_*.pt` cho E0+, `E3_*.pt` cho E3+.
- Tên file phải **bắt đầu bằng mã thực nghiệm**, ví dụ `E3_fold1.pt`.
- Thực nghiệm không có bản mở rộng sẽ tự dùng bản baseline khi chuyển chế độ.

**Bước 2. Kiểm tra checkpoint mà không mở camera.**

```bash
python demo.py --self-test
```

**Bước 3. Chạy demo.**

```bash
python demo.py                         # webcam mặc định, chế độ baseline
python demo.py --camera 1              # chọn webcam khác
python demo.py --mode extended         # khởi động ở chế độ mở rộng (E3+)
python demo.py --signer-seen yes       # ghi chú người demo có trong dữ liệu huấn luyện
python demo.py --camera-url http://192.168.1.10:8080/video   # camera điện thoại qua IP
```

Lần chạy đầu, chương trình tự tải `hand_landmarker.task` về `models/`, nên cần có mạng một lần.

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

**Cách dùng**

- Đưa **đúng một bàn tay** vào khung và giữ khoảng 8 khung để hệ thống khóa tay.
  Muốn đổi tay, rút hết tay ra khỏi khung rồi chỉ đưa tay muốn dùng vào.
- Giữ mỗi ký hiệu khoảng 2 giây. Thời gian này đủ để có cửa sổ 45 khung và 4 lần dự đoán liên tiếp giống nhau.
- Một ký hiệu chỉ được nhận khi thỏa cả ba ngưỡng: xác suất top-1 ≥ 75%, margin ≥ 35%, đồng thuận ≥ 75% (6/8 phiếu).
  Nếu không đạt, giao diện hiện `?` và ghi rõ tiêu chí chưa đạt.
- Xác suất hiển thị là đầu ra softmax, không phải độ tin cậy đã hiệu chỉnh.

Xem các tùy chọn khác: `python demo.py --help`.

> Accuracy trong bảng trên lấy từ đánh giá LOSO. Không dùng demo để chấm lại trên người đã có trong dữ liệu:
> người đó nằm trong dữ liệu huấn luyện của 3/4 mô hình fold, nên kết quả sẽ cao giả tạo.

---

## 4. Tái tạo toàn bộ quy trình

### 4.1. Quay dữ liệu

```bash
python src/record.py
```

Nhập tên người (`hau` / `khoi` / `tai` / `vy`) và block (`A` / `B`) khi được hỏi.
Mỗi video dài 3 giây, có đếm ngược 5 giây trước khi ghi.

| Nhóm lớp | Số mẫu mỗi người | Mỗi block |
|---|---|---|
| 22 chữ cái cơ bản | 4 | 2 |
| Đ, 6 chữ có dấu phụ, 5 thanh điệu | 6 | 3 |
| **Tổng** | **160** | |

Chữ Đ được quay 6 mẫu theo quy cách thu thập, dù là ký hiệu tĩnh.

| Phím | Tác dụng |
|---|---|
| `SPACE` | Đếm ngược rồi ghi mẫu hiện tại |
| `N` | Bỏ qua, đẩy mẫu xuống cuối hàng đợi để quay sau |
| `L` | Chế độ quay dài (clip đánh vần) |
| `Q` | Lưu tiến độ và thoát; chạy lại `record.py` sẽ tiếp tục đúng chỗ dừng |

File `progress_{người}_{block}.json` ở thư mục gốc ghi lại các mẫu đã quay xong.
Xóa file này nếu muốn quay lại từ đầu.

Sau khi quay xong, nên đặt `raw/` ở chế độ chỉ đọc và sao lưu sang nơi khác:

```bash
chmod -R a-w raw/      # macOS / Linux
```

### 4.2. Trích landmark

```bash
python src/extract_landmarks.py --source raw
```

- Mỗi video được lấy mẫu lại về 45 khung và lưu thành một file `.npy` kích thước `(45, 63)`.
- Khung không phát hiện được bàn tay được gán toàn 0.
- File `.npy` chỉ lấy mẫu lại theo thời gian, chưa chuẩn hóa tọa độ. Mỗi thực nghiệm tự chuẩn hóa theo cách riêng.
- Script bỏ qua các file `.npy` đã có, nên có thể chạy lại nếu bị ngắt giữa chừng.

### 4.3. Kiểm tra chất lượng và EDA

```bash
python src/check_landmarks.py     # liệt kê mẫu có hơn 20% số khung mất tay (chỉ để kiểm tra, không loại mẫu)
python scripts/eda.py             # hình cho báo cáo (300 dpi), lưu vào outputs/eda/
python scripts/eda.py --titles    # bản có tiêu đề trong hình, dùng cho slide
```

`outputs/eda/` gồm:

| File | Nội dung |
|---|---|
| `01_so_mau_nguoi_lop.png` | Số mẫu theo người và theo lớp |
| `02_chat_luong_du_lieu.png` | Phân bố tỉ lệ khung mất tay, tổng hợp lỗi tệp |
| `03_chat_luong_theo_nguoi.png` | Tỉ lệ mất tay và tay sát mép theo người |
| `04_nang_luong_chuyen_dong.png` | Năng lượng chuyển động nhóm tĩnh / động và từng lớp |
| `eda_summary.txt` | Các con số dùng trong báo cáo |
| `so_mau_nguoi_lop.csv`, `chi_tiet_mau.csv` | Số liệu chi tiết theo lớp và theo từng mẫu |

### 4.4. Huấn luyện và đánh giá

Các notebook trong `notebooks/` huấn luyện và đánh giá E0, E0+, E1, E2, E3, E3+ theo giao thức LOSO 4 fold.

| Thực nghiệm | Cách chọn mô hình |
|---|---|
| E0, E0+, E1 | Checkpoint tốt nhất trên 15% validation tách theo mẫu từ 3 người huấn luyện |
| E2, E3, E3+ | Số epoch chọn bằng inner subject-CV trên 3 người huấn luyện, rồi huấn luyện lại |

Người kiểm tra không được dùng để chọn mô hình trong cả hai cách.
Mọi thực nghiệm dùng tăng cường lật ngang khi huấn luyện và TTA lật ngang khi kiểm tra.

Mỗi notebook xuất checkpoint 4 fold và file kết quả JSON (accuracy, Macro-F1, độ trễ).
Chép checkpoint sang `models/baseline/` hoặc `models/extended/` như ở mục 3 để chạy demo.

---

## 5. Xử lý lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `Không mở được camera index 0` | Thử `--camera 1` hoặc `--camera 2`; đóng ứng dụng khác đang dùng webcam |
| `[E3] không có checkpoint baseline -> bỏ qua` | Kiểm tra tên file trong `models/baseline/` có bắt đầu bằng `E3` không |
| Lỗi khi tải `hand_landmarker.task` | Kiểm tra kết nối mạng, hoặc tải thủ công và đặt vào `models/` |
| Chữ trên giao diện không có dấu | `pip install pillow` |
| Landmark rung nhiều | Nhấn `S` để đổi chế độ làm mượt; tăng ánh sáng và dùng nền trơn |
| FPS thấp | Tắt TTA bằng `--no-tta`; giảm tần suất dự đoán bằng `--stride 5` |
| `UnicodeEncodeError` trên terminal Windows | Chạy `chcp 65001` trước khi chạy script |

---

## 6. Nhóm thực hiện

Nhóm 9 — Học phần Thị giác máy tính và ứng dụng, Trường Đại học Công nghiệp TP. Hồ Chí Minh.
Giảng viên hướng dẫn: TS. Lê Thị Vĩnh Thanh.

| Thành viên | Phụ trách chính |
|---|---|
| Tô Thanh Hậu (nhóm trưởng) | Công cụ quay dữ liệu, trích landmark, EDA, khung LOSO, E0/E0+, demo |
| Võ Tấn Tài | E1, so sánh E0–E1 (H1) |
| Đặng Thế Vỹ | E2, so sánh E1–E2 (H2) |
| Nguyễn Thanh Khôi | E3, E3+, so sánh E2–E3 (H3), video demo |

Cả 4 thành viên cùng quay dữ liệu.
