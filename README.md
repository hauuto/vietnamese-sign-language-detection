# Nhận dạng ngôn ngữ ký hiệu tiếng Việt

Nhận dạng ngôn ngữ ký hiệu tiếng Việt (34 lớp) bằng MediaPipe + MLP/Bi-LSTM.

## 1. Setup môi trường 

```bash
# Cần Python 3.10 hoặc 3.11 (mediapipe hiện chưa hỗ trợ 3.12+ ổn định)
python3 --version

python3 -m venv .venv
.venv/Scripts/activate

pip install -r requirements.txt
```

Kiểm tra cài đặt đúng:
```bash
python3 -c "import cv2, mediapipe, numpy, torch; print('OK')"
python3 src/classes.py
```


## 2. Quay dữ liệu

```bash
python3 src/record.py
```
Nhập tên (`hau`/`khoi`/`tai`/`vy`) và block (`A`/`B`) khi được hỏi.

| Phím | Tác dụng |
|---|---|
| `SPACE` | đếm ngược 3 giây rồi ghi mẫu hiện tại |
| `N` | bỏ qua, đẩy xuống cuối hàng đợi, quay lại sau |
| `L` | chuyển sang chế độ quay dài (10 clip đánh vần 4 giây/người) |
| `Q` | lưu tiến độ và thoát — **chạy lại `record.py` sẽ tiếp tục đúng chỗ dừng**, không mất gì |

File `progress_{person}_{block}.json` sinh tự động ở thư mục gốc, ghi lại
mẫu nào đã xong. Xóa file này nếu thật sự muốn quay lại từ đầu.

Sau khi quay xong, đặt `raw/` về chỉ đọc để tránh ghi đè nhầm:
```bash
chmod -R a-w raw/
```
Rồi **sao lưu ngay `raw/` sang ổ cứng hoặc cloud khác**, trước khi tắt máy.

## 3. Trích landmark (13/09)

```bash
chmod u+w -R raw/          # mở quyền ghi tạm để đọc, hoặc bỏ qua nếu hệ điều hành không chặn đọc file chỉ-đọc
python3 src/extract_landmarks.py --source raw
python3 src/extract_landmarks.py --source raw_external   # sau buổi mời người ngoài
```

Lần chạy đầu tiên sẽ tự tải model `hand_landmarker.task` (~10MB, cần mạng
một lần). Script bỏ qua file `.npy` đã có sẵn nên **chạy lại được giữa
chừng** nếu bị ngắt hoặc máy treo.

Cuối log sẽ in danh sách file có tỉ lệ mất tay trên 20% — đây là các mẫu
nên xem xét quay bù trong sáng 13/09.

## 4. Cấu trúc thư mục

```
vsl-project/
├── src/
│   ├── classes.py            # NGUỒN DUY NHẤT cho danh sách 34 lớp — sửa ở đây
│   ├── record.py              # script quay (chạy ngày 12/09)
│   └── extract_landmarks.py   # script trích landmark (chạy ngày 13/09)
├── raw/{person}/               # video gốc — KHÔNG vào git
├── raw_external/{person}/      # video người ngoài lab (P3) — KHÔNG vào git
├── landmarks/{raw,raw_external}/{person}/  # .npy đã trích — KHÔNG vào git
├── models/                     # hand_landmarker.task tự tải về — KHÔNG vào git
└── requirements.txt
```

## 5. Việc CHƯA có trong repo này, cần làm tiếp theo lịch

- `train_mlp.py` (E0, E1) — ngày 13/09
- `train_lstm.py` (E2, Bi-LSTM) — ngày 14/09
- `cross_subject.py` (P2, P3) — ngày 15/09
- `demo.py` (cửa sổ trượt + điểm năng lượng + demo realtime) — ngày 16/09

