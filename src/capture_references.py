"""
Chụp ẢNH THAM CHIẾU cho 34 cử chỉ — làm việc này TRƯỚC khi quay dữ liệu thật.

Đây chính là bước "chốt bảng tham chiếu" trong kế hoạch (mục 01, cổng chặn
lúc 14:00). Ảnh chụp ra sẽ được record.py hiện cạnh khung hình trực tiếp,
để không ai phải nhớ 34 cử chỉ.

Chạy: python src/capture_references.py

- Lớp TĨNH (22 chữ cái cơ bản): chụp 1 ảnh — tư thế tay đứng yên.
- Lớp ĐỘNG (12 lớp: dấu phụ + thanh dấu): chụp 2 ảnh — điểm BẮT ĐẦU và
  điểm KẾT THÚC của chuyển động. record.py sẽ hiện cả hai để người quay
  biết quỹ đạo tay phải đi từ đâu đến đâu, không chỉ một tư thế tĩnh.

Phím tắt: SPACE = chụp, N = bỏ qua lớp này (làm sau), Q = thoát (chạy lại
tiếp tục từ ảnh còn thiếu, ảnh đã có không bị chụp đè).

QUAN TRỌNG: người chụp ảnh mẫu nên là người đã xác minh đúng cử chỉ theo
tài liệu VSL. Ảnh sai chuẩn ở đây sẽ khiến CẢ BỐN NGƯỜI quay sai theo.
"""
import cv2
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from classes import ALL_CLASSES, DONG, display_name

ROOT = Path(__file__).resolve().parent.parent
REF_DIR = ROOT / "reference_images"


def draw_hud(frame, lines, color=(255, 255, 255)):
    y = 30
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
        y += 28


def targets_for(code):
    """Trả về danh sách (đường dẫn ảnh, nhãn phụ) cần chụp cho 1 lớp."""
    if code in DONG:
        return [
            (REF_DIR / f"{code}_start.jpg", "DIEM BAT DAU"),
            (REF_DIR / f"{code}_end.jpg", "DIEM KET THUC"),
        ]
    return [(REF_DIR / f"{code}.jpg", "tu the tinh")]


def main():
    REF_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("KHONG mo duoc webcam.")
        return

    # gom toàn bộ ảnh cần chụp thành 1 hàng đợi phẳng, bỏ qua ảnh đã có sẵn
    queue = []
    for code in ALL_CLASSES:
        for path, sub in targets_for(code):
            if not path.exists():
                queue.append((code, path, sub))

    if not queue:
        print("Da chup du anh tham chieu cho ca 34 lop.")
        return

    print(f"Con {len(queue)} anh can chup. SPACE=chup, N=bo qua, Q=thoat")

    while queue:
        code, path, sub = queue[0]
        ok, frame = cap.read()
        if not ok:
            continue
        disp = frame.copy()
        draw_hud(disp, [
            f"Con lai {len(queue)} anh",
            f">> {display_name(code)}  [{code}]  ({sub})",
            "SPACE=chup  N=bo qua  Q=thoat",
        ])
        cv2.imshow("Chup anh tham chieu", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n"):
            queue.append(queue.pop(0))
        elif key == ord(" "):
            cv2.imwrite(str(path), frame)
            print(f"Da luu: {path.name}")
            queue.pop(0)

    cap.release()
    cv2.destroyAllWindows()
    remaining = sum(1 for code in ALL_CLASSES for p, _ in targets_for(code) if not p.exists())
    print(f"Con thieu {remaining} anh. Chay lai script de tiep tuc.")


if __name__ == "__main__":
    main()
