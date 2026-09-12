"""
Script quay dữ liệu VSL.

Chạy: python src/record.py

Luồng: nhập tên người quay -> chọn block A/B -> hệ thống nạp/tạo file tiến độ
progress_{person}_{block}.json -> lần lượt hiện từng lớp theo thứ tự đã xáo trộn
-> bấm SPACE để đếm ngược và ghi -> hệ thống DỪNG LẠI CHỜ XÁC NHẬN
-> bấm SPACE để xác nhận đạt, hoặc R để ghi lại (chưa đạt) -> chỉ sau khi xác
nhận mới tính là xong và sang mẫu kế tiếp.

Phím tắt khi đang CHỜ CHỌN mẫu để quay:
  SPACE : đếm ngược rồi ghi mẫu hiện tại
  N     : bỏ qua mẫu hiện tại, quay lại sau (đẩy xuống cuối hàng đợi)
  L     : chuyển chế độ quay dài (clip đánh vần) — bấm lại L để quay về chế độ thường
  Q     : lưu tiến độ và thoát, chạy lại sẽ tiếp tục đúng chỗ dừng

Phím tắt khi đang CHỜ XÁC NHẬN mẫu vừa ghi:
  SPACE : xác nhận đạt, tính là xong, sang mẫu kế tiếp
  R     : chưa đạt, đếm ngược và ghi lại đè lên đúng file cũ, rồi chờ xác nhận tiếp
  Q     : thoát, KHÔNG tính mẫu này là xong (lần sau sẽ quay lại từ mẫu này)

Quan trọng:
  - Video ghi vào raw/{person}/  — KHÔNG đưa vào git (xem .gitignore)
  - Tên file: {code}_{person}_{block}_{seq:03d}.mp4
  - progress_*.json ghi lại danh sách đã xong, xóa file này = quay lại từ đầu
  - Không có ảnh tham chiếu — chỉ hiện tên chữ trên màn hình

Ghi chú kỹ thuật: cv2.putText (OpenCV) KHÔNG hỗ trợ tiếng Việt có dấu — font
Hershey của nó chỉ vẽ được ASCII, ký tự có dấu bị vẽ sai nét và chồng lên
nhau. Script này dùng Pillow (PIL) để vẽ chữ, hỗ trợ Unicode đầy đủ.
"""
import cv2
import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from classes import (
    build_order, display_name, CLIP_SECONDS, COUNTDOWN_SECONDS,
    SPELLING_CLIP_SECONDS, PEOPLE,
)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw"

# --- Font hỗ trợ tiếng Việt: thử lần lượt các font phổ biến trên Windows/Linux/macOS ---
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]
_FONT_CACHE = {}
_font_warned = False


def _get_font(size):
    global _font_warned
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            font = ImageFont.truetype(path, size)
            _FONT_CACHE[size] = font
            return font
    if not _font_warned:
        print("CANH BAO: khong tim thay font TrueType nao, chu co dau tieng Viet co the hien sai.")
        _font_warned = True
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def draw_hud(frame_bgr, text_lines, color=(255, 255, 255), font_size=18):
    """Vẽ chữ (hỗ trợ Unicode/tiếng Việt có dấu) lên khung hình qua Pillow.
    color dùng thứ tự BGR để tương thích với cách gọi cũ của OpenCV.
    Có nền đen bán trong suốt phía sau mỗi dòng để chữ không bị lẫn vào hình nền,
    và trả về một MẢNG MỚI (không sửa frame gốc tại chỗ) — luôn dùng giá trị trả về."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img, "RGBA")
    font = _get_font(font_size)
    fill = (color[2], color[1], color[0])  # BGR -> RGB

    y = 10
    line_h = font_size + 12
    for line in text_lines:
        bbox = draw.textbbox((14, y), line, font=font)
        pad = 4
        draw.rectangle(
            [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
            fill=(0, 0, 0, 160),
        )
        draw.text((14, y), line, font=font, fill=fill)
        y += line_h

    out_rgb = np.array(pil_img)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def get_progress_path(person, block):
    return ROOT / f"progress_{person}_{block}.json"


def load_progress(person, block):
    p = get_progress_path(person, block)
    if p.exists():
        return json.loads(p.read_text())
    return {"done": []}  # danh sách "code_seq" đã ghi xong


def save_progress(person, block, done_set):
    get_progress_path(person, block).write_text(
        json.dumps({"done": sorted(done_set)}, ensure_ascii=False, indent=2)
    )


def probe_fps(cap, seconds=2.0):
    """Đo FPS thật của webcam bằng cách đếm khung hình, không tin số cap.get() báo cáo."""
    n = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        ok, _ = cap.read()
        if ok:
            n += 1
    elapsed = time.time() - t0
    return n / elapsed if elapsed > 0 else 30.0


def record_clip(cap, out_path, duration_s, fps_estimate, label_text):
    """Ghi theo THỜI GIAN (không theo số khung hình cố định) — bù cho FPS khác nhau giữa các máy."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ok, probe = cap.read()
    h, w = probe.shape[:2] if ok else (480, 640)
    writer = cv2.VideoWriter(str(out_path), fourcc, max(fps_estimate, 10), (w, h))

    t0 = time.time()
    n_frames = 0
    while time.time() - t0 < duration_s:
        ok, frame = cap.read()
        if not ok:
            continue
        writer.write(frame)  # ghi RAW, không có overlay
        n_frames += 1
        remaining = duration_s - (time.time() - t0)
        disp = draw_hud(frame, [f"ĐANG GHI: {label_text}", f"còn lại {remaining:0.1f}s"], color=(0, 0, 255))
        cv2.imshow("VSL Recorder", disp)
        cv2.waitKey(1)
    writer.release()
    return n_frames


def main():
    print("Mã người hợp lệ:", ", ".join(PEOPLE))
    person = input("Ten nguoi quay (vd: hau): ").strip().lower()
    if person not in PEOPLE:
        print(f"Canh bao: '{person}' khong nam trong danh sach {PEOPLE}, van tiep tuc.")
    block = ""
    while block not in ("A", "B"):
        block = input("Block (A/B): ").strip().upper()

    person_dir = RAW_DIR / person
    person_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_order(person, block)
    progress = load_progress(person, block)
    done = set(progress["done"])

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DSHOW giam do tre so voi backend mac dinh tren Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # may khong ho tro DSHOW thi lui ve mac dinh
    if not cap.isOpened():
        print("KHONG mo duoc webcam. Kiem tra lai thiet bi.")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # khong de driver don khung hinh cu lai
    # LUU Y: khong ep dinh dang MJPG nua — mot so webcam khong ho tro tot qua DSHOW,
    # gay hinh bi vo/nhoe. Neu may ban van lag sau ban vá nay, bao lai de bat MJPG co dieu kien.

    print("Dang do FPS that cua may (2 giay)...")
    fps = probe_fps(cap)
    print(f"FPS do duoc: {fps:.1f}")
    if fps < 15:
        print("CANH BAO: FPS duoi 15, can xem xet doi may khac truoc khi quay that.")

    # Kiem tra lai voi file THAT tren dia theo CA HAI CHIEU, khong chi tin file progress_*.json:
    # (1) progress ghi "da xong" nhung file .mp4 bi xoa (vd do quay hong) -> loai khoi done,
    #     de no tu dong quay lai vao hang doi.
    # (2) file .mp4 co that tren dia (da quay va co dung dinh dang ten) nhung progress KHONG
    #     ghi la xong -> tu dong nhan la da xong. Truong hop nay xay ra khi progress_*.json bi
    #     mat/ghi de rieng voi video (vd dong bo OneDrive bi xung dot, tat may dot ngot ngay
    #     sau khi ghi nhung truoc khi kip luu progress).
    verified_done = set()
    missing_but_marked = []
    recovered_from_disk = []
    for code, seq in tasks:
        key = f"{code}_{seq:03d}"
        video_path = person_dir / f"{code}_{person}_{block}_{seq:03d}.mp4"
        file_exists = video_path.exists() and video_path.stat().st_size > 0
        if key in done:
            if file_exists:
                verified_done.add(key)
            else:
                missing_but_marked.append(video_path.name)
        elif file_exists:
            verified_done.add(key)
            recovered_from_disk.append(video_path.name)
    if missing_but_marked:
        print(f"CANH BAO: {len(missing_but_marked)} file da bi xoa nhung progress ghi la xong:")
        for name in missing_but_marked:
            print("  -", name)
        print("Da tu dong dua lai vao hang doi de quay lai.")
    if recovered_from_disk:
        print(f"CANH BAO: {len(recovered_from_disk)} file da co san tren dia nhung progress KHONG ghi la xong (co the do progress bi mat/ghi de):")
        for name in recovered_from_disk:
            print("  -", name)
        print("Da tu dong danh dau la xong, khong bat quay lai. Neu file nao chat luong khong dat,")
        print("hay tu xoa file .mp4 do bang tay roi chay lai script — no se tu quay lai vao hang doi.")
    if missing_but_marked or recovered_from_disk:
        save_progress(person, block, verified_done)
    done = verified_done

    long_mode = False
    long_idx = 1
    queue = [t for t in tasks if f"{t[0]}_{t[1]:03d}" not in done]
    pending = None  # dict {"out_path","label","dur"}: mẫu vừa ghi, đang chờ người dùng xác nhận đạt hay chưa

    def countdown_and_record(out_path, label, dur):
        """Đếm ngược rồi ghi 1 clip vào out_path (ghi đè nếu đã tồn tại)."""
        for c in range(COUNTDOWN_SECONDS, 0, -1):
            t_start = time.time()
            while time.time() - t_start < 1.0:
                ok, f2 = cap.read()
                if ok:
                    f2disp = draw_hud(f2, [f">> {label}", f"chuẩn bị... {c}"], color=(0, 165, 255))
                    cv2.imshow("VSL Recorder", f2disp)
                cv2.waitKey(1)
        record_clip(cap, out_path, dur, fps, label)
        print(f"Da luu: {out_path.name}")

    print(f"Con lai {len(queue)}/{len(tasks)} mau. SPACE=ghi, N=bo qua, L=che do dai, Q=thoat")

    while True:
        if pending is not None:
            # Đang chờ xác nhận mẫu vừa ghi — KHÔNG cho chuyển sang mẫu khác cho đến khi quyết định.
            ok, frame = cap.read()
            if not ok:
                continue
            disp = draw_hud(frame, [
                f">> {pending['label']}",
                "Da ghi xong. Dat chua?",
                "SPACE=Đạt, sang mẫu kế   R=Chưa đạt, ghi lại   Q=Thoát (không tính xong)",
            ], color=(0, 255, 0))
            cv2.imshow("VSL Recorder", disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                countdown_and_record(pending["out_path"], pending["label"], pending["dur"])
                # vẫn ở trạng thái chờ xác nhận, chờ người dùng bấm lại
            elif key == ord(" "):
                if long_mode:
                    long_idx += 1
                else:
                    code, seq = queue[0]
                    done.add(f"{code}_{seq:03d}")
                    save_progress(person, block, done)
                    queue.pop(0)
                pending = None
            continue

        if long_mode:
            label = f"clip đánh vần #{long_idx}"
            out_path = person_dir / f"spell_{person}_{long_idx:02d}.mp4"
        else:
            if not queue:
                print("HET HANG DOI. Da quay xong toan bo block nay.")
                break
            code, seq = queue[0]
            label = f"{display_name(code)}  [{code}]  mẫu {seq}"
            out_path = person_dir / f"{code}_{person}_{block}_{seq:03d}.mp4"

        ok, frame = cap.read()
        if not ok:
            continue
        remain_txt = "chế độ đánh vần" if long_mode else f"còn {len(queue)} mẫu"
        disp = draw_hud(frame, [
            f"Người: {person}   Block: {block}   ({remain_txt})",
            f">> {label}",
            "SPACE=ghi   N=bỏ qua   L=đổi chế độ   Q=thoát",
        ])
        cv2.imshow("VSL Recorder", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("l"):
            long_mode = not long_mode
        elif key == ord("n") and not long_mode and queue:
            queue.append(queue.pop(0))  # đẩy xuống cuối, quay lại sau
        elif key == ord(" "):
            # Đếm ngược, ĐỌC LIÊN TỤC trong lúc chờ thay vì chặn bằng waitKey dài.
            # Nếu không đọc liên tục, driver webcam (đặc biệt trên Windows) dồn khung hình
            # cũ vào bộ đệm, và khi bắt đầu ghi thật thì mấy khung đầu tiên là hình bị trễ.
            dur = SPELLING_CLIP_SECONDS if long_mode else CLIP_SECONDS
            countdown_and_record(out_path, label, dur)
            pending = {"out_path": out_path, "label": label, "dur": dur}

    cap.release()
    cv2.destroyAllWindows()
    print(f"Ket thuc. Da hoan thanh {len(done)}/{len(tasks)} mau cho block {block}.")


if __name__ == "__main__":
    main()
