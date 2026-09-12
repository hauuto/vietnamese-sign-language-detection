"""
Script quay dữ liệu VSL.

Chạy: python src/record.py

Luồng: nhập tên người quay -> chọn block A/B -> hệ thống nạp/tạo file tiến độ
progress_{person}_{block}.json -> lần lượt hiện từng lớp theo thứ tự đã xáo trộn
-> bấm SPACE để đếm ngược và ghi 1.5s -> tự động sang mẫu kế tiếp.

Phím tắt:
  SPACE : bắt đầu ghi mẫu hiện tại
  R     : ghi lại mẫu VỪA XONG (đè lên file cũ)
  N     : bỏ qua mẫu hiện tại, quay lại sau (đẩy xuống cuối hàng đợi)
  L     : chuyển chế độ quay dài (clip đánh vần 4s) — bấm lại L để quay về chế độ thường
  Q     : lưu tiến độ và thoát, chạy lại sẽ tiếp tục đúng chỗ dừng

Quan trọng:
  - Video ghi vào raw/{person}/  — KHÔNG đưa vào git (xem .gitignore)
  - Tên file: {code}_{person}_{block}_{seq:03d}.mp4
  - progress_*.json ghi lại danh sách đã xong, xóa file này = quay lại từ đầu
"""
import cv2
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from classes import build_order, display_name, CLIP_SECONDS, SPELLING_CLIP_SECONDS, PEOPLE

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw"


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


def draw_hud(frame, text_lines, color=(255, 255, 255)):
    y = 30
    for line in text_lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
        y += 28
    return frame


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
        writer.write(frame)
        n_frames += 1
        disp = frame.copy()
        remaining = duration_s - (time.time() - t0)
        draw_hud(disp, [f"DANG GHI: {label_text}", f"con lai {remaining:0.1f}s"], color=(0, 0, 255))
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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("KHONG mo duoc webcam. Kiem tra lai thiet bi.")
        return

    print("Dang do FPS that cua may (2 giay)...")
    fps = probe_fps(cap)
    print(f"FPS do duoc: {fps:.1f}")
    if fps < 15:
        print("CANH BAO: FPS duoi 15, can xem xet doi may khac truoc khi quay that.")

    long_mode = False
    long_idx = 1
    queue = [t for t in tasks if f"{t[0]}_{t[1]:03d}" not in done]

    print(f"Con lai {len(queue)}/{len(tasks)} mau. SPACE=ghi, R=ghi lai, N=bo qua, L=che do dai, Q=thoat")

    while True:
        if long_mode:
            label = f"clip danh van #{long_idx}"
            out_path = person_dir / f"spell_{person}_{long_idx:02d}.mp4"
        else:
            if not queue:
                print("HET HANG DOI. Da quay xong toan bo block nay.")
                break
            code, seq = queue[0]
            label = f"{display_name(code)}  [{code}]  mau {seq}"
            out_path = person_dir / f"{code}_{person}_{block}_{seq:03d}.mp4"

        ok, frame = cap.read()
        if not ok:
            continue
        remain_txt = "che do dai (danh van)" if long_mode else f"con {len(queue)} mau"
        draw_hud(frame, [
            f"Nguoi: {person}  Block: {block}  ({remain_txt})",
            f">> {label}",
            "SPACE=ghi  R=ghi lai  N=bo qua  L=doi che do  Q=thoat",
        ])
        cv2.imshow("VSL Recorder", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("l"):
            long_mode = not long_mode
        elif key == ord("n") and not long_mode and queue:
            queue.append(queue.pop(0))  # đẩy xuống cuối, quay lại sau
        elif key == ord(" "):
            # đếm ngược 3 giây trước khi ghi thật
            for c in [3, 2, 1]:
                ok, f2 = cap.read()
                if ok:
                    draw_hud(f2, [f">> {label}", f"chuan bi... {c}"], color=(0, 165, 255))
                    cv2.imshow("VSL Recorder", f2)
                cv2.waitKey(700)
            dur = SPELLING_CLIP_SECONDS if long_mode else CLIP_SECONDS
            record_clip(cap, out_path, dur, fps, label)
            if long_mode:
                long_idx += 1
            else:
                done.add(f"{code}_{seq:03d}")
                save_progress(person, block, done)
                queue.pop(0)
            print(f"Da luu: {out_path.name}")
        elif key == ord("r"):
            print("Ghi lai mau vua roi: bam SPACE sau khi lui lai bang tay (chua tu dong lui hang doi).")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Ket thuc. Da hoan thanh {len(done)}/{len(tasks)} mau cho block {block}.")


if __name__ == "__main__":
    main()
