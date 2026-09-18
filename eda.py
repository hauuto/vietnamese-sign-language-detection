"""
EDA (phân tích khám phá dữ liệu) trên tập landmark đã trích.

Yêu cầu: pip install matplotlib   (numpy đã có sẵn trong requirements.txt)

Chạy: python eda.py
Kết quả: lưu toàn bộ biểu đồ vào thư mục eda_output/ (không in bảng số ra console nữa —
mọi thông tin đều được thể hiện dưới dạng biểu đồ).
"""
import re
import sys
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from classes import ALL_CLASSES, TINH, DONG, class_group, samples_for, display_name

LANDMARK_DIR = Path("landmarks/raw")
OUT_DIR = Path("eda_output")
FNAME_RE = re.compile(r"^([a-z_]+)_([a-z]+)_([AB])_(\d+)\.npy$")

# Bảng màu — xanh dương (chính) và cam (đối chiếu), theo hệ màu categorical
# đã được kiểm định (đủ khoảng cách nhận biết kể cả với người mù màu đỏ-lục).
MAU_XANH = "#2a78d6"
MAU_CAM = "#eb6834"
MAU_XAM = "#9a9a9a"

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "DejaVu Sans"  # có đủ dấu tiếng Việt


def load_all():
    records = []
    for f in sorted(LANDMARK_DIR.glob("*/*.npy")):
        m = FNAME_RE.match(f.name)
        if not m:
            print(f"CẢNH BÁO: không đọc được tên file theo đúng quy ước: {f.name}")
            continue
        code, person, block, seq = m.groups()
        arr = np.load(f)
        records.append({
            "path": f, "code": code, "person": person, "block": block,
            "seq": int(seq), "arr": arr,
        })
    return records


def bieu_do_so_luong(records):
    """So sánh số mẫu thực tế và kỳ vọng, theo từng người."""
    persons = sorted(set(r["person"] for r in records))
    count = defaultdict(int)
    for r in records:
        count[(r["code"], r["person"])] += 1

    tong_thuc_te = {p: 0 for p in persons}
    tong_ky_vong = {p: 0 for p in persons}
    for code in ALL_CLASSES:
        ky_vong_moi_nguoi = samples_for(code, "A") + samples_for(code, "B")
        for p in persons:
            tong_thuc_te[p] += count[(code, p)]
            tong_ky_vong[p] += ky_vong_moi_nguoi

    x = np.arange(len(persons))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, [tong_ky_vong[p] for p in persons], width=w,
           label="Kỳ vọng", color=MAU_XAM)
    ax.bar(x + w / 2, [tong_thuc_te[p] for p in persons], width=w,
           label="Thực tế", color=MAU_XANH)
    for i, p in enumerate(persons):
        ax.text(i - w / 2, tong_ky_vong[p] + 2, str(tong_ky_vong[p]), ha="center", fontsize=9)
        ax.text(i + w / 2, tong_thuc_te[p] + 2, str(tong_thuc_te[p]), ha="center", fontsize=9,
                color=MAU_XANH if tong_thuc_te[p] == tong_ky_vong[p] else "#e34948")
    ax.set_xticks(x)
    ax.set_xticklabels(persons)
    ax.set_ylabel("Số mẫu")
    ax.set_title("Số mẫu thực tế so với kỳ vọng, theo từng người")
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_so_luong_mau.png", dpi=120)
    plt.close(fig)

    # Chi tiết các tổ hợp (lớp, người) bị lệch so với kỳ vọng
    lech = []
    for code in ALL_CLASSES:
        ky_vong = samples_for(code, "A") + samples_for(code, "B")
        for p in persons:
            thuc_te = count[(code, p)]
            if thuc_te != ky_vong:
                lech.append((f"{display_name(code)} / {p}", thuc_te - ky_vong))
    if lech:
        lech.sort(key=lambda x: x[1])
        labels = [x[0] for x in lech]
        values = [x[1] for x in lech]
        fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(lech))))
        colors = [MAU_CAM if v < 0 else MAU_XANH for v in values]
        ax.barh(labels, values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Chênh lệch so với kỳ vọng (âm = thiếu, dương = thừa)")
        ax.set_title("Các tổ hợp (lớp, người) lệch số mẫu so với kỳ vọng")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "02_lech_so_mau.png", dpi=120)
        plt.close(fig)


def bieu_do_chat_luong(records):
    """Tỉ lệ bước bị mất tay (toàn số 0) sau resample — phân bố + số file lỗi."""
    loi_shape = 0
    loi_nan = 0
    ty_le_mat_tay = []
    for r in records:
        arr = r["arr"]
        if arr.shape[0] != 45 or arr.shape[1] not in (63, 126):
            loi_shape += 1
            continue
        if np.isnan(arr).any():
            loi_nan += 1
            continue
        ty_le = float(np.all(arr == 0, axis=1).mean())
        r["ty_le_mat_tay"] = ty_le
        ty_le_mat_tay.append(ty_le)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) phân bố tỉ lệ mất tay
    ax = axes[0]
    arr_tl = np.array(ty_le_mat_tay) * 100
    ax.hist(arr_tl, bins=30, color=MAU_XANH, edgecolor="white")
    ax.axvline(30, color="#e34948", linestyle="--", linewidth=1.2, label="Ngưỡng cảnh báo 30%")
    ax.set_xlabel("Tỉ lệ bước bị mất tay (%)")
    ax.set_ylabel("Số lượng mẫu")
    ax.set_title("Phân bố tỉ lệ mất tay trên toàn bộ dữ liệu")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    # (b) tổng số file theo tình trạng
    so_loi = int((arr_tl > 30).sum())
    so_binh_thuong = len(arr_tl) - so_loi
    ax = axes[1]
    cats = ["Bình thường", "Mất tay > 30%", "Sai shape", "Có NaN"]
    vals = [so_binh_thuong, so_loi, loi_shape, loi_nan]
    colors = [MAU_XANH, "#e34948", "#e34948", "#e34948"]
    bars = ax.bar(cats, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, str(v), ha="center", fontsize=9)
    ax.set_ylabel("Số file")
    ax.set_title("Tổng hợp chất lượng file .npy")
    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_chat_luong_du_lieu.png", dpi=120)
    plt.close(fig)


def bieu_do_khoang_gia_tri(records):
    """Tỉ lệ khung hình có toạ độ x/y vượt ra ngoài [0,1] — theo từng người."""
    per_person_oor = defaultdict(int)
    per_person_total = defaultdict(int)
    for r in records:
        arr = r["arr"]
        real_mask = ~np.all(arr == 0, axis=1)
        if not real_mask.any():
            continue
        real = arr[real_mask]
        pts = real.reshape(real.shape[0], -1, 3)
        x, y = pts[:, :, 0], pts[:, :, 1]
        oor = ((x < -0.02) | (x > 1.02) | (y < -0.02) | (y > 1.02))
        per_person_oor[r["person"]] += int(oor.any(axis=1).sum())
        per_person_total[r["person"]] += real.shape[0]

    persons = sorted(per_person_total.keys())
    ty_le = [100 * per_person_oor[p] / per_person_total[p] if per_person_total[p] else 0 for p in persons]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(persons, ty_le, color=MAU_CAM)
    for b, v in zip(bars, ty_le):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}%", ha="center", fontsize=9)
    ax.axhline(5, color="#e34948", linestyle="--", linewidth=1.2, label="Mốc 5% — nên kiểm tra lại khung hình")
    ax.set_ylabel("Tỉ lệ khung hình tay ở gần/ngoài mép khung hình (%)")
    ax.set_title("Toạ độ tay vượt khung hình, theo từng người")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_khoang_gia_tri_toa_do.png", dpi=120)
    plt.close(fig)


def nang_luong_chuyen_dong(arr):
    real_mask = ~np.all(arr == 0, axis=1)
    real = arr[real_mask]
    if real.shape[0] < 2:
        return None
    diffs = np.diff(real, axis=0)
    return float(np.linalg.norm(diffs, axis=1).mean())


def bieu_do_chuyen_dong(records):
    """So sánh năng lượng chuyển động: lớp tĩnh vs lớp động — bằng chứng cho lựa chọn LSTM/Bi-LSTM."""
    for r in records:
        e = nang_luong_chuyen_dong(r["arr"])
        if e is not None:
            r["chuyen_dong"] = e

    tinh = [r["chuyen_dong"] for r in records if "chuyen_dong" in r and class_group(r["code"]) == "tinh"]
    dong = [r["chuyen_dong"] for r in records if "chuyen_dong" in r and class_group(r["code"]) == "dong"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bp = ax.boxplot([tinh, dong], tick_labels=["Tĩnh (22 lớp)", "Động (12 lớp)"],
                     patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], [MAU_XANH, MAU_CAM]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel("Năng lượng chuyển động trung bình / mẫu")
    ax.set_title("So sánh chuyển động: nhóm lớp tĩnh và nhóm lớp động")
    ax.spines[["top", "right"]].set_visible(False)

    # Trung bình theo từng lớp động, để phát hiện lớp có chuyển động bất thường thấp
    per_class = defaultdict(list)
    for r in records:
        if "chuyen_dong" in r and class_group(r["code"]) == "dong":
            per_class[r["code"]].append(r["chuyen_dong"])
    means = sorted(((display_name(c), np.mean(v)) for c, v in per_class.items()), key=lambda x: x[1])

    ax = axes[1]
    labels = [x[0] for x in means]
    values = [x[1] for x in means]
    colors = [MAU_CAM if v < np.mean(tinh) else MAU_XANH for v in values]
    ax.barh(labels, values, color=colors)
    ax.axvline(np.mean(tinh), color="black", linestyle="--", linewidth=1,
               label="Trung bình nhóm tĩnh (mốc so sánh)")
    ax.set_xlabel("Năng lượng chuyển động trung bình")
    ax.set_title("Từng lớp động, so với mốc trung bình nhóm tĩnh")
    ax.legend(loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_nang_luong_chuyen_dong.png", dpi=120)
    plt.close(fig)


def bieu_do_theo_nguoi(records):
    """Tỉ lệ mất tay trung bình, so sánh giữa 4 người — phát hiện người quay bất thường."""
    by_person = defaultdict(list)
    for r in records:
        if "ty_le_mat_tay" in r:
            by_person[r["person"]].append(r["ty_le_mat_tay"])

    persons = sorted(by_person.keys())
    means = [100 * np.mean(by_person[p]) for p in persons]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [MAU_CAM if m > 15 else MAU_XANH for m in means]
    bars = ax.bar(persons, means, color=colors)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)
    ax.axhline(15, color="#e34948", linestyle="--", linewidth=1.2, label="Mốc 15% — cao bất thường")
    ax.set_ylabel("Tỉ lệ mất tay trung bình (%)")
    ax.set_title("So sánh chất lượng dữ liệu giữa 4 người quay")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_so_sanh_theo_nguoi.png", dpi=120)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    records = load_all()
    if not records:
        print(f"Không tìm thấy file .npy nào trong {LANDMARK_DIR}. Đã chạy src/extract_landmarks.py chưa?")
        return

    bieu_do_so_luong(records)
    bieu_do_chat_luong(records)
    bieu_do_khoang_gia_tri(records)
    bieu_do_chuyen_dong(records)
    bieu_do_theo_nguoi(records)

    print(f"Đã lưu toàn bộ biểu đồ vào thư mục: {OUT_DIR.resolve()}")
    print("  01_so_luong_mau.png          — số mẫu thực tế vs kỳ vọng")
    print("  02_lech_so_mau.png           — chi tiết tổ hợp (lớp, người) bị lệch (nếu có)")
    print("  03_chat_luong_du_lieu.png    — phân bố mất tay + tổng hợp lỗi")
    print("  04_khoang_gia_tri_toa_do.png — toạ độ vượt khung hình, theo người")
    print("  05_nang_luong_chuyen_dong.png— tĩnh vs động, bằng chứng chọn LSTM/Bi-LSTM")
    print("  06_so_sanh_theo_nguoi.png    — chất lượng dữ liệu theo từng người")


if __name__ == "__main__":
    main()