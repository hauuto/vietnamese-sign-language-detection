"""
EDA (phân tích khám phá dữ liệu) trên tập landmark đã trích — bản vẽ lại cho báo cáo Word.

Thay đổi so với bản cũ:
  - Ảnh 300 dpi, khổ vừa trang A4 (rộng ~16 cm), chữ 9–10 pt; không đặt tiêu đề trong hình
    (tiêu đề nằm ở chú thích Word). Muốn có tiêu đề để dùng cho slide: --titles.
  - Hai biểu đồ số mẫu (4 cột 160 và 136 thanh) được thay bằng MỘT heatmap 4 người × 34 lớp.
  - Ngưỡng mất tay thống nhất với check_landmarks.py (20%); mốc 30% vẫn được vẽ để đối chiếu
    với số liệu đã viết trong báo cáo. Bỏ các mốc 5% / 15% không có căn cứ.
  - Tỉ lệ mất tay và tỉ lệ tay sát mép khung được gộp vào một hình (hai panel).
  - Toàn bộ con số dùng trong báo cáo được ghi ra outputs/eda/eda_summary.txt và các file CSV.

Yêu cầu: pip install matplotlib   (numpy đã có trong requirements.txt)
Chạy:    python eda.py            (hoặc python eda.py --titles cho bản dùng trên slide)
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from classes import ALL_CLASSES, class_group, display_name  # noqa: E402

# Theo Thông tư 17/2020/TT-BGDĐT, chữ đ là ký hiệu TĨNH. Khi quay, nhóm xếp đ vào nhóm
# 6 mẫu/người cùng các chữ có dấu phụ; ở đây chỉ đổi nhóm để phân tích, dữ liệu giữ nguyên.
TINH_THEO_CHUAN = {"dd"}


def nhom(code):
    """Nhóm tĩnh/động theo chuẩn ngôn ngữ (dùng cho EDA), khác với quy cách số mẫu khi quay."""
    return "tinh" if code in TINH_THEO_CHUAN else class_group(code)

LANDMARK_DIR = Path("../landmarks/raw")
OUT_DIR = Path("outputs/eda")
FNAME_RE = re.compile(r"^([a-z_]+)_([a-z]+)_([AB])_(\d+)\.npy$")

SEQ_LEN = 45
NGUONG_CANH_BAO = 0.20     # trùng check_landmarks.py
NGUONG_NANG = 0.30         # mốc đã dùng trong báo cáo (10 mẫu)
BIEN_KHUNG = 0.02          # tọa độ ngoài [-0.02, 1.02] được xem là sát/ra ngoài mép khung

# Khổ hình theo trang A4 (lề 2 cm hai bên -> ~16 cm = 6.3 in)
W_FULL = 6.3
MAU_XANH = "#2a78d6"
MAU_CAM = "#eb6834"
MAU_DO = "#c0392b"
MAU_XAM = "#6b6b6b"

plt.rcParams.update({
    "font.family": "DejaVu Sans",   # có đủ dấu tiếng Việt
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

SHOW_TITLES = False
NANG_LUONG_DD = float("nan")


def tieu_de(ax, text):
    if SHOW_TITLES:
        ax.set_title(text)


def luu(fig, ten):
    path = OUT_DIR / ten
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Đọc dữ liệu + chỉ số cho từng mẫu
# ---------------------------------------------------------------------------
def load_all():
    records, bo_qua = [], []
    for f in sorted(LANDMARK_DIR.glob("*/*.npy")):
        m = FNAME_RE.match(f.name)
        if not m:
            bo_qua.append(f.name)
            continue
        code, person, block, seq = m.groups()
        records.append({"path": f, "code": code, "person": person, "block": block,
                        "seq": int(seq), "arr": np.load(f)})
    if bo_qua:
        print(f"CẢNH BÁO: {len(bo_qua)} file sai quy ước tên, ví dụ: {bo_qua[:3]}")
    return records


def tinh_chi_so(records):
    """Gắn các chỉ số chất lượng/chuyển động vào từng record. Trả về số file lỗi shape/NaN."""
    loi_shape = loi_nan = 0
    for r in records:
        arr = r["arr"]
        r["hop_le"] = False
        if arr.ndim != 2 or arr.shape[0] != SEQ_LEN or arr.shape[1] not in (63, 126):
            loi_shape += 1
            continue
        if np.isnan(arr).any():
            loi_nan += 1
            continue
        r["hop_le"] = True
        mat = np.all(arr == 0, axis=1)
        r["ty_le_mat_tay"] = float(mat.mean())

        real = arr[~mat]
        r["so_khung_that"] = int(real.shape[0])
        if real.shape[0]:
            pts = real.reshape(real.shape[0], -1, 3)
            x, y = pts[:, :, 0], pts[:, :, 1]
            ngoai = (x < -BIEN_KHUNG) | (x > 1 + BIEN_KHUNG) | (y < -BIEN_KHUNG) | (y > 1 + BIEN_KHUNG)
            r["so_khung_sat_mep"] = int(ngoai.any(axis=1).sum())
        else:
            r["so_khung_sat_mep"] = 0
        # Năng lượng chuyển động: độ dời trung bình giữa hai khung có tay liên tiếp (tọa độ thô).
        r["chuyen_dong"] = float(np.linalg.norm(np.diff(real, axis=0), axis=1).mean()) if real.shape[0] >= 2 else None
    return loi_shape, loi_nan


# ---------------------------------------------------------------------------
# Hình 1 — số mẫu theo (người, lớp): một heatmap thay cho hai biểu đồ cũ
# ---------------------------------------------------------------------------
def hinh_so_mau(records, persons):
    count = defaultdict(int)
    for r in records:
        count[(r["person"], r["code"])] += 1
    # Lớp tĩnh trước, lớp động sau. Lớp đ thuộc nhóm tĩnh nhưng được quay 6 mẫu/người.
    classes = [c for c in ALL_CLASSES if nhom(c) == "tinh"] + \
              [c for c in ALL_CLASSES if nhom(c) == "dong"]
    M = np.array([[count[(p, c)] for c in classes] for p in persons])

    fig, ax = plt.subplots(figsize=(W_FULL, 0.42 * len(persons) + 0.9))
    vals = sorted(set(M.flatten()))
    palette = ["#d9e7f7", "#9cc2ec", MAU_XANH, "#1b4f8f"]
    cmap = ListedColormap(palette[:max(1, len(vals))])
    idx = np.searchsorted(vals, M)
    ax.imshow(idx, cmap=cmap, aspect="auto", vmin=0, vmax=max(1, len(vals) - 1))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if idx[i, j] >= 2 else "black")
    n_tinh = sum(nhom(c) == "tinh" for c in classes)
    ax.axvline(n_tinh - 0.5, color="black", linewidth=1.2)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([display_name(c) for c in classes], rotation=90)
    ax.set_yticks(range(len(persons)))
    ax.set_yticklabels([f"{p} ({M[i].sum()})" for i, p in enumerate(persons)])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text((n_tinh - 1) / 2, -0.8, f"{n_tinh} lớp tĩnh", ha="center", fontsize=8)
    ax.text(n_tinh + (len(classes) - n_tinh - 1) / 2, -0.8, f"{len(classes) - n_tinh} lớp động",
            ha="center", fontsize=8)
    tieu_de(ax, "Số mẫu theo từng người và từng lớp")
    p = luu(fig, "01_so_mau_nguoi_lop.png")

    with open(OUT_DIR / "so_mau_nguoi_lop.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Người"] + [display_name(c) for c in classes] + ["Tổng"])
        for i, pr in enumerate(persons):
            w.writerow([pr] + list(M[i]) + [M[i].sum()])
    return p, M


# ---------------------------------------------------------------------------
# Hình 2 — chất lượng file: phân bố tỉ lệ mất tay + tổng hợp tình trạng
# ---------------------------------------------------------------------------
def hinh_chat_luong(records, loi_shape, loi_nan):
    tl = np.array([r["ty_le_mat_tay"] for r in records if r["hop_le"]]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.6), gridspec_kw={"width_ratios": [1.3, 1]})

    ax = axes[0]
    ax.hist(tl, bins=np.arange(0, 102.5, 2.5), color=MAU_XANH, edgecolor="white", linewidth=0.5)
    ax.set_yscale("symlog", linthresh=10)   # phần lớn mẫu ở 0% -> thang log để thấy phần đuôi
    ax.axvline(NGUONG_CANH_BAO * 100, color=MAU_CAM, linestyle="--", linewidth=1,
               label=f"Ngưỡng cảnh báo {NGUONG_CANH_BAO * 100:.0f}%")
    ax.axvline(NGUONG_NANG * 100, color=MAU_DO, linestyle=":", linewidth=1.2,
               label=f"Mốc mất tay nặng {NGUONG_NANG * 100:.0f}%")
    ax.set_xlabel("Tỉ lệ khung mất tay trong mẫu (%)")
    ax.set_ylabel("Số mẫu (thang log)")
    ax.legend(frameon=False, loc="upper right")
    tieu_de(ax, "Phân bố tỉ lệ mất tay")

    ax = axes[1]
    n_20 = int((tl > NGUONG_CANH_BAO * 100).sum())
    n_30 = int((tl > NGUONG_NANG * 100).sum())
    # "> 30%" là tập con của "> 20%" -> ghi rõ trên nhãn
    cats = ["Có NaN", "Sai kích thước", f"Mất tay > {NGUONG_NANG * 100:.0f}%",
            f"Mất tay > {NGUONG_CANH_BAO * 100:.0f}%\n(gồm cả > {NGUONG_NANG * 100:.0f}%)",
            f"Mất tay ≤ {NGUONG_CANH_BAO * 100:.0f}%"]
    vals = [loi_nan, loi_shape, n_30, n_20, len(tl) - n_20]
    colors = [MAU_XAM, MAU_XAM, MAU_DO, MAU_CAM, MAU_XANH]
    bars = ax.barh(cats, vals, color=colors, height=0.6)
    ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_xlabel("Số file")
    ax.set_xlim(0, max(vals) * 1.2)
    tieu_de(ax, "Tổng hợp chất lượng file")
    fig.tight_layout()
    return luu(fig, "02_chat_luong_du_lieu.png"), n_20, n_30


# ---------------------------------------------------------------------------
# Hình 3 — chất lượng theo người: mất tay (trên) và tay sát/ra ngoài mép khung (dưới)
# ---------------------------------------------------------------------------
def hinh_theo_nguoi(records, persons):
    mat, mep = {}, {}
    for p in persons:
        rs = [r for r in records if r["hop_le"] and r["person"] == p]
        mat[p] = 100 * np.mean([r["ty_le_mat_tay"] for r in rs]) if rs else 0.0
        tong = sum(r["so_khung_that"] for r in rs)
        mep[p] = 100 * sum(r["so_khung_sat_mep"] for r in rs) / tong if tong else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.4), sharex=True)
    for ax, data, color, label in ((axes[0], mat, MAU_XANH, "Khung mất tay (%)"),
                                   (axes[1], mep, MAU_CAM, "Khung tay sát mép (%)")):
        bars = ax.bar(persons, [data[p] for p in persons], color=color, width=0.6)
        ax.bar_label(bars, labels=[f"{data[p]:.1f}" for p in persons], padding=2, fontsize=8)
        ax.set_ylabel(label)
        ax.set_ylim(0, max(data.values()) * 1.2 + 0.5)
    tieu_de(axes[0], "Mất tay theo người")
    tieu_de(axes[1], "Tay sát mép khung theo người")
    fig.tight_layout()
    return luu(fig, "03_chat_luong_theo_nguoi.png"), mat, mep


# ---------------------------------------------------------------------------
# Hình 4 — năng lượng chuyển động: tĩnh vs động và từng lớp động
# ---------------------------------------------------------------------------
def hinh_chuyen_dong(records):
    tinh = [r["chuyen_dong"] for r in records
            if r["hop_le"] and r["chuyen_dong"] is not None and nhom(r["code"]) == "tinh"]
    dong = [r["chuyen_dong"] for r in records
            if r["hop_le"] and r["chuyen_dong"] is not None and nhom(r["code"]) == "dong"]
    moc = float(np.mean(tinh))

    per_class = defaultdict(list)
    for r in records:
        if r["hop_le"] and r["chuyen_dong"] is not None and nhom(r["code"]) == "dong":
            per_class[r["code"]].append(r["chuyen_dong"])
    means = sorted(((display_name(c), float(np.mean(v))) for c, v in per_class.items()), key=lambda x: x[1])

    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.8), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    bp = ax.boxplot([tinh, dong], tick_labels=[f"Tĩnh\n({len(tinh)} mẫu)", f"Động\n({len(dong)} mẫu)"],
                    patch_artist=True, widths=0.5, flierprops={"markersize": 3})
    for patch, color in zip(bp["boxes"], [MAU_XANH, MAU_CAM]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_ylabel("Năng lượng chuyển động / mẫu")
    tieu_de(ax, "Nhóm tĩnh và nhóm động")

    ax = axes[1]
    labels = [x[0] for x in means]
    values = [x[1] for x in means]
    ax.barh(labels, values, color=[MAU_XAM if v < moc else MAU_CAM for v in values], height=0.65)
    ax.axvline(moc, color="black", linestyle="--", linewidth=1, label="Trung bình nhóm tĩnh")
    ax.set_xlabel("Năng lượng chuyển động trung bình")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_xlim(0, max(values + [moc]) * 1.08)
    tieu_de(ax, "Từng lớp động")
    fig.tight_layout()
    vals_dd = [r["chuyen_dong"] for r in records if r["hop_le"] and r["chuyen_dong"] is not None and r["code"] == "dd"]
    global NANG_LUONG_DD
    NANG_LUONG_DD = float(np.mean(vals_dd)) if vals_dd else float("nan")
    tren_moc = sum(v > moc for v in values)
    duoi_moc = [l for l, v in means if v <= moc]
    return luu(fig, "04_nang_luong_chuyen_dong.png"), (float(np.median(tinh)), float(np.median(dong)),
                                                        moc, tren_moc, len(values), duoi_moc)


# ---------------------------------------------------------------------------
def main():
    global SHOW_TITLES
    ap = argparse.ArgumentParser(description="EDA tập landmark VSL — ảnh cho báo cáo")
    ap.add_argument("--titles", action="store_true", help="Thêm tiêu đề trong hình (dùng cho slide)")
    SHOW_TITLES = ap.parse_args().titles

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_all()
    if not records:
        print(f"Không tìm thấy file .npy nào trong {LANDMARK_DIR}. Đã chạy src/extract_landmarks.py chưa?")
        return
    persons = sorted({r["person"] for r in records})
    loi_shape, loi_nan = tinh_chi_so(records)

    f1, M = hinh_so_mau(records, persons)
    f2, n_20, n_30 = hinh_chat_luong(records, loi_shape, loi_nan)
    f3, mat, mep = hinh_theo_nguoi(records, persons)
    f4, (med_t, med_d, moc, tren, n_dong, duoi) = hinh_chuyen_dong(records)

    # Tệp chi tiết từng mẫu, để tra lại khi viết báo cáo
    with open(OUT_DIR / "chi_tiet_mau.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "lop", "nguoi", "block", "hop_le", "ty_le_mat_tay_%", "khung_sat_mep", "chuyen_dong"])
        for r in records:
            w.writerow([r["path"].name, r["code"], r["person"], r["block"], r["hop_le"],
                        f"{100 * r.get('ty_le_mat_tay', float('nan')):.1f}", r.get("so_khung_sat_mep", ""),
                        "" if r.get("chuyen_dong") is None else f"{r['chuyen_dong']:.5f}"])

    lines = [
        f"Tổng số file: {len(records)} | người: {', '.join(persons)} | lớp: {M.shape[1]}",
        "Số mẫu mỗi người: " + ", ".join(f"{p}={M[i].sum()}" for i, p in enumerate(persons)),
        f"Số mẫu mỗi (người, lớp): tối thiểu {M.min()}, tối đa {M.max()}",
        f"File sai kích thước: {loi_shape} | có NaN: {loi_nan}",
        f"Mẫu mất tay > {NGUONG_CANH_BAO * 100:.0f}%: {n_20} | > {NGUONG_NANG * 100:.0f}%: {n_30}",
        "Tỉ lệ mất tay trung bình theo người (%): " + ", ".join(f"{p}={mat[p]:.1f}" for p in persons),
        "Tỉ lệ khung tay sát/ra ngoài mép theo người (%): " + ", ".join(f"{p}={mep[p]:.1f}" for p in persons),
        f"Trung vị năng lượng chuyển động: tĩnh={med_t:.4f}, động={med_d:.4f}; mốc trung bình tĩnh={moc:.4f}",
        f"Lớp động có năng lượng TB cao hơn mốc tĩnh: {tren}/{n_dong}; thấp hơn: {', '.join(duoi) or 'không có'}",
        f"Lớp đ (xếp vào nhóm tĩnh theo TT 17/2020): năng lượng TB={NANG_LUONG_DD:.4f} so với mốc tĩnh {moc:.4f}",
    ]
    (OUT_DIR / "eda_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nĐã lưu vào {OUT_DIR.resolve()}:")
    for f in (f1, f2, f3, f4):
        print("  ", f.name)
    print("   so_mau_nguoi_lop.csv, chi_tiet_mau.csv, eda_summary.txt")


if __name__ == "__main__":
    main()