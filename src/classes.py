"""
Danh sách 34 lớp ký hiệu VSL và số mẫu cần quay theo từng block.
Đây là NGUỒN DUY NHẤT cho danh sách lớp — record.py và extract_landmarks.py
đều import từ đây, không hard-code lại ở nơi khác.
"""

# 22 chữ cái không dấu phụ — cử chỉ tĩnh
TINH = [
    "a", "b", "c", "d", "e", "g", "h", "i", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "x", "y",
]

# 7 chữ có dấu phụ — xếp vào nhóm động (VN Telex)
DAU_PHU = ["aw", "aa", "ee", "oo", "ow", "uw", "dd"]  # ă â ê ô ơ ư đ

# 5 thanh điệu — động
THANH = ["tone_s", "tone_f", "tone_r", "tone_x", "tone_j"]  # sắc huyền hỏi ngã nặng

DONG = DAU_PHU + THANH  # 12 lớp động
ALL_CLASSES = TINH + DONG  # 34 lớp, thứ tự cố định để tái lập được

assert len(TINH) == 22 and len(DONG) == 12 and len(ALL_CLASSES) == 34

# Số mẫu mỗi lớp mỗi người, chia theo block A/B (đã giảm theo yêu cầu ngày 12/09:
# tĩnh dễ nên ít lần lặp, động là bằng chứng chính nên giữ nhiều hơn một chút)
SAMPLES_PER_BLOCK = {
    "A": {"tinh": 2, "dong": 3},
    "B": {"tinh": 2, "dong": 3},
}
# Tổng mỗi lớp mỗi người: tĩnh 4, động 6

# 10 lớp dùng cho phép đo P3 (người ngoài lab) — xem mục 08
P3_CLASSES = ["a", "m", "n", "s", "o", "l", "tone_s", "tone_j", "aw", "ow"]
P3_SAMPLES_PER_CLASS = 10

CLIP_SECONDS = 3.0          # độ dài một mẫu ký hiệu, tính theo THỜI GIAN không theo số khung hình
COUNTDOWN_SECONDS = 5        # thời gian đếm ngược trước khi ghi thật (tăng vì không còn ảnh mẫu để nhớ tư thế)
RESAMPLE_STEPS = 45         # số bước sau khi lấy mẫu lại — cố định bất kể FPS máy quay
SPELLING_CLIP_SECONDS = 4.0
SPELLING_CLIPS_PER_PERSON = 10

PEOPLE = ["hau", "khoi", "tai", "vy"]

# Nhãn hiển thị tiếng Việt để show lên màn hình lúc quay (không dùng trong tên file)
DISPLAY = {
    "aw": "ă", "aa": "â", "ee": "ê", "oo": "ô", "ow": "ơ", "uw": "ư", "dd": "đ",
    "tone_s": "dấu SẮC (á)", "tone_f": "dấu HUYỀN (à)", "tone_r": "dấu HỎI (ả)",
    "tone_x": "dấu NGÃ (ã)", "tone_j": "dấu NẶNG (ạ)",
}


def display_name(code: str) -> str:
    return DISPLAY.get(code, code.upper())


def class_group(code: str) -> str:
    return "tinh" if code in TINH else "dong"


def samples_for(code: str, block: str) -> int:
    return SAMPLES_PER_BLOCK[block][class_group(code)]


def build_order(person: str, block: str, seed_extra: int = 0):
    """Danh sách (mã_lớp, số_thứ_tự_mẫu) đã xáo trộn, xáo khác nhau theo người+block."""
    import random
    tasks = []
    for code in ALL_CLASSES:
        n = samples_for(code, block)
        for i in range(1, n + 1):
            tasks.append((code, i))
    rng = random.Random(f"{person}-{block}-{seed_extra}")
    rng.shuffle(tasks)
    return tasks


if __name__ == "__main__":
    total_tinh = sum(SAMPLES_PER_BLOCK[b]["tinh"] for b in "AB") * len(TINH)
    total_dong = sum(SAMPLES_PER_BLOCK[b]["dong"] for b in "AB") * len(DONG)
    print(f"34 lớp: {len(TINH)} tĩnh + {len(DONG)} động")
    print(f"Mẫu/người: {total_tinh + total_dong} ({total_tinh} tĩnh + {total_dong} động)")
    print(f"Tổng toàn nhóm ({len(PEOPLE)} người): {(total_tinh + total_dong) * len(PEOPLE)}")
