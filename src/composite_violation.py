# -*- coding: utf-8 -*-
"""위반_교차오염 합성 데이터 생성기
   (도마 색상 × 불일치 식재료를 합성해 "위반" 클래스 이미지를 만듭니다)

▶ 무슨 도구인가?
  실제로는 거의 존재하지 않는 "색상-재료 불일치" 사진을,
  이미 갖고 있는 (1) 빈 도마 사진 + (2) 배경 제거된 식재료 사진을
  합성(Poisson blending)해서 만들어줍니다.

▶ 사전 준비물 (직접 찍거나 크롤링해서 준비)
  1) 빈 도마 사진 (재료 없이 도마만) — 색상별로 각 5~10장 정도
     예) BOARD_DIR/red/*.jpg, BOARD_DIR/blue/*.jpg, BOARD_DIR/green/*.jpg
  2) 식재료 사진 — 이미 모으신 "정상" 클래스 사진에서 재사용 가능
     예) INGREDIENT_DIR/meat/*.jpg, INGREDIENT_DIR/fish/*.jpg, INGREDIENT_DIR/vegetable/*.jpg

▶ 어떻게 쓰나?
  1) pip install rembg opencv-python pillow numpy --break-system-packages
     (rembg 첫 실행 시 배경제거 모델(u2net) 자동 다운로드, 수십 MB, 인터넷 필요)
  2) 아래 [설정]만 본인 경로/조합에 맞게 고친다
  3) python composite_violation.py 실행

▶ 만들 조합 (MISMATCH_PAIRS에서 정의)
  - 빨간 도마(육류용) + 채소/생선 → 위반
  - 파란 도마(생선용) + 육류/채소 → 위반
  - 초록 도마(채소용) + 육류/생선 → 위반
  (같은 색-재료 정상 조합은 절대 만들지 않도록 주의)

▶ 주의 (중요)
  - 합성본만으로 학습하면 "재료-도마 불일치" 대신 "합성 흔적"을 학습할 위험이 있습니다.
  - 가능하면 실사 5~10장을 확보해 검증(validation)/테스트셋에는 실사 위주로 넣으세요.
  - 생성된 이미지를 열어 부자연스러운 경계/그림자가 심한 것은 직접 골라내세요.
"""
import os
import glob
import random
import cv2
import numpy as np
from PIL import Image

# ============================================================
#  [설정] ▼▼▼ 본인 경로/조합에 맞게 여기만 고치세요 ▼▼▼
# ============================================================

# (1) 빈 도마 사진들이 들어있는 폴더 (색상별 하위폴더)
BOARD_DIR = r"C:\choi\비전 프로젝트\labeling\composite\boards"          # boards/red, boards/blue, boards/green

# (2) 식재료 사진들이 들어있는 폴더 (재료별 하위폴더) — 정상 클래스에서 모은 사진 재사용 가능
INGREDIENT_DIR = r"C:\choi\비전 프로젝트\labeling\composite\ingredients"  # ingredients/meat, ingredients/fish, ingredients/vegetable

# (3) 결과 저장 폴더
OUT_DIR = r"C:\choi\비전 프로젝트\labeling\composite\output"

# (4) 배경 제거 캐시 폴더 (한 번 제거한 식재료는 재사용, 매번 새로 안 돌림)
CUTOUT_CACHE_DIR = r"C:\choi\비전 프로젝트\labeling\composite\_cutout_cache"

# (5) 색상별 도마 - "불일치"로 취급할 재료 목록
#     예) 빨간 도마(육류용)에 채소나 생선을 올리면 위반
MISMATCH_PAIRS = {
    "red":   ["vegetable", "fish"],
    "blue":  ["meat", "vegetable"],
    "green": ["meat", "fish"],
}

# (6) 조합당 생성할 장수 (예: red+vegetable 20장, red+fish 20장 ...)
PER_COMBO = 20

# (5-2) 정상 조합(색-재료가 맞는 경우)도 같은 방식으로 합성 — 정상 클래스 데이터 보강용
NORMAL_PAIRS = {
    "red":   ["meat"],
    "blue":  ["fish"],
    "green": ["vegetable"],
}
PER_COMBO_NORMAL = 10

# (7) 재료 크기/위치 랜덤 범위
SCALE_RANGE = (0.5, 0.85)     # 도마 대비 재료 크기 비율
POS_JITTER = 0.15             # 도마 중앙 기준 위치를 얼마나 흔들지 (비율)

# ============================================================
#  ▲▲▲ 여기까지만 고치면 됩니다. 아래는 안 건드려도 됩니다. ▲▲▲
# ============================================================


def imread_unicode(path):
    """한글 등 유니코드 경로에서도 동작하는 imread 대체 함수"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """한글 등 유니코드 경로에서도 동작하는 imwrite 대체 함수"""
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def get_bg_remover():
    """rembg 세션은 한 번만 만들어 재사용 (모델 로딩 비용 절감)"""
    from rembg import remove, new_session
    session = new_session("u2net")
    return lambda img_bytes: remove(img_bytes, session=session)


def cutout_ingredient(src_path, remover):
    """식재료 사진의 배경을 제거해 RGBA(알파채널 포함) 이미지로 반환.
       이미 제거한 적 있으면 캐시에서 불러옴."""
    os.makedirs(CUTOUT_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(
        CUTOUT_CACHE_DIR, os.path.splitext(os.path.basename(src_path))[0] + "_cut.png"
    )
    if os.path.exists(cache_path):
        return Image.open(cache_path).convert("RGBA")

    with open(src_path, "rb") as f:
        input_bytes = f.read()
    output_bytes = remover(input_bytes)
    cutout = Image.open(__import__("io").BytesIO(output_bytes)).convert("RGBA")
    cutout.save(cache_path, "PNG")
    return cutout


def tight_crop_alpha(cutout_img):
    """알파 채널 기준으로 여백을 잘라내 재료만 딱 맞게 크롭"""
    arr = np.array(cutout_img)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0 or len(ys) == 0:
        return cutout_img
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return cutout_img.crop((x0, y0, x1 + 1, y1 + 1))


def detect_board_bbox(board_bgr):
    """도마 사진에서 실제 도마가 차지하는 영역(bbox)을 감지.
       배경(흰 여백 등)과 색이 다른 픽셀들의 바운딩 박스를 도마 영역으로 간주."""
    h, w = board_bgr.shape[:2]
    corners = np.array([
        board_bgr[0, 0], board_bgr[0, w - 1],
        board_bgr[h - 1, 0], board_bgr[h - 1, w - 1],
    ], dtype=np.float32)
    bg_color = np.median(corners, axis=0)
    diff = np.linalg.norm(board_bgr.astype(np.float32) - bg_color, axis=2)
    fg_mask = (diff > 25).astype(np.uint8) * 255
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    ys, xs = np.where(fg_mask > 0)
    if len(xs) < 100:  # 배경과 거의 구분이 안 되면(=도마가 화면을 꽉 채움) 전체를 도마로 간주
        return 0, 0, w, h
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def composite_one(board_bgr, ingredient_rgba, scale, offset_ratio):
    """도마(BGR, OpenCV) 위에 식재료(RGBA, PIL)를 Poisson blending으로 합성"""
    bx, by, bw, bh = detect_board_bbox(board_bgr)  # 도마의 실제 영역 (여백 제외)

    # 재료 크기 조정: 도마 짧은 변 기준 비율로 스케일
    target_w = int(min(bh, bw) * scale)
    iw, ih = ingredient_rgba.size
    ratio = target_w / iw
    new_w, new_h = target_w, int(ih * ratio)

    # 세로로 아주 길쭉한 재료(대파, 오이 등)는 new_h가 도마 세로 길이를 넘어설 수 있음
    # -> 폭 기준 대신 도마에 실제로 들어가는 한도 내로 다시 스케일 (가로세로 비율 유지)
    max_h = int(bh * 0.85)
    max_w = int(bw * 0.85)
    if new_h > max_h or new_w > max_w:
        shrink = min(max_h / new_h, max_w / new_w)
        new_w = max(1, int(new_w * shrink))
        new_h = max(1, int(new_h * shrink))

    ingredient_rgba = ingredient_rgba.resize((new_w, new_h), Image.LANCZOS)

    ing_arr = np.array(ingredient_rgba)
    ing_bgr = cv2.cvtColor(ing_arr[:, :, :3], cv2.COLOR_RGB2BGR)
    ing_alpha = ing_arr[:, :, 3]

    # 재료가 "도마 실제 영역" 경계를 넘지 않도록 중심 좌표 계산 (약간의 랜덤 지터 포함)
    margin_x = min(new_w // 2 + 5, bw // 2)
    margin_y = min(new_h // 2 + 5, bh // 2)
    cx = bx + bw // 2 + int(offset_ratio[0] * bw)
    cy = by + bh // 2 + int(offset_ratio[1] * bh)
    cx = int(np.clip(cx, bx + margin_x, max(bx + margin_x, bx + bw - margin_x)))
    cy = int(np.clip(cy, by + margin_y, max(by + margin_y, by + bh - margin_y)))

    # seamlessClone(Poisson blending)은 재료 색/질감을 도마 쪽과 섞어버려서
    # "재료가 도마에 녹아있는" 것처럼 보이는 부작용이 있음 (NORMAL/MIXED 둘 다 마찬가지).
    # -> 대신 재료를 원래 색 그대로 오려 붙이고(직접 알파 합성), 그림자만 자연스럽게 넣는 방식으로 변경.
    #    실제로 물체를 표면에 올려놨을 때의 느낌에 더 가깝다.

    # 반투명한 경계(배경 제거가 애매하게 된 부분)는 제외하고 확실한 전경만 사용
    hard_mask = np.zeros((new_h, new_w), dtype=np.uint8)
    hard_mask[ing_alpha > 160] = 255
    kernel = np.ones((5, 5), np.uint8)
    hard_mask = cv2.morphologyEx(hard_mask, cv2.MORPH_OPEN, kernel)
    hard_mask = cv2.morphologyEx(hard_mask, cv2.MORPH_CLOSE, kernel)

    if hard_mask.sum() == 0:
        return None  # 알파가 전부 비어있으면(배경제거 실패) 스킵
    if hard_mask.mean() / 255.0 < 0.03:
        return None  # 전경 비율이 너무 작으면(배경 제거가 거의 실패한 경우) 스킵

    # 가장자리만 살짝 블러해서 부드러운 알파(anti-aliasing)를 만듦 — 내부는 그대로 불투명 유지
    soft_alpha = cv2.GaussianBlur(hard_mask, (7, 7), 0).astype(np.float32) / 255.0

    img_h, img_w = board_bgr.shape[:2]
    result = board_bgr.copy()

    y0, y1 = cy - new_h // 2, cy - new_h // 2 + new_h
    x0, x1 = cx - new_w // 2, cx - new_w // 2 + new_w

    # --- 그림자: 재료 모양을 살짝 아래-오른쪽으로 밀고 크게 블러해서 어둡게 깔아준다 ---
    shadow_offset = max(3, min(new_w, new_h) // 20)
    shadow_full = np.zeros(board_bgr.shape[:2], dtype=np.float32)
    sy0, sx0 = y0 + shadow_offset, x0 + shadow_offset
    sy0c, sx0c = max(0, sy0), max(0, sx0)
    sy1c, sx1c = min(img_h, sy0 + new_h), min(img_w, sx0 + new_w)
    if sy1c > sy0c and sx1c > sx0c:
        msy0, msx0 = sy0c - sy0, sx0c - sx0
        shadow_full[sy0c:sy1c, sx0c:sx1c] = soft_alpha[msy0:msy0 + (sy1c - sy0c), msx0:msx0 + (sx1c - sx0c)]
    shadow_full = cv2.GaussianBlur(shadow_full, (0, 0), sigmaX=shadow_offset * 1.5)
    shadow_strength = 0.35
    result = (result.astype(np.float32) * (1 - shadow_full[:, :, None] * shadow_strength)).astype(np.uint8)

    # --- 재료를 원래 색 그대로 올려 붙이기 (직접 알파 합성) ---
    iy0, iy1 = max(0, -y0), new_h - max(0, y1 - img_h)
    ix0, ix1 = max(0, -x0), new_w - max(0, x1 - img_w)
    y0c, y1c = max(0, y0), min(img_h, y1)
    x0c, x1c = max(0, x0), min(img_w, x1)

    if y1c <= y0c or x1c <= x0c:
        return result  # 겹치는 영역이 없으면 그림자만 있는 도마 반환

    ing_crop = ing_bgr[iy0:iy1, ix0:ix1]
    alpha_crop = soft_alpha[iy0:iy1, ix0:ix1][:, :, None]
    result[y0c:y1c, x0c:x1c] = (
        ing_crop * alpha_crop + result[y0c:y1c, x0c:x1c] * (1 - alpha_crop)
    ).astype(np.uint8)
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("배경 제거 모델 로딩 중 (최초 1회는 다운로드 시간 소요)...", flush=True)
    remover = get_bg_remover()

    total_made = 0
    # (위반 조합, 정상 조합) 둘 다 같은 방식으로 생성 — 라벨 접두사/장수만 다름
    combo_sets = [
        (MISMATCH_PAIRS, PER_COMBO, "위반", "불일치"),
        (NORMAL_PAIRS, PER_COMBO_NORMAL, "정상", "정상 조합"),
    ]

    for pairs, per_combo, prefix, desc in combo_sets:
        for board_color, ingredient_list in pairs.items():
            board_paths = glob.glob(os.path.join(BOARD_DIR, board_color, "*.*"))
            if not board_paths:
                print(f"  ! {board_color} 도마 사진이 없습니다: {os.path.join(BOARD_DIR, board_color)}")
                continue

            for ingredient_name in ingredient_list:
                ing_paths = glob.glob(os.path.join(INGREDIENT_DIR, ingredient_name, "*.*"))
                if not ing_paths:
                    print(f"  ! {ingredient_name} 재료 사진이 없습니다: "
                          f"{os.path.join(INGREDIENT_DIR, ingredient_name)}")
                    continue

                print(f"\n===== {board_color} 도마 + {ingredient_name} ({desc}) =====", flush=True)
                made = 0
                attempts = 0
                while made < per_combo and attempts < per_combo * 3:
                    attempts += 1
                    board_path = random.choice(board_paths)
                    ing_path = random.choice(ing_paths)

                    board_bgr = imread_unicode(board_path)
                    if board_bgr is None:
                        continue

                    try:
                        cutout = cutout_ingredient(ing_path, remover)
                        cutout = tight_crop_alpha(cutout)
                    except Exception as e:
                        print(f"    ! 배경 제거 실패 ({os.path.basename(ing_path)}): {e}")
                        continue

                    scale = random.uniform(*SCALE_RANGE)
                    offset = (
                        random.uniform(-POS_JITTER, POS_JITTER),
                        random.uniform(-POS_JITTER, POS_JITTER),
                    )
                    result = composite_one(board_bgr, cutout, scale, offset)
                    if result is None:
                        continue

                    out_name = f"{prefix}_{board_color}_{ingredient_name}_{made + 1:03d}.jpg"
                    imwrite_unicode(os.path.join(OUT_DIR, out_name), result)
                    made += 1
                    total_made += 1

                print(f"  -> {made}장 생성 (시도 {attempts}회)", flush=True)

    print(f"\n완료! 총 {total_made}장 생성됨 -> {OUT_DIR}", flush=True)
    print("생성된 이미지를 열어 경계가 부자연스럽거나 이상한 것은 직접 삭제해 선별하세요.", flush=True)


if __name__ == "__main__":
    main()
