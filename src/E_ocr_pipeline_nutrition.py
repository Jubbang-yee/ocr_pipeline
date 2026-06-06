# pip install opencv-python paddlepaddle paddleocr numpy ultralytics

import cv2
import numpy as np
import os
import re
import logging
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ─────────────────────────────────
# 전역 모델 초기화
# ─────────────────────────────────
logging.getLogger('ppocr').setLevel(logging.ERROR)

print("모델 초기화 중...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yolo_model = YOLO(os.path.join(BASE_DIR, "nutrition_best.pt"))
paddle_ocr  = PaddleOCR(lang='korean', enable_mkldnn=False)
print("모델 초기화 완료")


# ═══════════════════════════════════════════════════════════════
# STEP 1. 이미지 진단
# ═══════════════════════════════════════════════════════════════

def diagnose_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score      = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness_mean = float(np.mean(gray))
    brightness_std  = float(np.std(gray))

    hist        = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_smooth = np.convolve(hist, np.ones(10)/10, mode='same')
    peaks = np.where(
        (hist_smooth[1:-1] > hist_smooth[:-2]) &
        (hist_smooth[1:-1] > hist_smooth[2:])
    )[0] + 1
    peaks          = peaks[hist_smooth[peaks] > hist_smooth.max() * 0.1]
    contrast_score = float(peaks[-1] - peaks[0]) if len(peaks) >= 2 else 0.0
    glare_ratio    = float(np.sum(gray > 230) / gray.size)

    diagnosis = {
        "blur_score"      : blur_score,
        "brightness_mean" : brightness_mean,
        "brightness_std"  : brightness_std,
        "contrast_score"  : contrast_score,
        "glare_ratio"     : glare_ratio,
        "is_blurry"       : blur_score < 80,
        "is_dark"         : brightness_mean < 80,
        "is_uneven_light" : brightness_std > 70,
        "has_glare"       : glare_ratio > 0.15,
        "low_contrast"    : contrast_score < 60,
    }

    print("\n  [이미지 진단]")
    for k, v in diagnosis.items():
        print(f"    {k:20s}: {v:.2f}" if isinstance(v, float) else f"    {k:20s}: {v}")

    return diagnosis


# ═══════════════════════════════════════════════════════════════
# STEP 2. 원근 보정
# ═══════════════════════════════════════════════════════════════

def correct_perspective(image):
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged   = cv2.Canny(blurred, 30, 120)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:5]:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts     = approx.reshape(4, 2).astype(np.float32)
            s       = pts.sum(axis=1)
            d       = np.diff(pts, axis=1).flatten()
            ordered = np.array([
                pts[np.argmin(s)], pts[np.argmin(d)],
                pts[np.argmax(s)], pts[np.argmax(d)],
            ], dtype=np.float32)

            w = max(np.linalg.norm(ordered[1]-ordered[0]),
                    np.linalg.norm(ordered[2]-ordered[3]))
            h = max(np.linalg.norm(ordered[3]-ordered[0]),
                    np.linalg.norm(ordered[2]-ordered[1]))

            if w * h < image.shape[0] * image.shape[1] * 0.1:
                continue

            dst    = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
            M      = cv2.getPerspectiveTransform(ordered, dst)
            warped = cv2.warpPerspective(image, M, (int(w), int(h)), flags=cv2.INTER_CUBIC)
            print(f"  [원근 보정] {image.shape[1]}x{image.shape[0]} → {int(w)}x{int(h)}")
            return warped

    print("  [원근 보정] 사각형 윤곽 미검출 → 원본 유지")
    return image


# ═══════════════════════════════════════════════════════════════
# STEP 3. 기울기 보정
# ═══════════════════════════════════════════════════════════════

def correct_skew(image):
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=100, minLineLength=100, maxLineGap=10)
    if lines is None:
        return image

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))
            if -45 < angle < 45:
                angles.append(angle)

    if not angles:
        return image

    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return image

    h, w    = image.shape[:2]
    M       = cv2.getRotationMatrix2D((w//2, h//2), median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    print(f"  [기울기 보정] {median_angle:.2f}°")
    return rotated


# ═══════════════════════════════════════════════════════════════
# STEP 4. 진단 기반 전처리
# ═══════════════════════════════════════════════════════════════

def apply_glare_removal(gray):
    _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask    = cv2.dilate(mask, kernel, iterations=1)
    return cv2.inpaint(gray, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

def apply_unsharp_mask(gray, sigma=3.0, strength=1.8):
    blur = cv2.GaussianBlur(gray, (0,0), sigma)
    return cv2.addWeighted(gray, strength, blur, -(strength-1), 0)

def preprocess_for_paddle(image, diagnosis):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if diagnosis["has_glare"]:
        gray = apply_glare_removal(gray)
        print("  [전처리] 반사 제거 적용")

    if diagnosis["is_blurry"]:
        gray = apply_unsharp_mask(gray)
        print("  [전처리] 언샤프 마스킹 적용")

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ═══════════════════════════════════════════════════════════════
# STEP 5. 해상도 보장
# ═══════════════════════════════════════════════════════════════

def ensure_resolution(image, target_char_height=40):
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    if regions:
        heights = [cv2.boundingRect(r.reshape(-1,1,2))[3] for r in regions]
        avg_char_height = np.median(heights)

        if avg_char_height < target_char_height:
            scale = target_char_height / avg_char_height
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            print(f"  [해상도 보정] 글자높이 {avg_char_height:.1f}px → {target_char_height}px 기준 {scale:.2f}배 확대")
        elif avg_char_height > target_char_height * 3:
            scale = (target_char_height * 3) / avg_char_height
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            print(f"  [해상도 보정] 글자높이 {avg_char_height:.1f}px → 축소")

    return image


# ═══════════════════════════════════════════════════════════════
# STEP 6. YOLO 탐지
# ═══════════════════════════════════════════════════════════════

def detect_with_yolo(image):
    results  = yolo_model(image, verbose=False)
    detected = []

    for box in results[0].boxes:
        cls_id   = int(box.cls[0])
        cls_name = yolo_model.names[cls_id]
        conf     = float(box.conf[0])

        if conf > 0.3:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected.append((x1, y1, x2, y2, cls_name))
            print(f"  YOLO 탐지: {cls_name} ({conf:.2f})")

    if detected:
        return detected, True

    h, w = image.shape[:2]
    print("  YOLO 탐지 실패 → 전체 이미지 사용")
    return [(0, 0, w, h, 'full')], False


# ═══════════════════════════════════════════════════════════════
# STEP 7. PaddleOCR 실행
# ═══════════════════════════════════════════════════════════════

def merge_split_rows(ocr_results, y_threshold=15):
    """y좌표가 비슷하고 x좌표가 연속인 bbox 병합"""
    if not ocr_results:
        return ocr_results

    merged = []
    used = set()

    for i, (bbox_i, text_i, conf_i) in enumerate(ocr_results):
        if i in used:
            continue
        cy_i = sum(p[1] for p in bbox_i) / len(bbox_i)
        cx_end_i = max(p[0] for p in bbox_i)

        for j, (bbox_j, text_j, conf_j) in enumerate(ocr_results):
            if j <= i or j in used:
                continue
            cy_j = sum(p[1] for p in bbox_j) / len(bbox_j)
            cx_start_j = min(p[0] for p in bbox_j)

            if abs(cy_i - cy_j) < y_threshold and abs(cx_end_i - cx_start_j) < 80:
                text_i = text_i + text_j
                conf_i = (conf_i + conf_j) / 2
                used.add(j)

        merged.append((bbox_i, text_i, conf_i))
        used.add(i)

    return merged


def run_paddleocr(image):
    result = paddle_ocr.predict(image)
    if not result:
        print("  [PaddleOCR] 결과 없음")
        return []

    parsed = []
    for res in result:
        polys  = res.get("dt_polys",   [])
        texts  = res.get("rec_texts",  [])
        scores = res.get("rec_scores", [])
        for bbox, text, conf in zip(polys, texts, scores):
            if float(conf) >= 0.5 and text.strip():
                bbox_list = bbox.tolist() if hasattr(bbox, "tolist") else list(bbox)
                parsed.append((bbox_list, text.strip(), float(conf)))

    parsed.sort(key=lambda x: min(p[1] for p in x[0]))
    parsed = merge_split_rows(parsed)

    avg_conf = sum(c for _, _, c in parsed) / len(parsed) if parsed else 0.0
    print(f"  [PaddleOCR] 검출: {len(parsed)}개  평균conf: {avg_conf:.2f}")
    return parsed


# ═══════════════════════════════════════════════════════════════
# STEP 8. OCR 결과 → 행 단위 텍스트 묶기
# ═══════════════════════════════════════════════════════════════

def group_into_rows(ocr_results, y_threshold=15):
    if not ocr_results:
        return []

    items = []
    for bbox, text, conf in ocr_results:
        cy = sum(p[1] for p in bbox) / len(bbox)
        cx = sum(p[0] for p in bbox) / len(bbox)
        items.append({"text": text, "conf": conf, "cx": cx, "cy": cy})

    items.sort(key=lambda x: x["cy"])

    rows        = []
    current_row = [items[0]]
    for item in items[1:]:
        if abs(item["cy"] - current_row[-1]["cy"]) < y_threshold:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]
    rows.append(current_row)

    result = []
    for row in rows:
        row_sorted = sorted(row, key=lambda x: x["cx"])
        result.append({
            "texts": [r["text"] for r in row_sorted],
            "cy"   : row_sorted[0]["cy"],
        })

    return result


# ═══════════════════════════════════════════════════════════════
# STEP 9. 영양성분 구조화 파싱
# ═══════════════════════════════════════════════════════════════

NUTRIENT_KEYWORDS = {
    "열량", "칼로리",
    "탄수화물", "당류", "식이섬유",
    "단백질",
    "지방", "포화지방", "트랜스지방", "불포화지방",
    "나트륨", "콜레스테롤",
    "칼슘", "철", "철분",
    "비타민A", "비타민B", "비타민B1", "비타민B2", "비타민B6", "비타민B12",
    "비타민C", "비타민D", "비타민E", "비타민K",
    "엽산", "나이아신", "판토텐산", "비오틴",
    "마그네슘", "아연", "셀레늄", "망간", "구리", "요오드", "불소", "크롬",
    "칼륨", "몰리브덴",
    "오메가", "EPA", "DHA",
    "코엔자임", "루테인", "지아잔틴",
    "글루코사민", "콘드로이친",
    "셀렌", "플라보노이드",
}

UNIT_PATTERN = re.compile(
    r'(\d+[\.,]?\d*)\s*(kcal|cal|mg|μg|ug|mcg|ml|IU|NE|g)'
)

def _extract_value(text):
    # 괄호 안 % 제거
    text = re.sub(r'\(\s*[\d\.,]+\s*%?\s*\)', '', text)
    match = UNIT_PATTERN.search(text)
    if match:
        return match.group(0).strip()
    # 단위만 있는 경우 (예: "탄수화물g")
    unit_only = re.search(r'\b(kcal|cal|g|mg|μg|ug|mcg)\b', text)
    if unit_only:
        before = text[:unit_only.start()].strip()
        num = re.search(r'[\d]+[\.,]?\d*$', before)
        if num:
            return num.group(0) + unit_only.group(0)
        return '0' + unit_only.group(0)
    num_only = re.search(r'\d+[\.,]?\d*', text)
    if num_only:
        return num_only.group(0)
    return None

def clean_ocr_text(text):
    """OCR 오인식 교정"""
    text = re.sub(r'(?<=\s)O(?=\s*kcal|g|mg)', '0', text)
    text = re.sub(r':O\s', ':0 ', text)
    text = re.sub(r'(\d+)\s*r\b', r'\1mg', text)
    text = text.replace('ug', 'μg').replace('mcg', 'μg')
    return text

def parse_nutrition_rows(rows):
    results = []
    used_kw = set()

    # 1단계: 행 단위 파싱 (순서 유지)
    for row in rows:
        line = clean_ocr_text(" ".join(row["texts"]))
        line = re.sub(r'([가-힣])\s([가-힣])', r'\1\2', line)
        cy   = row["cy"]

        for seg in re.split(r'[,，]', line):
            seg = seg.strip()
            if not seg:
                continue
            value = _extract_value(seg)
            if not value:
                continue
            seg_clean = seg.replace(" ", "")
            for kw in sorted(NUTRIENT_KEYWORDS, key=len, reverse=True):
                if kw.replace(" ", "") in seg_clean and kw not in used_kw:
                    results.append((cy, kw, value))
                    used_kw.add(kw)
                    break

    # 2단계: 인접 행 2개씩 합쳐서 재파싱 (셀/렌 같은 쪼개진 경우 보완)
    for i in range(len(rows) - 1):
        combined = " ".join(rows[i]["texts"]) + " " + " ".join(rows[i+1]["texts"])
        combined = clean_ocr_text(combined)
        combined = re.sub(r'([가-힣])\s([가-힣])', r'\1\2', combined)
        cy = rows[i]["cy"]

        for seg in re.split(r'[,，]', combined):
            seg = seg.strip()
            if not seg:
                continue
            value = _extract_value(seg)
            if not value:
                continue
            seg_clean = seg.replace(" ", "")
            for kw in sorted(NUTRIENT_KEYWORDS, key=len, reverse=True):
                if kw.replace(" ", "") in seg_clean and kw not in used_kw:
                    results.append((cy, kw, value))
                    used_kw.add(kw)
                    break

    # y좌표 기준 정렬 후 딕셔너리 변환
    results.sort(key=lambda x: x[0])
    return {kw: value for _, kw, value in results}


# ═══════════════════════════════════════════════════════════════
# 전체 파이프라인
# ═══════════════════════════════════════════════════════════════

def process_nutrition_image(image: np.ndarray, bbox: dict = None, save_debug=False):
    if image is None:
        return None

    base_name = "upload"

    # ── 1. 기하 보정 ──────────────────────────────────────────
    print("\n" + "="*55)
    print("STEP 1. 기하 보정 (기울기)")
    print("="*55)
    image = correct_skew(image)

    # ── 2. YOLO 탐지 ──────────────────────────────────────────
    if bbox:
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        image = image[y:y+h, x:x+w]
        print(f"  [bbox crop] x={x} y={y} w={w} h={h}")
        regions = [(0, 0, image.shape[1], image.shape[0], "bbox")]
        yolo_success = True
    else:
        print("\n" + "="*55)
        print("STEP 2. YOLO 영양성분표 탐지")
        print("="*55)
        regions, yolo_success = detect_with_yolo(image)

    all_nutrition = {}

    for idx, (*coords, label) in enumerate(regions):
        x1, y1, x2, y2 = coords
        print(f"\n{'='*55}")
        print(f"STEP 3~7: 영역 [{idx+1}: {label}]")
        print("="*55)

        cropped = image[y1:y2, x1:x2]

        print("\n[3] 이미지 진단")
        diagnosis = diagnose_image(cropped)

        print("\n[4] 해상도 보장")
        cropped = ensure_resolution(cropped)

        img_h, img_w = cropped.shape[:2]
        y_thr = int(np.clip(img_h * 0.015, 10, 60))
        print(f"  [동적 임계값] y_threshold={y_thr}px")

        print("\n[5] 전처리")
        preprocessed = preprocess_for_paddle(cropped, diagnosis)

        if save_debug:
            cv2.imwrite(f"debug_{base_name}_{label}_preprocessed.jpg", preprocessed)
            print(f"  디버그 이미지 저장")

        print("\n[6] PaddleOCR 실행")
        ocr_results = run_paddleocr(preprocessed)

        print("\n[7] 행 단위 묶기 → 영양성분 파싱")
        rows      = group_into_rows(ocr_results, y_threshold=y_thr)
        nutrition = parse_nutrition_rows(rows)
        all_nutrition.update(nutrition)

    print("\n" + "="*55)
    print("최종 영양성분 추출 결과")
    print("="*55)
    if all_nutrition:
        for nutrient, value in all_nutrition.items():
            print(f"  {nutrient:15s}: {value}")
    else:
        print("  추출된 영양성분 없음")

    return {
        "nutrition"    : all_nutrition,
        "yolo_success" : yolo_success,
    }


# ─────────────────────────────────
# 실행
# ─────────────────────────────────
if __name__ == "__main__":
    import sys

    target_image = sys.argv[1] if len(sys.argv) > 1 else "nutrition_sample.jpg"

    image = cv2.imread(target_image)
    if image is None:
        print(f"  [오류] 파일 없음: {target_image}")
    else:
        h, w = image.shape[:2]
        if w > 1500:
            scale = 1500 / w
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            print(f"  이미지 축소: {w}x{h} → {int(w*scale)}x{int(h*scale)}")

        result = process_nutrition_image(image, save_debug=False)

        print("\n" + "="*55)
        print("최종 결과 요약")
        print("="*55)
        if result:
            print(f"YOLO: {'성공' if result['yolo_success'] else '실패(전체이미지)'}")
            print("\n[영양성분표]")
            for nutrient, value in result["nutrition"].items():
                print(f"  {nutrient:15s}: {value}")
        else:
            print("  결과 없음")