# pip install opencv-python easyocr typing_extensions torch numpy ultralytics pytesseract
# pip install transformers torchvision Pillow timm

# pytesseract 에 대해서는 별도의 설치가 필요. https://github.com/UB-Mannheim/tesseract/wiki

import cv2
import easyocr
import numpy as np
import os
import ssl
import threading
import pytesseract
import hashlib
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO
from difflib import SequenceMatcher
import time
import json
from google import genai

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

GEMINI_API_KEY2 = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY2:
    raise ValueError("GEMINI_API_KEY 환경변수를 설정해주세요.")

GEMINI_MODEL   = "gemini-2.5-flash"
_gemini_client = genai.Client(api_key=GEMINI_API_KEY2)

ssl._create_default_https_context = ssl._create_unverified_context

# ─────────────────────────────────
# Tesseract 경로 자동 탐색
# ─────────────────────────────────
def _find_tesseract():
    import shutil, platform
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path and os.path.isfile(env_path):
        return env_path
    if platform.system() == "Windows":
        win_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.isfile(win_default):
            return win_default
    found = shutil.which("tesseract")
    if found:
        return found
    raise FileNotFoundError(
        "Tesseract를 찾을 수 없습니다. "
        "환경변수 TESSERACT_CMD에 실행 경로를 지정하거나 "
        "시스템 PATH에 tesseract가 있어야 합니다."
    )

pytesseract.pytesseract.tesseract_cmd = _find_tesseract()

# ─────────────────────────────────
# 조기 종료 기준
# ─────────────────────────────────
EARLY_STOP_MIN_RESULTS  = 5
EARLY_STOP_CONF_THRESH  = 0.80
LOW_PERF_CONF_THRESH    = 0.40   # 연속 저성능 버전 판단 기준
LOW_PERF_CONSECUTIVE    = 2      # 연속 N개 저성능이면 스킵

# ─────────────────────────────────
# 전역 싱글톤: 모델 + MSER
# ─────────────────────────────────
print("모델 초기화 중...")
yolo_model = YOLO("best.pt")
ocr_reader  = easyocr.Reader(['ko', 'en'], gpu=False)
_MSER       = cv2.MSER_create(delta=5, min_area=60, max_area=14400)
print("모델 초기화 완료")

# ─────────────────────────────────
# EasyOCR 결과 캐시 → LRU 방식으로 메모리 누수 방지
# ─────────────────────────────────
class LRUCache(OrderedDict):
    def __init__(self, maxsize=64):
        self.maxsize = maxsize
        super().__init__()

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            self.popitem(last=False)

_ocr_cache: LRUCache = LRUCache(maxsize=64)

def _img_hash(img: np.ndarray) -> str:
    return hashlib.md5(img.tobytes()).hexdigest()


# ═══════════════════════════════════════════════════════════════
# STEP 1. 이미지 진단
# ═══════════════════════════════════════════════════════════════

def diagnose_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
           if len(image.shape) == 3 else image

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
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
              if len(image.shape) == 3 else image
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
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
            if len(image.shape) == 3 else image
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
# STEP 4. 진단 기반 전처리 선택 + 실행
# ═══════════════════════════════════════════════════════════════

def apply_glare_removal(gray):
    _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask    = cv2.dilate(mask, kernel, iterations=1)
    return cv2.inpaint(gray, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

def apply_clahe(gray, clip=3.0, grid=(8,8)):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(gray)

def apply_unsharp_mask(gray, sigma=3.0, strength=1.8):
    blur = cv2.GaussianBlur(gray, (0,0), sigma)
    return cv2.addWeighted(gray, strength, blur, -(strength-1), 0)

def apply_adaptive_threshold(gray):
    denoised = cv2.GaussianBlur(gray, (3,3), 0)
    binary   = cv2.adaptiveThreshold(denoised, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

def apply_otsu(gray):
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def apply_background_removal(gray):
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(cleaned)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def apply_tophat(gray):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def apply_lab_enhance(image):
    if len(image.shape) == 2:
        return apply_clahe(image, clip=2.0)
    lab     = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    l_eq    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    bgr     = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_Lab2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def select_and_build_versions(image, diagnosis):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
           if len(image.shape) == 3 else image

    is_clean = (
        not diagnosis["is_blurry"]
        and not diagnosis["has_glare"]
        and not diagnosis["is_uneven_light"]
        and not diagnosis["low_contrast"]
    )
    if is_clean:
        print("  [전처리] 고품질 이미지 → 최소 버전(2개)")
        return [
            ("원본그레이",   gray),
            ("적응형이진화", apply_adaptive_threshold(gray)),
        ]

    versions = [
        ("원본그레이",   gray),
        ("CLAHE",       apply_clahe(gray)),
        ("적응형이진화", apply_adaptive_threshold(gray)),
        ("배경제거강화", apply_background_removal(gray)),
    ]

    if diagnosis["has_glare"]:
        deglared = apply_glare_removal(gray)
        versions += [
            ("반사제거",        deglared),
            ("반사제거+CLAHE",  apply_clahe(deglared)),
            ("반사제거+적응형", apply_adaptive_threshold(deglared)),
        ]
        print("  [전처리 선택] 반사 제거 추가")

    if diagnosis["is_blurry"]:
        versions += [
            ("언샤프마스킹",       apply_unsharp_mask(gray)),
            ("언샤프+적응형이진화", apply_adaptive_threshold(apply_unsharp_mask(gray))),
        ]
        print("  [전처리 선택] 언샤프 마스킹 추가")

    if diagnosis["is_uneven_light"]:
        versions += [
            ("CLAHE_강화", apply_clahe(gray, clip=4.0, grid=(4,4))),
            ("Lab강조",    apply_lab_enhance(image)),
        ]
        print("  [전처리 선택] 조명 균일화 강화 추가")

    if diagnosis["low_contrast"]:
        versions.append(("TopHat형태학", apply_tophat(gray)))
        print("  [전처리 선택] TopHat 형태학 추가")

    if not diagnosis["low_contrast"] and not diagnosis["has_glare"]:
        versions.append(("Otsu이진화", apply_otsu(gray)))
        print("  [전처리 선택] Otsu 이진화 추가")

    print(f"  [전처리 버전 수] {len(versions)}개")
    return versions


# ═══════════════════════════════════════════════════════════════
# STEP 5. 해상도 보장
# ═══════════════════════════════════════════════════════════════

def ensure_resolution(image, target_char_height=40):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
           if len(image.shape) == 3 else image

    regions, _ = _MSER.detectRegions(gray)

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
# STEP 6. 텍스트 영역 검출 (MSER)
# ═══════════════════════════════════════════════════════════════

def detect_text_regions(gray, y_cluster_px=30):
    regions, _ = _MSER.detectRegions(gray)

    if not regions:
        h, w = gray.shape
        return [(0, 0, w, h)]

    hulls = [cv2.convexHull(r.reshape(-1,1,2)) for r in regions]
    rects = [cv2.boundingRect(h) for h in hulls]
    if not rects:
        h, w = gray.shape
        return [(0, 0, w, h)]

    rects_sorted  = sorted(rects, key=lambda r: r[1])
    row_groups    = []
    current_group = [rects_sorted[0]]

    for rect in rects_sorted[1:]:
        if abs(rect[1] - current_group[-1][1]) < y_cluster_px:
            current_group.append(rect)
        else:
            row_groups.append(current_group)
            current_group = [rect]
    row_groups.append(current_group)

    img_area = gray.shape[0] * gray.shape[1]
    roi_list = []

    for group in row_groups:
        x_min = min(r[0] for r in group)
        y_min = min(r[1] for r in group)
        x_max = max(r[0]+r[2] for r in group)
        y_max = max(r[1]+r[3] for r in group)
        w, h  = x_max - x_min, y_max - y_min

        roi_area = w * h
        if roi_area < 100 or roi_area > img_area * 0.95:
            continue

        roi_list.append((x_min, y_min, w, h))

    if not roi_list:
        h, w = gray.shape
        return [(0, 0, w, h)]

    return roi_list


# ═══════════════════════════════════════════════════════════════
# STEP 7. OCR 실행 (EasyOCR + Tesseract)
# ═══════════════════════════════════════════════════════════════

def is_noise_text(text):
    text = text.strip()

    if len(text) < 2:
        return True

    meaningful = sum(1 for c in text if c.isalnum() or '\uAC00' <= c <= '\uD7A3')
    if len(text) > 0 and meaningful / len(text) < 0.5:
        return True

    has_korean = any('\uAC00' <= c <= '\uD7A3' for c in text)
    has_alpha  = any(c.isalpha() for c in text)
    if not has_korean and not has_alpha:
        return True

    special_chars = set('()[]{}:;,./\\|=-+*&^%$#@!~`"\'=><')
    if all(c in special_chars or c == ' ' for c in text):
        return True

    return False

def _should_early_stop(results):
    if len(results) < EARLY_STOP_MIN_RESULTS:
        return False
    return sum(c for _,_,c in results) / len(results) >= EARLY_STOP_CONF_THRESH

def run_easyocr(processed_img):
    key = _img_hash(processed_img)
    if key in _ocr_cache:
        return _ocr_cache[key]

    results = ocr_reader.readtext(
        processed_img, paragraph=False,
        text_threshold=0.5, low_text=0.4,
        contrast_ths=0.1, adjust_contrast=0.5,
    )
    filtered = [(bbox, text, conf) for bbox, text, conf in results
                if text.strip() and conf >= 0.3 and not is_noise_text(text)]

    _ocr_cache[key] = filtered
    return filtered

def run_tesseract(processed_img):
    config = '--oem 3 --psm 6 -l kor+eng'
    try:
        data = pytesseract.image_to_data(processed_img, config=config,
                                         output_type=pytesseract.Output.DICT)
    except Exception as e:
        print(f"  [Tesseract 오류] {e}")
        return []

    results = []
    for i, text in enumerate(data['text']):
        text = text.strip()
        conf = int(data['conf'][i])
        if not text or conf < 50 or is_noise_text(text):
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        results.append(([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], text, conf/100.0))
    return results


def _ocr_one_version(args, stop_flag: threading.Event):
    """
    stop_flag가 set되면 즉시 빈 결과 반환.
    future.cancel()은 이미 실행 중인 스레드를 멈추지 못하므로
    플래그 방식으로 대체한다.
    """
    ver_name, processed = args
    if stop_flag.is_set():
        return ver_name, [], []
    easy = run_easyocr(processed)
    if stop_flag.is_set():
        return ver_name, easy, []
    tess = run_tesseract(processed)
    return ver_name, easy, tess


def run_ocr_with_early_stop(versions, roi_list):
    """
    [BUG FIX] ROI 좌표 이중 보정 수정:
      기존 코드는 ROI 크롭 결과에 rx1/ry1을 더한 좌표를 all_results에 바로 추가했다.
      그런데 process_image의 to_global()은 이 all_results 전체에 scale 역변환 + YOLO offset을
      적용한다. 즉, 버전 OCR 결과(easy/tess)는 to_global()을 거치는 반면,
      ROI 크롭 결과는 거치지 않아 YOLO offset이 누락되고 scale도 맞지 않는 문제가 있었다.

      수정: ROI 결과도 "크롭+해상도보정 이미지 기준 절대 좌표"로만 변환해서 반환.
            scale 역변환과 YOLO offset 합산은 process_image의 to_global()에서 일괄 처리.
    """
    all_results       = []
    version_score_map = {}
    stopped_early     = False
    stop_flag         = threading.Event()

    with ThreadPoolExecutor(max_workers=min(4, len(versions))) as executor:
        future_map = {
            executor.submit(_ocr_one_version, v, stop_flag): v[0]
            for v in versions
        }

        for future in as_completed(future_map):
            ver_name = future_map[future]
            try:
                ver_name, easy, tess = future.result()
            except Exception as e:
                print(f"  [{ver_name}] 오류: {e}")
                continue

            combined = easy + tess
            all_results.extend(combined)

            avg_conf = (sum(c for _,_,c in combined) / len(combined)) if combined else 0.0
            version_score_map[ver_name] = avg_conf

            print(f"  [{ver_name}] easy:{len(easy)} tess:{len(tess)} "
                  f"avg_conf:{avg_conf:.2f} | 누적:{len(all_results)}")

            if _should_early_stop(all_results):
                print(f"  ★ 조기 종료: '{ver_name}'에서 신뢰도 조건 충족")
                stop_flag.set()
                stopped_early = True
                break

            scores_so_far = list(version_score_map.values())
            if (len(scores_so_far) >= LOW_PERF_CONSECUTIVE
                    and all(s < LOW_PERF_CONF_THRESH
                            for s in scores_so_far[-LOW_PERF_CONSECUTIVE:])):
                print(f"  ★ 저성능 연속 {LOW_PERF_CONSECUTIVE}회 → 나머지 버전 스킵")
                stop_flag.set()
                stopped_early = True
                break

    # ── ROI 크롭 OCR (조기 종료되지 않은 경우만) ──────────────
    if not stopped_early:
        if version_score_map:
            best_name = max(version_score_map, key=version_score_map.get)
            best_ver  = next(v for v in versions if v[0] == best_name)
        else:
            best_ver = versions[0]
        _, best_processed = best_ver

        for (rx, ry, rw, rh) in roi_list:
            pad  = 8
            rx1  = max(0, rx-pad);  ry1 = max(0, ry-pad)
            rx2  = min(best_processed.shape[1], rx+rw+pad)
            ry2  = min(best_processed.shape[0], ry+rh+pad)
            roi_crop = best_processed[ry1:ry2, rx1:rx2]
            if roi_crop.size == 0:
                continue

            for (bbox, text, conf) in run_easyocr(roi_crop):
                # [BUG FIX] 크롭 내 좌표(p[0], p[1])에 ROI 오프셋(rx1, ry1)만 더해
                #   "크롭+해상도보정 이미지 기준 절대 좌표"로 변환.
                #   scale 역변환 + YOLO offset 합산은 process_image의 to_global()에서 일괄 처리.
                #   (기존: rx1/ry1 더한 좌표를 all_results에 직접 추가 → to_global() 미적용 경로)
                adjusted_bbox = [
                    [p[0] + rx1, p[1] + ry1]
                    for p in bbox
                ]
                all_results.append((adjusted_bbox, text, conf))

        if _should_early_stop(all_results):
            print("  ★ 조기 종료: ROI 처리 후 조건 충족")
            stopped_early = True

    return all_results, stopped_early


# ═══════════════════════════════════════════════════════════════
# STEP 8. 결과 통합 (중복 제거 + 행 단위 병합)
# ═══════════════════════════════════════════════════════════════

def get_bbox_center(bbox):
    return sum(p[0] for p in bbox)/len(bbox), sum(p[1] for p in bbox)/len(bbox)

def get_bbox_x_extent(bbox):
    xs = [p[0] for p in bbox]
    return min(xs), max(xs)

def is_similar(a, b, threshold=0.85):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

def deduplicate(all_results):
    sorted_results = sorted(all_results, key=lambda x: x[2], reverse=True)
    seen_exact: set  = set()
    seen_fuzzy: list = []
    deduped          = []

    for bbox, text, conf in sorted_results:
        key = text.strip().lower()
        if not key or len(key) < 2 or is_noise_text(text):
            continue

        if key in seen_exact:
            continue

        is_dup = False
        for seen in seen_fuzzy:
            if abs(len(key) - len(seen)) > max(len(key), len(seen)) * 0.3:
                continue
            if is_similar(key, seen):
                is_dup = True
                break

        if not is_dup:
            seen_exact.add(key)
            seen_fuzzy.append(key)
            deduped.append((bbox, text, conf))

    return deduped

def merge_by_row(results, y_threshold=20, x_gap_threshold=60):
    if not results:
        return []

    items = []
    for bbox, text, conf in results:
        cx, cy = get_bbox_center(bbox)
        x_start, x_end = get_bbox_x_extent(bbox)
        items.append({
            "bbox"   : bbox,
            "text"   : text,
            "conf"   : conf,
            "cx"     : cx,
            "cy"     : cy,
            "x_start": x_start,
            "x_end"  : x_end,
        })

    items_sorted = sorted(items, key=lambda x: x["cy"])
    rows, current_row = [], [items_sorted[0]]

    for item in items_sorted[1:]:
        if abs(item["cy"] - current_row[-1]["cy"]) < y_threshold:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]
    rows.append(current_row)

    merged = []
    for row in rows:
        row_sorted = sorted(row, key=lambda x: x["x_start"])
        current    = row_sorted[0]
        for nxt in row_sorted[1:]:
            gap = nxt["x_start"] - current["x_end"]
            if gap < x_gap_threshold:
                # [BUG FIX] merge_by_row cy 갱신:
                #   기존: cy를 current["cy"]로 고정 → 병합이 길어질수록 cy가 첫 항목 기준에
                #         고정되어 행 분리 판단(y_threshold 비교)이 틀어지는 문제.
                #   수정: cy를 두 항목의 평균으로 점진 갱신.
                #         (x_end도 nxt 기준으로 갱신 — 기존과 동일하게 유지)
                current = {
                    "bbox"   : current["bbox"],
                    "text"   : current["text"] + " " + nxt["text"],
                    "conf"   : (current["conf"] + nxt["conf"]) / 2,
                    "cx"     : nxt["cx"],
                    "cy"     : (current["cy"] + nxt["cy"]) / 2.0,  # ← FIX: 평균 cy 갱신
                    "x_start": current["x_start"],
                    "x_end"  : nxt["x_end"],
                }
            else:
                merged.append((current["bbox"], current["text"], current["conf"]))
                current = nxt
        merged.append((current["bbox"], current["text"], current["conf"]))

    return merged


# ═══════════════════════════════════════════════════════════════
# STEP 9. YOLO 탐지 (파인튜닝 모델)
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
# 동적 임계값 계산
# ═══════════════════════════════════════════════════════════════

def calc_dynamic_thresholds(image_h, image_w):
    y_threshold     = int(np.clip(image_h * 0.015, 10, 60))
    x_gap_threshold = int(np.clip(image_w * 0.040, 30, 150))
    y_cluster_px    = int(np.clip(image_h * 0.020, 15, 80))
    print(f"  [동적 임계값] y_threshold={y_threshold}px  "
          f"x_gap={x_gap_threshold}px  y_cluster={y_cluster_px}px")
    return y_threshold, x_gap_threshold, y_cluster_px


# ═══════════════════════════════════════════════════════════════
# 전체 파이프라인
# ═══════════════════════════════════════════════════════════════
def remove_similar_duplicates(text: str) -> str:
    """
    완전 동일 중복 토큰만 제거. 유사 중복은 LLM에 위임.
    """
    tokens = text.split()
    seen = []
    for t in tokens:
        if t not in seen:
            seen.append(t)
    return " ".join(seen)


def refine_with_llm(raw_ocr_text: str, ocr_items: list) -> dict | None:
    if not raw_ocr_text.strip():
        print("  [LLM 정제] OCR 텍스트 없음 → 건너뜀")
        return None

    # LLM 호출 전 부분 포함 중복 제거
    preprocessed_text = remove_similar_duplicates(raw_ocr_text)
    print(f"  [중복 전처리] {raw_ocr_text}")
    print(f"           → {preprocessed_text}")

    top_items = sorted(ocr_items, key=lambda x: x[2], reverse=True)[:10]
    top_texts = "\n".join(f"  ({conf:.2f}) {text}" for _, text, conf in top_items)

    prompt = f"""당신은 의약품·영양제 패키지 OCR 전문 교정가입니다.

[OCR 전체 텍스트]
{preprocessed_text}

[신뢰도 높은 항목 상위 10개]
{top_texts}

[한국 주요 제약회사 참고 목록]
유한양행, GC녹십자, 광동제약, 종근당, 한미약품, 대웅제약, 동아제약, 동아쏘시오홀딩스,
셀트리온, 보령, HK이노엔, 동국제약, JW중외제약, 동아에스티, 제일약품, 일동제약,
SK바이오팜, 한독, 동화약품, 삼진제약, 부광약품, 안국약품, 경동제약, 대원제약,
명인제약, 일양약품, 한림제약, 조아제약, 파마리서치, 영진약품, 신풍제약

[교정 규칙]
※ 아래 규칙은 우선순위 순서입니다. 반드시 모두 준수하세요.

1. 브랜드명 교정
   - 위 제약회사 목록과 1글자만 달라도 목록의 회사명으로 교정
   - 목록에 없는 경우 가장 유사한 이름으로 교정

2. 제품명 교정
   - 실존하는 의약품·영양제 제품명 기준으로 오탈자 교정
   - 후보 간 차이가 나는 부분(예: '리'↔'티')은 탈락시키지 말고 실존 제품명 기준으로 올바른 음절을 선택할 것
   - OCR 후보가 여러 개면 가장 긴 토큰을 기준으로 삼고, 실존 제품명과 대조해 채택
   - 음절 탈락·공백 삽입 금지 — 입력보다 짧은 제품명 출력 금지
   - 영문·숫자·기호(F, Fx, Plus, 1mg, 500mg 등) 절대 삭제 금지
   - 수식어(식물성, 고함량, 발효 등)는 제품명에 포함

3. 노이즈 처리
   - 제거 대상: 판독 불가 문자열(한글·영문·숫자가 무작위 혼합된 것)만 해당
   - 확신할 수 없으면 제거하지 말고 포함
   - 교정 근거 없는 내용 추가 금지

아래 JSON 형식으로만 반환하세요.
마크다운 코드블록(```)이나 다른 설명 없이 순수 JSON만 출력하세요.

{{
  "product_name": "교정된 제품명",
  "brand": "교정된 제조사 또는 브랜드명 (불명확하면 null)",
  "cleaned_text": "노이즈 제거 + 오탈자 교정된 전체 텍스트"
}}"""

    MAX_RETRIES = 3
    RETRY_DELAY = 5

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  Gemini API 호출 중... (시도 {attempt}/{MAX_RETRIES})")
            response = _gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            text = response.text.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)

            print(f"  [LLM 정제 완료]")
            print(f"    제품명  : {parsed.get('product_name')}")
            print(f"    브랜드  : {parsed.get('brand')}")
            return parsed

        except json.JSONDecodeError as e:
            print(f"  [LLM JSON 파싱 오류] {e}")
            return None
        except Exception as e:
            err_msg = str(e)
            if ("503" in err_msg or "UNAVAILABLE" in err_msg) and attempt < MAX_RETRIES:
                wait = RETRY_DELAY * (2 ** (attempt - 1))
                print(f"  [503 과부하] {wait}초 후 재시도...")
                time.sleep(wait)
            else:
                print(f"  [LLM 호출 오류] {e}")
                return None

    return None
    
def process_image(image: np.ndarray, bbox: dict = None, save_debug=True, debug_prefix: str = "upload"):
    if image is None:
        return None

    # ── 1. 기하 보정 ──────────────────────────────────────────
    print("\n" + "="*55)
    print("STEP 1. 기하 보정 (원근 + 기울기)")
    print("="*55)
    image = correct_perspective(image)
    image = correct_skew(image)

    # ── 2. YOLO 탐지 ──────────────────────────────────────────
    # ── bbox가 있으면 crop 후 YOLO/MSER 스킵 ──
    if bbox:
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        image = image[y:y+h, x:x+w]
        print(f"  [bbox crop] x={x} y={y} w={w} h={h}")
        # YOLO 스킵, 전체 이미지를 단일 영역으로 처리
        regions = [(0, 0, image.shape[1], image.shape[0], "bbox")]
        yolo_success = True
    else:
        # 기존 YOLO 탐지
        print("\n" + "="*55)
        print("STEP 2. YOLO 객체 탐지 (파인튜닝 모델)")
        print("="*55)
        regions, yolo_success = detect_with_yolo(image)

    all_ocr_results = []
    class_ocr_map   = defaultdict(list)

    img_h, img_w         = image.shape[:2]
    y_thr, x_gap, _      = calc_dynamic_thresholds(img_h, img_w)

    for idx, (x1, y1, x2, y2, label) in enumerate(regions):
        crop_offset_x = x1
        crop_offset_y = y1

        print(f"\n{'='*55}")
        print(f"STEP 3~7: 영역 [{idx+1}: {label}]  "
            f"크롭 오프셋=({crop_offset_x}, {crop_offset_y})")
        print("="*55)

        cropped = image[y1:y2, x1:x2]

        print("\n[3] 이미지 진단")
        diagnosis = diagnose_image(cropped)

        print("\n[4] 해상도 보장")
        cropped_resized = ensure_resolution(cropped)

        orig_h, orig_w = cropped.shape[:2]
        new_h,  new_w  = cropped_resized.shape[:2]
        scale_x = new_w / orig_w if orig_w > 0 else 1.0
        scale_y = new_h / orig_h if orig_h > 0 else 1.0
        cropped = cropped_resized

        img_h, img_w            = cropped.shape[:2]
        y_thr, x_gap, y_cluster = calc_dynamic_thresholds(img_h, img_w)

        print("\n[5] 전처리 버전 선택 및 생성")
        versions = select_and_build_versions(cropped, diagnosis)

        if save_debug:
            safe_label = label.replace(" ", "_").replace("/", "-")
            for ver_name, ver_img in versions:
                cv2.imwrite(
                    f"debug_{debug_prefix}_{idx}_{safe_label}_{ver_name}.jpg",
                    ver_img
                )
            print(f"  디버그 이미지 {len(versions)}개 저장")

        print("\n[6] MSER 텍스트 영역 검출")
        gray_for_mser = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) \
                        if len(cropped.shape) == 3 else cropped
        roi_list = detect_text_regions(gray_for_mser, y_cluster_px=y_cluster)
        print(f"  검출된 텍스트 행 그룹: {len(roi_list)}개")

        print("\n[7] OCR 실행 (EasyOCR + Tesseract, 병렬화 + 조기 종료)")
        region_results, stopped = run_ocr_with_early_stop(versions, roi_list)
        if stopped:
            print("  → 조기 종료 적용됨")

        def to_global(bbox):
            return [
                [int(p[0] / scale_x) + crop_offset_x,
                int(p[1] / scale_y) + crop_offset_y]
                for p in bbox
            ]

        region_results_global = [
            (to_global(bbox), text, conf)
            for bbox, text, conf in region_results
        ]

        del versions

        all_ocr_results.extend(region_results_global)

        deduped_region = deduplicate(region_results_global)
        merged_region  = merge_by_row(deduped_region, y_threshold=y_thr, x_gap_threshold=x_gap)
        class_ocr_map[label].extend(merged_region)

    # ── 8. OCR 결과 통합 ──────────────────────────────────────
    print("\n" + "="*55)
    print("STEP 8. OCR 결과 통합")
    print("="*55)

    deduped  = deduplicate(all_ocr_results)
    merged   = merge_by_row(deduped, y_threshold=y_thr, x_gap_threshold=x_gap)
    raw_text = " ".join(t for _,t,_ in merged)

    print(f"\n  중복 제거 전: {len(all_ocr_results)}개")
    print(f"  중복 제거 후: {len(deduped)}개")
    print(f"  행 병합 후:   {len(merged)}개")

    # ── 9. 최종 OCR 결과 출력 ────────────────────────────────
    print("\n" + "="*55)
    print("STEP 9. 최종 OCR 결과 (클래스별)")
    print("="*55)

    if not yolo_success:
        print("\n  ※ YOLO 탐지 실패 → 전체 이미지 OCR 결과:")
        for _, text, conf in sorted(class_ocr_map["full"], key=lambda x: x[2], reverse=True):
            print(f"    ({conf:.2f}) {text}")
    else:
        for label, results in class_ocr_map.items():
            if label == "full":
                continue
            print(f"\n  [{label}]")
            if results:
                for _, text, conf in sorted(results, key=lambda x: x[2], reverse=True):
                    print(f"    ({conf:.2f}) {text}")
            else:
                print("    탐지된 결과 없음")

    # ── 10. LLM 텍스트 정제 ───────────────────────────────────
    print("\n" + "="*55)
    print(f"STEP 10. LLM 텍스트 정제 ({GEMINI_MODEL})")
    print("="*55)

    llm_result = refine_with_llm(raw_text, merged)

    return {
        "class_ocr_map"    : dict(class_ocr_map),
        "raw_ocr_text"     : raw_text,
        "ocr_items"        : merged,
        "yolo_success"     : yolo_success,
        "llm_refined_text" : llm_result.get("product_name") if llm_result else None,
        "llm_result"       : llm_result,
    }
