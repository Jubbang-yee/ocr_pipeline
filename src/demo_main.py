"""
demo_main.py
============
제품명 파이프라인(E_ocr_pipeline_product)과
영양성분 파이프라인(E_ocr_pipeline_nutrition)을
각각 다른 이미지로 실행하는 데모 스크립트.

사용법:
    python demo_main.py --product 제품이미지.jpg --nutrition 영양성분이미지.jpg

이미지를 지정하지 않으면 기본값(product_sample.jpg / nutrition_sample.jpg)을 사용합니다.
"""

import argparse
import sys
import os
import cv2
import json
import time

# ─────────────────────────────────────────────────────────
# 경로 설정: 이 파일과 같은 폴더에 파이프라인 파일이 있어야 함
# ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def load_image(path: str, label: str, max_width: int = 1200):
    """이미지 로드 및 기본 검증"""
    if not os.path.isfile(path):
        print(f"  [오류] {label} 이미지를 찾을 수 없습니다: {path}")
        return None
    img = cv2.imread(path)
    if img is None:
        print(f"  [오류] {label} 이미지를 읽을 수 없습니다: {path}")
        return None
    h, w = img.shape[:2]
    print(f"  [{label}] 이미지 로드 완료: {w}x{h}  경로: {path}")

    # 이미지가 너무 크면 축소 (EasyOCR/PaddleOCR 인식률 향상)
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        new_h, new_w = img.shape[:2]
        print(f"  [{label}] 이미지 축소: {w}x{h} → {new_w}x{new_h}")

    return img


def run_product_pipeline(image_path: str):
    """제품명 OCR 파이프라인 실행"""
    print("\n" + "=" * 60)
    print("  [1/2]  제품명 OCR 파이프라인 시작")
    print("=" * 60)

    from E_ocr_pipeline_product import process_image

    image = load_image(image_path, "제품명")
    if image is None:
        return None

    start = time.time()
    result = process_image(image, save_debug=True, debug_prefix="product")
    elapsed = time.time() - start

    print(f"\n  ✅ 제품명 파이프라인 완료  ({elapsed:.1f}초)")
    if result:
        llm = result.get("llm_result") or {}
        print(f"     제품명  : {llm.get('product_name', 'N/A')}")
        print(f"     브랜드  : {llm.get('brand', 'N/A')}")
        print(f"     원본OCR : {result.get('raw_ocr_text', '')[:80]}...")

    return result


def run_nutrition_pipeline(image_path: str):
    """영양성분 OCR 파이프라인 실행"""
    print("\n" + "=" * 60)
    print("  [2/2]  영양성분 OCR 파이프라인 시작")
    print("=" * 60)

    from E_ocr_pipeline_nutrition import process_nutrition_image

    image = load_image(image_path, "영양성분", max_width=1500)
    if image is None:
        return None

    start = time.time()
    result = process_nutrition_image(image)
    elapsed = time.time() - start

    print(f"\n  ✅ 영양성분 파이프라인 완료  ({elapsed:.1f}초)")
    if result:
        nutrients = result.get("nutrition") or {}
        if nutrients:
            print("     [영양성분 결과]")
            for k, v in list(nutrients.items())[:8]:
                print(f"       {k}: {v}")
        else:
            print("     원본 OCR:", str(result)[:120])

    return result


def save_results(product_result, nutrition_result, out_path="results/ocr_output.json"):
    """결과를 JSON으로 저장"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    output = {
        "product_pipeline": {
            "llm_result"   : product_result.get("llm_result")   if product_result else None,
            "raw_ocr_text" : product_result.get("raw_ocr_text") if product_result else None,
            "yolo_success" : product_result.get("yolo_success") if product_result else None,
        },
        "nutrition_pipeline": nutrition_result if nutrition_result else None,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n   결과 저장 완료: {out_path}")


def print_final_summary(product_result, nutrition_result):
    """최종 결과 요약 출력"""
    print("\n" + "=" * 60)
    print("   최종 인식 결과 요약")
    print("=" * 60)

    # 제품명 / 브랜드
    llm = (product_result or {}).get("llm_result") or {}
    print(f"\n   제품명  : {llm.get('product_name', '인식 실패')}")
    print(f"    브랜드  : {llm.get('brand', '인식 실패')}")

    # 영양성분표
    print("\n   영양성분표")
    nutrients = (nutrition_result or {}).get("nutrition") or {}
    if nutrients:
        for k, v in nutrients.items():
            print(f"    {k:15s}: {v}")
    else:
        print("    인식 실패")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="MimediQ OCR 파이프라인 데모"
    )
    parser.add_argument(
        "--product",
        default="product_sample.jpg",
        help="제품명 인식에 사용할 이미지 경로 (기본값: product_sample.jpg)",
    )
    parser.add_argument(
        "--nutrition",
        default="nutrition_sample.jpg",
        help="영양성분 인식에 사용할 이미지 경로 (기본값: nutrition_sample.jpg)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  MimediQ OCR 파이프라인 데모")
    print("=" * 60)
    print(f"  제품명 이미지   : {args.product}")
    print(f"  영양성분 이미지 : {args.nutrition}")

    # ── 파이프라인 실행 ───────────────────────────────────────
    product_result   = run_product_pipeline(args.product)
    nutrition_result = run_nutrition_pipeline(args.nutrition)

    # ── 결과 저장 ─────────────────────────────────────────────
    save_results(product_result, nutrition_result)

    # ── 최종 요약 출력 ────────────────────────────────────────
    print_final_summary(product_result, nutrition_result)


if __name__ == "__main__":
    main()