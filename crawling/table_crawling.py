# pip install icrawler Pillow

import os
import time
import shutil
from icrawler.builtin import BingImageCrawler
from pathlib import Path

# ─────────────────────────────────
# 설정
# ─────────────────────────────────
SAVE_DIR      = "dataset/nutrition_label_images_2"
MAX_PER_QUERY = 100

os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────
# 검색 쿼리 목록 (한국 제품 위주)
# ─────────────────────────────────
QUERIES = [
    "한국 영양제 영양성분표",
    "한국 영양성분표 클로즈업"
]

# ─────────────────────────────────
# 크롤링 함수
# ─────────────────────────────────
def crawl_bing_images(
    queries,
    save_dir,
    max_num=30,
    min_size=(300, 300)
):
    total = 0

    for idx, query in enumerate(queries):
        print(f"\n[{idx+1}/{len(queries)}] 검색: '{query}'")

        temp_dir = os.path.join(save_dir, "temp_" + query.replace(" ", "_"))
        os.makedirs(temp_dir, exist_ok=True)

        try:
            crawler = BingImageCrawler(
                storage={"root_dir": temp_dir},
                feeder_threads=1,
                parser_threads=1,
                downloader_threads=4,
            )
            crawler.crawl(
                keyword=query,
                max_num=max_num,
                min_size=min_size,
                file_idx_offset=0,
            )

            images = (
                list(Path(temp_dir).glob("*.jpg")) +
                list(Path(temp_dir).glob("*.jpeg")) +
                list(Path(temp_dir).glob("*.png")) +
                list(Path(temp_dir).glob("*.webp"))
            )

            for img_path in images:
                new_name = f"img_{total:04d}{img_path.suffix}"
                new_path = os.path.join(save_dir, new_name)
                shutil.move(str(img_path), new_path)
                total += 1

            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"  → {len(images)}장 수집 (누적: {total}장)")

        except Exception as e:
            print(f"  → 오류: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            continue

        time.sleep(2)

    return total

# ─────────────────────────────────
# 실행
# ─────────────────────────────────
if __name__ == "__main__":
    print("=" * 40)
    print("Bing 이미지 크롤링 시작 (한국 제품)")
    print(f"저장 위치:    {SAVE_DIR}")
    print(f"쿼리 수:      {len(QUERIES)}개")
    print(f"쿼리당 최대:  {MAX_PER_QUERY}장")
    print(f"예상 수집량:  {len(QUERIES) * MAX_PER_QUERY}장")
    print("=" * 40)

    total = crawl_bing_images(
        queries=QUERIES,
        save_dir=SAVE_DIR,
        max_num=MAX_PER_QUERY,
        min_size=(300, 300),
    )

    print("\n" + "=" * 40)
    print("크롤링 완료")
    print(f"최종 이미지 수: {total}장")
    print(f"저장 위치:      {SAVE_DIR}")
    print("=" * 40)
    print("\n다음 단계: Roboflow 어노테이션")
    print("https://roboflow.com")