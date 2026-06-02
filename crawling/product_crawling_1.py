import os
import time
import shutil
from icrawler.builtin import BingImageCrawler
from pathlib import Path

# ─────────────────────────────────
# 설정
# ─────────────────────────────────
SAVE_DIR      = "dataset/raw_images"
MAX_PER_QUERY = 30

os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────
# 검색 쿼리 목록
# ─────────────────────────────────
QUERIES = [
    # ── 한국 영양제 브랜드 + 제품명 ──
    "종근당건강 오메가3 박스",
    "유한양행 비타민C 영양제",
    "GNM자연의품격 영양제",
    "닥터린 비타민D 박스",
    "뉴트리원 영양제 패키지",
    "한미약품 영양제 박스",
    "일동제약 비타민 박스",
    "고려은단 비타민C",
    "건강기능식품 박스 앞면",
    "홍삼정 원데이타임 박스",

    # ── 영양제 종류별 (한국어) ──
    "비타민C 영양제 박스 앞면",
    "비타민D 영양제 패키지",
    "오메가3 영양제 박스",
    "유산균 프로바이오틱스 박스",
    "마그네슘 영양제 포장",
    "루테인 영양제 박스",
    "콜라겐 영양제 패키지",
    "철분 영양제 박스",
    "아연 영양제 포장",
    "엽산 영양제 박스",
    "코엔자임Q10 영양제",
    "밀크씨슬 간 영양제 박스",
    "글루코사민 관절 영양제",
    "프로틴 단백질 보충제 박스",
    "멀티비타민 영양제 박스",

    # ── 영양제 종류별 (영어) ──
    "vitamin C supplement box front",
    "vitamin D supplement bottle label",
    "omega3 fish oil supplement packaging",
    "probiotics supplement box front",
    "magnesium supplement bottle",
    "lutein supplement packaging",
    "collagen supplement box",
    "iron supplement packaging front",
    "zinc supplement bottle label",
    "folic acid supplement box",
    "CoQ10 supplement packaging",
    "milk thistle supplement box",
    "glucosamine supplement packaging",
    "whey protein powder packaging front",
    "multivitamin supplement bottle label",
    "biotin supplement box front",
    "calcium supplement packaging",

    # ── 제품 형태별 ──
    "영양제 캡슐 병 앞면",
    "영양제 정제 박스 앞면",
    "영양제 파우치 포장",
    "supplement capsule bottle front label",
    "supplement tablet bottle label",
    "vitamin gummy packaging front",
    "supplement powder jar label",

    # ── 패키징 특성 ──
    "supplement packaging product name large text",
    "health supplement box close up product name",
    "vitamin bottle front label product name",
    "supplement packaging brand name visible",
    "health product box front clear label",

    # ── 화장품/의약품 (다양성 확보) ──
    "화장품 박스 제품명 앞면",
    "스킨케어 패키지 앞면",
    "의약품 박스 포장 앞면",
    "일반의약품 박스 제품명",
    "cosmetic product box front label",
    "skincare packaging product name visible",
    "medicine box packaging front label",
    "pharmaceutical box product name",
    "drug packaging front label clear",

    # ── 해외 영양제 브랜드 ──
    "NOW Foods supplement packaging",
    "Nature Made vitamin bottle label",
    "Solgar supplement box front",
    "GNC supplement packaging front",
    "Centrum multivitamin box front",
    "Kirkland supplement bottle label",
    "Garden of Life supplement packaging",
    "Swisse supplement box front",
    "Blackmores supplement packaging",
    "Jamieson vitamin box front",
]

# ─────────────────────────────────
# 크롤링 함수
# ─────────────────────────────────
def crawl_google_images(
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

            # 임시 폴더 이미지를 메인 폴더로 이동 + 파일명 통일
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

        # 구글 차단 방지 딜레이
        time.sleep(2)

    return total

# ─────────────────────────────────
# 품질 필터링
# ─────────────────────────────────
def filter_images(save_dir, min_width=300, min_height=300):
    from PIL import Image

    images = (
        list(Path(save_dir).glob("*.jpg")) +
        list(Path(save_dir).glob("*.jpeg")) +
        list(Path(save_dir).glob("*.png")) +
        list(Path(save_dir).glob("*.webp"))
    )

    print(f"\n품질 필터링 시작 ({len(images)}장)")
    removed = 0

    for img_path in images:
        try:
            img = Image.open(img_path)
            w, h = img.size

            # 너무 작은 이미지 제거
            if w < min_width or h < min_height:
                os.remove(img_path)
                removed += 1
                continue

            # 극단적인 비율 제거 (파노라마 등)
            ratio = w / h
            if ratio > 5 or ratio < 0.2:
                os.remove(img_path)
                removed += 1

        except Exception:
            # 깨진 이미지 제거
            os.remove(img_path)
            removed += 1

    remaining = len([
        f for f in Path(save_dir).iterdir()
        if f.is_file()
    ])
    print(f"  제거: {removed}장 / 남은 이미지: {remaining}장")
    return remaining

# ─────────────────────────────────
# 실행
# ─────────────────────────────────
if __name__ == "__main__":
    print("=" * 40)
    print("구글 이미지 크롤링 시작")
    print(f"저장 위치:    {SAVE_DIR}")
    print(f"쿼리 수:      {len(QUERIES)}개")
    print(f"쿼리당 최대:  {MAX_PER_QUERY}장")
    print(f"예상 수집량:  {len(QUERIES) * MAX_PER_QUERY}장")
    print("=" * 40)

    total = crawl_google_images(
        queries=QUERIES,
        save_dir=SAVE_DIR,
        max_num=MAX_PER_QUERY,
        min_size=(300, 300),
    )

    remaining = filter_images(SAVE_DIR)

    print("\n" + "=" * 40)
    print("크롤링 완료")
    print(f"최종 이미지 수: {remaining}장")
    print(f"저장 위치:      {SAVE_DIR}")
    print("=" * 40)
    print("\n다음 단계: Roboflow 어노테이션")
    print("https://roboflow.com")