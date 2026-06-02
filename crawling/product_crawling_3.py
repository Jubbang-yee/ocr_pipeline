import os
import time
import shutil
from icrawler.builtin import BingImageCrawler
from pathlib import Path

# ─────────────────────────────────
# 설정
# ─────────────────────────────────
SAVE_DIR      = "dataset/raw_images_3"
MAX_PER_QUERY = 20

os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────
# 검색 쿼리 목록 (한국 제품 위주)
# ─────────────────────────────────
QUERIES = [
    # ── 추가 한국 영양제 브랜드 ──
    "삼양사 큐원 영양제 박스",
    "한독 비타민 박스",
    "동아제약 박카스 박스",
    "동아제약 영양제 박스",
    "JW중외제약 영양제",
    "녹십자 비타민 박스",
    "일양약품 영양제 박스",
    "유유제약 비타민 박스",
    "안국약품 영양제 박스",
    "경동제약 영양제 박스",
    "명인제약 영양제 박스",
    "신신제약 영양제 박스",
    "비타할로 영양제 박스",
    "뉴트리데이 영양제",
    "종근당 헬씨업 영양제",
    "바이오가이아 유산균 한국",
    "컬처렐 유산균 한국 박스",

    # ── 제품 형태 다양화 ──
    "한국 영양제 유리병 앞면",
    "한국 영양제 플라스틱 통 앞면",
    "한국 영양제 스틱형 파우치",
    "한국 영양제 블리스터 포장",
    "한국 영양제 액상 앰플",
    "건강기능식품 병 라벨 한국",
    "한국 영양제 캔 포장",

    # ── 세부 성분별 추가 ──
    "한국 NAC 영양제 박스",
    "한국 GABA 영양제 박스",
    "한국 알파리포산 박스",
    "한국 스피루리나 영양제",
    "한국 클로렐라 영양제 박스",
    "한국 노니 건강기능식품",
    "한국 아르기닌 영양제 박스",
    "한국 시서스 다이어트 박스",
    "한국 포스파티딜세린 박스",
    "한국 레시틴 영양제 박스",
    "한국 쏘팔메토 영양제 박스",
    "한국 크랜베리 영양제 박스",
    "한국 감마리놀렌산 영양제",
    "한국 폴리코사놀 박스",
    "한국 테아닌 영양제 박스",
    "한국 트립토판 영양제 박스",

    # ── 대상별 영양제 (성분 중복 제거) ──
    "임산부 영양제 박스 한국",
    "어린이 멀티비타민 한국 박스",
    "시니어 영양제 박스 한국",
    "남성 영양제 박스 한국",
    "여성 영양제 박스 한국",
    "수험생 영양제 박스 한국",
    "갱년기 영양제 박스 한국",
    "피부 영양제 박스 한국",

    # ── 추가 일반의약품 ──
    "속쓰림약 한국 박스",
    "변비약 한국 박스",
    "지사제 한국 박스",
    "파스 한국 박스",
    "비염약 한국 박스",
    "인공눈물 한국 박스",
    "소염진통제 한국 박스",
    "해열제 한국 박스",
    "상처치료제 한국 박스",
    "탈모약 한국 박스",
    "여드름약 한국 박스",
    "아토피 크림 한국 박스",
    "한국 구충제 박스",
    "멀미약 한국 박스",
    "수면유도제 한국 박스",
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