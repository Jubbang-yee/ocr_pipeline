import os
import time
import shutil
from icrawler.builtin import BingImageCrawler
from pathlib import Path

# ─────────────────────────────────
# 설정
# ─────────────────────────────────
SAVE_DIR      = "dataset/raw_images_2"
MAX_PER_QUERY = 15

os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────
# 검색 쿼리 목록 (한국 제품 위주)
# ─────────────────────────────────
QUERIES = [
    # ── 한국 영양제 브랜드 ──
    "종근당건강 오메가3 박스",
    "종근당건강 비타민D 패키지",
    "종근당건강 루테인 박스",
    "종근당건강 프로바이오틱스",
    "고려은단 비타민C 박스",
    "고려은단 멀티비타민 패키지",
    "고려은단 마그네슘 박스",
    "유한양행 비타민C 영양제",
    "유한양행 칼슘 마그네슘 박스",
    "일동제약 비타민 박스",
    "일동제약 지큐랩 영양제",
    "한미약품 영양제 박스",
    "대웅제약 비타민 박스",
    "동국제약 영양제 패키지",
    "광동제약 비타민 박스",
    "뉴트리원 영양제 패키지",
    "뉴트리원 콜라겐 박스",
    "닥터린 비타민D 박스",
    "닥터린 오메가3 패키지",
    "GNM자연의품격 영양제",
    "GNM자연의품격 비타민C 박스",
    "에스더포뮬러 영양제 박스",
    "에스더포뮬러 콜라겐",
    "내츄럴플러스 영양제",
    "나우푸드 한국 영양제 박스",
    "셀트리온 건강기능식품",
    "보령 영양제 박스",
    "동화약품 비타민 박스",
    "제일약품 영양제 패키지",
    "CJ웰케어 영양제 박스",
    "CJ 리턴업 영양제",
    "풀무원건강생활 영양제",
    "KGC인삼공사 정관장 홍삼",
    "정관장 홍삼정 에브리타임 박스",
    "홍삼정 원데이타임 박스",
    "천호엔케어 영양제 박스",

    # ── 한국 영양제 종류별 ──
    "한국 비타민C 1000mg 박스",
    "한국 비타민D 영양제 박스",
    "한국 오메가3 영양제 박스",
    "한국 유산균 프로바이오틱스 박스",
    "한국 마그네슘 영양제 박스",
    "한국 루테인 영양제 박스",
    "한국 콜라겐 영양제 패키지",
    "한국 철분 영양제 박스",
    "한국 아연 영양제 박스",
    "한국 엽산 영양제 박스",
    "한국 코엔자임Q10 박스",
    "한국 밀크씨슬 영양제 박스",
    "한국 글루코사민 영양제",
    "한국 멀티비타민 영양제 박스",
    "한국 칼슘 영양제 박스",
    "한국 식물성 오메가3 박스",
    "한국 MSM 관절 영양제",
    "한국 나이아신 영양제 박스",
    "한국 비오틴 영양제 박스",
    "한국 건강기능식품 박스 앞면",

    # ── 한국 일반의약품 브랜드 ──
    "타이레놀 박스 한국",
    "판피린 박스",
    "판콜에이 박스",
    "게보린 박스",
    "이부프로펜 한국 박스",
    "아세트아미노펜 한국 약 박스",
    "부루펜 시럽 박스",
    "어린이 타이레놀 박스",
    "베아제 소화제 박스",
    "훼스탈 소화제 박스",
    "개비스콘 한국 박스",
    "겔포스 제산제 박스",
    "우황청심원 박스",
    "동화 활명수 박스",
    "까스활명수 박스",
    "인사돌 잇몸약 박스",
    "이가탄 잇몸약 박스",
    "센시아 잇몸약 박스",
    "클라리틴 알레르기약 박스",
    "지르텍 알레르기약 박스",
    "알레그라 한국 박스",
    "코푸시럽 감기약 박스",
    "테라플루 감기약 박스",
    "콜대원 감기약 박스",
    "화콜 감기약 박스",
    "베나드릴 한국 박스",
    "피부연고 한국 약 박스",
    "후시딘 연고 박스",
    "마데카솔 연고 박스",
    "리도멕스 연고 박스",
    "무좀약 한국 박스",
    "락토핏 유산균 박스",
    "장대원 유산균 박스",
    "듀오락 유산균 박스",

    # ── 한국 한방/홍삼 제품 ──
    "한국 홍삼 건강기능식품 박스",
    "한국 한방 영양제 박스",
    "공진단 박스 한국",
    "쌍화탕 박스 한국",
    "경옥고 박스 한국",
    "한국 녹용 건강식품 박스",

    # ── 한국 다이어트/슬리밍 ──
    "한국 다이어트 보조제 박스",
    "한국 체지방 감소 건강기능식품",
    "카르니틴 한국 영양제 박스",
    "가르시니아 한국 다이어트 박스",
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