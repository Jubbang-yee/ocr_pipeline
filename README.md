##  ocr_pipeline
약품패키지 ocr 파이프라인

의약품·영양제 패키지 이미지에서 **제품명**과 **영양성분표**를 자동 추출하는 멀티 OCR 파이프라인입니다.

> 전체 프로젝트 MimediQ의 이미지 인식 파트를 담당하며, 독립 모듈로 구성되어 있습니다.

---

##  주요 기능

- **제품명 인식** (`ocr_pipeline_product.py`)
  - YOLO 기반 텍스트 영역 탐지
  - EasyOCR + Tesseract 병렬 실행 및 결과 통합
  - Gemini 2.5 Flash LLM을 통한 오탈자 교정 및 브랜드명 정제

- **영양성분표 인식** (`ocr_pipeline_nutrition.py`)
  - YOLO 기반 영양성분표 영역 탐지
  - PaddleOCR 기반 텍스트 추출
  - 행 단위 파싱으로 영양소명·수치 구조화

---

##  파이프라인 구조

```
이미지 입력
    │
    ▼
[STEP 1] 기하 보정 (원근 보정 + 기울기 보정)
    │
    ▼
[STEP 2] YOLO 객체 탐지 (파인튜닝 모델)
    │
    ▼
[STEP 3] 이미지 진단 (블러·반사·조명 분석)
    │
    ▼
[STEP 4] 적응형 전처리 (진단 결과 기반 자동 선택)
    │
    ▼
[STEP 5] 해상도 보장 (MSER 기반 글자 크기 자동 조정)
    │
    ▼
[STEP 6] OCR 실행 (EasyOCR + Tesseract 병렬 / PaddleOCR)
    │
    ▼
[STEP 7] 결과 통합 (중복 제거 + 행 단위 병합)
    │
    ▼
[STEP 8] LLM 정제 (Gemini 2.5 Flash) ← 제품명 파이프라인만
    │
    ▼
최종 출력 (제품명 / 영양성분 딕셔너리)
```

---

##  기술 스택

| 분야 | 기술 |
|------|------|
| 객체 탐지 | YOLOv8 (파인튜닝) |
| OCR | EasyOCR, Tesseract, PaddleOCR |
| 이미지 처리 | OpenCV, MSER |
| LLM 정제 | Google Gemini 2.5 Flash |
| 딥러닝 | PyTorch, Transformers |

---

##  실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. Tesseract 설치

[Tesseract 다운로드](https://github.com/UB-Mannheim/tesseract/wiki)에서 설치 후 환경변수 설정

### 3. 환경변수 설정

```bash
cp .env.example .env
```

`.env` 파일에 Gemini API 키 입력:
```
GEMINI_API_KEY=your_api_key_here
```

### 4. 모델 파일 다운로드

| 파일 | 용도 | 다운로드 |
|------|------|----------|
| `best.pt` | 제품명 영역 탐지 | [Google Drive](#) |
| `nutrition_best.pt` | 영양성분표 탐지 | [Google Drive](#) |

다운로드 후 `src/` 폴더에 위치

### 5. 실행

```python
import cv2
from ocr_pipeline_product import process_image

image = cv2.imread("your_image.jpg")
result = process_image(image)

print(result["llm_result"])
# {'product_name': '비타민C 1000', 'brand': '유한양행', 'cleaned_text': '...'}
```

---

##  프로젝트 구조

```
ocr_pipeline/
├── src/
│   ├── ocr_pipeline_product.py      # 제품명 인식 파이프라인
│   └── ocr_pipeline_nutrition.py    # 영양성분표 인식 파이프라인
├── demo/                            # 시연 GIF (추가 예정)
├── results/                         # 출력 결과 샘플 (추가 예정)
├── .env.example                     # 환경변수 설정 예시
├── requirements.txt
└── README.md
```
