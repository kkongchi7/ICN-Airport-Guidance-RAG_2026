# ✈️ 인천공항 지능형 안내 RAG 시스템 
> **Intelligent Incheon Airport RAG Guidance System**
> LangChain LCEL 라우터, 경량화 오픈소스 LLM(Qwen2.5-3B), H3 공간 인덱싱 기반의 로컬 RAG 파이프라인

본 프로젝트는 인천국제공항의 실시간 항공편 스케줄, 공항 시설물 POI, 그리고 지역별 공항버스 노선 데이터를 통합하여 사용자의 자연어 질의에 맞춤형 정보를 제공하는 **지능형 대화형 안내 시스템**입니다. 외부 API 의존도를 낮추고 데이터 보안성을 확보하기 위해 **오픈소스 LLM의 로컬 양자화 배포** 및 **Uber H3 공간 인덱싱**을 결합한 하이브리드 RAG 프레임워크를 구현하였습니다.

---

## 🚀 주요 특징 (Key Features)

- **로컬오픈소스 LLM 4비트 양자화**: `Qwen/Qwen2.5-3B-Instruct` 모델을 `BitsAndBytes` 4-bit 양자화(`NF4`, `torch.float16`) 상태로 VRAM에 로드하여, 저사양 로컬 인프라에서도 고성능 추론이 가능한 온프레미스 환경을 구축했습니다.
- **E5 다국어 임베딩 & 문서 접두어 최적화**: 의미론적 검색 성능을 극대화하기 위해 `intfloat/multilingual-e5-base` 모델을 사용하며, 모델의 특성에 맞춰 문서 저장 시 `passage: `, 질의 시 `query: ` 접두어를 자동 부가하는 전처리 파이프라인을 적용했습니다.
- **H3 기반 공간 구역 그룹화 (Resolution 10)**: 공항 시설물의 위경도 좌표를 Uber H3 해시(해상도 10)로 변환하고, `(Building, Floor, Cell)` 구조로 공간 인덱싱 파일(`spatial_index.json`)을 생성하여 428개의 핵심 공간 구역으로 데이터를 압축 관리합니다.
- **강인한 데이터 수집 정책 (Hourly Fallback)**: 동적 API 크롤링 시 세션 및 레이아웃 토큰을 실시간 추적하며, 하루 단일 요청 실패 시 24시간 분할 요청으로 자동 전환되는 안정적인 항공편 수집 로직을 반영했습니다.

---

## 🛠 기술 스택 (Tech Stack)

### Framework & LLM
- **LLM**: Qwen/Qwen2.5-3B-Instruct (로컬 HuggingFacePipeline 연동)
- **Quantization**: BitsAndBytes (4-bit NF4 Quantization)
- **Orchestration**: LangChain Core (LCEL - LangChain Expression Language)

### Data & Embedding
- **Vector DB**: ChromaDB (PersistentClient 기반 로컬 디스크 저장)
- **Embedding Model**: `intfloat/multilingual-e5-base` (SentenceTransformer)
- **Spatial Indexing**: Uber H3 (v3.7.7)
- **Data Parsing**: BeautifulSoup4 (lxml), Pandas, JSON/CSV

---

## 🏗 시스템 아키텍처 (Architecture)

시스템은 사용자가 자연어 질의를 입력하면 5가지 의도로 분류한 뒤, LangChain `RunnableBranch`를 통해 도메인별 검색 체인으로 라우팅합니다.

[ 사용자 질의 입력 ]
│
▼
┌──────────────────────────────────────────────┐
│  1. Intent Classification (Qwen2.5-3B)       │ ──► 오직 단일 대문자 카테고리만 반환
└──────────────────────────────────────────────┘
│
├───────────────┬───────────────┬───────────────┐
▼               ▼               ▼               ▼
┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────┐
│  [FACILITY]  ││   [NEARBY]   ││   [FLIGHT]   ││    [BUS]     │
│ 단순 시설위치││ H3 공간 인덱스││ 항공편 스케줄││ 공항버스 노선│
│  ChromaDB    ││  지하/지상 층 ││   정밀 매칭  ││  경유지/시간 │
└──────────────┘└──────────────┘└──────────────┘└──────────────┘
│               │               │               │
└───────────────┼───────────────┴───────────────┘
▼
┌──────────────────────────────────────────────┐
│       2. Domain-Specific ChromaDB Query      │ ──► 벡터 및 메타데이터 하이브리드 검색
└──────────────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────┐
│  3. Final Response Generation (Local LLM)   │ ──► 공항 안내원 톤앤매너로 답변 출력
└──────────────────────────────────────────────┘


### 🎯 의도 분류 카테고리 (5종)
1. **FACILITY**: 화장실, 식당, 라운지 등 단순 공항 시설 위치 탐색
2. **NEARBY**: 특정 장소 '근처/주변/옆/가까운'에 있는 시설을 찾는 공간 맥락 탐색 (H3 인덱스 연동)
3. **FLIGHT**: 항공편, 비행기 시간표, 게이트, 카운터 정보 등
4. **BUS**: 공항 버스 노선, 대중교통 탑승 위치 및 시간표 관련
5. **NONE**: 그 외 공항과 무관한 일반 질문

---

## 📂 데이터베이스 및 컬렉션 구조

프로젝트 실행 시 로컬 디스크에 다음과 같은 구조로 벡터 DB와 공간 인덱스가 빌드됩니다.

| 데이터 도메인 | 사용 원천 파일 | ChromaDB 컬렉션명 | 저장 경로 |
| :--- | :--- | :--- | :--- |
| **공항 시설물** | `spoi_formatted_with_category.json` | `facilities` | `/content/chroma_facilities` |
| **항공편 스케줄** | `incheon_departures.csv` / `incheon_arrivals.csv` | `flights` | `/content/chroma_flights` |
| **공항 버스** | `airport_bus_routes.csv` | `bus_routes` | `./chroma_bus_db` |
| **공간 인덱스** | 위경도 기반 H3 해시 매핑 데이터 | - (JSON 구조체) | `/content/spatial_index.json` |

---

## ⚙️ 시작하기 (Quick Start)

### 1. 필수 패키지 설치
로컬 환경에 오픈소스 LLM 가동 및 양자화를 위한 필수 라이브러리를 설치합니다.
```bash
pip install chromadb sentence-transformers transformers openai tqdm pandas beautifulsoup4 lxml
pip install h3==3.7.7 accelerate bitsandbytes langchain-huggingface

# 파이프라인 인프라 가동 예시
from langchain_core.runnables import RunnableLambda

# 디버그 모드가 포함된 final_chain 호출
result = final_chain.invoke({"query": "내일 출발하는 뉴욕 가는 비행기 찾아줘"})
print(result)
