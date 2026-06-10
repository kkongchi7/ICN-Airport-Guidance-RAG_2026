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

## 🛠 시스템 아키텍처 (System Architecture)

본 시스템은 사용자의 자연어 질의를 분석하여 최적의 도메인 지식에 접근하고, 공간적 맥락을 결합하여 답변을 생성하는 3단계 파이프라인(의도 분류 ➔ 하이브리드 검색 ➔ LLM 답변 생성)으로 구성되어 있습니다.

```mermaid
graph TD

    classDef user fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef router fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef search fill:#f1f8e9,stroke:#689f38,stroke-width:2px;
    classDef llm fill:#ede7f6,stroke:#5e35b1,stroke-width:2px;
    classDef db fill:#eceff1,stroke:#455a64,stroke-width:2px;

    User([사용자 질의 입력]):::user

    subgraph Intent_Routing["1. 의도 분류 및 라우팅"]
        Router{"LangChain LCEL\n의도 분류 라우터"}:::router
    end

    subgraph Hybrid_Search_Engine["2. 도메인별 특화 검색 엔진"]
        FlightSearch["항공편 검색 모듈"]:::search
        BusSearch["대중교통 검색 모듈"]:::search
        FacilitySearch["시설물 의미 검색"]:::search
        NearbySearch["H3 기반 공간 검색"]:::search
    end

    subgraph Data_Layer["데이터 레이어"]
        FlightDB[("ChromaDB\n항공편 DB")]:::db
        BusDB[("ChromaDB\n교통 DB")]:::db
        FacilityDB[("ChromaDB\n시설 Vector DB")]:::db
        H3Index["H3 Geo-Indexing\n공간 인덱스"]:::db
    end

    subgraph Generation_Layer["3. 컨텍스트 합성 및 답변 생성"]
        Prompt["프롬프트 엔지니어링\nContext + Prompt"]:::llm
        LLM["오픈소스 LLM\nQwen 2.5"]:::llm
        Answer([최종 안내 답변 제공]):::user
    end

    User --> Router

    Router -- FLIGHT --> FlightSearch
    Router -- BUS --> BusSearch
    Router -- FACILITY --> FacilitySearch
    Router -- NEARBY --> NearbySearch
    Router -- NONE --> Prompt

    FlightSearch --> FlightDB
    BusSearch --> BusDB
    FacilitySearch --> FacilityDB
    NearbySearch --> H3Index
    H3Index --> FacilityDB

    FlightSearch --> Prompt
    BusSearch --> Prompt
    FacilitySearch --> Prompt
    NearbySearch --> Prompt

    Prompt --> LLM
    LLM --> Answer
```

### 🎯 의도 분류 카테고리 (5종)

1. **FACILITY**: 화장실, 식당, 라운지 등 단순 공항 시설 위치 탐색
2. **NEARBY**: 특정 장소 근처/주변/옆/가까운 시설 탐색 (H3 인덱스 연동)
3. **FLIGHT**: 항공편, 비행기 시간표, 게이트, 카운터 정보
4. **BUS**: 공항버스 노선, 탑승 위치 및 시간표
5. **NONE**: 공항과 무관한 일반 질문

---

## ⚙️ 시작하기 (Quick Start)

### 1. 필수 패키지 설치

```bash
pip install chromadb sentence-transformers transformers openai tqdm pandas beautifulsoup4 lxml
pip install h3==3.7.7 accelerate bitsandbytes langchain-huggingface
```

### 2. 파이프라인 실행 예시

```python
from langchain_core.runnables import RunnableLambda

# 디버그 모드가 포함된 final_chain 호출
result = final_chain.invoke(
    {"query": "내일 출발하는 뉴욕 가는 비행기 찾아줘"}
)

print(result)
```

---

## 📱 모바일 애플리케이션 확장 구상 (Application Development Plan)

본 RAG 파이프라인의 연구 성과를 확장하여 실제 공항 이용객이 현장에서 사용할 수 있는 대화형 스마트폰 애플리케이션의 세부 아키텍처와 UI/UX를 구상하고 있습니다.

### 1) 시스템 및 데이터 아키텍처 확장

#### 로컬 경량화 데이터베이스 통합 (Android Room/SQLite)

- 네트워크 연결이 불안정한 공항 내부 환경을 고려
- BUS 및 시설 데이터는 Room DB에 내장
- 오프라인 상태에서도 빠른 조회 지원

#### 하이브리드 동기화 시스템

- 실시간 항공편 정보는 서버 API 처리
- 정적 데이터는 모바일 내부 DB 처리
- 서버 비용 및 응답 지연 최소화

### 2) 핵심 기능 및 인터페이스 (UI/UX)

#### 대화형 AI 타임라인 인터페이스

- ChatGPT 및 Gemini와 유사한 대화형 UI
- 자연어 기반 질의응답

#### 위치 기반 실시간 컨텍스트 매핑

- "25번 게이트 앞인데 근처 화장실 있어?"
- H3 K-ring 기반 주변 시설 탐색
- 미니맵 및 이동 경로 시각화 제공

#### 상황 맞춤형 톤앤매너

- 비행기 시간 문의
- 시설 탐색 문의
- 상황별 맞춤 응답 생성

---

## 📸 애플리케이션 화면 예시

### 예시 화면

![메인 화면](Screenshot_54.png)
