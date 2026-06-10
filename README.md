# ICN-Airport-Guidance-RAG_2026
Updated version of original ICN RAG project

# ✈️ 인천공항 지능형 안내 RAG 시스템 
> **Intelligent Incheon Airport RAG Guidance System**
> LangChain LCEL 라우터와 H3 공간 인덱싱을 결합한 도메인 특화 RAG 파이프라인

본 프로젝트는 인천국제공항의 방대한 항공편, 시설물, 대중교통 데이터를 통합하여 사용자의 자연어 질의에 맞춤형 정보를 제공하는 **LLM 기반의 지능형 대화형 안내 시스템**입니다. 기존 규칙 기반 챗봇의 한계를 극복하고, 공항이라는 다중 복합 공간의 특성을 반영하기 위해 **공간 인덱싱(Geo-Indexing)** 기술을 융합한 하이브리드 RAG 프레임워크를 제안합니다.

---

## 🚀 주요 특징 (Key Features)

- **LLM 기반 지능형 라우팅**: 사용자의 복합적인 자연어 질의를 분석하여 해당 도메인(항공편, 시설물, 버스 등)으로 정확히 분기합니다.
- **공간 맥락 인식 (Spatial-Aware RAG)**: 우버(Uber)의 H3 육각형 격자 시스템을 활용하여, 특정 탑승구나 위치 기준 "근처 화장실/카페"를 정확하게 탐색합니다.
- **하이브리드 데이터 아키텍처**: 의미론적 검색을 위한 벡터 데이터베이스와 대용량 정적·정형 데이터 관리를 위한 관계형 데이터베이스를 효율적으로 병행 운용합니다.
- **강인한 데이터 스크래핑**: 정기적인 토큰 갱신 및 hourly fallback 전략을 통해 공항 포털의 동적 항공편 데이터를 완전하게 수집 및 전처리하였습니다.

## 🛠 기술 스택 (Tech Stack)

### Framework & LLM
- **Framework**: LangChain (LCEL - LangChain Expression Language)
- **Orchestration**: LangGraph (확장 아키텍처 적용)
- **LLM**: OpenAI GPT-4o-mini, Google Gemini 1.5 / 2.5 Flash

### Data & Embedding
- **Vector DB**: ChromaDB
- **Relational DB**: SQLite (모바일 호환 및 정적 데이터 관리)
- **Embedding Model**: `intfloat/multilingual-e5-base` (다국어 의미 검색 최적화)
- **Spatial Indexing**: Uber H3 (Hexagonal Hierarchical Spatial Index)

---

## 🏗 시스템 아키텍처 (Architecture)

시스템은 사용자의 자연어 입력 시 크게 **3단계의 파이프라인**을 거쳐 최종 응답을 생성합니다.
