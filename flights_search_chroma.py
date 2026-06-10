# -*- coding: utf-8 -*-
"""
flights_search_chroma.py
ChromaDB 기반 항공편 검색 (FAISS 알고리즘 로직 유지 + where 필터 + CITY/AIRLINE 매핑 완전 통합)
"""

import re
import chromadb
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from sentence_transformers import SentenceTransformer
from city_airline_map import CITY_MAP, AIRLINE_MAP


# ===============================
# ⚙️ 설정
# ===============================
CHROMA_PATH = "/content/chroma_flights"
COLLECTION_NAME = "flights"
MODEL_NAME = "intfloat/multilingual-e5-base"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)
model = SentenceTransformer(MODEL_NAME)


# ===============================
# 🧭 유틸 함수
# ===============================

KST = timezone(timedelta(hours=9))

def is_count_query(query: str) -> bool:
    """'몇 개', '몇 편', '몇 대', '몇 건' 등의 질의인지 판별"""
    return bool(re.search(r"(몇\s*(편|개|대|건))", query))

def flights_search(query: str, collection):
    # --- 1️⃣ 질의 파싱 (도시, 날짜, 방향 등) ---
    parsed = parse_query(query)  # 예: {'city': '도쿄', 'direction': '출발', 'date': '20251014'}
    is_count = is_count_query(query)

    where = {"$and": [
        {"날짜": {"$eq": parsed["date"]}},
        {"arr_or_dep": {"$eq": parsed["direction"]}},
    ]}

    if parsed.get("city"):
        where["$and"].append({"arr_city": {"$in": CITY_MAP.get(parsed["city"], [parsed["city"]])}})

    # --- 2️⃣ Chroma 검색 수행 ---
    results = collection.get(where=where, include=["metadatas", "documents", "ids"])
    filtered_count = len(results["ids"]) if results and "ids" in results else 0

    # --- 3️⃣ Count 질의 응답 처리 ---
    if is_count:
        city_txt = parsed.get("city") or ""
        direction_txt = parsed.get("direction") or ""
        date_txt = parsed.get("date") or datetime.now(KST).strftime("%Y%m%d")

        answer = f"{date_txt} 기준 인천국제공항에서 {city_txt}로 {direction_txt}하는 항공편은 총 {filtered_count}편입니다!"
        return {"mode": "flight", "count": filtered_count, "answer": answer}

    # --- 4️⃣ 일반 검색 질의일 경우 기존 로직으로 ---
    # (상위 k개 문서만 출력 등)
    return {"mode": "flight", "results": results}

def normalize_gate_field(meta: dict) -> dict:
    """'게이트' 필드가 있다면 '탑승구'로 통일"""
    if "게이트" in meta and "탑승구" not in meta:
        meta["탑승구"] = meta["게이트"]
    return meta


def _normalize_gate(gate_val):
    """게이트/탑승구 값 포맷 정규화 (float → int or string)"""
    if gate_val is None:
        return "-"
    try:
        # float(111.0) → "111"
        if isinstance(gate_val, float) and gate_val.is_integer():
            return str(int(gate_val))
        # "111.0" 같은 문자열 처리
        if isinstance(gate_val, str):
            if re.fullmatch(r"\d+\.0+", gate_val):
                return gate_val.split(".")[0]
            return gate_val
        return str(gate_val)
    except Exception:
        return str(gate_val)


def parse_relative_date_yyyymmdd(query: str) -> str | None:
    """'오늘/내일/모레/글피/어제' 등 상대 날짜를 YYYYMMDD로 변환"""
    q = query.strip().lower()
    KST = timezone(timedelta(hours=9))
    today = datetime.now(KST).date()

    offset = 0
    if any(k in q for k in ["오늘", "금일", "today"]):
        offset = 0
    elif any(k in q for k in ["내일", "tomorrow"]):
        offset = 1
    elif "모레" in q:
        offset = 2
    elif "글피" in q:
        offset = 3
    elif any(k in q for k in ["어제", "yesterday"]):
        offset = -1

    if offset != 0 or "오늘" in q:
        d = today + timedelta(days=offset)
        return d.strftime("%Y%m%d")

    # 절대 날짜 인식 (예: 10월 13일, 2025-10-13 등)
    m = re.search(r"(\d{4})[-./]?\s?(\d{1,2})[-./]?\s?(\d{1,2})", q)
    if m:
        y, mo, d = map(int, m.groups())
        return datetime(y, mo, d).strftime("%Y%m%d")

    m2 = re.search(r"(\d{1,2})월\s*(\d{1,2})일", q)
    if m2:
        mo, d = map(int, m2.groups())
        y = today.year
        return datetime(y, mo, d).strftime("%Y%m%d")

    return None


def infer_direction(query: str) -> str | None:
    """출발/도착 방향 추론"""
    q = query.lower()
    if any(k in q for k in ["도착", "입국", "돌아오", "오는"]):
        return "도착"
    if any(k in q for k in ["출발", "떠나", "가는", "출국"]):
        return "출발"
    return None


def extract_city_aliases(text: str) -> tuple[str | None, list[str]]:
    """CITY_MAP에서 일치 도시 및 모든 alias 반환"""
    q = text.lower()
    for city, aliases in CITY_MAP.items():
        all_aliases = [city] + aliases
        for alias in all_aliases:
            if alias.lower() in q:
                return city, all_aliases
    return None, []


def extract_airline_aliases(text: str) -> tuple[str | None, list[str]]:
    """AIRLINE_MAP에서 일치 항공사 및 모든 alias 반환"""
    q = text.lower()
    for airline, aliases in AIRLINE_MAP.items():
        all_aliases = [airline] + aliases
        for alias in all_aliases:
            if alias.lower() in q:
                return airline, all_aliases
    return None, []


def soft_match(meta: dict, key_candidates: list[str], needles: list[str]) -> bool:
    """메타데이터의 여러 필드 중 일부라도 needles를 포함하면 True"""
    text = []
    for k in key_candidates:
        v = meta.get(k)
        if v:
            text.append(str(v).lower())
    blob = " ".join(text)
    return all(n.lower() in blob for n in needles)


def rerank_with_heuristics(results, destination_terms=None, airline_terms=None):
    """도시/항공사 일치 시 가중치 부여"""
    destination_terms = destination_terms or []
    airline_terms = airline_terms or []
    out = []
    for r in results:
        score = r["score"]
        meta = r["meta"]
        if destination_terms and soft_match(meta, ["목적지", "destination", "출발지"], destination_terms):
            score += 0.05
        if airline_terms and soft_match(meta, ["항공사", "airline"], airline_terms):
            score += 0.03
        r2 = dict(r)
        r2["score"] = round(score, 4)
        out.append(r2)
    return sorted(out, key=lambda x: x["score"], reverse=True)

def extract_airline(query: str) -> str | None:
    """
    질의문에서 항공사명이나 항공사 코드를 추출
    ex) '대한항공 체크인 카운터' → '대한항공'
        'KE123편 게이트' → '대한항공'
        '아시아나 OZ123' → '아시아나'
    """
    q = query.strip().upper()

    # 1️⃣ 항공사명으로 탐색
    for airline, aliases in AIRLINE_MAP.items():
        for alias in aliases:
            if alias.upper() in q:
                return airline

    # 2️⃣ 편명 코드 (예: KE, OZ, 7C 등)
    m = re.search(r"\b([A-Z]{2})\d{1,4}\b", q)
    if m:
        prefix = m.group(1)
        for airline, aliases in AIRLINE_MAP.items():
            if prefix in aliases:
                return airline

    return None

# ===============================
# 🔍 Chroma 검색 함수
# ===============================

def search_flights_chroma(
    query: str,
    k: int = 10,
    min_score: float = 0.50,
    direction: str | None = None,
):
    """
    Chroma 기반 항공편 검색
    - 출발/도착 방향, 도시/항공사 alias 매칭, 날짜 자동 필터링(기본=오늘)
    - FAISS 시절의 논리 구조를 유지하면서 Chroma where 필터 적용
    """
    # 1️⃣ 질의 분석
    direction = direction or infer_direction(query)
    date_yyyymmdd = parse_relative_date_yyyymmdd(query)

    # ✅ 날짜 명시 안 되어 있으면 '오늘'로 자동 설정
    if not date_yyyymmdd:
        date_yyyymmdd = datetime.now().strftime("%Y%m%d")

    query_city, city_aliases = extract_city_aliases(query)
    query_airline, airline_aliases = extract_airline_aliases(query)

    print(f"\n[DEBUG] date={date_yyyymmdd}, direction={direction}, city={query_city}, airline={query_airline}")

    # 2️⃣ where 조건 구성
    filters = []
    if date_yyyymmdd:
        filters.append({"날짜": {"$eq": int(date_yyyymmdd)}})
    if direction:
        filters.append({"arr_or_dep": {"$eq": direction}})

    if len(filters) == 1:
        where_clause = filters[0]
    elif len(filters) > 1:
        where_clause = {"$and": filters}
    else:
        where_clause = None

    print(f"[DEBUG] where={where_clause}")

    # 3️⃣ 쿼리 임베딩 생성 및 검색
    q_emb = model.encode([query], normalize_embeddings=True)
    res = collection.query(
        query_embeddings=q_emb,
        n_results=k * 15,   # 넉넉하게 검색 후 후처리
        where=where_clause
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    # 4️⃣ 초기 후보 필터링
    prelim = []
    for doc, meta, dist in zip(docs, metas, dists):
        score = 1 - dist
        if score >= min_score:
            meta = normalize_gate_field(meta)  # 게이트 필드 통합
            prelim.append({"score": round(score, 4), "text": doc, "meta": meta})

    # 5️⃣ alias 기반 세부 필터링
    filtered = []
    for r in prelim:
        meta = r["meta"]
        origin_str = str(meta.get("출발지") or "").lower()
        dest_str = str(meta.get("목적지") or "").lower()
        airline_str = str(meta.get("항공사") or "").lower()

        city_ok, airline_ok = True, True

        if query_city:
            aliases = [a.lower() for a in city_aliases]
            if direction == "도착":
                # 도착편 → 출발지 기준
                city_ok = any(a in origin_str for a in aliases)
            else:
                # 출발편 → 목적지 기준
                city_ok = any(a in dest_str for a in aliases)

        if query_airline:
            aliases = [a.lower() for a in airline_aliases]
            airline_ok = any(a in airline_str for a in aliases)

        if city_ok and airline_ok:
            filtered.append(r)

    # 6️⃣ 동일 항공편 중복 제거 (운항편명 기준)
    unique = []
    seen = set()
    for r in filtered:
        fn = r["meta"].get("운항편명")
        if fn not in seen:
            seen.add(fn)
            unique.append(r)

    # 7️⃣ 재랭킹 (도시/항공사 일치 가중치)
    reranked = rerank_with_heuristics(
        unique,
        destination_terms=[query_city] if query_city else None,
        airline_terms=[query_airline] if query_airline else None,
    )

    print(f"✅ {len(reranked)} results after filtering")
    return reranked



# ===============================
# 💬 프롬프트 빌더
# ===============================

def build_flight_prompt(query: str, retrieved: list[dict]) -> str:
    """LLM용 프롬프트 생성 (count 질의는 전체 개수 전달)"""
    is_count = is_count_query(query)
    total_count = len(retrieved)
    cards = []

    # count 질의가 아니면 상위 5개만 보여줌
    shown_results = retrieved if is_count else retrieved[:5]

    for r in shown_results:
        m = r["meta"]
        airline = m.get("항공사", "-")
        flight_no = m.get("운항편명", "-")
        arr_or_dep = m.get("arr_or_dep", "-")
        terminal = m.get("터미널", "-")
        status = m.get("출발현황") or m.get("도착현황") or "정보 없음"
        dep_time = m.get("출발시간")
        arr_time = m.get("도착시간")
        gate = m.get("탑승구") or m.get("게이트") or "-"
        counter = m.get("체크인 카운터") or "-"

        # 경로 문자열
        if arr_or_dep == "출발":
            route = f"인천 → {m.get('목적지') or '-'}"
        elif arr_or_dep == "도착":
            route = f"{m.get('출발지') or '-'} → 인천"
        else:
            route = m.get("목적지") or m.get("출발지") or "-"

        # 시간 표기
        if arr_or_dep == "출발":
            time_info = f"출발 시간: {dep_time or '-'}"
        elif arr_or_dep == "도착":
            time_info = f"도착 시간: {arr_time or '-'}"
        else:
            time_info = f"{dep_time or '-'} → {arr_time or '-'}"

        cards.append(
            f"- [{airline}] {flight_no}편 ({arr_or_dep}) | {route} | {time_info} | "
            f"터미널 {terminal} | 게이트 {gate} | 카운터 {counter} | 상태: {status}"
        )

    ctx = "\n".join(cards)
    prompt = f"""
당신은 인천국제공항 안내 챗봇입니다.
아래의 항공편 데이터를 참고하여 사용자 질문에 정확하고 친절하게 답하세요.

[사용자 질문]
{query}

[검색된 항공편 목록]
{ctx}

위 데이터를 근거로 답변을 작성하세요.
- 여러 편이 있으면 시간 순서대로 요약하세요.
- 항공편명, 출발/도착 시간, 터미널, 상태를 포함하세요.
- 검색된 모든 항공편 정보를 빠짐없이 각각 설명하세요.
- 단, 개수를 물어보는 질의에는 검색된 문서의 개수만 알려주고 상세하게 답하지 마세요.
- 질의에 충실하게 답해주세요.
- 활기차고 친절한 태도로 응대해주세요.
- 데이터 바깥 추측은 금지하며 확인되지 않은 정보는 언급하지 마세요.
    """.strip()

    return prompt

# =============================================================
# 🚀 [추가된 메인 인터페이스] 라우터 연결 및 오픈소스 LLM 주입
# =============================================================
def handle_flight_query(query: str, llm_client) -> str:
    """
    라우터에서 호출하는 항공편 도메인 메인 함수.
    항공편을 검색하고 주입된 오픈소스 LLM을 통해 최종 답변을 생성합니다.
    """
    # 1. 항공편 검색 (Chroma + 메타데이터 필터링 + 정규화)
    hits = search_flights_chroma(query)

    # 2. 결과가 없을 경우 안전한 Fallback 반환
    if not hits:
        # count 질의인데 0개인 경우를 위한 처리
        if is_count_query(query):
            return f"해당 조건으로 검색된 항공편은 총 0편입니다."
        return f"'{query}'에 해당하는 항공편 정보를 찾을 수 없습니다."

    # 3. LLM 프롬프트 생성
    prompt = build_flight_prompt(query, hits)

    # 4. 주입받은 LangChain 오픈소스 LLM(Qwen) 호출 (.invoke 사용)
    response = llm_client.invoke(prompt)

    # 5. 객체 또는 문자열 결과 안전하게 추출
    if hasattr(response, 'content'):
        return response.content.strip()
    return str(response).strip()