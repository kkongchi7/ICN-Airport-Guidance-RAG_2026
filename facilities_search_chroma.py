# -*- coding: utf-8 -*-
"""
facilities_search_chroma.py
인천공항 시설 검색 + H3 기반 근처 탐색 (Chroma + Spatial Index 통합 버전)
"""

import re, math, json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from facility_category_map import normalize_category_query

# ===== 기본 설정 =====
CHROMA_PATH = "/content/chroma_facilities"
COLLECTION_NAME = "facilities"
MODEL_NAME = "intfloat/multilingual-e5-base"

SPATIAL_PATH = "/content/spatial_index.json"
FAC_PATH = "/content/spoi_formatted_with_category.json"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)
model = SentenceTransformer(MODEL_NAME)

# -------------------------------------------------------------
# 🔧 유틸 함수
# -------------------------------------------------------------
def _normalize(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip())

def _floor_eq(a, b):
    if not a or not b:
        return False
    aa = str(a).replace(" ", "").upper().replace("층", "").replace("F", "")
    bb = str(b).replace(" ", "").upper().replace("층", "").replace("F", "")
    return aa == bb

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# -------------------------------------------------------------
# 🎯 질의 힌트 추출
# -------------------------------------------------------------
def parse_query_hints(q: str):
    ql = q.lower().replace(" ", "")
    term, floor, cat = None, None, None

    if re.search(r"(제?2터미널|2터|t2|terminal2)", ql): term = "T2"
    elif re.search(r"(제?1터미널|1터|t1|terminal1)", ql): term = "T1"
    elif "concourse" in ql or "탑승동" in ql: term = "Concourse"

    if m := re.search(r"(\d)\s*층", q): floor = f"{m.group(1)}층"
    elif re.search(r"b1|지하\s*1", q): floor = "지하1층"
    elif re.search(r"b2|지하\s*2", q): floor = "지하2층"

    if any(k in ql for k in ["카페","coffee","스타벅스","투썸","할리스","공차","엔제리너스","폴바셋"]): cat = "카페/음료"
    elif any(k in ql for k in ["흡연","smoking"]): cat = "흡연실"
    elif any(k in ql for k in ["화장실","toilet","restroom"]): cat = "화장실"
    elif any(k in ql for k in ["편의점","cu","gs25","세븐일레븐","이마트24"]): cat = "편의점"
    elif any(k in ql for k in ["라운지","lounge","kal","스카이허브","아시아나"]): cat = "라운지"

    return {"terminal": term, "floor": floor, "category": cat}

# -------------------------------------------------------------
# 🧠 리랭킹 함수
# -------------------------------------------------------------
def _facility_bonus(meta, query: str) -> float:
    hints = parse_query_hints(query)
    score = 0.0
    if hints["terminal"] and meta.get("building_alias") == hints["terminal"]:
        score += 0.3
    if hints["floor"] and _floor_eq(meta.get("floor"), hints["floor"]):
        score += 0.2
    if hints["category"] and hints["category"] in (meta.get("category") or ""):
        score += 0.3
    return score

# -------------------------------------------------------------
# 🔍 메인 검색 (+카테고리 자동 감지 + 디버그)
# -------------------------------------------------------------
def search_facilities_chroma(query: str, k: int = 10, min_score: float = 0.6, debug: bool = True):
    query_norm = _normalize(query)
    q_emb = model.encode([query_norm], normalize_embeddings=True)
    hints = parse_query_hints(query_norm)

    detected_cat = normalize_category_query(query_norm)
    if detected_cat and not hints.get("category"):
        hints["category"] = detected_cat
        if debug:
            print(f"[DEBUG] Auto-detected category from map: {detected_cat}")

    where_clauses = []
    if hints["terminal"]:
        where_clauses.append({"building_alias": {"$eq": hints["terminal"]}})
    if hints["floor"]:
        where_clauses.append({"floor": {"$eq": hints["floor"]}})
    if hints["category"]:
        where_clauses.append({"category": {"$eq": hints["category"]}})
    
    if not where_clauses:
        where = None
    elif len(where_clauses) == 1:
        where = where_clauses[0]
    else:
        where = {"$and": where_clauses}

    if debug:
        print(f"[DEBUG] term={hints['terminal']}, floor={hints['floor']}, category={hints['category']}")
        print(f"[DEBUG] where={where}")

    res = collection.query(query_embeddings=q_emb, n_results=k * 3, where=where)
    docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]

    hits = []
    for doc, meta, dist in zip(docs, metas, dists):
        base = 1 - dist
        bonus = _facility_bonus(meta, query_norm)
        score = base + 0.2 * bonus
        if score >= min_score:
            hits.append({"score": round(score, 4), "text": doc, "meta": meta})

    if debug:
        print(f"✅ {len(hits)} results after filtering")

    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:k]

# -------------------------------------------------------------
# 📍 근처 검색 (H3 기반 spatial index 사용)
# -------------------------------------------------------------
try:
    import h3
except ImportError:
    h3 = None

BUILDING_ALIASES = {
    "제1여객터미널": {"제1여객터미널", "제1터미널", "t1", "tt1", "terminal1"},
    "제2여객터미널": {"제2여객터미널", "제2터미널", "t2", "tt2", "terminal2"},
    "탑승동": {"탑승동", "concourse", "c", "tc", "탑승동동편", "탑승동서편"},
}

def normalize_building_name(name: str) -> str:
    if not name:
        return None
    n = name.strip().lower()
    for std, variants in BUILDING_ALIASES.items():
        if any(v.lower() == n for v in variants):
            return std
    return name

def is_nearby_pattern(q: str) -> bool:
    return any(k in q for k in ["근처", "주변", "가까운", "옆", "맞은편"])
    
def prefer_boarding_gate(hits):
    """
    nearby search에서 '게이트' 질의일 경우
    도착게이트보다 '탑승구'를 anchor로 우선 선택
    """
    boarding = []
    others = []

    for h in hits:
        name = (h["meta"].get("name") or "")
        if "탑승구" in name:
            boarding.append(h)
        else:
            others.append(h)

    # 탑승구가 하나라도 있으면 그중 score 최고
    if boarding:
        boarding.sort(key=lambda x: x["score"], reverse=True)
        return boarding[0]

    # 없으면 기존 최고 점수 유지
    return hits[0]


# ===== 핵심 함수 =====
def structured_nearby_any(query: str, max_k: int = 10, start_ring: int = 2, final_ring: int = 15, max_distance_m: int = 250):
    """
    H3 기반 spatial index를 이용한 근처 시설 탐색 (다단계 ring 확장)
    """
    #from nearby_search_spatial import haversine as _haversine

    # 1️⃣ 질의 파싱
    m = re.search(r"(.+?)\s*(근처|주변|가까운|옆|맞은편)\s*(.+)", query)
    if not m:
        return {"error": "패턴 불인식", "results": []}
    anchor_txt, _, target_txt = m.groups()

    # 2️⃣ 앵커 검색
    anc_hits = search_facilities_chroma(anchor_txt, k=3)
    if not anc_hits:
        return {"error": "앵커 없음", "results": []}
    anchor_hit = prefer_boarding_gate(anc_hits)
    anchor = anchor_hit["meta"]


    # 3️⃣ 좌표 필드 인식 확장
    lat = anchor.get("lat") or anchor.get("poiLatitude") or anchor.get("latitude") or anchor.get("위도")
    lon = anchor.get("lon") or anchor.get("poiLongitude") or anchor.get("longitude") or anchor.get("경도")
    if not lat or not lon:
        return {"error": "앵커 좌표 없음", "results": []}

    lat, lon = float(lat), float(lon)
    building = normalize_building_name(anchor.get("building"))
    floor = anchor.get("floor")

    # 4️⃣ 타깃 카테고리 정규화
    target_category = normalize_category_query(target_txt)
    if not target_category:
        target_category = target_txt.strip()

    # 5️⃣ 데이터 로드
    facilities = json.loads(Path(FAC_PATH).read_text())
    items = facilities.get("items", facilities)
    spatial = json.loads(Path(SPATIAL_PATH).read_text())
    mode = spatial.get("mode", "h3")

    # 6️⃣ 근처 탐색 (다단계 ring 확장)
    results = []
    debug_log = []
    seen_ids = set()

    if mode == "h3":
        anchor_cell = h3.geo_to_h3(lat, lon, spatial.get("h3_res", 12))

        for ring in range(start_ring, final_ring + 1):
            nearby_cells = list(h3.k_ring(anchor_cell, ring))
            cand_count, kept_count = 0, 0

            for entry in spatial["keys"]:
                if (
                    normalize_building_name(entry["building"]) == building
                    and (not floor or entry["floor"] == floor)
                    and entry["cell"] in nearby_cells
                ):
                    for fid in entry["ids"]:
                        if fid in seen_ids:
                            continue
                        f = next((x for x in items if x["vsid"] == fid), None)
                        if not f:
                            continue
                        cat = f.get("category", "")
                        nm = f.get("name", "")
                        cand_count += 1
                        # 부분 일치 허용
                        if (
                            target_category not in cat
                            and cat not in target_category
                            and target_category not in nm
                        ):
                            continue

                        d = _haversine(lat, lon, f["poiLatitude"], f["poiLongitude"])
                        if d <= max_distance_m:
                            kept_count += 1
                            seen_ids.add(fid)
                            results.append({
                                "meta": f,                     # ✅ 핵심 수정
                                "distance_m": round(d, 1),
                                "score": round(1 / (1 + d / 1000), 6)
                            })

            debug_log.append(f"(ring={ring}, cands={cand_count}, kept={kept_count})")

            # 🔹 일정 수 이상 찾으면 조기 종료
            if len(results) >= max_k:
                break

    # 7️⃣ 결과 정렬 및 디버그 출력
    results.sort(key=lambda x: x["distance_m"])
    print(f"[DEBUG] Nearby step summary: {len(results)} results | {' '.join(debug_log)}")

    return {
        "anchor": {
            "id": anchor.get("vsid") or anchor.get("id"),
            "name": anchor.get("name"),
            "building": anchor.get("building"),
            "floor": anchor.get("floor"),
            "lat": lat,
            "lon": lon,
        },
        "results": results,
        "note": f"mode={mode}, start_ring={start_ring}, final_ring={final_ring}, same_bf={(not floor)}, category='{target_category}' | dbg:{' '.join(debug_log)}"
    }



# -------------------------------------------------------------
# 💬 LLM 프롬프트
# -------------------------------------------------------------
def build_facility_prompt(query: str, hits: list[dict]) -> str:
    if not hits:
        return f"'{query}'에 해당하는 시설 정보를 찾을 수 없습니다."

    cards = []
    for h in hits[:5]:
        m = h.get("meta", {})
        name = m.get("name") or "-"
        building = m.get("building_alias") or m.get("building") or "-"
        floor = m.get("floor") or "-"
        category = m.get("category") or "-"
        loc = m.get("loc") or "-"
        goods = m.get("goods") or "-"
        tel = m.get("tel") or m.get("telNo") or "-"
        opn = m.get("opnTm") or m.get("openTime") or "-"
        cls = m.get("clsTm") or m.get("closeTime") or "-"
        hours = "정보없음"
        if opn != "-" or cls != "-":
            if (
                (opn in ("0000", "00:00") and cls in ("2400", "24:00"))
                or (opn in ("0000", "00:00") and cls in ("0000", "00:00"))
                or (opn == "24:00" and cls == "24:00")
            ):
                hours = "상시 운영"
            elif opn != "-" and cls != "-":
                hours = f"{opn} ~ {cls}"
            else:
                hours = opn if opn != "-" else cls

        dist_txt = ""
        if h.get("distance_m") is not None:
            dist_txt = f" | 거리: 약 {int(h['distance_m'])}m"
        
        cards.append(
            f"- 시설명: {name} | 터미널: {building} | 층: {floor} | "
            f"카테고리: {category} | 위치: {loc} | 영업시간: {hours} | "
            f"전화번호: {tel} | 취급품목: {goods}{dist_txt}"
        )

    ctx = "\n".join(cards)
    prompt = f"""
당신은 인천국제공항 안내 챗봇입니다.
아래의 시설 데이터를 참고하여 사용자 질문에 정확하고 친절하게 답하세요.

[사용자 질문]
{query}

[검색된 시설 목록]
{ctx}

위 데이터를 근거로 답변을 작성하세요.
- 여러 시설이 검색되면 위치나 층, 터미널 기준으로 자연스럽게 요약하세요.
- 시설명, 터미널(T1/T2/Concourse), 층, 위치, 영업시간, 전화번호, 거리를 반드시 포함하세요.
- 취급품목(goods)이나 주요 특징이 있으면 함께 설명하세요.
- 검색된 모든 시설을 빠짐없이 각각 언급하세요.
- 단, 개수를 물어보는 질의에는 시설의 개수만 알려주고 상세 설명은 생략하세요.
- 활기차고 친절한 어조로 답변하세요.
- 데이터 외 추측은 금지하며, 확인되지 않은 정보는 언급하지 마세요.
- 비어있거나 찾을 수 없는 항목(ex. 전화번호, 거리)은 언급하지 마세요

[답변]:
    """.strip()
    return prompt
    
    
# =============================================================
# 🚀 [추가된 메인 인터페이스] 라우터 연결 및 오픈소스 LLM 주입
# =============================================================
def handle_facility_query(query: str, llm_client) -> str:
    """
    라우터에서 호출하는 시설 도메인 메인 함수.
    공간 탐색(Nearby)이 필요한지 판별하고, 주입된 LLM을 통해 최종 답변을 생성합니다.
    """
    # 1. 검색 모듈 자동 분기 ('근처/주변' 키워드가 있으면 공간 탐색, 없으면 일반 벡터 검색)
    if is_nearby_pattern(query):
        search_result = structured_nearby_any(query)
        hits = search_result.get("results", [])
        
        # 앵커(기준점)를 못 찾았거나 근처에 결과가 없으면 일반 검색으로 안전하게 Fallback(대체)
        if not hits:
            hits = search_facilities_chroma(query)
    else:
        hits = search_facilities_chroma(query)

    # 2. LLM 프롬프트 생성
    prompt = build_facility_prompt(query, hits)

    # 3. 주입받은 LangChain 오픈소스 LLM(Qwen) 호출 (.invoke 사용)
    response = llm_client.invoke(prompt)

    # 4. 객체 또는 문자열 결과 안전하게 추출
    if hasattr(response, 'content'):
        return response.content.strip()
    return str(response).strip()