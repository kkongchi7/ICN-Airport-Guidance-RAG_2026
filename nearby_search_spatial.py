# -*- coding: utf-8 -*-
"""
nearby_search_spatial.py
------------------------------------
spatial_index.json을 이용한 근처 시설 검색 엔진
anchor: 기준 시설(탑승구, 라운지 등)
target: 찾고자 하는 시설(카페, 화장실 등)
------------------------------------
"""

import json, math
from pathlib import Path

try:
    import h3
except ImportError:
    h3 = None


# ===== 경로 설정 =====
SPATIAL_PATH = "/content/spatial_index.json"
FAC_PATH = "/content/spoi_formatted_with_category.json"


# ===== 좌표 기반 거리 계산 =====
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ===== 데이터 로드 =====
def load_facilities():
    data = json.loads(Path(FAC_PATH).read_text())
    return data.get("items", data)


def load_spatial_index():
    return json.loads(Path(SPATIAL_PATH).read_text())


# ===== 근처 탐색 =====
def find_nearby(anchor_meta: dict, target_category: str, ring: int = 2, max_distance_m: int = 200):
    """anchor 주변에서 target_category에 해당하는 시설 검색"""
    facilities = load_facilities()
    spatial = load_spatial_index()
    mode = spatial.get("mode")

    lat, lon = anchor_meta.get("poiLatitude"), anchor_meta.get("poiLongitude")
    building = anchor_meta.get("building")
    floor = anchor_meta.get("floor")

    if not lat or not lon:
        return {"error": "앵커에 좌표가 없습니다.", "results": []}

    results = []

    if mode == "h3":
        # H3 셀 계산
        if hasattr(h3, "geo_to_h3"):
            anchor_cell = h3.geo_to_h3(lat, lon, spatial.get("h3_res", 12))
        else:
            anchor_cell = h3.latlng_to_cell(lat, lon, spatial.get("h3_res", 12))
        nearby_cells = list(h3.k_ring(anchor_cell, ring))

        # building/floor 동일한 셀 찾기
        for entry in spatial["keys"]:
            if (
                entry["building"] == building
                and entry["floor"] == floor
                and entry["cell"] in nearby_cells
            ):
                for fid in entry["ids"]:
                    f = next((x for x in facilities if x["vsid"] == fid), None)
                    if not f:
                        continue
                    if f.get("category") != target_category:
                        continue
                    d = haversine(lat, lon, f["poiLatitude"], f["poiLongitude"])
                    if d <= max_distance_m:
                        results.append({"meta": f, "distance_m": round(d)})

    elif mode == "grid":
        grid_size_m = spatial["grid_size_m"]
        origin = spatial["origin"]
        ref_lat = spatial["ref_lat_deg"]
        M_PER_DEG_LAT = 111320.0
        M_PER_DEG_LON = M_PER_DEG_LAT * math.cos(math.radians(ref_lat))

        def cell_for(lat_, lon_):
            dx_m = (lat_ - origin["lat"]) * M_PER_DEG_LAT
            dy_m = (lon_ - origin["lon"]) * M_PER_DEG_LON
            gx = int(round(dx_m / grid_size_m))
            gy = int(round(dy_m / grid_size_m))
            return f"{gx}:{gy}"

        gx, gy = cell_for(lat, lon)
        nearby_cells = [f"{gx+dx}:{gy+dy}" for dx in range(-ring, ring+1) for dy in range(-ring, ring+1)]

        for entry in spatial["keys"]:
            if (
                entry["building"] == building
                and entry["floor"] == floor
                and entry["cell"] in nearby_cells
            ):
                for fid in entry["ids"]:
                    f = next((x for x in facilities if x["vsid"] == fid), None)
                    if not f:
                        continue
                    if f.get("category") != target_category:
                        continue
                    d = haversine(lat, lon, f["poiLatitude"], f["poiLongitude"])
                    if d <= max_distance_m:
                        results.append({"meta": f, "distance_m": round(d)})

    results.sort(key=lambda x: x["distance_m"])
    return {"error": None, "anchor": anchor_meta, "results": results}


# ===== 테스트 실행 =====
if __name__ == "__main__":
    # 예시: 225번 게이트 주변 카페 찾기
    anchor = {
        "name": "탑승구 225",
        "building": "제 2 여객터미널",
        "floor": "3층",
        "poiLatitude": 37.4662110213611,
        "poiLongitude": 126.4308745042174
    }
    result = find_nearby(anchor, target_category="카페/음료", ring=2, max_distance_m=200)
    if result["error"]:
        print("❌ Error:", result["error"])
    else:
        print(f"✅ {len(result['results'])}개 시설 탐색됨")
        for r in result["results"][:5]:
            m = r["meta"]
            print(f"- {m['poiNm']} ({m['category']}) | {r['distance_m']}m")
