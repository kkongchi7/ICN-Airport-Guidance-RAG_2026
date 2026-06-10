# -*- coding: utf-8 -*-
"""
facility_category_map.py
PATTERNS 기반 질의어 → 표준 카테고리 매핑 사전
"""
import re

# ===== 카테고리 키워드 매핑 =====
CATEGORY_KEYWORDS = {
    "화장실": ["화장실", "restroom", "toilet", "bathroom"],
    "대피소": ["대피소", "shelter"],
    "환승장": ["환승장", "transfer", "환승", "transfer zone"],
    "보험": ["보험", "insurance"],
    "서점": ["서점", "bookstore", "도서", "신문", "잡지"],
    "흡연실": ["흡연실", "흡연", "전자담배", "smoking room", "cigarette"],
    "숙박시설/호텔": ["호텔", "객실", "숙박", "lodging", "transit hotel", "hotel"],
    "안경점": ["안경", "렌즈", "optical", "eyewear", "lens"],
    "라운지": ["라운지", "lounge", "스카이허브", "KAL", "아시아나라운지", "Sky Hub"],
    "면세점": ["면세점", "면세", "duty free", "신라면세점", "롯데면세점", "신세계면세점"],
    "보안검색": ["보안검색", "검색대", "security check", "security", "검색"],
    "체크인카운터": ["체크인카운터", "check-in counter", "체크인", "카운터"],
    "입국심사": ["입국심사", "immigration arrival", "입국심사대", "arrival immigration"],
    "출국심사": ["출국심사", "immigration departure", "출국심사대", "departure immigration"],
    "입국장": ["입국장", "도착장", "arrivals", "arrival hall"],
    "출국장": ["출국장", "출발장", "departures", "departure hall"],
    "게이트/탑승구": ["게이트", "탑승구", "gate", "boarding gate", "GATE"],
    "출입구": ["출입구", "entrance", "gate entrance"],
    "약국/의료": ["약국", "pharmacy", "drugstore", "medicine"],
    "의료": ["의무실", "응급", "의료", "clinic", "medical", "emergency room"],
    "통신/로밍": ["와이파이", "wifi", "로밍", "유심", "SIM", "통신", "mobile", "internet"],
    "금융/환전": [
        "은행", "환전", "외환", "ATM", "우리은행", "신한", "국민", "농협", "하나",
        "IBK", "KB", "NH", "currency exchange", "bank"
    ],
    "수하물/보관": [
        "수하물", "유실물", "보관소", "보관", "baggage", "lost and found",
        "locker", "보관함"
    ],
    "유아/어린이": [
        "유아", "수유실", "수유", "키즈", "어린이", "놀이시설", "stroller", "kids", "baby"
    ],
    "카페/음료": [
        "카페", "커피", "cafe", "coffee", "tea", "베이커리", "bakery", "던킨",
        "파리바게뜨", "뚜레쥬르", "폴 바셋", "paul bassett", "스타벅스",
        "투썸", "이디야", "음료"
    ],
    "식음/음식점/식당": [
        "KFC", "맥도날드", "버거킹", "롯데리아", "서브웨이", "피자", "파스타",
        "스시", "라멘", "라면", "우동", "돈카츠", "한식", "중식", "일식", "양식",
        "분식", "식당", "레스토랑", "restaurant", "푸드코트", "food court",
        "국수", "요리", "밥", "버거", "고기"
    ],
    "편의점": ["편의점", "CU", "GS25", "세븐일레븐", "7-ELEVEN", "이마트24"],
    "우편/택배": ["우체국", "post", "ems", "택배", "courier", "parcel", "배송"],
    "편의시설": [
        "샤워", "샤워실", "사우나", "수면실", "마사지",
        "비즈니스센터", "휴게실", "rest area", "business center"
    ],
    "셔틀트레인/교통": [
        "셔틀", "셔틀트레인", "shuttle train", "무인궤도열차", "열차", "APM"
    ],
    "대중교통": [
        "버스", "리무진", "공항철도", "AREX", "철도", "지하철", "택시", "KTX",
        "rail", "subway", "train"
    ],
    "주차": ["주차", "parking", "주차장"],
    "렌터카": ["렌터카", "렌트카", "rental car", "rent-a-car"],
    "쇼핑": [
        "기념품", "souvenir", "선물", "액세서리", "잡화", "토이", "완구",
        "화장품", "코스메틱", "패션", "의류", "슈즈", "시계", "쥬얼리",
        "명품", "쇼핑", "매장", "store"
    ],
    "안내/편의": [
        "안내", "information", "help desk", "인포", "여권", "민원",
        "발급", "신고", "고객센터", "service center"
    ],
}

# ===== 카테고리 정규화 함수 =====
def normalize_category_query(query: str) -> str:
    """자유 질의어를 표준 카테고리로 정규화"""
    q = query.lower()
    q = re.sub(r"[^가-힣a-z0-9\s]", " ", q)   # 특수문자 제거
    q = re.sub(r"\s+", " ", q).strip()         # 중복 공백 제거

    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in sorted(kws, key=len, reverse=True):  # 긴 키워드부터 탐색
            if kw.lower() in q:
                return cat
    return None


# ===== 다중 카테고리 탐지 (선택용) =====
def normalize_category_query_multi(query: str) -> list[str]:
    """질의어에서 발견된 모든 카테고리 반환"""
    q = query.lower()
    q = re.sub(r"[^가-힣a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    found = []
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in q:
                found.append(cat)
                break
    return list(dict.fromkeys(found))  # 중복 제거


# ===== 단독 실행 테스트 =====
if __name__ == "__main__":
    tests = [
        "화장실 어디 있어?",
        "면세점 근처 카페 찾아줘",
        "롯데리아 가고 싶어",
        "휴게실이나 라운지 있어?",
        "버거킹 위치 알려줘"
    ]
    for q in tests:
        print(f"{q} → {normalize_category_query(q)} | {normalize_category_query_multi(q)}")
