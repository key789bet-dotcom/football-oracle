"""
Client gọi SofaScore (qua RapidAPI, host: sofascore.p.rapidapi.com).

LƯU Ý: Mỗi nhà cung cấp SofaScore trên RapidAPI có tên endpoint hơi khác nhau.
Hãy mở tab "Endpoints" trong trang RapidAPI của bạn và đối chiếu các đường dẫn
+ tên tham số bên dưới (PATH_*). Phần parse đã được viết "phòng thủ": tự dò
homeTeam/awayTeam ở bất kỳ đâu trong JSON nên ít phụ thuộc cấu trúc cụ thể.

Output được CHUẨN HÓA về đúng định dạng mà predictor.py & frontend đang dùng,
nên không phải sửa các file khác.
"""
import os
import time
import httpx
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
SOFA_HOST = os.getenv("SOFA_HOST", "sofascore.p.rapidapi.com")
CACHE_TTL = int(os.getenv("CACHE_TTL", "60"))
BASE_URL = f"https://{SOFA_HOST}"

# --- Đường dẫn endpoint: đối chiếu với RapidAPI dashboard của bạn ---
PATH_LIVE = "/matches/list-live"          # ?sport=football  (nếu provider hỗ trợ)
PATH_BY_DATE = "/matches/list"            # ?date=dd/mm/yyyy
PATH_TEAM_LAST = "/teams/get-last-matches"  # ?teamId=..&pageIndex=0
PATH_H2H = "/matches/get-h2h-events"      # ?customId=..

_cache: dict[str, tuple[float, dict]] = {}


def _headers() -> dict:
    return {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": SOFA_HOST}


def _get(path: str, params: dict | None = None) -> dict:
    params = params or {}
    key = path + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    now = time.time()
    if key in _cache and now - _cache[key][0] < CACHE_TTL:
        return _cache[key][1]
    if not RAPIDAPI_KEY:
        raise RuntimeError("Chưa có RAPIDAPI_KEY. Tạo file .env từ .env.example.")
    with httpx.Client(timeout=15) as client:
        resp = client.get(f"{BASE_URL}{path}", headers=_headers(), params=params)
        if resp.status_code != 200:
            # Hiện rõ thông báo từ RapidAPI (vd "not subscribed", "endpoint not found")
            raise RuntimeError(f"HTTP {resp.status_code} từ {path} -> {resp.text[:300]}")
        data = resp.json()
    _cache[key] = (now, data)
    return data


# ---------------- Bóc tách & chuẩn hóa ----------------
def _extract_events(data) -> list[dict]:
    """Dò đệ quy mọi object trông giống 'event' (có homeTeam & awayTeam)."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            if "homeTeam" in node and "awayTeam" in node:
                found.append(node)
            else:
                for v in node.values():
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    return found


def _score(side: dict) -> int | None:
    """SofaScore: homeScore/awayScore là dict {'current': n, ...} hoặc số."""
    if isinstance(side, dict):
        return side.get("current", side.get("display"))
    return side if isinstance(side, int) else None


def normalize_event(ev: dict) -> dict:
    """Đổi 1 event SofaScore -> định dạng 'slim fixture' của app."""
    home, away = ev.get("homeTeam", {}), ev.get("awayTeam", {})
    status = ev.get("status", {})
    ts = ev.get("startTimestamp")
    return {
        "fixture_id": ev.get("id"),
        "custom_id": ev.get("customId"),  # cần cho h2h
        "date": datetime.utcfromtimestamp(ts).isoformat() if ts else None,
        "status": status.get("description") or status.get("type", ""),
        "league": (ev.get("tournament") or {}).get("name", ""),
        "home": {"id": home.get("id"), "name": home.get("name"), "logo": ""},
        "away": {"id": away.get("id"), "name": away.get("name"), "logo": ""},
        "goals": {"home": _score(ev.get("homeScore")), "away": _score(ev.get("awayScore"))},
    }


def to_predictor_match(ev: dict) -> dict:
    """Đổi event -> dict mà predictor.expected_goals_from_form mong đợi."""
    return {
        "goals": {"home": _score(ev.get("homeScore")), "away": _score(ev.get("awayScore"))},
        "teams": {
            "home": {"id": ev.get("homeTeam", {}).get("id")},
            "away": {"id": ev.get("awayTeam", {}).get("id")},
        },
    }


# ---------------- API công khai (cùng tên với api_client) ----------------
def get_live_fixtures() -> list[dict]:
    data = _get(PATH_LIVE, {"sport": "football"})
    return [normalize_event(e) for e in _extract_events(data)]


def get_fixtures_by_date(date_iso: str) -> list[dict]:
    """date_iso = 'YYYY-MM-DD' -> SofaScore thường dùng dd/mm/yyyy."""
    d = datetime.fromisoformat(date_iso)
    data = _get(PATH_BY_DATE, {"date": d.strftime("%d/%m/%Y")})
    return [normalize_event(e) for e in _extract_events(data)]


def get_team_last_matches(team_id: int, last: int = 10) -> list[dict]:
    """Trả list theo định dạng predictor (goals + teams)."""
    data = _get(PATH_TEAM_LAST, {"teamId": team_id, "pageIndex": 0})
    events = _extract_events(data)[:last]
    return [to_predictor_match(e) for e in events]


def get_h2h(custom_id: str) -> list[dict]:
    """Các trận đối đầu lịch sử giữa 2 đội (định dạng predictor)."""
    if not custom_id:
        return []
    data = _get(PATH_H2H, {"customId": custom_id})
    return [to_predictor_match(e) for e in _extract_events(data)]
