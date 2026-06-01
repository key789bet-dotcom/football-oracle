"""
Client gọi API-Football (qua RapidAPI).
Docs: https://www.api-football.com/documentation-v3
Có cache đơn giản trong RAM để tránh vượt giới hạn free tier.
"""
import os
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST", "api-football-v1.p.rapidapi.com")
CACHE_TTL = int(os.getenv("CACHE_TTL", "60"))
BASE_URL = f"https://{RAPIDAPI_HOST}/v3"

_cache: dict[str, tuple[float, dict]] = {}


def _headers() -> dict:
    return {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
    }


def _get(path: str, params: dict | None = None) -> dict:
    """GET có cache. Key cache = path + params."""
    params = params or {}
    cache_key = path + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    now = time.time()
    if cache_key in _cache:
        ts, data = _cache[cache_key]
        if now - ts < CACHE_TTL:
            return data

    if not RAPIDAPI_KEY:
        raise RuntimeError("Chưa có RAPIDAPI_KEY. Hãy tạo file .env từ .env.example.")

    with httpx.Client(timeout=15) as client:
        resp = client.get(f"{BASE_URL}{path}", headers=_headers(), params=params)
        resp.raise_for_status()
        data = resp.json()

    _cache[cache_key] = (now, data)
    return data


def get_live_fixtures() -> list[dict]:
    """Các trận đang diễn ra (realtime)."""
    data = _get("/fixtures", {"live": "all"})
    return data.get("response", [])


def get_fixtures_by_date(date: str) -> list[dict]:
    """Trận theo ngày, định dạng YYYY-MM-DD."""
    data = _get("/fixtures", {"date": date})
    return data.get("response", [])


def get_team_last_matches(team_id: int, last: int = 10) -> list[dict]:
    """N trận gần nhất của 1 đội, dùng để tính phong độ."""
    data = _get("/fixtures", {"team": team_id, "last": last})
    return data.get("response", [])


def get_odds(fixture_id: int) -> list[dict]:
    """Tỉ lệ kèo (odds) của 1 trận."""
    data = _get("/odds", {"fixture": fixture_id})
    return data.get("response", [])
