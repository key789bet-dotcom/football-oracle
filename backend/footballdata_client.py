"""
Client gọi football-data.org (API v4) — MIỄN PHÍ, không qua RapidAPI.

Đăng ký token: https://www.football-data.org/client/register  (nhận token qua email)
Xác thực bằng header: X-Auth-Token: <token>

Gói free: ~12 giải lớn (Ngoại hạng Anh, La Liga, Serie A, Bundesliga, Ligue 1,
Champions League, World Cup...), giới hạn ~10 request/phút (đã có cache).
Gói free KHÔNG có odds -> tool sẽ dùng Poisson + H2H.

Output được chuẩn hóa về đúng định dạng app, nên không phải sửa file khác.
"""
import os
import time
import httpx
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("FOOTBALLDATA_TOKEN", "")
CACHE_TTL = int(os.getenv("CACHE_TTL", "60"))
BASE_URL = "https://api.football-data.org/v4"

_cache: dict[str, tuple[float, dict]] = {}

# football-data dùng các trạng thái: SCHEDULED, TIMED, IN_PLAY, PAUSED, FINISHED...
LIVE_STATUSES = {"IN_PLAY", "PAUSED", "LIVE"}

# 12 giải trong gói FREE (endpoint /matches free-tier cần lọc theo giải mới ra dữ liệu)
# WC,CL,BL1,DED,BSA,PD,FL1,ELC,PPL,EC,SA,PL
FREE_COMPETITIONS = "2000,2001,2002,2003,2013,2014,2015,2016,2017,2018,2019,2021"


def _get(path: str, params: dict | None = None) -> dict:
    params = params or {}
    key = path + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    now = time.time()
    if key in _cache and now - _cache[key][0] < CACHE_TTL:
        return _cache[key][1]
    if not TOKEN:
        raise RuntimeError("Chưa có FOOTBALLDATA_TOKEN. Đăng ký tại football-data.org rồi điền vào .env.")
    with httpx.Client(timeout=15) as client:
        resp = client.get(f"{BASE_URL}{path}", headers={"X-Auth-Token": TOKEN}, params=params)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code} từ {path} -> {resp.text[:300]}")
        data = resp.json()
    _cache[key] = (now, data)
    return data


# ---------------- Chuẩn hóa ----------------
def _goals(match: dict) -> dict:
    """Lấy tỉ số: ưu tiên fullTime, nếu trống thì halfTime."""
    score = match.get("score", {})
    ft = score.get("fullTime", {})
    ht = score.get("halfTime", {})
    home = ft.get("home")
    away = ft.get("away")
    if home is None and away is None:
        home, away = ht.get("home"), ht.get("away")
    return {"home": home, "away": away}


def normalize_match(m: dict) -> dict:
    """Đổi 1 match football-data -> định dạng 'slim fixture' của app."""
    home, away = m.get("homeTeam", {}), m.get("awayTeam", {})
    ts = m.get("utcDate")
    return {
        "fixture_id": m.get("id"),
        # dùng id trận làm custom_id để lấy h2h (head2head theo id trận)
        "custom_id": str(m.get("id")) if m.get("id") else None,
        "date": ts,
        "status": m.get("status", ""),
        "league": (m.get("competition") or {}).get("name", ""),
        "league_code": (m.get("competition") or {}).get("code", ""),
        "home": {"id": home.get("id"), "name": home.get("name"), "logo": home.get("crest", "")},
        "away": {"id": away.get("id"), "name": away.get("name"), "logo": away.get("crest", "")},
        "goals": _goals(m),
    }


def to_predictor_match(m: dict) -> dict:
    """Đổi match -> dict mà predictor.expected_goals_from_form mong đợi."""
    g = _goals(m)
    return {
        "goals": g,
        "teams": {
            "home": {"id": m.get("homeTeam", {}).get("id")},
            "away": {"id": m.get("awayTeam", {}).get("id")},
        },
    }


# ---------------- API công khai (cùng tên với các client khác) ----------------
def get_live_fixtures() -> list[dict]:
    data = _get("/matches", {"status": "LIVE", "competitions": FREE_COMPETITIONS})
    return [normalize_match(m) for m in data.get("matches", [])]


def get_fixtures_by_date(date_iso: str) -> list[dict]:
    """date_iso = 'YYYY-MM-DD'. Lọc theo 12 giải gói free để có dữ liệu."""
    data = _get("/matches", {"dateFrom": date_iso, "dateTo": date_iso,
                             "competitions": FREE_COMPETITIONS})
    return [normalize_match(m) for m in data.get("matches", [])]


def get_competition_matches(code: str, date_from: str | None = None,
                            date_to: str | None = None) -> list[dict]:
    """Lấy trận theo GIẢI (cách chạy ổn định nhất trên gói free).
    code: PL, PD, SA, BL1, FL1, BSA, CL, DED, PPL, ELC, EC, WC."""
    params = {}
    if date_from:
        params["dateFrom"] = date_from
    if date_to:
        params["dateTo"] = date_to
    data = _get(f"/competitions/{code}/matches", params)
    return [normalize_match(m) for m in data.get("matches", [])]


def get_team_last_matches(team_id: int, last: int = 10) -> list[dict]:
    data = _get(f"/teams/{team_id}/matches", {"status": "FINISHED", "limit": last})
    return [to_predictor_match(m) for m in data.get("matches", [])]


def get_h2h(custom_id: str) -> list[dict]:
    """custom_id ở đây là id trận; football-data trả lịch sử đối đầu theo trận."""
    if not custom_id:
        return []
    data = _get(f"/matches/{custom_id}/head2head", {"limit": 10})
    return [to_predictor_match(m) for m in data.get("matches", [])]
