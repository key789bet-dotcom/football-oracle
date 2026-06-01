"""
Client lấy KÈO THẬT từ the-odds-api.com (free 500 request/tháng).

Đăng ký key: https://the-odds-api.com/  -> điền ODDS_API_KEY vào .env
Trả: tỉ lệ 1X2 (h2h) + Tài/Xỉu (totals) đã KHỬ BIÊN nhà cái -> xác suất thị trường.

Ghép trận theo TÊN đội (football-data và the-odds-api đặt tên hơi khác nhau)
bằng so khớp gần đúng (bỏ dấu, bỏ hậu tố FC/EC/SC...).
"""
import os
import re
import time
import unicodedata
import httpx
from dotenv import load_dotenv

load_dotenv()

ODDS_KEY = os.getenv("ODDS_API_KEY", "")
CACHE_TTL = int(os.getenv("ODDS_TTL", "600"))  # 10 phút (tiết kiệm quota)
BASE = "https://api.the-odds-api.com/v4"

# football-data code -> sport key của the-odds-api
SPORT_KEYS = {
    "WC": "soccer_fifa_world_cup",
    "BSA": "soccer_brazil_campeonato",
    "PL": "soccer_epl",
    "PD": "soccer_spain_la_liga",
    "SA": "soccer_italy_serie_a",
    "BL1": "soccer_germany_bundesliga",
    "FL1": "soccer_france_ligue_one",
    "CL": "soccer_uefa_champs_league",
    "DED": "soccer_netherlands_eredivisie",
    "PPL": "soccer_portugal_primeira_liga",
    "ELC": "soccer_england_efl_champ",
}

_cache: dict[str, tuple[float, list]] = {}
_STOP = {"fc", "ec", "sc", "cr", "se", "af", "ca", "cf", "fbpa", "club", "de",
         "futebol", "clube", "the", "afc", "cd", "ud", "rc", "ac", "as", "ssc", "us"}


def _norm(name: str) -> set:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    toks = re.findall(r"[a-z0-9]+", name.lower())
    return {t for t in toks if t not in _STOP and len(t) > 1}


def get_events(code: str) -> list:
    sport = SPORT_KEYS.get(code)
    if not sport or not ODDS_KEY:
        return []
    if code in _cache and time.time() - _cache[code][0] < CACHE_TTL:
        return _cache[code][1]
    url = f"{BASE}/sports/{sport}/odds"
    params = {"apiKey": ODDS_KEY, "regions": "eu,uk",
              "markets": "h2h,totals", "oddsFormat": "decimal"}
    with httpx.Client(timeout=15) as c:
        r = c.get(url, params=params)
        if r.status_code != 200:
            raise RuntimeError(f"the-odds-api HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
    _cache[code] = (time.time(), data)
    return data


def list_sports() -> list:
    """Gọi /sports (KHÔNG tốn quota) — xác minh key + xem giải nào đang hoạt động."""
    if not ODDS_KEY:
        raise RuntimeError("Chưa có ODDS_API_KEY trong .env")
    with httpx.Client(timeout=15) as c:
        r = c.get(f"{BASE}/sports", params={"apiKey": ODDS_KEY})
        if r.status_code != 200:
            raise RuntimeError(f"the-odds-api HTTP {r.status_code}: {r.text[:200]}")
        return r.json()


def match_event(events: list, home_name: str, away_name: str) -> dict | None:
    """Tìm trận khớp theo tên 2 đội (so khớp token gần đúng)."""
    h, a = _norm(home_name), _norm(away_name)
    best, best_score = None, 0
    for ev in events:
        eh, ea = _norm(ev.get("home_team", "")), _norm(ev.get("away_team", ""))
        score = len(h & eh) + len(a & ea)
        if score > best_score and (h & eh) and (a & ea):
            best, best_score = ev, score
    return best


def _devig(odds: dict) -> dict:
    inv = {k: 1.0 / v for k, v in odds.items() if v}
    s = sum(inv.values())
    return {k: round(v / s, 4) for k, v in inv.items()} if s else {}


def market_probs(event: dict, home_name: str, away_name: str) -> dict | None:
    """Xác suất 1X2 từ kèo (lấy bookmaker đầu tiên có h2h, đã khử biên)."""
    if not event:
        return None
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk.get("key") != "h2h":
                continue
            o = {}
            for out in mk.get("outcomes", []):
                nm = out["name"]
                if nm == event.get("home_team"):
                    o["home"] = out["price"]
                elif nm == event.get("away_team"):
                    o["away"] = out["price"]
                elif nm.lower() in ("draw", "tie"):
                    o["draw"] = out["price"]
            if {"home", "draw", "away"} <= set(o):
                p = _devig(o)
                p["bookmaker"] = bk.get("title")
                p["raw_odds"] = o
                return p
    return None


def market_totals(event: dict) -> dict | None:
    """Tài/Xỉu thị trường ở mức gần 2.5 (đã khử biên)."""
    if not event:
        return None
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk.get("key") != "totals":
                continue
            best = None
            for out in mk.get("outcomes", []):
                pt = out.get("point")
                if pt is None:
                    continue
                if best is None or abs(pt - 2.5) < abs(best - 2.5):
                    best = pt
            if best is None:
                continue
            od = {}
            for out in mk.get("outcomes", []):
                if out.get("point") == best:
                    od[out["name"].lower()] = out["price"]
            if "over" in od and "under" in od:
                p = _devig(od)
                return {"line": best, "over": p.get("over"), "under": p.get("under"),
                        "bookmaker": bk.get("title")}
    return None
