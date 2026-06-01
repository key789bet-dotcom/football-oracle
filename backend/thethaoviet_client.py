"""
Client cho API thethaoviet.vip — endpoint CHI TIẾT 1 TRẬN theo ID:
    GET https://api.thethaoviet.vip/api/p/fixtures/{id}/detail

Ưu điểm: có `status_elapsed` = PHÚT THẬT đang đá + phạt góc/thẻ (summary) + tỉ số live.
Dùng cho phân tích IN-PLAY theo phút chính xác (chính xác hơn ước lượng từ giờ bóng lăn).

Lưu ý: API này có thể chặn request không phải trình duyệt. Đã thêm header giả lập.
Nếu backend của bạn vẫn không gọi được, thường do mạng/Cloudflare — thử mạng khác.
"""
import time
import httpx

BASE = "https://api.thethaoviet.vip/api/p"
CACHE_TTL = 20  # live nên cache ngắn

_cache: dict[str, tuple[float, dict]] = {}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://thethaoviet.vip/",
    "Origin": "https://thethaoviet.vip",
}


def _get(path: str) -> dict:
    now = time.time()
    if path in _cache and now - _cache[path][0] < CACHE_TTL:
        return _cache[path][1]
    with httpx.Client(timeout=15, headers=_HEADERS) as c:
        r = c.get(f"{BASE}{path}")
        if r.status_code != 200:
            raise RuntimeError(f"thethaoviet HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
    _cache[path] = (now, data)
    return data


def get_detail_raw(fixture_id: int) -> dict:
    return _get(f"/fixtures/{fixture_id}/detail")


def _norm_item(fx: dict, summary: dict | None = None) -> dict | None:
    """Chuẩn hóa 1 trận (kèm phút thật + phạt góc/thẻ).
    fx: object có các trường id/status_short/home_team... (đã bóc khỏi 'fixture' nếu cần)."""
    if not fx:
        return None
    # nếu lỡ truyền cả object {fixture, summary} thì bóc ra
    if "fixture" in fx and isinstance(fx["fixture"], dict):
        summary = summary or fx.get("summary")
        fx = fx["fixture"]
    h, a = fx.get("home_team", {}), fx.get("away_team", {})
    summ = summary or fx.get("summary", {}) or {}
    return {
        "fixture_id": fx.get("id"),
        "league": (fx.get("league") or {}).get("name", ""),
        "league_id": fx.get("league_id"),
        "status": fx.get("status_short", ""),
        "status_long": fx.get("status_long", ""),
        "minute": fx.get("status_elapsed") or 0,
        "date": fx.get("date"),
        "home": {"id": h.get("id"), "name": h.get("name"), "logo": h.get("logo", "")},
        "away": {"id": a.get("id"), "name": a.get("name"), "logo": a.get("logo", "")},
        "goals": {"home": fx.get("goals_home"), "away": fx.get("goals_away")},
        "halftime": {"home": fx.get("score_halftime_home"), "away": fx.get("score_halftime_away")},
        "corners": {"home": summ.get("homeCorners", 0), "away": summ.get("awayCorners", 0)},
        "cards": {"home_yellow": summ.get("homeYellow", 0), "home_red": summ.get("homeRed", 0),
                  "away_yellow": summ.get("awayYellow", 0), "away_red": summ.get("awayRed", 0)},
    }


def parse_odds(odds: dict) -> dict | None:
    """Bóc KÈO THẬT từ thethaoviet: 1×2 (đồng thuận, khử biên) + bảng chấp/Tài-Xỉu."""
    if not odds:
        return None
    mw = odds.get("match_winner") or []
    ah = odds.get("asian_handicap") or []
    ou = odds.get("over_under") or []

    # 1X2: gộp theo nhà cái -> khử biên từng nhà -> lấy trung bình (đồng thuận)
    books = {}
    for o in mw:
        try:
            books.setdefault(o["bookmaker_name"], {})[o["value_name"].lower()] = float(o["odd_value"])
        except (ValueError, TypeError, KeyError):
            continue
    probs_list, h2h_table = [], []
    for b, od in books.items():
        if {"home", "draw", "away"} <= set(od):
            inv = {k: 1 / od[k] for k in ("home", "draw", "away")}
            s = sum(inv.values())
            probs_list.append({k: inv[k] / s for k in inv})
            h2h_table.append({"book": b, "home": od["home"], "draw": od["draw"], "away": od["away"]})
    market_probs = None
    if probs_list:
        n = len(probs_list)
        market_probs = {k: round(sum(p[k] for p in probs_list) / n, 4)
                        for k in ("home", "draw", "away")}

    def pair_table(arr, k1, k2):
        out = {}
        for o in arr:
            b = o.get("bookmaker_name")
            out.setdefault(b, {"book": b, "line": o.get("handicap")})
            try:
                out[b][o["value_name"].lower()] = float(o["odd_value"])
            except (ValueError, TypeError, KeyError):
                pass
        return [r for r in out.values() if k1 in r and k2 in r]

    return {
        "market_probs": market_probs,
        "n_books": len(h2h_table),
        "h2h": h2h_table,
        "ah": pair_table(ah, "home", "away"),       # mỗi dòng: book, line, home, away
        "ou": pair_table(ou, "over", "under"),       # mỗi dòng: book, line, over, under
    }


def normalize(raw: dict) -> dict | None:
    """Cho endpoint detail: data là dict {fixture, odds, summary} hoặc mảng."""
    data = raw.get("data")
    if isinstance(data, list):
        data = data[0] if data else None
    if not data:
        return None
    if isinstance(data, dict) and "fixture" in data:
        m = _norm_item(data["fixture"], data.get("summary"))
        if m:
            m["odds"] = parse_odds(data.get("odds") or {})
        return m
    return _norm_item(data)


def get_detail(fixture_id: int) -> dict | None:
    return normalize(get_detail_raw(fixture_id))


def get_fixtures_by_date(date_iso: str) -> list[dict]:
    """Danh sách trận theo ngày YYYY-MM-DD. Thử vài kiểu URL phổ biến."""
    candidates = [f"/fixtures/by-date?date={date_iso}",
                  f"/fixtures?date={date_iso}", f"/fixtures/date/{date_iso}"]
    last_err = None
    for path in candidates:
        try:
            raw = _get(path)
        except Exception as e:
            last_err = e
            continue
        data = raw.get("data") if isinstance(raw, dict) else raw
        if isinstance(data, list) and data:
            return [m for m in (_norm_item(x) for x in data) if m]
    if last_err:
        raise last_err
    return []


LIVE_STATUS = {"1H", "2H", "HT", "ET", "BT", "P", "LIVE", "INT", "SUSP"}


def get_live(date_iso: str | None = None) -> list[dict]:
    """Trận ĐANG ĐÁ: lấy danh sách hôm nay rồi lọc theo trạng thái live."""
    from datetime import date as _d
    day = date_iso or _d.today().isoformat()
    try:
        allm = get_fixtures_by_date(day)
    except Exception:
        return []
    return [m for m in allm if str(m.get("status", "")).upper() in LIVE_STATUS]
