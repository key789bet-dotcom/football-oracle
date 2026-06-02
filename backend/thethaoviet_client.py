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
ODDS_BASE = "https://api.thethaoviet.vip/api/odds"
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


def get_corner_odds(fixture_id: int) -> list[dict]:
    """Probe các bet_id phổ biến cho PHẠT GÓC TÀI/XỈU và trả bảng:
    [{book, line, over, under}, ...]. Rỗng nếu API không có corner odds cho fixture đó.
    bet_id thường dùng: 45 (API-Football corners), 14, 13, 8 — thử tuần tự."""
    now = time.time()
    ck = f"_corner:{fixture_id}"
    if ck in _cache and now - _cache[ck][0] < CACHE_TTL:
        return _cache[ck][1]
    out_raw = []
    with httpx.Client(timeout=10, headers=_HEADERS) as c:
        for bid in [45, 14, 13, 8, 60, 81, 82, 83]:
            try:
                r = c.get(f"{ODDS_BASE}/prematch", params={"fixture": fixture_id, "bet": bid})
                if r.status_code != 200: continue
                d = r.json().get("data", []) or []
                if not d: continue
                sample = d[0]
                vn = (sample.get("value_name") or "").lower()
                if vn in ("over", "under", "tài", "xỉu", "tai", "xiu") and sample.get("handicap") is not None:
                    out_raw = d
                    break
            except Exception:
                continue
    # parse → bảng book/line/over/under
    table = {}
    for o in out_raw:
        try:
            b = o.get("bookmaker_name")
            ln = o.get("handicap")
            vn = (o.get("value_name") or "").lower()
            key = vn if vn in ("over", "under") else ("over" if vn in ("tài", "tai") else "under")
            table.setdefault((b, ln), {"book": b, "line": ln})[key] = float(o["odd_value"])
        except (ValueError, TypeError, KeyError):
            continue
    result = [r for r in table.values() if "over" in r and "under" in r]
    _cache[ck] = (now, result)
    return result


def _get_odds_full_raw(fixture_id: int, prefer: str = "auto") -> dict:
    """Lấy ĐẦY ĐỦ odds 1×2 + AH + O/U từ endpoint /api/odds/prematch & /live.
    Endpoint này có 10-15 bookmakers (đầy đủ hơn /detail).
    prefer: 'prematch' | 'live' | 'auto' (live trước, rỗng → prematch)."""
    now = time.time()
    ck = f"_odds_full:{fixture_id}:{prefer}"
    if ck in _cache and now - _cache[ck][0] < CACHE_TTL:
        return _cache[ck][1]
    results = {"match_winner": [], "asian_handicap": [], "over_under": []}
    bet_ids = {"match_winner": 1, "asian_handicap": 4, "over_under": 5}
    sources = []
    if prefer == "live":
        sources = ["live", "prematch"]
    elif prefer == "prematch":
        sources = ["prematch"]
    else:
        sources = ["prematch"]  # default: prematch ổn định hơn

    with httpx.Client(timeout=15, headers=_HEADERS) as c:
        for src in sources:
            for key, bid in bet_ids.items():
                if results[key]:  # đã có data, skip
                    continue
                try:
                    url = f"{ODDS_BASE}/{src}"
                    params = {"fixture": fixture_id, "bet": bid} if src == "prematch" else {"bet": bid}
                    r = c.get(url, params=params)
                    if r.status_code != 200:
                        continue
                    data = r.json().get("data", []) or []
                    if src == "live":
                        # Live trả ALL matches → filter
                        data = [o for o in data if o.get("fixture_id") == fixture_id or o.get("fixture") == fixture_id]
                    results[key] = data
                except Exception:
                    continue
    _cache[ck] = (now, results)
    return results


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
    """Cho endpoint detail: data là dict {fixture, odds, summary} hoặc mảng.
    FALLBACK: nếu /detail trả odds rỗng → fetch từ /api/odds/prematch (10+ bookmakers)."""
    data = raw.get("data")
    if isinstance(data, list):
        data = data[0] if data else None
    if not data:
        return None
    if isinstance(data, dict) and "fixture" in data:
        m = _norm_item(data["fixture"], data.get("summary"))
        if m:
            # Parse odds từ /detail trước
            parsed = parse_odds(data.get("odds") or {})
            # Nếu rỗng → fallback sang endpoint /api/odds/prematch (đầy đủ hơn)
            if not parsed or not parsed.get("n_books"):
                try:
                    fid = m.get("fixture_id")
                    status = (m.get("status") or "").upper()
                    prefer = "live" if status in ("1H", "2H", "HT", "ET", "P") else "prematch"
                    full_raw = _get_odds_full_raw(fid, prefer=prefer)
                    parsed = parse_odds(full_raw)
                except Exception as e:
                    print(f"[thethaoviet] odds fallback err: {e}")
            m["odds"] = parsed
        return m
    return _norm_item(data)


def get_stats_raw(fixture_id: int) -> dict:
    """Thử các endpoint stats khác nhau để lấy CORNER/SHOTS/POSSESSION thật.
    /detail's summary chỉ có cho giải lớn → giải nhỏ trả 0. Thử thêm:
    - /fixtures/{id}/statistics
    - /fixtures/{id}/stats
    - /fixtures/{id}/events  (parse events kiểu 'Corner')
    - /matches/{id}/details
    Trả {corners: {home, away}, shots: {...}, found_path: 'path that worked'} hoặc {}."""
    now = time.time()
    ck = f"_stats:{fixture_id}"
    if ck in _cache and now - _cache[ck][0] < CACHE_TTL:
        return _cache[ck][1]

    candidates = [
        f"/fixtures/{fixture_id}/statistics",
        f"/fixtures/{fixture_id}/stats",
        f"/fixtures/{fixture_id}/events",
        f"/matches/{fixture_id}/details",
        f"/matches/{fixture_id}/statistics",
        f"/livescore/{fixture_id}",
    ]
    result = {}
    with httpx.Client(timeout=10, headers=_HEADERS) as c:
        for path in candidates:
            try:
                r = c.get(f"{BASE}{path}")
                if r.status_code != 200: continue
                data = r.json().get("data") if isinstance(r.json(), dict) else r.json()
                if not data: continue
                # Tìm corner trong response (nhiều shape có thể có)
                corner_h = corner_a = None
                # Shape 1: list events có type 'Corner'
                if isinstance(data, list):
                    corner_h = sum(1 for e in data if isinstance(e, dict) and
                                   (e.get("type") or e.get("event_type") or "").lower() == "corner"
                                   and ("home" in str(e.get("team") or e.get("side") or "").lower()))
                    corner_a = sum(1 for e in data if isinstance(e, dict) and
                                   (e.get("type") or e.get("event_type") or "").lower() == "corner"
                                   and ("away" in str(e.get("team") or e.get("side") or "").lower()))
                # Shape 2: dict có 'home'/'away' keys với stats
                if isinstance(data, dict):
                    # Tìm key chứa 'corner'
                    def deep_find(obj, hint):
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                if hint in str(k).lower():
                                    return v
                                found = deep_find(v, hint)
                                if found is not None: return found
                        elif isinstance(obj, list):
                            for item in obj:
                                found = deep_find(item, hint)
                                if found is not None: return found
                        return None
                    c_val = deep_find(data, "corner")
                    if isinstance(c_val, dict):
                        corner_h = c_val.get("home") or c_val.get("h")
                        corner_a = c_val.get("away") or c_val.get("a")
                if corner_h is not None or corner_a is not None:
                    result = {
                        "corners": {"home": int(corner_h or 0), "away": int(corner_a or 0)},
                        "found_path": path,
                    }
                    break
            except Exception as e:
                continue
    _cache[ck] = (now, result)
    return result


def get_detail(fixture_id: int) -> dict | None:
    """Lấy chi tiết trận + fallback stats nếu summary corners trống."""
    m = normalize(get_detail_raw(fixture_id))
    if not m:
        return m
    # Fallback corner: nếu /detail summary cả home+away = 0 (có thể missing data) →
    # thử các endpoint stats khác. Nếu tìm được → override.
    ch = (m.get("corners") or {}).get("home", 0)
    ca = (m.get("corners") or {}).get("away", 0)
    minute = m.get("minute", 0)
    # Chỉ override nếu trận đã đá (minute>5) và current = 0-0 (thường trống data)
    if minute and minute > 5 and ch == 0 and ca == 0:
        try:
            stats = get_stats_raw(fixture_id)
            if stats.get("corners"):
                m["corners"] = stats["corners"]
                m["_corner_source"] = stats.get("found_path", "fallback")
        except Exception as e:
            print(f"[thethaoviet] stats fallback err: {e}")
    return m


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
