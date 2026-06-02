"""
FastAPI backend cho tool dự đoán bóng đá.
Chạy:  uvicorn backend.main:app --reload   (từ thư mục gốc project)
Docs tự động: http://127.0.0.1:8000/docs
"""
from datetime import date as _date
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import os
from dotenv import load_dotenv
load_dotenv()  # PHẢI nạp .env TRƯỚC khi đọc DATA_PROVIDER

import time
from . import predictor, engine, mlmodel, analytics, tracker, oddsapi_client, thethaoviet_client

# Chọn nguồn dữ liệu: "footballdata" | "sofascore" | "apifootball"
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "apifootball").lower()
if DATA_PROVIDER == "footballdata":
    from . import footballdata_client as provider
elif DATA_PROVIDER == "sofascore":
    from . import sofascore_client as provider
else:
    from . import api_client as provider

app = FastAPI(title="Football Predictor API")

# Cho phép frontend gọi (mở rộng theo nhu cầu)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _slim_fixture(fx: dict) -> dict:
    """Rút gọn dữ liệu trận API-Football để gửi cho frontend."""
    return {
        "fixture_id": fx["fixture"]["id"],
        "custom_id": None,
        "date": fx["fixture"]["date"],
        "status": fx["fixture"]["status"]["short"],
        "league": fx["league"]["name"],
        "home": {"id": fx["teams"]["home"]["id"], "name": fx["teams"]["home"]["name"],
                 "logo": fx["teams"]["home"]["logo"]},
        "away": {"id": fx["teams"]["away"]["id"], "name": fx["teams"]["away"]["name"],
                 "logo": fx["teams"]["away"]["logo"]},
        "goals": fx.get("goals", {}),
    }


def _to_slim(raw: list[dict]) -> list[dict]:
    """footballdata/sofascore đã trả sẵn slim; chỉ API-Football cần rút gọn."""
    if DATA_PROVIDER in ("footballdata", "sofascore"):
        return raw
    return [_slim_fixture(f) for f in raw]


@app.get("/api/matches")
def matches(live: bool = Query(False), date: str | None = None):
    """Danh sách trận: live=true cho trận đang đá, hoặc theo ngày YYYY-MM-DD."""
    try:
        if live:
            raw = provider.get_live_fixtures()
        else:
            raw = provider.get_fixtures_by_date(date or _date.today().isoformat())
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"count": len(raw), "matches": _to_slim(raw)}


# ---------------- LỚP ENGINE: fit theo giải, có cache ----------------
_engine_cache: dict[str, tuple[float, dict]] = {}
ENGINE_TTL = 600  # 10 phút


def _season(code: str) -> dict:
    """Lấy cả mùa giải, fit ratings + train ML + tên đội. Cache 10 phút."""
    now = time.time()
    if code in _engine_cache and now - _engine_cache[code][0] < ENGINE_TTL:
        return _engine_cache[code][1]
    if not hasattr(provider, "get_competition_matches"):
        raise HTTPException(status_code=400, detail="Nguồn dữ liệu không hỗ trợ engine theo giải.")
    matches = provider.get_competition_matches(code)
    ratings = engine.fit_ratings(matches)
    names = {}
    for m in matches:
        if m.get("home", {}).get("id"):
            names[m["home"]["id"]] = m["home"]["name"]
        if m.get("away", {}).get("id"):
            names[m["away"]["id"]] = m["away"]["name"]
    model = mlmodel.train(matches, ratings)
    data = {"matches": matches, "ratings": ratings, "names": names, "model": model}
    _engine_cache[code] = (now, data)
    return data


def _blend(*sources):
    weights = [0.5, 0.3, 0.2]  # DC, ML, H2H
    acc = {"home": 0.0, "draw": 0.0, "away": 0.0}
    uw = 0.0
    for s, w in zip(sources, weights):
        if not s:
            continue
        for k in acc:
            acc[k] += s[k] * w
        uw += w
    if uw == 0:
        return {"home": 1/3, "draw": 1/3, "away": 1/3}
    return {k: v / uw for k, v in acc.items()}


@app.get("/api/predict/{fixture_id}")
def predict_fixture(request: Request, fixture_id: int, home_id: int, away_id: int,
                    code: str | None = None, custom_id: str | None = None):
    """Dự đoán dùng ENGINE (Dixon-Coles + ML + Monte Carlo) fit từ cả mùa giải `code`.
    Nếu không có code, quay về model cũ (last-10 qua API).
    PAYWALL: user phải trả điểm; admin xem free; user mua rồi → xem lại free."""
    _charge_match(request, fixture_id)
    # --- ENGINE mode (khuyên dùng, chỉ tốn 1 request mùa giải đã cache) ---
    if code and hasattr(provider, "get_competition_matches"):
        try:
            s = _season(code)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))
        ratings = s["ratings"]
        lh, la = engine.expected_goals(ratings, home_id, away_id)
        mat = engine.dc_matrix(lh, la)
        ana = engine.analyse_matrix(mat)
        p_dc = ana["probs"]
        p_ml = mlmodel.predict_proba(s["model"], ratings, home_id, away_id)
        # h2h từ chính mùa giải (các trận 2 đội gặp nhau)
        h2h_games = [m for m in s["matches"]
                     if {m.get("home", {}).get("id"), m.get("away", {}).get("id")} == {home_id, away_id}]
        p_h2h = predictor.h2h_probs(
            [engine_to_predictor(m) for m in h2h_games], home_id) if h2h_games else None
        final = _blend(p_dc, p_ml, p_h2h)

        # --- KÈO THẬT từ the-odds-api: blend + tính value ---
        hn = s["names"].get(home_id, ""); an = s["names"].get(away_id, "")
        p_market = market_tot = None; value = None
        try:
            events = oddsapi_client.get_events(code)
            ev = oddsapi_client.match_event(events, hn, an) if events else None
            if ev:
                p_market = oddsapi_client.market_probs(ev, hn, an)
                market_tot = oddsapi_client.market_totals(ev)
        except Exception:
            p_market = None
        n_hist = ratings.get("n", 0)
        model_informed = n_hist >= 30 and p_ml is not None
        if p_market:
            mk_probs = {k: p_market[k] for k in ("home", "draw", "away")}
            model_only = dict(final)
            # nếu model có đủ dữ liệu: blend 55% thị trường. Nếu thiếu (vd World Cup): tin kèo 90%.
            w_mkt = 0.55 if model_informed else 0.90
            final = {k: w_mkt * mk_probs[k] + (1 - w_mkt) * model_only[k] for k in mk_probs}
            sm = sum(final.values()); final = {k: v / sm for k, v in final.items()}
            # value CHỈ có ý nghĩa khi model đủ dữ liệu để tin
            value = {k: round(model_only[k] - mk_probs[k], 4) for k in mk_probs} if model_informed else None

        pick = max(final, key=final.get)
        market = {"top_scores": ana["top_scores"], "over_2_5": ana["over_2_5"],
                  "under_2_5": ana["under_2_5"], "btts_yes": ana["btts_yes"],
                  "btts_no": ana["btts_no"], "total_xg": round(lh + la, 2)}
        mc = engine.monte_carlo(lh, la, 10000)
        tier = predictor._confidence_tier(final[pick])
        grid = [[round(mat[i][j], 4) for j in range(6)] for i in range(6)]  # heatmap 0-5
        bets = engine.betting_lines(lh, la, mat)
        market["over_under"] = bets["over_under"]
        market["handicap"] = bets["handicap"]
        market["fair_handicap"] = bets["fair_handicap"]
        market["ou_pick"] = bets["ou_pick"]
        market["ah_pick"] = bets["ah_pick"]
        market["corner"] = engine.corner_pick(lh + la, 0, 0)
        verdict = predictor.build_verdict(final, market, lh, la)
        ahp, oup = bets["ah_pick"], bets["ou_pick"]
        verdict.append(f"Kèo chấp: {ahp['team']} chấp {ahp['line']:+g} — thắng kèo {ahp['cover']*100:.1f}%.")
        verdict.append(f"Tài/Xỉu {oup['line']}: nghiêng {oup['side']} — {oup['prob']*100:.1f}%.")
        if p_market:
            verdict.append(f"Kèo thật ({p_market.get('bookmaker','?')}): "
                           f"N {p_market['home']*100:.0f}% / H {p_market['draw']*100:.0f}% / "
                           f"K {p_market['away']*100:.0f}% — đã blend vào kết quả.")
            if value:
                vk = max(value, key=value.get)
                if value[vk] > 0.05:
                    lab = {"home": "Đội nhà", "draw": "Hòa", "away": "Đội khách"}[vk]
                    verdict.append(f"⚠ VALUE: model đánh giá {lab} cao hơn nhà cái {value[vk]*100:.1f}% "
                                   f"(chỗ có thể có giá trị — vẫn rủi ro).")
            else:
                verdict.append("Lưu ý: giải này model thiếu dữ liệu phong độ nên ưu tiên kèo nhà cái; "
                               "value tạm ẩn (không đáng tin).")
        else:
            verdict.append("Kèo thật: chưa lấy được (chưa có ODDS_API_KEY hoặc giải/trận không có kèo) "
                           "— đang dùng 100% xác suất mô hình.")
        resp = {
            "fixture_id": fixture_id, "engine": "dixon-coles",
            "expected_goals": {"home": lh, "away": la},
            "probabilities": {k: round(v, 4) for k, v in final.items()},
            "sources": {
                "dixon_coles": {k: round(v, 4) for k, v in p_dc.items()},
                "ml": {k: round(v, 4) for k, v in p_ml.items()} if p_ml else None,
                "h2h": {k: round(v, 4) for k, v in p_h2h.items()} if p_h2h else None,
                "market": {k: p_market[k] for k in ("home", "draw", "away")} if p_market else None,
            },
            "market_odds": ({"bookmaker": p_market.get("bookmaker"), "odds": p_market.get("raw_odds")}
                            if p_market else None),
            "market_totals": market_tot,
            "value": value,
            "model_informed": model_informed,
            "monte_carlo": mc,
            "prediction": pick,
            "prediction_label": {"home": "Đội nhà thắng", "draw": "Hòa", "away": "Đội khách thắng"}[pick],
            "confidence": round(final[pick], 4),
            "confidence_tier": tier,
            "market": market,
            "verdict": verdict,
            "ratings": {"home": ratings["teams"].get(home_id), "away": ratings["teams"].get(away_id)},
            "score_grid": grid,
        }
        # tự ghi sổ track record — CHỈ với trận chưa đá (không ghi hồi tố)
        meta = next((m for m in s["matches"] if m.get("fixture_id") == fixture_id), None)
        finished = meta and str(meta.get("status", "")).upper() in ("FINISHED", "AWARDED")
        if meta and not finished:
            tracker.log_prediction({
                "fixture_id": fixture_id, "code": code,
                "home": s["names"].get(home_id, str(home_id)),
                "away": s["names"].get(away_id, str(away_id)),
                "date": meta.get("date"),
                "pick": pick, "probs": {k: round(v, 4) for k, v in final.items()},
                "confidence": round(final[pick], 4), "confidence_tier": tier,
            })
        return resp

    # --- Fallback: model cũ qua API ---
    try:
        home_matches = provider.get_team_last_matches(home_id, last=10)
        away_matches = provider.get_team_last_matches(away_id, last=10)
        odds = provider.get_odds(fixture_id) if hasattr(provider, "get_odds") else None
        h2h = provider.get_h2h(custom_id) if (custom_id and hasattr(provider, "get_h2h")) else None
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    result = predictor.predict(home_matches, away_matches, home_id, away_id, odds, h2h)
    result["fixture_id"] = fixture_id
    return result


def engine_to_predictor(m: dict) -> dict:
    """Đổi slim match -> shape predictor (cho h2h_probs)."""
    return {"goals": m.get("goals", {}),
            "teams": {"home": {"id": m.get("home", {}).get("id")},
                      "away": {"id": m.get("away", {}).get("id")}}}


def _charge_match(request: Request, fixture_id: int):
    """Helper paywall — trừ điểm user (admin free, đã mua → free).
    Raise HTTPException(402) nếu thiếu điểm. Trả về (cost, paid_now) để log nếu cần."""
    try:
        from . import users as _udb
        user = getattr(request.state, "user", None)
        if not user or user["role"] == "admin":
            return 0, False
        if _udb.has_paid(user["id"], fixture_id):
            return 0, False
        cost = _udb.match_cost(fixture_id)
        ok, msg, _bal = _udb.mark_paid(user["id"], fixture_id, cost)
        if not ok:
            raise HTTPException(status_code=402, detail=msg)
        return cost, True
    except HTTPException:
        raise
    except Exception as _e:
        print(f"[paywall] warning: {_e}")
        return 0, False


@app.get("/api/live_predict/{fixture_id}")
def live_predict(request: Request, fixture_id: int, home_id: int, away_id: int,
                 gh: int = 0, ga: int = 0, minute: int = 0, code: str | None = None):
    """Xác suất IN-PLAY: cập nhật theo tỉ số hiện tại (gh-ga) và phút đã đá.
    Gọi lại mỗi phút để có phân tích liên tục."""
    _charge_match(request, fixture_id)
    # bàn kỳ vọng cả trận: dùng chỉ số giải nếu có, không thì mặc định
    lh, la = 1.35, 1.15
    if code:
        try:
            r = _season(code)["ratings"]
            if r.get("teams"):
                lh, la = engine.expected_goals(r, home_id, away_id)
        except Exception:
            pass
    live = engine.live_inplay(lh, la, gh, ga, minute)
    pick = max(live["probs"], key=live["probs"].get)
    live["fixture_id"] = fixture_id
    live["prediction"] = pick
    live["prediction_label"] = {"home": "Đội nhà thắng", "draw": "Hòa", "away": "Đội khách thắng"}[pick]
    live["confidence"] = live["probs"][pick]
    # nhận định nhanh theo tình huống
    diff = gh - ga
    note = []
    if minute >= 80 and diff != 0:
        note.append("Cuối trận, tỉ số chênh lệch → kết quả gần như định đoạt.")
    elif minute >= 80 and diff == 0:
        note.append("Cuối trận hòa → khả năng cao chia điểm.")
    elif diff == 0:
        note.append("Đang hòa → còn nhiều thời gian, kèo mở.")
    else:
        lead = "Đội nhà" if diff > 0 else "Đội khách"
        note.append(f"{lead} đang dẫn, còn {90-minute}' → lợi thế nghiêng về {lead.lower()}.")
    live["note"] = note
    return live


@app.get("/api/tv_raw")
def tv_raw(path: str = "/fixtures"):
    """Dò endpoint thethaoviet: gọi thẳng path bạn nhập và trả raw JSON.
    VD: /api/tv_raw?path=/fixtures?date=2026-06-01  hoặc  /api/tv_raw?path=/livescore"""
    try:
        raw = thethaoviet_client._get(path)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    keys = list(raw.keys()) if isinstance(raw, dict) else "list"
    data = raw.get("data") if isinstance(raw, dict) else raw
    sample = None
    if isinstance(data, list) and data:
        sample = data[0]
    elif isinstance(data, dict):
        sample = {k: data[k] for k in list(data)[:8]}
    return {"path": path, "top_keys": keys,
            "data_type": type(data).__name__,
            "data_len": len(data) if isinstance(data, (list, dict)) else None,
            "first_item": sample}


@app.get("/api/tv_matches")
def tv_matches(date: str | None = None, live: bool = False):
    """Danh sách trận từ thethaoviet (theo ngày hoặc đang đá) — để hiện hết, khỏi gõ ID."""
    try:
        if live:
            data = thethaoviet_client.get_live()
        else:
            data = thethaoviet_client.get_fixtures_by_date(date or _date.today().isoformat())
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Không gọi được thethaoviet: {e}")
    return {"count": len(data), "matches": data}


@app.get("/api/tv_debug/{fixture_id}")
def tv_debug(fixture_id: int):
    """DEBUG: trả raw response + parsed result để check field mapping."""
    try:
        raw = thethaoviet_client.get_detail_raw(fixture_id)
        data = raw.get("data")
        if isinstance(data, list): data = data[0] if data else {}
        if not isinstance(data, dict): data = {}

        # Cũng test _norm_item parse
        parsed = thethaoviet_client.get_detail(fixture_id)

        return {
            "ok": True,
            "raw_summary": data.get("summary", {}),
            "raw_odds_count": {
                "match_winner": len((data.get("odds") or {}).get("match_winner", [])),
                "asian_handicap": len((data.get("odds") or {}).get("asian_handicap", [])),
                "over_under": len((data.get("odds") or {}).get("over_under", [])),
            },
            "fixture_status": (data.get("fixture") or {}).get("status_short"),
            "PARSED_corners": parsed.get("corners") if parsed else None,
            "PARSED_cards": parsed.get("cards") if parsed else None,
            "PARSED_goals": parsed.get("goals") if parsed else None,
            "PARSED_odds_books": (parsed.get("odds") or {}).get("n_books") if parsed else None,
            "sample_h2h": (parsed.get("odds") or {}).get("h2h", [])[:3] if parsed else None,
        }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()[-500:]}


def _pick_status(side: str, line: float, current: float, minute: int, finished: bool,
                 exp_remaining: float = 0.0) -> dict:
    """Cho 1 pick (TÀI/XỈU/NHÀ/KHÁCH), line, dữ liệu hiện tại → trạng thái.
    Trả: {label, color, hint}."""
    side = (side or "").upper()
    is_over = side in ("TÀI", "OVER", "T", "NHÀ", "HOME")
    is_under = side in ("XỈU", "UNDER", "U", "KHÁCH", "AWAY")
    line = float(line) if line is not None else 0
    cur = float(current)

    # Đã kết thúc → trúng/trật
    if finished:
        if cur > line:
            if is_over:
                return {"label": "✓ ĐÚNG", "color": "neon", "hint": f"Kết quả {cur} > {line} → TÀI thắng"}
            return {"label": "✗ SAI", "color": "bad", "hint": f"Kết quả {cur} > {line} → TÀI thắng (pick XỈU sai)"}
        if cur < line:
            if is_under:
                return {"label": "✓ ĐÚNG", "color": "neon", "hint": f"Kết quả {cur} < {line} → XỈU thắng"}
            return {"label": "✗ SAI", "color": "bad", "hint": f"Kết quả {cur} < {line} → XỈU thắng (pick TÀI sai)"}
        return {"label": "= HÒA VỐN", "color": "dim", "hint": f"Đúng line {line}"}

    # Còn đá: dự đoán cuối trận
    expected_final = cur + exp_remaining
    # TÀI: nếu expected > line → đang theo
    if is_over:
        if cur > line:
            return {"label": "✓ ĐÃ ĐỦ", "color": "neon", "hint": f"Hiện {cur} > {line} → TÀI gần chắc thắng"}
        if expected_final > line + 0.5:
            return {"label": "▶ ĐANG THEO", "color": "neon", "hint": f"Hiện {cur:.0f} phút {minute}', dự kiến cuối ~{expected_final:.1f} > {line}"}
        if expected_final < line - 0.5:
            return {"label": "⚠ ĐANG LỆCH", "color": "bad", "hint": f"Dự kiến chỉ {expected_final:.1f} < {line} → nên chuyển sang XỈU"}
        return {"label": "⚖ 50/50", "color": "warn", "hint": f"Dự kiến cuối ~{expected_final:.1f} ≈ line {line}"}
    if is_under:
        if cur >= line:
            return {"label": "✗ ĐÃ THUA", "color": "bad", "hint": f"Hiện {cur} ≥ {line} → XỈU thua"}
        if expected_final < line - 0.5:
            return {"label": "▶ ĐANG THEO", "color": "neon", "hint": f"Dự kiến cuối ~{expected_final:.1f} < {line}"}
        if expected_final > line + 0.5:
            return {"label": "⚠ ĐANG LỆCH", "color": "bad", "hint": f"Dự kiến ~{expected_final:.1f} > {line} → nên chuyển sang TÀI"}
        return {"label": "⚖ 50/50", "color": "warn", "hint": f"Dự kiến ~{expected_final:.1f} ≈ line {line}"}
    return {"label": "?", "color": "dim", "hint": ""}


@app.get("/api/tv_live/{fixture_id}")
def tv_live(request: Request, fixture_id: int):
    """Phân tích IN-PLAY theo PHÚT THẬT từ thethaoviet.vip (gọi lại mỗi phút).
    PAYWALL: trừ điểm user (admin free, đã mua xem lại free)."""
    _charge_match(request, fixture_id)
    try:
        m = thethaoviet_client.get_detail(fixture_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Không gọi được thethaoviet: {e}")
    if not m:
        raise HTTPException(status_code=404, detail="Không có dữ liệu trận này.")
    gh = m["goals"]["home"] or 0
    ga = m["goals"]["away"] or 0
    minute = m["minute"] or 0
    lh, la = 1.35, 1.15  # bàn kỳ vọng trung bình giải (nguồn này chưa có phong độ)

    # ma trận TỈ SỐ CUỐI dựa trên tỉ số hiện tại + thời gian còn lại
    M, rem_h, rem_a = engine.live_matrix(lh, la, gh, ga, minute)
    ana = engine.analyse_matrix(M)
    final = ana["probs"]

    # KÈO THẬT từ thethaoviet
    odds = m.get("odds")
    mp = odds.get("market_probs") if odds else None
    value = None  # nguồn này chưa có phong độ -> model không đủ tin để tính value
    if mp and minute == 0:   # chưa đá: tin kèo nhà cái (đồng thuận) gần như hoàn toàn
        model_only = dict(final)
        final = {k: 0.85 * mp[k] + 0.15 * model_only[k] for k in mp}
        sm = sum(final.values()); final = {k: v / sm for k, v in final.items()}
    # đang đá: giữ mô hình in-play (kèo nhà cái là kèo trước trận, đã cũ) — chỉ hiện tham khảo

    pick = max(final, key=final.get)
    lh_eff, la_eff = gh + rem_h, ga + rem_a
    bets = engine.betting_lines(lh_eff, la_eff, M)
    mc = engine.monte_carlo_live(lh, la, gh, ga, minute, 10000)
    tier = predictor._confidence_tier(final[pick])
    grid = [[round(M[i][j], 4) for j in range(6)] for i in range(6)]
    market = {"top_scores": ana["top_scores"], "over_2_5": ana["over_2_5"],
              "under_2_5": ana["under_2_5"], "btts_yes": ana["btts_yes"],
              "btts_no": ana["btts_no"], "total_xg": round(lh_eff + la_eff, 2),
              "over_under": bets["over_under"], "handicap": bets["handicap"],
              "fair_handicap": bets["fair_handicap"], "ou_pick": bets["ou_pick"],
              "ah_pick": bets["ah_pick"],
              "corner": engine.corner_pick(lh_eff + la_eff,
                        (m["corners"]["home"] or 0) + (m["corners"]["away"] or 0), minute)}

    # ========== PRE-MATCH PICKS + VERIFICATION ==========
    # Pick ban đầu (phút 0, 0-0) — model dự đoán trước trận
    M_pre, _, _ = engine.live_matrix(lh, la, 0, 0, 0)
    bets_pre = engine.betting_lines(lh, la, M_pre)

    # ⭐ LINE THẬT TỪ BOOKMAKER (consensus median) — thay vì 2.5/0 hardcoded
    real_ou_line = engine.consensus_line(odds.get("ou", []) if odds else [], "line")
    real_ah_line = engine.consensus_line(odds.get("ah", []) if odds else [], "line")
    # Nếu API có data → dùng line thật; không thì giữ default từ engine
    if real_ou_line is not None:
        ou_real_pre = engine.over_under_at_line(M_pre, real_ou_line)
        ou_real_live = engine.over_under_at_line(M, real_ou_line)
        bets_pre["ou_pick"] = ou_real_pre
        bets["ou_pick"] = ou_real_live
        market["ou_pick"] = ou_real_live
    if real_ah_line is not None:
        ah_real_pre = engine.asian_handicap_at_line(M_pre, real_ah_line)
        ah_real_live = engine.asian_handicap_at_line(M, real_ah_line)
        # Map về schema cũ {team, side, line, cover}
        for o in (ah_real_pre, ah_real_live):
            o["team"] = "Đội nhà" if o["side"] == "home" else "Đội khách"
        bets_pre["ah_pick"] = ah_real_pre
        bets["ah_pick"] = ah_real_live
        market["ah_pick"] = ah_real_live

    # Corner: API thethaoviet không có corner odds → dùng heuristic engine.corner_pick
    corner_pre = engine.corner_pick(lh + la, 0, 0)

    # Trạng thái hiện tại
    cur_corners_total = (m["corners"]["home"] or 0) + (m["corners"]["away"] or 0)
    cur_goals_total = gh + ga
    score_diff = gh - ga
    finished = (m["status"] or "").upper() in ("FT", "AET", "PEN")

    # Verification 3 picks
    # 1. Corner T/X
    corner_line_pre = corner_pre.get("line", 9.5)
    corner_side_pre = corner_pre.get("pick", "TÀI")
    # góc kỳ vọng còn lại
    corner_rem = max(0, (90 - minute) / 90) * max(0, corner_pre.get("exp_corners", 10) - cur_corners_total)
    corner_status = _pick_status(corner_side_pre, corner_line_pre, cur_corners_total, minute, finished, exp_remaining=corner_rem)

    # 2. AH (cược chấp) — pick theo home/away
    ah_pre = bets_pre.get("ah_pick", {})
    ah_line = ah_pre.get("line", 0)
    ah_side = "NHÀ" if ah_pre.get("side") == "home" else ("KHÁCH" if ah_pre.get("side") == "away" else "?")
    # Adjusted diff cho pick:
    # nếu pick NHÀ chấp -1.5 → diff_eff = score_diff + (-1.5). Cộng kỳ vọng còn lại
    if ah_pre.get("side") == "home":
        adj_now = score_diff + ah_line
        adj_rem = rem_h - rem_a
    else:
        adj_now = -score_diff - ah_line
        adj_rem = rem_a - rem_h
    ah_status = _pick_status(
        "TÀI" if adj_now + adj_rem > 0 else "XỈU",  # bị quên: chỉ cần coverage
        0,  # threshold
        adj_now, minute, finished, exp_remaining=adj_rem
    )
    # Reuse with cover prob
    if finished:
        ah_status = {
            "label": "✓ ĐÚNG" if adj_now > 0 else ("✗ SAI" if adj_now < 0 else "= HÒA VỐN"),
            "color": "neon" if adj_now > 0 else ("bad" if adj_now < 0 else "dim"),
            "hint": f"Tỉ số {gh}-{ga} → kèo chấp {'thắng' if adj_now > 0 else ('thua' if adj_now < 0 else 'hoà')}"
        }
    else:
        adj_final = adj_now + adj_rem
        if adj_now > 0:
            ah_status = {"label": "▶ ĐANG THEO", "color": "neon", "hint": f"Hiện cộng {ah_line:+g}: {adj_now:+.1f} → kèo đang thắng"}
        elif adj_final > 0.5:
            ah_status = {"label": "▶ ĐANG THEO", "color": "neon", "hint": f"Dự kiến cuối cộng line: {adj_final:+.1f} > 0"}
        elif adj_final < -0.5:
            ah_status = {"label": "⚠ ĐANG LỆCH", "color": "bad", "hint": f"Dự kiến {adj_final:+.1f} < 0 → nên chuyển cửa ngược"}
        else:
            ah_status = {"label": "⚖ 50/50", "color": "warn", "hint": f"Dự kiến gần line {adj_final:+.1f}"}

    # 3. O/U bàn thắng
    ou_pre = bets_pre.get("ou_pick", {})
    ou_line = ou_pre.get("line", 2.5)
    ou_side = "TÀI" if ou_pre.get("side", "").lower() in ("tài","over","t") else "XỈU"
    ou_rem = max(0, rem_h + rem_a)
    ou_status = _pick_status(ou_side, ou_line, cur_goals_total, minute, finished, exp_remaining=ou_rem)

    # Picks summary
    picks_tracking = {
        "corner": {
            "pre_match": {"side": corner_side_pre, "line": corner_line_pre, "prob": corner_pre.get("prob"),
                          "exp_final": corner_pre.get("exp_corners")},
            "current": {"actual": cur_corners_total, "minute": minute, "exp_remaining": round(corner_rem, 1)},
            "status": corner_status,
            "live_pick": market["corner"],  # tính lại với data hiện tại
        },
        "ah": {
            "pre_match": {"side": ah_side, "line": ah_line, "cover": ah_pre.get("cover")},
            "current": {"score": f"{gh}-{ga}", "minute": minute, "adj": round(adj_now, 2)},
            "status": ah_status,
            "live_pick": bets["ah_pick"],
        },
        "ou": {
            "pre_match": {"side": ou_side, "line": ou_line, "prob": ou_pre.get("prob")},
            "current": {"actual": cur_goals_total, "minute": minute, "exp_remaining": round(ou_rem, 1)},
            "status": ou_status,
            "live_pick": bets["ou_pick"],
        },
    }
    verdict = predictor.build_verdict(final, market, lh_eff, la_eff)
    diff = gh - ga
    if minute >= 80 and diff != 0:
        verdict.insert(0, f"PHÚT {minute}: tỉ số {gh}-{ga}, sắp hết giờ → kết quả gần như định đoạt.")
    elif diff == 0 and minute > 0:
        verdict.insert(0, f"PHÚT {minute}: đang hòa {gh}-{ga}, còn ~{90-minute}' → kèo còn mở.")
    elif minute > 0:
        lead = "Đội nhà" if diff > 0 else "Đội khách"
        verdict.insert(0, f"PHÚT {minute}: {lead} dẫn {gh}-{ga}, còn ~{90-minute}'.")
    ahp, oup = bets["ah_pick"], bets["ou_pick"]
    verdict.append(f"Kèo chấp: {ahp['team']} chấp {ahp['line']:+g} — thắng kèo {ahp['cover']*100:.1f}%.")
    verdict.append(f"Tài/Xỉu {oup['line']}: nghiêng {oup['side']} — {oup['prob']*100:.1f}%.")
    tc = m["corners"]["home"] + m["corners"]["away"]
    return {
        "fixture_id": fixture_id, "source": "diendanbongda.com (phút thật)", "engine": "in-play",
        "league": m["league"], "status": m["status"], "status_long": m["status_long"],
        "home": m["home"], "away": m["away"], "score": m["goals"], "minute": minute,
        "expected_goals": {"home": round(lh_eff, 2), "away": round(la_eff, 2)},
        "remaining_xg": {"home": rem_h, "away": rem_a},
        "probabilities": {k: round(v, 4) for k, v in final.items()},
        "prediction": pick,
        "prediction_label": {"home": "Đội nhà thắng", "draw": "Hòa", "away": "Đội khách thắng"}[pick],
        "confidence": round(final[pick], 4), "confidence_tier": tier,
        "monte_carlo": mc, "market": market, "score_grid": grid, "verdict": verdict,
        "sources": {"dixon_coles": {k: round(v, 4) for k, v in ana["probs"].items()},
                    "ml": None, "h2h": None,
                    "market": mp if mp else None},
        "value": value,
        "model_informed": False,
        "real_odds": odds,   # kèo thật từ thethaoviet (1×2 + chấp + Tài/Xỉu theo nhà cái)
        "ratings": {"home": None, "away": None},
        "corners": m["corners"], "total_corners": tc, "cards": m["cards"],
        "picks_tracking": picks_tracking,
    }


@app.get("/api/standings")
def standings(code: str = "BSA"):
    """Bảng xếp hạng tính từ các trận đã đá của giải."""
    try:
        s = _season(code)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"code": code, "table": engine.standings(s["matches"], s["names"])}


@app.get("/api/backtest")
def backtest(code: str = "BSA"):
    """Đo độ chính xác: dự đoán lại mọi trận đã đá, so với kết quả thật."""
    try:
        s = _season(code)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"code": code, **mlmodel.backtest(s["matches"], s["ratings"], s["model"])}


@app.get("/api/competition")
def competition_matches(code: str = "BSA", days_back: int = 10, days_ahead: int = 60):
    """Lấy trận theo giải trong khoảng [hôm nay - days_back, hôm nay + days_ahead]."""
    if not hasattr(provider, "get_competition_matches"):
        raise HTTPException(status_code=400, detail="Nguồn dữ liệu hiện tại không hỗ trợ chọn giải.")
    from datetime import timedelta
    today = _date.today()
    df = (today - timedelta(days=days_back)).isoformat()
    dt = (today + timedelta(days=days_ahead)).isoformat()
    try:
        raw = provider.get_competition_matches(code, df, dt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    # sắp xếp theo ngày tăng dần
    raw.sort(key=lambda m: m.get("date") or "")
    return {"count": len(raw), "matches": raw}


@app.get("/api/validate")
def validate(code: str = "BSA"):
    """Kiểm định OUT-OF-SAMPLE (walk-forward) + đường calibration + log-loss."""
    try:
        s = _season(code)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    res = analytics.walk_forward(s["matches"])
    res["code"] = code
    # benchmark vs nhà cái: gói free không có odds -> để khung, ghi rõ
    res["market"] = {"available": False,
                     "note": "Gói free football-data.org không cung cấp odds. "
                             "Cắm nguồn odds (the-odds-api...) vào để so log-loss với thị trường."}
    return res


@app.get("/api/track")
def track():
    """Sổ track record: đối chiếu kết quả các trận đã đá rồi trả thống kê + lịch sử."""
    rows = tracker.recent(9999)
    codes = {r.get("code") for r in rows if r.get("result") is None and r.get("code")}
    results_by_code = {}
    for c in codes:
        try:
            s = _season(c)
            results_by_code[c] = {m["fixture_id"]: (m["goals"]["home"], m["goals"]["away"])
                                  for m in s["matches"]
                                  if m.get("goals", {}).get("home") is not None
                                  and m.get("goals", {}).get("away") is not None}
        except Exception:
            continue
    stats = tracker.reconcile(results_by_code)
    return {"stats": stats, "recent": tracker.recent(40)}


@app.get("/api/odds")
def odds_check(code: str = "PL"):
    """Chẩn đoán the-odds-api: liệt kê trận đang có kèo (chỉ trận sắp đá vài ngày tới)."""
    try:
        events = oddsapi_client.get_events(code)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    if not events:
        return {"code": code, "count": 0,
                "note": "Không có kèo. Lý do thường gặp: chưa có ODDS_API_KEY, "
                        "giải này the-odds-api không hỗ trợ, hoặc không có trận nào trong vài ngày tới. "
                        "the-odds-api CHỈ ra kèo cho trận sắp đá gần (không có cho trận đã đá / xa nhiều tuần)."}
    out = []
    for ev in events[:20]:
        has = any(m.get("key") == "h2h" for b in ev.get("bookmakers", []) for m in b.get("markets", []))
        out.append({"home": ev.get("home_team"), "away": ev.get("away_team"),
                    "time": ev.get("commence_time"), "bookmakers": len(ev.get("bookmakers", [])),
                    "has_h2h": has})
    return {"code": code, "count": len(events), "events": out}


@app.get("/api/odds_sports")
def odds_sports():
    """Xác minh ODDS_API_KEY + liệt kê giải bóng đá đang có dữ liệu."""
    try:
        sports = oddsapi_client.list_sports()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    soccer = [{"key": s["key"], "title": s["title"], "active": s.get("active")}
              for s in sports if s.get("group") == "Soccer"]
    active = [s for s in soccer if s["active"]]
    return {"key_ok": True, "total_soccer": len(soccer),
            "active_soccer_count": len(active), "active_soccer": active}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/debug")
def debug(path: str = "/matches/list-live", params: str = "sport=football"):
    """
    Gọi thẳng 1 endpoint SofaScore và trả về RAW JSON để xem cấu trúc.
    Ví dụ:
      /api/debug?path=/matches/list-live&params=sport=football
      /api/debug?path=/matches/list&params=date=31/05/2026
    """
    if not hasattr(provider, "_get"):
        return {"error": "provider hiện tại không hỗ trợ debug"}
    p = dict(kv.split("=", 1) for kv in params.split("&") if "=" in kv)
    try:
        raw = provider._get(path, p)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    # tóm tắt để dễ đọc
    def shape(node, depth=0):
        if depth > 2:
            return "..."
        if isinstance(node, dict):
            return {k: shape(v, depth + 1) for k, v in list(node.items())[:8]}
        if isinstance(node, list):
            return [shape(node[0], depth + 1)] if node else []
        return type(node).__name__
    return {"top_level_keys": list(raw.keys()) if isinstance(raw, dict) else "not-dict",
            "shape": shape(raw), "raw": raw}


# ==================== AUTH SYSTEM (multi-user + session DB) ====================
# Đăng ký không hỗ trợ — chỉ admin tạo user. Admin đầu tiên auto-bootstrap khi DB rỗng.
# .env tùy chọn:
#   ADMIN_USERNAME=admin   (default 'admin')
#   ADMIN_PASSWORD=...     (default 'admin123' — ĐỔI NGAY khi deploy!)
from fastapi import Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse

from . import users as users_db

# Khởi tạo DB + bootstrap admin lúc app khởi động
users_db.bootstrap_admin()

_COOKIE_NAME = "oracle_session"
# Path mở (không cần login): landing, public API, login flow, assets
_OPEN_PATHS = {"/", "/login", "/auth", "/logout", "/favicon.ico", "/health",
               "/landing.html"}
_OPEN_PREFIXES = ("/api/public/", "/assets/", "/api/tv_debug/")
_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")


def _current_user(request: Request):
    """Lấy user từ session cookie. Trả dict hoặc None."""
    token = request.cookies.get(_COOKIE_NAME)
    return users_db.get_session_user(token) if token else None


def _render_login(error: str = "", status_code: int = 200):
    """Đọc login.html từ frontend/ + inject error nếu có."""
    path = os.path.join(_frontend_dir, "login.html")
    try:
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
    except FileNotFoundError:
        html = "<h1>login.html không tìm thấy</h1>"
    if error:
        html = html.replace("<!--ERR-->", f'<div class="err">{error}</div>')
    else:
        html = html.replace("<!--ERR-->", "")
    return HTMLResponse(html, status_code=status_code)


@app.middleware("http")
async def _auth_gate(request: Request, call_next):
    path = request.url.path
    # Path mở: landing, login, public api, favicon, health → cho qua
    if path in _OPEN_PATHS or any(path.startswith(p) for p in _OPEN_PREFIXES):
        return await call_next(request)
    user = _current_user(request)
    # Admin paths cần role admin
    if path.startswith("/admin") or path.startswith("/api/admin"):
        if not user:
            if path.startswith("/api"):
                return JSONResponse({"detail": "Chưa đăng nhập"}, status_code=401)
            return RedirectResponse("/login")
        if user.get("role") != "admin":
            if path.startswith("/api"):
                return JSONResponse({"detail": "Cần quyền admin"}, status_code=403)
            return HTMLResponse("<h1 style='color:#ff3b5c;font-family:monospace;padding:40px'>403 — Chỉ admin mới truy cập được trang này</h1>", status_code=403)
        # Gắn user vào request để route lấy
        request.state.user = user
        return await call_next(request)
    # Path thường — phải có user
    if not user:
        if path.startswith("/api"):
            return JSONResponse({"detail": "Chưa đăng nhập"}, status_code=401)
        return RedirectResponse("/login")
    request.state.user = user
    return await call_next(request)


# ============ AUTH ROUTES ============

@app.get("/login", response_class=HTMLResponse)
def _login_page(request: Request):
    # Nếu đã đăng nhập → redirect về home
    if _current_user(request):
        return RedirectResponse("/")
    return _render_login()


@app.post("/auth")
async def _auth(
    request: Request,
    username: str = Form(""),
    password: str = Form(""),
):
    user = users_db.verify_user(username.strip(), password)
    if not user:
        return _render_login("Sai tài khoản hoặc mật khẩu", status_code=401)
    ua = request.headers.get("user-agent", "")[:200]
    token, exp = users_db.create_session(user["id"], days=30, user_agent=ua)
    target = "/admin" if user["role"] == "admin" else "/app"
    r = RedirectResponse(target, status_code=303)
    r.set_cookie(
        _COOKIE_NAME, token,
        httponly=True, max_age=30 * 86400, samesite="lax",
        secure=False,  # đổi True nếu chỉ HTTPS
    )
    return r


# ============ ROOT + APP ROUTES ============

@app.get("/", response_class=HTMLResponse)
def _root(request: Request):
    """Trang root: nếu đã đăng nhập → /app; chưa → landing page public."""
    if _current_user(request):
        return RedirectResponse("/app")
    landing = os.path.join(_frontend_dir, "landing.html")
    if os.path.isfile(landing):
        return FileResponse(landing)
    return HTMLResponse("<h1>FOOTBALL ORACLE</h1><a href='/login'>Đăng nhập</a>")


@app.get("/app", response_class=HTMLResponse)
def _app_page():
    """Trang app chính (middleware đã verify auth)."""
    return FileResponse(os.path.join(_frontend_dir, "index.html"))


# ============ PUBLIC API (không cần đăng nhập) ============

@app.get("/api/public/stats")
def _public_stats():
    """Stats cho landing page: accuracy + total predictions + user count."""
    try:
        st = tracker.stats()
        usr = users_db.stats()
        return {
            "accuracy": st.get("accuracy"),
            "total_predictions": st.get("total_logged", 0),
            "resolved": st.get("resolved", 0),
            "pending": st.get("pending", 0),
            "total_users": usr.get("total_users", 0),
        }
    except Exception as e:
        return {"accuracy": None, "total_predictions": 0, "resolved": 0, "pending": 0, "total_users": 0, "error": str(e)}


@app.get("/api/public/leaderboard")
def _public_leaderboard():
    """Top 15 prediction gần đây để show track record (giấu sensitive info)."""
    try:
        rows = tracker.recent(15)
        items = []
        for r in (rows or []):
            items.append({
                "home": r.get("home", ""),
                "away": r.get("away", ""),
                "pick": r.get("pick"),
                "confidence": r.get("confidence"),
                "result": r.get("result"),
                "correct": r.get("correct"),
            })
        return {"items": items}
    except Exception as e:
        return {"items": [], "error": str(e)}


@app.get("/logout")
def _logout(request: Request):
    token = request.cookies.get(_COOKIE_NAME)
    if token:
        users_db.delete_session(token)
    r = RedirectResponse("/login")
    r.delete_cookie(_COOKIE_NAME)
    return r


# ============ USER INFO ============

@app.get("/api/me")
def _me(request: Request):
    """Trả info user đang đăng nhập (bất kỳ role nào) + số điểm."""
    u = request.state.user
    pts = users_db.get_points(u["id"])
    return {"id": u["id"], "username": u["username"], "role": u["role"], "points": pts}


@app.get("/api/match_cost/{fixture_id}")
def _match_cost(request: Request, fixture_id: int):
    """Trả cost xem phân tích + đã thanh toán chưa."""
    u = request.state.user
    cost = users_db.match_cost(fixture_id)
    is_admin = u["role"] == "admin"
    paid = is_admin or users_db.has_paid(u["id"], fixture_id)
    return {
        "fixture_id": fixture_id,
        "cost": cost,
        "paid": paid,
        "is_admin": is_admin,
        "points": users_db.get_points(u["id"]),
    }


# ============ ADMIN PAGE ============

@app.get("/admin", response_class=HTMLResponse)
def _admin_page():
    """Trang quản lý user (middleware đã check admin role)."""
    return FileResponse(os.path.join(_frontend_dir, "admin.html"))


# ============ ADMIN API ============

@app.get("/api/admin/me")
def _admin_me(request: Request):
    u = request.state.user
    return {"id": u["id"], "username": u["username"], "role": u["role"]}


@app.get("/api/admin/stats")
def _admin_stats():
    return users_db.stats()


@app.get("/api/admin/users")
def _admin_users():
    return {"users": users_db.list_users()}


@app.post("/api/admin/users/create")
def _admin_user_create(
    username: str = Form(""),
    password: str = Form(""),
    role: str = Form("user"),
    note: str = Form(""),
):
    ok, msg = users_db.create_user(username.strip(), password, role.strip(), note.strip())
    return {"ok": ok, "msg": msg}


@app.post("/api/admin/users/delete")
def _admin_user_delete(request: Request, id: int = Form(...)):
    # Không cho admin tự xóa chính mình
    if request.state.user["id"] == id:
        return {"ok": False, "msg": "Không thể tự xóa tài khoản đang đăng nhập"}
    users_db.delete_user(id)
    return {"ok": True, "msg": "Đã xóa user"}


@app.post("/api/admin/users/password")
def _admin_user_password(id: int = Form(...), password: str = Form("")):
    ok, msg = users_db.update_password(id, password)
    return {"ok": ok, "msg": msg}


@app.post("/api/admin/users/role")
def _admin_user_role(request: Request, id: int = Form(...), role: str = Form("user")):
    # Không cho admin tự hạ quyền chính mình (tránh khóa cứng)
    if request.state.user["id"] == id and role != "admin":
        return {"ok": False, "msg": "Không thể tự hạ quyền của chính bạn"}
    ok, msg = users_db.update_role(id, role)
    return {"ok": ok, "msg": msg}


@app.post("/api/admin/users/points")
def _admin_user_points(
    id: int = Form(...),
    delta: int = Form(0),
    mode: str = Form("add"),   # "add" hoặc "set"
):
    """Cộng/trừ điểm (mode=add, delta có thể âm) hoặc set giá trị tuyệt đối (mode=set)."""
    if mode == "set":
        ok, msg = users_db.set_points(id, delta)
    else:
        ok, msg = users_db.add_points(id, delta)
    return {"ok": ok, "msg": msg, "points": users_db.get_points(id)}


# ============ BET SLIP TRACKER ============

@app.get("/api/bets/list")
def _bets_list(request: Request, status: str = ""):
    """List bet của user đang đăng nhập."""
    user = request.state.user
    return {"items": users_db.list_bets(user["id"], status=status)}


@app.get("/api/bets/stats")
def _bets_stats(request: Request):
    """Stats ROI / win rate / by league / by type."""
    user = request.state.user
    return users_db.bets_stats(user["id"])


@app.post("/api/bets/create")
async def _bets_create(request: Request):
    """Tạo bet mới. Body JSON với: fixture_id, home_team, away_team, league,
    bet_type, pick, line, stake, odd, note."""
    user = request.state.user
    try:
        data = await request.json()
    except Exception:
        # Hỗ trợ Form data fallback
        form = await request.form()
        data = dict(form)
    ok, msg, bid = users_db.create_bet(user["id"], data)
    return {"ok": ok, "msg": msg, "id": bid}


@app.post("/api/bets/{bet_id}/settle")
async def _bets_settle(request: Request, bet_id: int, status: str = Form(...)):
    user = request.state.user
    ok, msg = users_db.settle_bet(bet_id, user["id"], status.strip().lower())
    return {"ok": ok, "msg": msg}


@app.post("/api/bets/{bet_id}/update")
async def _bets_update(request: Request, bet_id: int):
    user = request.state.user
    try:
        data = await request.json()
    except Exception:
        form = await request.form()
        data = dict(form)
    ok, msg = users_db.update_bet(bet_id, user["id"], data)
    return {"ok": ok, "msg": msg}


@app.post("/api/bets/{bet_id}/delete")
def _bets_delete(request: Request, bet_id: int):
    user = request.state.user
    users_db.delete_bet(bet_id, user["id"])
    return {"ok": True, "msg": "Đã xóa cược"}


@app.get("/api/bets/timeline")
def _bets_timeline(request: Request):
    """Profit chart data — lãi/lỗ tích lũy theo settled date."""
    user = request.state.user
    return {"points": users_db.bets_timeline(user["id"])}


@app.get("/api/bets/export")
def _bets_export(request: Request):
    """Export CSV sổ cược."""
    from fastapi.responses import Response
    user = request.state.user
    csv = users_db.bets_export_csv(user["id"])
    filename = f"bets_{user['username']}_{int(time.time())}.csv"
    return Response(
        content="﻿" + csv,  # BOM cho Excel mở UTF-8 đúng
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/bets/bankroll")
def _bets_bankroll(request: Request, value: float = Form(...)):
    """Set bankroll (vốn ban đầu)."""
    user = request.state.user
    ok, msg = users_db.set_bankroll(user["id"], value)
    return {"ok": ok, "msg": msg, "bankroll": users_db.get_bankroll(user["id"])}


@app.get("/api/bets/kelly")
def _bets_kelly(request: Request, prob: float, odd: float, fraction: float = 0.25):
    """Kelly stake suggestion dựa trên bankroll user."""
    user = request.state.user
    br = users_db.get_bankroll(user["id"])
    stake = users_db.kelly_stake(br, prob, odd, fraction)
    return {"bankroll": br, "prob": prob, "odd": odd, "fraction": fraction, "stake": stake}


@app.post("/api/bets/auto_settle")
async def _bets_auto_settle(request: Request):
    """Auto-settle các pending bet có fixture_id — gọi API kết quả thật.
    Body: {fixture_ids: [123, 456]} hoặc rỗng (tự quét tất cả pending)."""
    user = request.state.user
    try:
        body = await request.json()
    except Exception:
        body = {}
    fixture_ids = body.get("fixture_ids") or []
    # Nếu không có fixture_ids, lấy từ pending bets của user
    if not fixture_ids:
        pending = users_db.list_bets(user["id"], status="pending")
        fixture_ids = list(set(b["fixture_id"] for b in pending if b.get("fixture_id")))
    if not fixture_ids:
        return {"ok": True, "updated": [], "msg": "Không có cược nào cần settle"}

    # Lấy kết quả từ thethaoviet_client
    results = {}
    for fid in fixture_ids[:30]:  # limit 30 fixtures/lần
        try:
            m = thethaoviet_client.get_detail(int(fid))
            if not m: continue
            status = (m.get("status") or "").upper()
            finished = status in ("FT", "AET", "PEN")
            home = m.get("goals", {}).get("home")
            away = m.get("goals", {}).get("away")
            if finished and home is not None and away is not None:
                results[int(fid)] = (int(home), int(away), True)
        except Exception:
            continue

    updated = users_db.auto_settle_bets(results)
    return {"ok": True, "updated": updated, "checked": len(fixture_ids), "msg": f"Đã settle {len(updated)} cược"}


# ============ FRONTEND STATIC (đặt CUỐI cùng) ============
# Phục vụ frontend tĩnh — index.html ở "/", các file khác theo path
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
