"""
FastAPI backend cho tool dự đoán bóng đá.
Chạy:  uvicorn backend.main:app --reload   (từ thư mục gốc project)
Docs tự động: http://127.0.0.1:8000/docs
"""
from datetime import date as _date
from fastapi import FastAPI, HTTPException, Query
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
def predict_fixture(fixture_id: int, home_id: int, away_id: int,
                    code: str | None = None, custom_id: str | None = None):
    """Dự đoán dùng ENGINE (Dixon-Coles + ML + Monte Carlo) fit từ cả mùa giải `code`.
    Nếu không có code, quay về model cũ (last-10 qua API)."""
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


@app.get("/api/live_predict/{fixture_id}")
def live_predict(fixture_id: int, home_id: int, away_id: int,
                 gh: int = 0, ga: int = 0, minute: int = 0, code: str | None = None):
    """Xác suất IN-PLAY: cập nhật theo tỉ số hiện tại (gh-ga) và phút đã đá.
    Gọi lại mỗi phút để có phân tích liên tục."""
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


@app.get("/api/tv_live/{fixture_id}")
def tv_live(fixture_id: int):
    """Phân tích IN-PLAY theo PHÚT THẬT từ thethaoviet.vip (gọi lại mỗi phút)."""
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


# ==================== ĐĂNG NHẬP (khóa trang cho riêng mình) ====================
# Đặt SITE_PASSWORD trong .env để bật khóa. Không đặt -> trang mở bình thường (dev).
import hmac as _hmac, hashlib as _hashlib
from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse

SITE_PASSWORD = os.getenv("SITE_PASSWORD", "")
_OPEN_PATHS = {"/login", "/auth", "/logout", "/favicon.ico", "/health"}


def _auth_token() -> str:
    return _hmac.new(("oracle::" + SITE_PASSWORD).encode(), b"v1", _hashlib.sha256).hexdigest()


_LOGIN_HTML = """<!DOCTYPE html><html lang="vi"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Đăng nhập · FOOTBALL ORACLE</title>
<style>
body{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;
font-family:'Courier New',monospace;background:#05070a;color:#00ff9c;
background-image:linear-gradient(rgba(0,255,156,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,156,.05) 1px,transparent 1px);background-size:32px 32px}
.box{border:1px solid rgba(0,255,156,.3);border-radius:8px;padding:30px 28px;width:320px;
background:linear-gradient(180deg,rgba(0,255,156,.06),transparent);box-shadow:0 0 30px rgba(0,255,156,.15)}
h1{font-size:18px;letter-spacing:2px;margin:0 0 4px;text-shadow:0 0 8px rgba(0,255,156,.6)}
.sub{font-size:11px;color:#3f6b5a;margin-bottom:18px;letter-spacing:1px}
input{width:100%;box-sizing:border-box;font-family:inherit;background:#0d141b;color:#00ff9c;
border:1px solid rgba(0,255,156,.3);padding:11px;border-radius:5px;font-size:14px;margin-bottom:12px}
button{width:100%;font-family:inherit;background:#00ff9c;color:#021;border:0;padding:11px;
border-radius:5px;font-size:14px;font-weight:700;letter-spacing:1px;cursor:pointer}
button:hover{box-shadow:0 0 14px rgba(0,255,156,.6)}
</style></head><body>
<form class="box" method="post" action="/auth">
  <h1>⛧ FOOTBALL ORACLE</h1>
  <div class="sub">Khu vực riêng tư · nhập mật khẩu để vào</div>
  <!--ERR-->
  <input type="password" name="password" placeholder="Mật khẩu" autofocus>
  <button type="submit">ĐĂNG NHẬP</button>
</form></body></html>"""


@app.middleware("http")
async def _auth_gate(request: Request, call_next):
    if not SITE_PASSWORD:                      # chưa đặt mật khẩu -> không khóa
        return await call_next(request)
    path = request.url.path
    if path in _OPEN_PATHS or request.cookies.get("oracle_auth") == _auth_token():
        return await call_next(request)
    if path.startswith("/api"):
        return JSONResponse({"detail": "Chưa đăng nhập"}, status_code=401)
    return RedirectResponse("/login")


@app.get("/login", response_class=HTMLResponse)
def _login_page():
    return _LOGIN_HTML.replace("<!--ERR-->", "")


@app.post("/auth")
async def _auth(request: Request):
    form = await request.form()
    if SITE_PASSWORD and _hmac.compare_digest(str(form.get("password", "")), SITE_PASSWORD):
        r = RedirectResponse("/", status_code=303)
        r.set_cookie("oracle_auth", _auth_token(), httponly=True, max_age=2592000, samesite="lax")
        return r
    err = '<div style="color:#ff3b5c;font-size:12px;margin-bottom:10px">✗ Sai mật khẩu</div>'
    return HTMLResponse(_LOGIN_HTML.replace("<!--ERR-->", err), status_code=401)


@app.get("/logout")
def _logout():
    r = RedirectResponse("/login")
    r.delete_cookie("oracle_auth")
    return r


# Phục vụ frontend tĩnh (đặt cuối để không che các route /api)
_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
