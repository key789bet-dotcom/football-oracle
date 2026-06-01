"""
Kiểm định TRUNG THỰC cho model (out-of-sample).

walk_forward: duyệt các trận theo thời gian, ở mỗi trận CHỈ fit chỉ số từ các trận
TRƯỚC đó rồi mới dự đoán -> không "nhìn trộm tương lai" như backtest in-sample.

Trả về: độ chính xác, Brier, log-loss, baseline (đoán mò), và dữ liệu CALIBRATION
(khi model nói X% thì thực tế xảy ra bao nhiêu %).
"""
from math import log
from . import engine

OUTCOMES = ["home", "draw", "away"]


def _result(gh, ga):
    return "home" if gh > ga else ("draw" if gh == ga else "away")


def walk_forward(matches: list[dict], warmup_frac: float = 0.35) -> dict:
    """matches: slim (có 'date'). Fit lũy tiến theo thời gian."""
    games = [m for m in matches
             if m.get("goals", {}).get("home") is not None
             and m.get("goals", {}).get("away") is not None
             and m.get("date")]
    games.sort(key=lambda m: m["date"])
    n = len(games)
    if n < 40:
        return {"n": 0, "error": "Không đủ trận đã đá để kiểm định out-of-sample."}

    start = int(n * warmup_frac)
    records = []  # (probs, actual)
    for i in range(start, n):
        history = games[:i]                      # CHỈ dùng quá khứ
        ratings = engine.fit_ratings(history)
        m = games[i]
        hid, aid = m["home"]["id"], m["away"]["id"]
        lh, la = engine.expected_goals(ratings, hid, aid)
        probs = engine.analyse_matrix(engine.dc_matrix(lh, la))["probs"]
        actual = _result(m["goals"]["home"], m["goals"]["away"])
        records.append((probs, actual))

    return _metrics(records, games[start:])


def _metrics(records, tested_games) -> dict:
    n = len(records)
    hit = 0
    brier = logloss = 0.0
    # calibration: gom theo từng dự đoán xác suất của TỪNG cửa
    bins = [{"sum_p": 0.0, "obs": 0, "cnt": 0} for _ in range(10)]
    for probs, actual in records:
        pick = max(probs, key=probs.get)
        if pick == actual:
            hit += 1
        for k in OUTCOMES:
            y = 1.0 if k == actual else 0.0
            p = min(max(probs[k], 1e-9), 1 - 1e-9)
            brier += (p - y) ** 2
            b = min(int(probs[k] * 10), 9)
            bins[b]["sum_p"] += probs[k]
            bins[b]["obs"] += y
            bins[b]["cnt"] += 1
        logloss += -log(min(max(probs[actual], 1e-9), 1))

    # baseline: luôn đoán lớp phổ biến nhất trong tập test
    cnt = {k: sum(1 for _, a in records if a == k) for k in OUTCOMES}
    baseline = max(cnt.values()) / n

    calib = []
    for i, b in enumerate(bins):
        if b["cnt"]:
            calib.append({"bin": f"{i*10}-{i*10+10}%",
                          "pred": round(b["sum_p"] / b["cnt"], 4),
                          "obs": round(b["obs"] / b["cnt"], 4),
                          "count": b["cnt"]})
    return {
        "n": n,
        "accuracy": round(hit / n, 4),
        "baseline": round(baseline, 4),
        "edge": round(hit / n - baseline, 4),
        "brier": round(brier / (n * 3), 4),     # trung bình trên 3 cửa
        "logloss": round(logloss / n, 4),
        "calibration": calib,
        "method": "walk-forward out-of-sample (chỉ học từ quá khứ)",
    }
