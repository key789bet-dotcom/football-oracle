"""
ML thuần Python: hồi quy softmax 3 lớp (thắng/hòa/thua) — KHÔNG cần scikit-learn/numpy.
Huấn luyện bằng gradient descent trên dữ liệu cả mùa giải.
Đặc trưng mỗi trận lấy từ chỉ số sức mạnh (engine.fit_ratings):
    x = [att_nhà, def_nhà, att_khách, def_khách]
Nhãn: 0=nhà thắng, 1=hòa, 2=khách thắng.

Cũng chứa backtest: dự đoán lại các trận đã đá và đo độ chính xác.
"""
from math import exp, log
from . import engine

CLASSES = ["home", "draw", "away"]


def _features(ratings, hid, aid):
    t = ratings["teams"]
    h = t.get(hid, {"att": 1.0, "def": 1.0})
    a = t.get(aid, {"att": 1.0, "def": 1.0})
    return [h["att"], h["def"], a["att"], a["def"]]


def _softmax(z):
    mx = max(z)
    e = [exp(v - mx) for v in z]
    s = sum(e)
    return [v / s for v in e]


def _standardize_fit(X):
    n, d = len(X), len(X[0])
    mean = [sum(row[k] for row in X) / n for k in range(d)]
    var = [sum((row[k] - mean[k]) ** 2 for row in X) / n for k in range(d)]
    std = [(v ** 0.5) or 1.0 for v in var]
    return mean, std


def _apply(x, mean, std):
    return [(x[k] - mean[k]) / std[k] for k in range(len(x))]


def train(matches, ratings, epochs=300, lr=0.3, l2=1e-3):
    """Train softmax. Trả model dict (weights, mean, std)."""
    games = engine._finished(matches)
    if len(games) < 20:
        return None
    X, y = [], []
    for hid, aid, gh, ga in games:
        X.append(_features(ratings, hid, aid))
        y.append(0 if gh > ga else (1 if gh == ga else 2))

    mean, std = _standardize_fit(X)
    Xs = [_apply(x, mean, std) for x in X]
    d = len(Xs[0])
    # weights[class][feature] + bias[class]
    W = [[0.0] * d for _ in range(3)]
    b = [0.0, 0.0, 0.0]
    n = len(Xs)

    for _ in range(epochs):
        gW = [[0.0] * d for _ in range(3)]
        gb = [0.0, 0.0, 0.0]
        for xi, yi in zip(Xs, y):
            z = [sum(W[c][k] * xi[k] for k in range(d)) + b[c] for c in range(3)]
            p = _softmax(z)
            for c in range(3):
                err = p[c] - (1.0 if c == yi else 0.0)
                for k in range(d):
                    gW[c][k] += err * xi[k]
                gb[c] += err
        for c in range(3):
            for k in range(d):
                W[c][k] -= lr * (gW[c][k] / n + l2 * W[c][k])
            b[c] -= lr * (gb[c] / n)

    return {"W": W, "b": b, "mean": mean, "std": std}


def predict_proba(model, ratings, hid, aid) -> dict | None:
    if not model:
        return None
    x = _apply(_features(ratings, hid, aid), model["mean"], model["std"])
    W, b = model["W"], model["b"]
    z = [sum(W[c][k] * x[k] for k in range(len(x))) + b[c] for c in range(3)]
    p = _softmax(z)
    return {"home": p[0], "draw": p[1], "away": p[2]}


# ---------------- BACKTEST ----------------
def backtest(matches, ratings, model=None) -> dict:
    """Dự đoán lại các trận đã đá, đo độ chính xác (in-sample) + Brier score."""
    games = engine._finished(matches)
    if not games:
        return {"n": 0}
    hit = 0
    brier = 0.0
    by_pick = {"home": [0, 0], "draw": [0, 0], "away": [0, 0]}  # [đúng, tổng]
    for hid, aid, gh, ga in games:
        lh, la = engine.expected_goals(ratings, hid, aid)
        probs = engine.analyse_matrix(engine.dc_matrix(lh, la))["probs"]
        if model:
            mlp = predict_proba(model, ratings, hid, aid)
            probs = {k: (probs[k] + mlp[k]) / 2 for k in probs}
        pick = max(probs, key=probs.get)
        actual = "home" if gh > ga else ("draw" if gh == ga else "away")
        if pick == actual:
            hit += 1
        by_pick[pick][1] += 1
        if pick == actual:
            by_pick[pick][0] += 1
        for k in probs:
            brier += (probs[k] - (1.0 if k == actual else 0.0)) ** 2
    n = len(games)
    return {
        "n": n,
        "accuracy": round(hit / n, 4),
        "brier": round(brier / n, 4),   # càng thấp càng tốt (0..2)
        "baseline": round(max(
            sum(1 for g in games if g[2] > g[3]),
            sum(1 for g in games if g[2] == g[3]),
            sum(1 for g in games if g[2] < g[3]),
        ) / n, 4),  # đoán mò theo lớp phổ biến nhất
        "by_pick": {k: {"correct": v[0], "total": v[1],
                        "rate": round(v[0] / v[1], 3) if v[1] else 0} for k, v in by_pick.items()},
    }
