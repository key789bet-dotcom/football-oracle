"""
Model dự đoán kết quả: kết hợp 3 nguồn tín hiệu
  1) Poisson  - dựa trên phong độ ghi/thủng lưới gần đây
  2) Odds     - xác suất ẩn trong tỉ lệ kèo (đã khử biên lợi nhuận nhà cái)
  3) ML       - (tùy chọn) model đã train trên dữ liệu lịch sử

Kết quả cuối = trung bình có trọng số của các nguồn có sẵn.
"""
from math import exp, factorial
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml_model.joblib")

# Trọng số trộn các nguồn (tự chỉnh theo độ tin cậy)
W_POISSON = 0.35
W_ODDS = 0.35
W_ML = 0.15
W_H2H = 0.15

MAX_GOALS = 8  # số bàn tối đa khi dựng ma trận xác suất


# ----------------------- 1) POISSON -----------------------
def _poisson_pmf(k: int, lam: float) -> float:
    return (lam ** k) * exp(-lam) / factorial(k)


def expected_goals_from_form(home_matches: list[dict], away_matches: list[dict],
                             home_id: int, away_id: int) -> tuple[float, float]:
    """Ước lượng số bàn kỳ vọng từ phong độ gần đây của 2 đội."""
    def avg_scored_conceded(matches, team_id):
        scored, conceded, n = 0, 0, 0
        for m in matches:
            goals = m.get("goals", {})
            teams = m.get("teams", {})
            if teams.get("home", {}).get("id") == team_id:
                gs, gc = goals.get("home"), goals.get("away")
            else:
                gs, gc = goals.get("away"), goals.get("home")
            if gs is None or gc is None:
                continue
            scored += gs
            conceded += gc
            n += 1
        if n == 0:
            return 1.3, 1.3  # giá trị mặc định trung bình giải đấu
        return scored / n, conceded / n

    h_att, h_def = avg_scored_conceded(home_matches, home_id)
    a_att, a_def = avg_scored_conceded(away_matches, away_id)

    # bàn kỳ vọng = tấn công đội này * phòng ngự đội kia, + lợi thế sân nhà
    home_xg = (h_att + a_def) / 2 * 1.15
    away_xg = (a_att + h_def) / 2 * 0.95
    return round(home_xg, 2), round(away_xg, 2)


def score_matrix(home_xg: float, away_xg: float) -> list[list[float]]:
    """Ma trận xác suất tỉ số i-j (i bàn nhà, j bàn khách)."""
    return [[_poisson_pmf(i, home_xg) * _poisson_pmf(j, away_xg)
             for j in range(MAX_GOALS + 1)] for i in range(MAX_GOALS + 1)]


def poisson_probs(home_xg: float, away_xg: float) -> dict:
    """Trả xác suất {home, draw, away} từ 2 giá trị bàn kỳ vọng."""
    m = score_matrix(home_xg, away_xg)
    p_home = p_draw = p_away = 0.0
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            if i > j:
                p_home += m[i][j]
            elif i == j:
                p_draw += m[i][j]
            else:
                p_away += m[i][j]
    total = p_home + p_draw + p_away
    return {"home": p_home / total, "draw": p_draw / total, "away": p_away / total}


def market_analysis(home_xg: float, away_xg: float) -> dict:
    """Phân tích kèo phụ từ ma trận Poisson: tỉ số khả dĩ, Tài/Xỉu, BTTS."""
    m = score_matrix(home_xg, away_xg)
    total = sum(sum(row) for row in m)
    scthat = []  # (xác suất, i, j)
    over25 = btts = 0.0
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = m[i][j] / total
            scthat.append((p, i, j))
            if i + j >= 3:
                over25 += p
            if i >= 1 and j >= 1:
                btts += p
    scthat.sort(reverse=True)
    top = [{"score": f"{i}-{j}", "prob": round(p, 4)} for p, i, j in scthat[:3]]
    return {
        "top_scores": top,
        "over_2_5": round(over25, 4),
        "under_2_5": round(1 - over25, 4),
        "btts_yes": round(btts, 4),
        "btts_no": round(1 - btts, 4),
        "total_xg": round(home_xg + away_xg, 2),
    }


# ----------------------- 2) ODDS -----------------------
def odds_to_probs(odds_response: list[dict]) -> dict | None:
    """
    Lấy odds 1X2 từ response API-Football, đổi sang xác suất và khử biên nhà cái.
    Trả None nếu không có dữ liệu odds.
    """
    try:
        bets = odds_response[0]["bookmakers"][0]["bets"]
    except (IndexError, KeyError, TypeError):
        return None

    market = next((b for b in bets if b.get("name") in ("Match Winner", "1X2")), None)
    if not market:
        return None

    vals = {v["value"].lower(): float(v["odd"]) for v in market["values"]}
    try:
        inv = {
            "home": 1 / vals["home"],
            "draw": 1 / vals["draw"],
            "away": 1 / vals["away"],
        }
    except KeyError:
        return None
    s = sum(inv.values())  # >1 do biên lợi nhuận
    return {k: v / s for k, v in inv.items()}


# ----------------------- 3) ML (tùy chọn) -----------------------
def ml_probs(features: list[float]) -> dict | None:
    """Dùng model đã train nếu có file ml_model.joblib. Xem train_model.py."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import joblib  # chỉ nạp khi thực sự cần ML
    except ImportError:
        return None
    model = joblib.load(MODEL_PATH)
    proba = model.predict_proba([features])[0]
    classes = list(model.classes_)  # ví dụ ['away','draw','home']
    out = {c: float(proba[i]) for i, c in enumerate(classes)}
    return {"home": out.get("home", 0), "draw": out.get("draw", 0), "away": out.get("away", 0)}


# ----------------------- 4) H2H (đối đầu lịch sử) -----------------------
def h2h_probs(h2h_matches: list[dict], home_id: int) -> dict | None:
    """Tỉ lệ thắng/hòa/thua dựa trên các trận đối đầu trực tiếp trước đây."""
    if not h2h_matches:
        return None
    h = d = a = 0
    for m in h2h_matches:
        g, t = m.get("goals", {}), m.get("teams", {})
        gh, ga = g.get("home"), g.get("away")
        if gh is None or ga is None:
            continue
        # quy về góc nhìn "đội nhà của trận sắp tới"
        home_is_home = t.get("home", {}).get("id") == home_id
        my, opp = (gh, ga) if home_is_home else (ga, gh)
        if my > opp:
            h += 1
        elif my == opp:
            d += 1
        else:
            a += 1
    total = h + d + a
    if total == 0:
        return None
    # làm mượt Laplace để tránh xác suất 0
    return {"home": (h + 1) / (total + 3),
            "draw": (d + 1) / (total + 3),
            "away": (a + 1) / (total + 3)}


# ----------------------- TRỘN KẾT QUẢ -----------------------
def blend(*sources: dict | None) -> dict:
    """Trộn các nguồn xác suất (bỏ qua None) theo trọng số tương ứng."""
    weights = [W_POISSON, W_ODDS, W_ML, W_H2H]
    acc = {"home": 0.0, "draw": 0.0, "away": 0.0}
    used_w = 0.0
    for src, w in zip(sources, weights):
        if src is None:
            continue
        for k in acc:
            acc[k] += src[k] * w
        used_w += w
    if used_w == 0:
        return {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}
    return {k: v / used_w for k, v in acc.items()}


def _confidence_tier(p: float) -> str:
    if p >= 0.60:
        return "RẤT CAO"
    if p >= 0.48:
        return "CAO"
    if p >= 0.40:
        return "TRUNG BÌNH"
    return "THẤP"


def build_verdict(final: dict, market: dict, home_xg: float, away_xg: float) -> list[str]:
    """Sinh các dòng 'nhận định' kiểu chuyên gia phân tích."""
    pick = max(final, key=final.get)
    label = {"home": "CỬA TRÊN (đội nhà)", "draw": "HÒA", "away": "CỬA DƯỚI (đội khách)"}[pick]
    lines = []
    lines.append(f"Kèo chính: {label} — xác suất {final[pick]*100:.1f}% "
                 f"[độ tin cậy {_confidence_tier(final[pick])}]")
    diff = abs(home_xg - away_xg)
    if diff < 0.4:
        lines.append("Cân bằng: hai đội sức mạnh tương đương, rủi ro cao, ưu tiên kèo hòa/đôi.")
    elif diff < 0.9:
        lines.append("Chênh lệch nhẹ: có lợi thế nhưng chưa áp đảo, nên kèo đôi để an toàn.")
    else:
        lines.append("Chênh lệch rõ: một đội vượt trội về kỳ vọng bàn thắng.")
    ou = "TÀI (Over 2.5)" if market["over_2_5"] >= 0.5 else "XỈU (Under 2.5)"
    op = max(market["over_2_5"], market["under_2_5"])
    lines.append(f"Tổng bàn: nghiêng {ou} — {op*100:.1f}% (xG tổng {market['total_xg']}).")
    btts = "CÓ" if market["btts_yes"] >= 0.5 else "KHÔNG"
    bp = max(market["btts_yes"], market["btts_no"])
    lines.append(f"Cả hai đội ghi bàn: {btts} — {bp*100:.1f}%.")
    ts = market["top_scores"][0]
    lines.append(f"Tỉ số khả dĩ nhất: {ts['score']} ({ts['prob']*100:.1f}%).")
    return lines


def predict(home_matches, away_matches, home_id, away_id,
            odds_response=None, h2h_matches=None) -> dict:
    """Hàm chính: trả dự đoán đầy đủ cho 1 trận."""
    home_xg, away_xg = expected_goals_from_form(home_matches, away_matches, home_id, away_id)
    p_pois = poisson_probs(home_xg, away_xg)
    p_odds = odds_to_probs(odds_response) if odds_response else None
    p_ml = ml_probs([home_xg, away_xg])
    p_h2h = h2h_probs(h2h_matches or [], home_id)

    final = blend(p_pois, p_odds, p_ml, p_h2h)
    pick = max(final, key=final.get)
    label = {"home": "Đội nhà thắng", "draw": "Hòa", "away": "Đội khách thắng"}[pick]
    market = market_analysis(home_xg, away_xg)

    return {
        "expected_goals": {"home": home_xg, "away": away_xg},
        "probabilities": {k: round(v, 4) for k, v in final.items()},
        "sources": {
            "poisson": {k: round(v, 4) for k, v in p_pois.items()},
            "odds": {k: round(v, 4) for k, v in p_odds.items()} if p_odds else None,
            "ml": {k: round(v, 4) for k, v in p_ml.items()} if p_ml else None,
            "h2h": {k: round(v, 4) for k, v in p_h2h.items()} if p_h2h else None,
        },
        "prediction": pick,
        "prediction_label": label,
        "confidence": round(final[pick], 4),
        "confidence_tier": _confidence_tier(final[pick]),
        "market": market,
        "verdict": build_verdict(final, market, home_xg, away_xg),
    }
