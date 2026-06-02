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


def build_phase_verdict(
    *,
    final: dict, market: dict,
    home_xg: float, away_xg: float,
    minute: int, status: str,
    home_score: int, away_score: int,
    corners_home: int = 0, corners_away: int = 0,
    cards_y: int = 0, cards_r: int = 0,
    home_name: str = "Đội nhà", away_name: str = "Đội khách",
    pre_market_probs: dict | None = None,
    real_ou_line: float | None = None,
    real_ah_line: float | None = None,
) -> list[str]:
    """Nhận định CHUYÊN SÂU theo PHASE — thay đổi theo phút thật + data live.

    Phase:
      0 = pre-match (minute=0, status NS/scheduled)
      1 = đầu trận (1-15')
      2 = giữa H1 (15-30')
      3 = cuối H1 (30-45')
      4 = HT
      5 = đầu H2 (46-60')
      6 = giữa H2 (60-75')
      7 = cuối trận (75-90+')
      8 = FT (đã kết thúc)
    """
    s = (status or "").upper()
    finished = s in ("FT", "AET", "PEN")
    if finished:
        phase = 8
    elif s == "HT" or (44 <= minute <= 46 and home_score == away_score):
        phase = 4 if s == "HT" else 3
    elif minute == 0 or s in ("NS", "TBD", ""):
        phase = 0
    elif minute < 15:
        phase = 1
    elif minute < 30:
        phase = 2
    elif minute < 45:
        phase = 3
    elif minute < 60:
        phase = 5
    elif minute < 75:
        phase = 6
    else:
        phase = 7

    diff = home_score - away_score
    abs_diff = abs(diff)
    total_g = home_score + away_score
    total_c = corners_home + corners_away
    rem_min = max(0, 90 - minute)
    pick = max(final, key=final.get)
    pick_label = {"home": home_name, "draw": "Hoà", "away": away_name}[pick]
    p_top = final[pick]

    lines = []

    # =============== PHASE 0: PRE-MATCH ===============
    if phase == 0:
        lines.append(f"📋 NHẬN ĐỊNH TRƯỚC TRẬN — {home_name} vs {away_name}")
        # Pick chính
        if p_top >= 0.55:
            lines.append(f"🎯 Kèo chính: ưu tiên {pick_label} — xác suất {p_top*100:.1f}% (độ tin cậy {_confidence_tier(p_top)}).")
        elif final["draw"] >= 0.30:
            lines.append(f"⚖ Trận cân: hoà {final['draw']*100:.1f}% — kèo đôi {pick_label.upper()} + HOÀ an toàn hơn.")
        else:
            lines.append(f"🌀 Kèo mở: {pick_label} chỉ {p_top*100:.1f}% — không có cửa thuận, hạn chế cược.")
        # xG comparison
        if abs(home_xg - away_xg) >= 0.8:
            stronger = home_name if home_xg > away_xg else away_name
            lines.append(f"💪 Sức mạnh: {stronger} vượt trội (xG {max(home_xg,away_xg):.2f} vs {min(home_xg,away_xg):.2f}).")
        else:
            lines.append(f"⚔ Sức mạnh ngang ngửa: xG {home_xg:.2f} - {away_xg:.2f}.")
        # Market consensus
        if pre_market_probs:
            mp = pre_market_probs
            top_mk = max(mp, key=mp.get)
            mk_label = {"home": home_name, "draw": "Hoà", "away": away_name}[top_mk]
            lines.append(f"📈 Kèo thị trường (đồng thuận N nhà cái): nghiêng {mk_label} {mp[top_mk]*100:.1f}%.")
            # Phát hiện divergence
            if abs(mp.get(pick, 0) - p_top) > 0.08:
                lines.append(f"⚠ Lệch model-thị trường: model {p_top*100:.0f}% vs thị trường {mp.get(pick,0)*100:.0f}% — kiểm tra kỹ.")
        # Tổng bàn
        ou_line = real_ou_line or 2.5
        ou_p = market.get(f"over_{ou_line}") or (market["over_2_5"] if ou_line == 2.5 else None)
        if ou_p:
            ou_side = "TÀI" if ou_p >= 0.5 else "XỈU"
            lines.append(f"🎯 Tài/Xỉu {ou_line}: nghiêng {ou_side} ({max(ou_p,1-ou_p)*100:.1f}%) · xG tổng {market.get('total_xg',0):.2f}.")
        # Strategy
        if p_top >= 0.55 and (pre_market_probs is None or abs((pre_market_probs.get(pick) or 0.5) - p_top) < 0.05):
            lines.append(f"✓ Chiến lược: vào sớm pre-match — cửa {pick_label} value rõ.")
        else:
            lines.append("⏳ Chiến lược: chờ in-play minute 15-30' để xem nhịp trận, vào kèo lệch.")
        return lines

    # =============== PHASE 1-7: IN-PLAY ===============
    # Header phase
    phase_names = {
        1: "ĐẦU TRẬN", 2: "GIỮA HIỆP 1", 3: "CUỐI HIỆP 1",
        4: "GIỜ NGHỈ", 5: "ĐẦU HIỆP 2", 6: "GIỮA HIỆP 2", 7: "CUỐI TRẬN",
    }
    pn = phase_names.get(phase, f"PHÚT {minute}")
    score_str = f"{home_score}-{away_score}"
    if diff > 0:
        leading_name = home_name
    elif diff < 0:
        leading_name = away_name
    else:
        leading_name = None

    if phase == 4:  # HT
        lines.append(f"⏸ {pn} · Tỉ số {score_str}")
    elif minute >= 90:
        lines.append(f"⏱ PHÚT 90+ · Tỉ số {score_str} · Bù giờ")
    else:
        lines.append(f"⏱ {pn} · PHÚT {minute} · Tỉ số {score_str} · Còn ~{rem_min}'")

    # Đánh giá tỉ số vs prediction pre-match
    if diff == 0:
        if phase >= 6:
            lines.append(f"🌀 Vẫn hoà sau {minute}' — kèo hoà tăng đáng kể, cân nhắc XỈU + HOÀ.")
        elif phase >= 2:
            lines.append(f"⚖ Đang hoà {score_str} — còn nhiều thời gian, kèo còn mở.")
        else:
            lines.append(f"⚖ Vào trận thận trọng — chưa có bàn.")
    else:
        if abs_diff >= 2:
            if phase >= 7:
                lines.append(f"🏆 {leading_name} dẫn {abs_diff} bàn cuối trận — gần như chắc thắng.")
            elif phase >= 5:
                lines.append(f"💎 {leading_name} dẫn {abs_diff} bàn — kèo {leading_name} thắng đã rất cao.")
            else:
                lines.append(f"🔥 {leading_name} dẫn sớm {abs_diff} bàn — áp đảo rõ rệt.")
        else:
            if phase >= 7:
                lines.append(f"⚠ {leading_name} dẫn sát nút {score_str} cuối trận — vẫn có rủi ro mất kèo nếu thủng lưới phút bù.")
            elif phase >= 5:
                lines.append(f"📊 {leading_name} dẫn nhẹ {score_str} sau giờ nghỉ — đang giữ lợi thế.")
            elif phase == 4:
                lines.append(f"📊 {leading_name} dẫn {score_str} hết H1 — có lợi thế đầu H2.")
            else:
                lines.append(f"⚡ {leading_name} mở tỉ số sớm — kèo nghiêng về {leading_name}.")

    # Live model probability — thay đổi do score+phút
    lines.append(f"🎯 Model live: {pick_label} thắng {p_top*100:.1f}% (đã update theo {score_str} + phút {minute}).")

    # Tổng bàn live
    expected_total = total_g + (home_xg + away_xg) * (rem_min / 90)
    ou_line = real_ou_line or 2.5
    if total_g > ou_line:
        lines.append(f"✓ T/X {ou_line}: TÀI đã đủ ({total_g} bàn > {ou_line}) — kèo TÀI thắng.")
    elif expected_total > ou_line + 0.5:
        lines.append(f"📈 T/X {ou_line}: hiện {total_g}, dự kiến cuối ~{expected_total:.1f} → vẫn theo TÀI.")
    elif expected_total < ou_line - 0.5:
        lines.append(f"📉 T/X {ou_line}: hiện {total_g}, dự kiến cuối ~{expected_total:.1f} → nghiêng XỈU.")
    else:
        lines.append(f"⚖ T/X {ou_line}: dự kiến cuối ~{expected_total:.1f} ≈ line → 50/50 rủi ro cao.")

    # Phạt góc nếu có data
    if total_c > 0:
        # Giả định 1.0 góc / 10 phút trung bình (~9 góc/trận)
        expected_corners_final = total_c + (total_c / max(minute, 1)) * rem_min if minute > 0 else total_c
        lines.append(f"🚩 Góc hiện tại: {corners_home}-{corners_away} (tổng {total_c}) · dự kiến cuối ~{expected_corners_final:.0f}.")

    # Thẻ
    if cards_r > 0:
        lines.append(f"🟥 {cards_r} thẻ đỏ — đội ít người yếu thế rõ.")
    if cards_y >= 6:
        lines.append(f"🟨 Trận căng ({cards_y} thẻ vàng) — rủi ro hiệp 2 còn thẻ đỏ.")

    # Strategy theo phase
    if phase == 1:
        lines.append("⏳ Chiến lược: quan sát nhịp, chưa vào kèo. Đợi 25'+.")
    elif phase == 2:
        if diff == 0 and abs(home_xg - away_xg) > 0.5:
            lines.append(f"💡 Cơ hội: đội mạnh chưa ghi → giá kèo TỐT hơn pre-match, có thể vào.")
        else:
            lines.append("📊 Đang vào nhịp — giữ vị thế, chờ HT điều chỉnh.")
    elif phase == 3:
        lines.append(f"⏰ Chuẩn bị HT — kèo HT 1×2 + HT O/U hợp lý nếu còn cửa.")
    elif phase == 4:
        if diff == 0:
            lines.append("⏸ HT hoà — H2 thường ít bàn hơn, ưu tiên XỈU + HOÀ/draw no bet.")
        else:
            lines.append(f"⏸ HT có bàn — H2 thường khó bùng nổ, nhưng đội thua sẽ đẩy cao đội hình.")
    elif phase == 5:
        lines.append("🔥 Đầu H2 — 10 phút đầu hay có bàn, theo dõi sát đà tấn công.")
    elif phase == 6:
        if diff == 0:
            lines.append("⚠ Hoà giữa H2 — kèo hoà tăng giá trị, cân nhắc Asian draw +0.")
        elif abs_diff == 1:
            lines.append("⚠ Cách biệt 1 bàn giữa H2 — đội thua sẽ liều, rủi ro huề/lội ngược cao.")
    elif phase == 7:
        if abs_diff >= 2:
            lines.append(f"🔒 Cuối trận chênh ≥2 bàn — kèo gần như chốt, chỉ vào những kèo siêu chắc.")
        elif diff == 0:
            lines.append("🌀 Hoà cuối trận — XỈU + HOÀ giá tăng mạnh phút 80+.")
        else:
            lines.append(f"⚠ Sát nút cuối trận — đội thua dồn người, rủi ro mất kèo phút 85+.")

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
