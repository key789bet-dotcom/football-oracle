"""
ENGINE phân tích chuyên sâu — fit sức mạnh đội bóng từ CẢ MÙA GIẢI rồi dự đoán.
Toàn bộ bằng Python thuần (math/random), không cần thư viện ngoài.
"""
from math import exp, factorial
import random

MAX_GOALS = 8
RHO = -0.08
HOME_ADV = 1.0


# ---------------- FIT RATINGS TỪ MÙA GIẢI ----------------
def _finished(matches):
    out = []
    for m in matches:
        g = m.get("goals", {})
        if g.get("home") is None or g.get("away") is None:
            continue
        if "teams" in m:
            t = m["teams"]
            hid, aid = t.get("home", {}).get("id"), t.get("away", {}).get("id")
        else:
            hid, aid = m.get("home", {}).get("id"), m.get("away", {}).get("id")
        if hid is None or aid is None:
            continue
        out.append((hid, aid, g["home"], g["away"]))
    return out


def fit_ratings(matches: list[dict]) -> dict:
    games = _finished(matches)
    if not games:
        return {"teams": {}, "lh": 1.35, "la": 1.15, "n": 0}
    n = len(games)
    lh = sum(g[2] for g in games) / n
    la = sum(g[3] for g in games) / n
    lh = max(lh, 0.3); la = max(la, 0.3)
    agg = {}
    def slot(tid):
        return agg.setdefault(tid, {"sh": 0, "ch": 0, "nh": 0, "sa": 0, "ca": 0, "na": 0})
    for hid, aid, gh, ga in games:
        h, a = slot(hid), slot(aid)
        h["sh"] += gh; h["ch"] += ga; h["nh"] += 1
        a["sa"] += ga; a["ca"] += gh; a["na"] += 1
    teams = {}
    for tid, s in agg.items():
        att_h = (s["sh"] / s["nh"] / lh) if s["nh"] else 1.0
        att_a = (s["sa"] / s["na"] / la) if s["na"] else 1.0
        def_h = (s["ch"] / s["nh"] / la) if s["nh"] else 1.0
        def_a = (s["ca"] / s["na"] / lh) if s["na"] else 1.0
        w = min(1.0, (s["nh"] + s["na"]) / 6)
        att = 1 + w * (((att_h + att_a) / 2) - 1)
        dfn = 1 + w * (((def_h + def_a) / 2) - 1)
        teams[tid] = {"att": round(att, 3), "def": round(dfn, 3)}
    return {"teams": teams, "lh": round(lh, 3), "la": round(la, 3), "n": n}


def expected_goals(ratings: dict, home_id: int, away_id: int) -> tuple[float, float]:
    t = ratings["teams"]
    h = t.get(home_id, {"att": 1.0, "def": 1.0})
    a = t.get(away_id, {"att": 1.0, "def": 1.0})
    lh = ratings["lh"] * h["att"] * a["def"]
    la = ratings["la"] * a["att"] * h["def"]
    return round(max(lh, 0.05), 3), round(max(la, 0.05), 3)


# ---------------- DIXON-COLES ----------------
def _pmf(k, lam):
    return (lam ** k) * exp(-lam) / factorial(k)


def _dc_tau(i, j, lh, la, rho):
    if i == 0 and j == 0: return 1 - lh * la * rho
    if i == 0 and j == 1: return 1 + lh * rho
    if i == 1 and j == 0: return 1 + la * rho
    if i == 1 and j == 1: return 1 - rho
    return 1.0


def dc_matrix(lh: float, la: float, rho: float = RHO) -> list[list[float]]:
    m = [[_pmf(i, lh) * _pmf(j, la) * _dc_tau(i, j, lh, la, rho)
          for j in range(MAX_GOALS + 1)] for i in range(MAX_GOALS + 1)]
    s = sum(sum(r) for r in m)
    return [[v / s for v in row] for row in m]


def analyse_matrix(m: list[list[float]]) -> dict:
    p_home = p_draw = p_away = over = btts = 0.0
    sc = []
    for i in range(len(m)):
        for j in range(len(m)):
            p = m[i][j]
            sc.append((p, i, j))
            if i > j: p_home += p
            elif i == j: p_draw += p
            else: p_away += p
            if i + j >= 3: over += p
            if i >= 1 and j >= 1: btts += p
    sc.sort(reverse=True)
    return {
        "probs": {"home": p_home, "draw": p_draw, "away": p_away},
        "top_scores": [{"score": f"{i}-{j}", "prob": round(p, 4)} for p, i, j in sc[:3]],
        "over_2_5": round(over, 4), "under_2_5": round(1 - over, 4),
        "btts_yes": round(btts, 4), "btts_no": round(1 - btts, 4),
    }


# ---------------- KÈO TÀI/XỈU & CHẤP ----------------
def over_under(m, lines=(0.5, 1.5, 2.5, 3.5)) -> dict:
    n = len(m)
    res = {}
    for L in lines:
        over = sum(m[i][j] for i in range(n) for j in range(n) if i + j > L)
        res[str(L)] = {"over": round(over, 4), "under": round(1 - over, 4)}
    return res


def over_under_at_line(m, line: float) -> dict:
    """Tài/Xỉu cho line bất kỳ (hỗ trợ nguyên/nửa/quarter)."""
    L = float(line)
    n = len(m)
    def at_one(L1):
        over = under = push = 0.0
        for i in range(n):
            for j in range(n):
                t = i + j
                if t > L1 + 1e-9: over += m[i][j]
                elif t < L1 - 1e-9: under += m[i][j]
                else: push += m[i][j]
        return over, under, push
    is_quarter = abs(L - round(L)) > 1e-6 and abs(L - (round(L * 2) / 2)) > 1e-6
    if is_quarter:
        L1 = round(L - 0.25, 2); L2 = round(L + 0.25, 2)
        o1, u1, p1 = at_one(L1); o2, u2, p2 = at_one(L2)
        over = (o1 + 0.5 * p1 + o2) / 2
        under = (u1 + 0.5 * p1 + u2) / 2
    else:
        o, u, p = at_one(L)
        if p > 1e-6:
            over = o + 0.5 * p; under = u + 0.5 * p
        else:
            over, under = o, u
    side = "TÀI" if over >= under else "XỈU"
    return {"line": L, "over": round(over, 4), "under": round(under, 4),
            "side": side, "prob": round(max(over, under), 4)}


def asian_handicap_at_line(m, line: float) -> dict:
    """AH cho line bất kỳ (hỗ trợ quarter)."""
    L = float(line)
    n = len(m)
    def at_one(L1):
        hw = aw = pu = 0.0
        for i in range(n):
            for j in range(n):
                d = (i + L1) - j
                if d > 1e-9: hw += m[i][j]
                elif d < -1e-9: aw += m[i][j]
                else: pu += m[i][j]
        return hw, aw, pu
    is_quarter = abs(L - round(L)) > 1e-6 and abs(L - (round(L * 2) / 2)) > 1e-6
    if is_quarter:
        L1 = round(L - 0.25, 2); L2 = round(L + 0.25, 2)
        h1, a1, p1 = at_one(L1); h2, a2, p2 = at_one(L2)
        home = (h1 + 0.5 * p1 + h2) / 2
        away = (a1 + 0.5 * p1 + a2) / 2
    else:
        h, a, p = at_one(L)
        if p > 1e-6:
            home = h + 0.5 * p; away = a + 0.5 * p
        else:
            home, away = h, a
    side = "home" if home >= away else "away"
    return {"line": L, "home": round(home, 4), "away": round(away, 4),
            "side": side, "cover": round(max(home, away), 4)}


def consensus_line(rows: list[dict], key: str = "line") -> float | None:
    """Trích line ĐỒNG THUẬN (median) từ bảng bookmaker."""
    vals = []
    for r in rows or []:
        try:
            v = r.get(key)
            if v is None: continue
            vals.append(float(v))
        except (ValueError, TypeError):
            continue
    if not vals: return None
    vals.sort()
    n = len(vals)
    if n % 2 == 1: return vals[n // 2]
    return round((vals[n // 2 - 1] + vals[n // 2]) / 2, 2)


def asian_handicap(m, lines) -> dict:
    n = len(m); res = {}
    for h in lines:
        hw = push = aw = 0.0
        for i in range(n):
            for j in range(n):
                d = (i + h) - j
                if d > 1e-9: hw += m[i][j]
                elif abs(d) < 1e-9: push += m[i][j]
                else: aw += m[i][j]
        res[_fmt(h)] = {"home": round(hw, 4), "push": round(push, 4), "away": round(aw, 4)}
    return res


def _fmt(x):
    return f"{x:+g}"


def corner_pick(total_goals_xg: float, cur_corners: int = 0, minute: int = 0,
                line: float | None = None) -> dict:
    """Kèo PHẠT GÓC T/X. LINE ĐỘNG: nếu line=None → tự chọn .5 gần nhất với exp_final.
    Truyền line khi API có corner odds thật."""
    base_full = max(5.0, min(14.0, 9.0 + (total_goals_xg - 2.5) * 1.6))
    frac = max(0.0, (90 - minute) / 90)
    exp_final = (cur_corners + base_full * frac) if minute > 0 else base_full
    if line is None:
        line = round(exp_final - 0.5) + 0.5
        line = max(4.5, min(14.5, line))
    L = float(line)
    if L != int(L):
        k_floor = int(L)
        cum = sum((exp_final ** i) * exp(-exp_final) / factorial(i) for i in range(k_floor + 1))
        over = max(0.0, min(1.0, 1 - cum))
        under = 1 - over
    else:
        k_int = int(L)
        p_lt = sum((exp_final ** i) * exp(-exp_final) / factorial(i) for i in range(k_int))
        p_eq = (exp_final ** k_int) * exp(-exp_final) / factorial(k_int)
        over = max(0.0, 1 - p_lt - p_eq) + 0.5 * p_eq
        under = p_lt + 0.5 * p_eq
    pick = "TÀI" if over >= under else "XỈU"
    return {"line": L, "exp_corners": round(exp_final, 1),
            "over": round(over, 4), "under": round(under, 4),
            "pick": pick, "prob": round(max(over, under), 4)}


def betting_lines(lh, la, m) -> dict:
    ou = over_under(m)
    diff = lh - la
    fav = "home" if diff >= 0 else "away"
    margin = abs(diff)
    fair = round(margin * 2) / 2
    lines = sorted(set([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5,
                        -fair, fair, -(fair + 0.5), -(fair - 0.5) if fair >= 0.5 else -0.5]))
    ah = asian_handicap(m, lines)
    ou25 = ou["2.5"]
    ou_side = "TÀI" if ou25["over"] >= ou25["under"] else "XỈU"
    ou_prob = max(ou25["over"], ou25["under"])
    home_line = -fair if fav == "home" else fair
    key = _fmt(home_line)
    cover = ah.get(key, {})
    if fav == "home":
        cover_prob = cover.get("home", 0); fav_label = "Đội nhà"
    else:
        cover_prob = cover.get("away", 0); fav_label = "Đội khách"
    return {
        "over_under": ou, "handicap": ah, "fair_handicap": fair,
        "ou_pick": {"side": ou_side, "line": 2.5, "prob": round(ou_prob, 4)},
        "ah_pick": {"team": fav_label, "side": fav, "line": home_line,
                    "cover": round(cover_prob, 4)},
    }


# ---------------- LIVE / IN-PLAY ----------------
def live_inplay(lh: float, la: float, gh: int, ga: int, minute: int) -> dict:
    minute = max(0, min(minute, 90))
    frac = max(0.0, (90 - minute) / 90)
    rem_h, rem_a = lh * frac, la * frac
    ph = pd = pa = 0.0
    over_cur = gh + ga
    p_over25 = 0.0
    for x in range(MAX_GOALS + 1):
        for y in range(MAX_GOALS + 1):
            p = _pmf(x, rem_h) * _pmf(y, rem_a)
            fh, fa = gh + x, ga + y
            if fh > fa: ph += p
            elif fh == fa: pd += p
            else: pa += p
            if (over_cur + x + y) >= 3: p_over25 += p
    s = ph + pd + pa or 1
    return {
        "minute": minute,
        "score": {"home": gh, "away": ga},
        "remaining_xg": {"home": round(rem_h, 2), "away": round(rem_a, 2)},
        "probs": {"home": round(ph / s, 4), "draw": round(pd / s, 4), "away": round(pa / s, 4)},
        "over_2_5": round(p_over25 / s, 4),
    }


def live_matrix(lh: float, la: float, gh: int, ga: int, minute: int) -> tuple:
    minute = max(0, min(minute, 90))
    frac = max(0.0, (90 - minute) / 90)
    rem_h, rem_a = lh * frac, la * frac
    M = [[0.0] * (MAX_GOALS + 1) for _ in range(MAX_GOALS + 1)]
    for x in range(MAX_GOALS + 1):
        for y in range(MAX_GOALS + 1):
            fh, fa = gh + x, ga + y
            if fh <= MAX_GOALS and fa <= MAX_GOALS:
                M[fh][fa] += _pmf(x, rem_h) * _pmf(y, rem_a)
    s = sum(sum(r) for r in M) or 1
    M = [[v / s for v in row] for row in M]
    return M, round(rem_h, 2), round(rem_a, 2)


def monte_carlo_live(lh, la, gh, ga, minute, n=10000) -> dict:
    minute = max(0, min(minute, 90))
    frac = max(0.0, (90 - minute) / 90)
    rem_h, rem_a = lh * frac, la * frac
    h = d = a = 0
    for _ in range(n):
        x, y = _rpois(rem_h), _rpois(rem_a)
        fh, fa = gh + x, ga + y
        if fh > fa: h += 1
        elif fh == fa: d += 1
        else: a += 1
    return {"sims": n, "home": round(h / n, 4), "draw": round(d / n, 4), "away": round(a / n, 4)}


# ---------------- MONTE CARLO ----------------
def _rpois(lam):
    L, k, p = exp(-lam), 0, 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= L:
            return k - 1


def monte_carlo(lh: float, la: float, n: int = 10000) -> dict:
    h = d = a = 0
    gh_tot = ga_tot = 0
    for _ in range(n):
        x, y = _rpois(lh), _rpois(la)
        gh_tot += x; ga_tot += y
        if x > y: h += 1
        elif x == y: d += 1
        else: a += 1
    return {
        "sims": n,
        "home": round(h / n, 4), "draw": round(d / n, 4), "away": round(a / n, 4),
        "avg_goals": round((gh_tot + ga_tot) / n, 2),
    }


# ---------------- BẢNG XẾP HẠNG ----------------
def standings(matches: list[dict], names: dict) -> list[dict]:
    tab = {}
    def row(tid):
        return tab.setdefault(tid, {"id": tid, "name": names.get(tid, str(tid)),
                                    "P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0})
    for hid, aid, gh, ga in _finished(matches):
        h, a = row(hid), row(aid)
        h["P"] += 1; a["P"] += 1
        h["GF"] += gh; h["GA"] += ga; a["GF"] += ga; a["GA"] += gh
        if gh > ga: h["W"] += 1; a["L"] += 1; h["Pts"] += 3
        elif gh < ga: a["W"] += 1; h["L"] += 1; a["Pts"] += 3
        else: h["D"] += 1; a["D"] += 1; h["Pts"] += 1; a["Pts"] += 1
    rows = list(tab.values())
    for r in rows:
        r["GD"] = r["GF"] - r["GA"]
    rows.sort(key=lambda r: (r["Pts"], r["GD"], r["GF"]), reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows
