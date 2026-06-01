"""
SỔ TRACK RECORD — lưu mọi dự đoán ra file JSON, tự đối chiếu kết quả khi trận xong.
Đây là thứ tạo niềm tin thật: một bảng thành tích không sửa được, lớn dần theo thời gian.

File lưu: backend/track_log.json
"""
import json
import os
import time
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "track_log.json")


def _load() -> list:
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save(rows: list):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)


def log_prediction(entry: dict):
    """Ghi 1 dự đoán. Chống trùng theo fixture_id (chỉ ghi lần đầu, giữ nguyên về sau)."""
    rows = _load()
    if any(r["fixture_id"] == entry["fixture_id"] for r in rows):
        return  # đã có -> không ghi đè (giữ tính bất biến của track record)
    entry["logged_at"] = datetime.utcnow().isoformat(timespec="seconds")
    entry["result"] = None
    entry["correct"] = None
    rows.append(entry)
    _save(rows)


def reconcile(results_by_code: dict) -> dict:
    """results_by_code: {code: {fixture_id: (gh, ga)}} — cập nhật kết quả cho mục đang chờ."""
    rows = _load()
    changed = False
    for r in rows:
        if r["result"] is not None:
            continue
        res = results_by_code.get(r.get("code"), {}).get(r["fixture_id"])
        if not res:
            continue
        gh, ga = res
        actual = "home" if gh > ga else ("draw" if gh == ga else "away")
        r["result"] = {"score": f"{gh}-{ga}", "outcome": actual}
        r["correct"] = (r["pick"] == actual)
        changed = True
    if changed:
        _save(rows)
    return stats(rows)


def stats(rows: list | None = None) -> dict:
    rows = rows if rows is not None else _load()
    resolved = [r for r in rows if r.get("correct") is not None]
    hit = sum(1 for r in resolved if r["correct"])
    nr = len(resolved)
    # tỉ lệ trúng theo mức tin cậy
    tiers = {}
    for r in resolved:
        t = r.get("confidence_tier", "?")
        tiers.setdefault(t, [0, 0])
        tiers[t][1] += 1
        if r["correct"]:
            tiers[t][0] += 1
    return {
        "total_logged": len(rows),
        "resolved": nr,
        "pending": len(rows) - nr,
        "accuracy": round(hit / nr, 4) if nr else None,
        "hits": hit,
        "by_tier": {k: {"correct": v[0], "total": v[1],
                        "rate": round(v[0] / v[1], 3) if v[1] else 0} for k, v in tiers.items()},
    }


def recent(limit: int = 30) -> list:
    rows = _load()
    rows.sort(key=lambda r: r.get("logged_at", ""), reverse=True)
    return rows[:limit]
