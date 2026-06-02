"""
backend/users.py — Module quản lý user + session bằng SQLite.
- Hash mật khẩu bằng pbkdf2_hmac (built-in Python).
- Session token random (secrets), lưu trong cookie httponly.
- Bootstrap admin tự động khi DB rỗng (đọc env ADMIN_USERNAME/ADMIN_PASSWORD).
"""
import os
import time
import sqlite3
import hashlib
import secrets
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")
PBKDF2_ITER = 120_000


@contextmanager
def _conn():
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA foreign_keys=ON")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db():
    """Tạo bảng + migration nếu schema cũ."""
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at INTEGER NOT NULL,
            last_login INTEGER,
            note TEXT,
            points INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS sessions(
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            user_agent TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);

        CREATE TABLE IF NOT EXISTS user_predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            fixture_id INTEGER NOT NULL,
            cost INTEGER NOT NULL,
            paid_at INTEGER NOT NULL,
            UNIQUE(user_id, fixture_id),
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_uprd_user ON user_predictions(user_id);

        CREATE TABLE IF NOT EXISTS user_bets(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            fixture_id INTEGER,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            league TEXT,
            bet_type TEXT NOT NULL,            -- '1x2' | 'hdp' | 'ou' | 'btts' | 'corner'
            pick TEXT NOT NULL,                -- 'home' | 'draw' | 'away' | 'over' | 'under' | 'yes' | 'no' | custom
            line REAL,                          -- handicap line hoặc total line
            stake REAL NOT NULL,                -- số tiền đặt
            odd REAL NOT NULL,                  -- tỷ lệ ăn (decimal)
            status TEXT NOT NULL DEFAULT 'pending',  -- 'pending' | 'won' | 'lost' | 'push' | 'half_won' | 'half_lost'
            profit REAL NOT NULL DEFAULT 0,    -- lãi/lỗ thực tế
            note TEXT,
            placed_at INTEGER NOT NULL,
            settled_at INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_bets_user ON user_bets(user_id);
        CREATE INDEX IF NOT EXISTS idx_bets_status ON user_bets(user_id, status);
        CREATE INDEX IF NOT EXISTS idx_bets_placed ON user_bets(user_id, placed_at DESC);

        -- ============ MATCH SESSIONS — chia vốn theo trận ============
        CREATE TABLE IF NOT EXISTS match_sessions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            fixture_id INTEGER NOT NULL,
            home_team TEXT, away_team TEXT, league TEXT,
            capital REAL NOT NULL,              -- vốn user nhập đầu trận
            allocations_json TEXT NOT NULL,     -- JSON list picks: [{type, side, line, prob, odd, stake, status, pnl}]
            total_pnl REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'open',-- 'open' | 'closed' | 'cancelled'
            created_at INTEGER NOT NULL,
            settled_at INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_msess_user ON match_sessions(user_id, status);
        CREATE INDEX IF NOT EXISTS idx_msess_fix ON match_sessions(user_id, fixture_id);
        """)
        # Migration: thêm cột points nếu users đã tồn tại từ trước nhưng chưa có
        try:
            c.execute("ALTER TABLE users ADD COLUMN points INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # column đã tồn tại
        # Migration: bankroll (vốn ban đầu để Kelly + ROI %)
        try:
            c.execute("ALTER TABLE users ADD COLUMN bankroll REAL NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass


def _hash_password(password: str, salt: str = None):
    if not salt:
        salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITER,
    )
    return h.hex(), salt


def create_user(username: str, password: str, role: str = "user", note: str = ""):
    """Tạo user mới. Trả (ok, message)."""
    username = (username or "").strip()
    password = (password or "").strip()
    if len(username) < 3:
        return False, "Username phải ít nhất 3 ký tự"
    if len(password) < 4:
        return False, "Mật khẩu phải ít nhất 4 ký tự"
    if role not in ("user", "admin"):
        role = "user"
    h, s = _hash_password(password)
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO users(username,password_hash,salt,role,created_at,note) VALUES(?,?,?,?,?,?)",
                (username, h, s, role, int(time.time()), note or ""),
            )
        return True, "Đã tạo user thành công"
    except sqlite3.IntegrityError:
        return False, "Username đã tồn tại"


def verify_user(username: str, password: str):
    """Kiểm tra login. Trả dict user hoặc None."""
    with _conn() as c:
        r = c.execute(
            "SELECT id, password_hash, salt, role FROM users WHERE username=?",
            (username,),
        ).fetchone()
        if not r:
            return None
        uid, ph, sl, role = r
        h, _ = _hash_password(password, sl)
        if not secrets.compare_digest(h, ph):
            return None
        c.execute("UPDATE users SET last_login=? WHERE id=?", (int(time.time()), uid))
        return {"id": uid, "username": username, "role": role}


def list_users():
    with _conn() as c:
        rows = c.execute(
            "SELECT id, username, role, created_at, last_login, note, points FROM users ORDER BY id"
        ).fetchall()
    return [
        {
            "id": r[0],
            "username": r[1],
            "role": r[2],
            "created_at": r[3],
            "last_login": r[4],
            "note": r[5] or "",
            "points": r[6] or 0,
        }
        for r in rows
    ]


def get_user(user_id: int):
    with _conn() as c:
        r = c.execute(
            "SELECT id, username, role, created_at, last_login, note, points FROM users WHERE id=?",
            (user_id,),
        ).fetchone()
    if not r:
        return None
    return {
        "id": r[0], "username": r[1], "role": r[2],
        "created_at": r[3], "last_login": r[4],
        "note": r[5] or "", "points": r[6] or 0,
    }


def get_points(user_id: int) -> int:
    with _conn() as c:
        r = c.execute("SELECT points FROM users WHERE id=?", (user_id,)).fetchone()
    return int(r[0]) if r else 0


def add_points(user_id: int, delta: int):
    """Cộng/trừ điểm. Không cho âm."""
    with _conn() as c:
        cur = c.execute("SELECT points FROM users WHERE id=?", (user_id,)).fetchone()
        if not cur:
            return False, "User không tồn tại"
        new_val = max(0, int(cur[0] or 0) + int(delta))
        c.execute("UPDATE users SET points=? WHERE id=?", (new_val, user_id))
    return True, f"OK (mới: {new_val}đ)"


def set_points(user_id: int, value: int):
    value = max(0, int(value))
    with _conn() as c:
        c.execute("UPDATE users SET points=? WHERE id=?", (value, user_id))
    return True, f"Đã set {value}đ"


def delete_user(user_id: int):
    """Xóa user (và sessions liên quan qua cascade)."""
    with _conn() as c:
        c.execute("DELETE FROM users WHERE id=?", (user_id,))


def update_password(user_id: int, new_password: str):
    if len(new_password or "") < 4:
        return False, "Mật khẩu phải ít nhất 4 ký tự"
    h, s = _hash_password(new_password)
    with _conn() as c:
        c.execute(
            "UPDATE users SET password_hash=?, salt=? WHERE id=?",
            (h, s, user_id),
        )
        # Đăng xuất khỏi mọi session khác sau khi đổi pass
        c.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
    return True, "Đã đổi mật khẩu"


def update_role(user_id: int, role: str):
    if role not in ("user", "admin"):
        return False, "Role không hợp lệ"
    with _conn() as c:
        c.execute("UPDATE users SET role=? WHERE id=?", (role, user_id))
    return True, "OK"


# ============== SESSIONS ==============

def create_session(user_id: int, days: int = 30, user_agent: str = ""):
    token = secrets.token_urlsafe(40)
    now = int(time.time())
    exp = now + days * 86400
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions(token,user_id,created_at,expires_at,user_agent) VALUES(?,?,?,?,?)",
            (token, user_id, now, exp, (user_agent or "")[:200]),
        )
        # Cleanup expired
        c.execute("DELETE FROM sessions WHERE expires_at<?", (now,))
    return token, exp


def get_session_user(token: str):
    if not token:
        return None
    with _conn() as c:
        r = c.execute(
            """SELECT u.id, u.username, u.role
               FROM sessions s JOIN users u ON u.id=s.user_id
               WHERE s.token=? AND s.expires_at>?""",
            (token, int(time.time())),
        ).fetchone()
    if r:
        return {"id": r[0], "username": r[1], "role": r[2]}
    return None


def delete_session(token: str):
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE token=?", (token,))


def list_user_sessions(user_id: int):
    """Liệt kê session đang active của 1 user."""
    with _conn() as c:
        rows = c.execute(
            "SELECT token, created_at, expires_at, user_agent FROM sessions WHERE user_id=? AND expires_at>? ORDER BY created_at DESC",
            (user_id, int(time.time())),
        ).fetchall()
    return [
        {"token_short": r[0][:8] + "...", "created_at": r[1], "expires_at": r[2], "user_agent": r[3]}
        for r in rows
    ]


# ============== BOOTSTRAP ==============

def bootstrap_admin():
    """Khởi tạo DB + tạo admin mặc định nếu DB chưa có user nào.
    Đọc env ADMIN_USERNAME (default 'admin') và ADMIN_PASSWORD (default 'admin123')."""
    init_db()
    with _conn() as c:
        count = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count == 0:
        u = os.getenv("ADMIN_USERNAME", "admin").strip() or "admin"
        p = os.getenv("ADMIN_PASSWORD", "admin123").strip() or "admin123"
        ok, msg = create_user(u, p, role="admin", note="Bootstrap admin")
        if ok:
            print(f"[users] ✓ Bootstrap admin '{u}' (đổi password ngay sau khi login!)")
        else:
            print(f"[users] ✗ Bootstrap admin lỗi: {msg}")


def stats():
    """Thống kê nhanh để dashboard."""
    now = int(time.time())
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        admins = c.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
        active_sessions = c.execute("SELECT COUNT(*) FROM sessions WHERE expires_at>?", (now,)).fetchone()[0]
        recent_logins = c.execute("SELECT COUNT(*) FROM users WHERE last_login>?", (now - 86400,)).fetchone()[0]
        total_points = c.execute("SELECT COALESCE(SUM(points),0) FROM users").fetchone()[0]
        total_unlocks = c.execute("SELECT COUNT(*) FROM user_predictions").fetchone()[0]
    return {
        "total_users": total,
        "admins": admins,
        "active_sessions": active_sessions,
        "logins_24h": recent_logins,
        "total_points": total_points,
        "total_unlocks": total_unlocks,
    }


# ============== MATCH COST + PAYMENT TRACKING ==============

def match_cost(fixture_id) -> int:
    """Cost xem phân tích 1 trận: deterministic random 5-20đ theo fixture_id.
    Cùng fixture_id → cùng cost (không thay đổi)."""
    try:
        fid = int(fixture_id)
    except (TypeError, ValueError):
        return 10  # fallback
    # Hash deterministic: hash số nguyên đơn giản
    h = (fid * 2654435761) & 0xFFFFFFFF   # Knuth multiplicative
    return 5 + (h % 16)   # 5..20


def has_paid(user_id: int, fixture_id: int) -> bool:
    """User đã trả phí cho fixture này chưa? (mua rồi xem lại free)."""
    with _conn() as c:
        r = c.execute(
            "SELECT 1 FROM user_predictions WHERE user_id=? AND fixture_id=?",
            (user_id, fixture_id),
        ).fetchone()
    return r is not None


def mark_paid(user_id: int, fixture_id: int, cost: int):
    """Trừ điểm + ghi nhận đã mua. Trả (ok, msg, points_left)."""
    with _conn() as c:
        # Lock row
        cur = c.execute("SELECT points FROM users WHERE id=?", (user_id,)).fetchone()
        if not cur:
            return False, "User không tồn tại", 0
        bal = int(cur[0] or 0)
        if bal < cost:
            return False, f"Thiếu điểm — cần {cost}đ, có {bal}đ", bal
        new_bal = bal - cost
        c.execute("UPDATE users SET points=? WHERE id=?", (new_bal, user_id))
        # Insert paid record (UNIQUE bảo vệ trùng)
        try:
            c.execute(
                "INSERT INTO user_predictions(user_id,fixture_id,cost,paid_at) VALUES(?,?,?,?)",
                (user_id, fixture_id, cost, int(time.time())),
            )
        except sqlite3.IntegrityError:
            # Đã có record (race condition), rollback trừ điểm
            c.execute("UPDATE users SET points=? WHERE id=?", (bal, user_id))
            return True, "Đã mua trước đó", bal
    return True, "Đã trừ điểm thành công", new_bal


def user_unlocks(user_id: int, limit: int = 50):
    """Liệt kê các trận user đã mua phân tích."""
    with _conn() as c:
        rows = c.execute(
            "SELECT fixture_id, cost, paid_at FROM user_predictions WHERE user_id=? ORDER BY paid_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [{"fixture_id": r[0], "cost": r[1], "paid_at": r[2]} for r in rows]


# ============== BET SLIP TRACKER ==============

VALID_BET_TYPES = ("1x2", "hdp", "ou", "btts", "corner")
VALID_STATUS = ("pending", "won", "lost", "push", "half_won", "half_lost")


def _bet_row_to_dict(r):
    return {
        "id": r[0], "user_id": r[1], "fixture_id": r[2],
        "home_team": r[3], "away_team": r[4], "league": r[5] or "",
        "bet_type": r[6], "pick": r[7], "line": r[8],
        "stake": r[9], "odd": r[10],
        "status": r[11], "profit": r[12], "note": r[13] or "",
        "placed_at": r[14], "settled_at": r[15],
    }


def create_bet(user_id: int, data: dict):
    """Tạo bet mới. Trả (ok, msg, bet_id)."""
    try:
        bet_type = (data.get("bet_type") or "1x2").strip().lower()
        if bet_type not in VALID_BET_TYPES:
            return False, f"bet_type không hợp lệ (phải là {VALID_BET_TYPES})", None
        pick = (data.get("pick") or "").strip().lower()
        if not pick:
            return False, "Thiếu pick", None
        stake = float(data.get("stake") or 0)
        odd = float(data.get("odd") or 0)
        if stake <= 0:
            return False, "Stake phải > 0", None
        if odd <= 1:
            return False, "Odd phải > 1.0 (decimal)", None
        home = (data.get("home_team") or "").strip()
        away = (data.get("away_team") or "").strip()
        if not home or not away:
            return False, "Thiếu tên 2 đội", None
        line = data.get("line")
        try:
            line = float(line) if line not in (None, "") else None
        except (TypeError, ValueError):
            line = None
        with _conn() as c:
            cur = c.execute(
                """INSERT INTO user_bets(user_id,fixture_id,home_team,away_team,league,
                                          bet_type,pick,line,stake,odd,status,profit,note,placed_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    user_id,
                    int(data["fixture_id"]) if data.get("fixture_id") else None,
                    home, away, (data.get("league") or "").strip()[:120],
                    bet_type, pick, line, stake, odd,
                    "pending", 0.0, (data.get("note") or "").strip()[:200],
                    int(time.time()),
                ),
            )
            bid = cur.lastrowid
        return True, "Đã tạo cược", bid
    except Exception as e:
        return False, f"Lỗi tạo cược: {e}", None


def list_bets(user_id: int, status: str = "", limit: int = 200):
    """Liệt kê bet của user. status = '' để lấy tất cả."""
    q = "SELECT id,user_id,fixture_id,home_team,away_team,league,bet_type,pick,line,stake,odd,status,profit,note,placed_at,settled_at FROM user_bets WHERE user_id=?"
    args = [user_id]
    if status and status in VALID_STATUS:
        q += " AND status=?"
        args.append(status)
    q += " ORDER BY placed_at DESC LIMIT ?"
    args.append(int(limit))
    with _conn() as c:
        rows = c.execute(q, args).fetchall()
    return [_bet_row_to_dict(r) for r in rows]


def get_bet(bet_id: int, user_id: int):
    """Get 1 bet (chỉ trả nếu thuộc user_id để bảo mật)."""
    with _conn() as c:
        r = c.execute(
            "SELECT id,user_id,fixture_id,home_team,away_team,league,bet_type,pick,line,stake,odd,status,profit,note,placed_at,settled_at FROM user_bets WHERE id=? AND user_id=?",
            (bet_id, user_id),
        ).fetchone()
    return _bet_row_to_dict(r) if r else None


def settle_bet(bet_id: int, user_id: int, new_status: str):
    """Đánh dấu thắng/thua/hoà. Tự tính profit dựa trên status."""
    if new_status not in VALID_STATUS:
        return False, f"status không hợp lệ"
    bet = get_bet(bet_id, user_id)
    if not bet:
        return False, "Bet không tồn tại"
    stake, odd = bet["stake"], bet["odd"]
    if new_status == "won":
        profit = stake * (odd - 1)
    elif new_status == "lost":
        profit = -stake
    elif new_status == "push":
        profit = 0
    elif new_status == "half_won":
        profit = stake * (odd - 1) / 2
    elif new_status == "half_lost":
        profit = -stake / 2
    else:  # pending
        profit = 0
    settled_at = int(time.time()) if new_status != "pending" else None
    with _conn() as c:
        c.execute(
            "UPDATE user_bets SET status=?, profit=?, settled_at=? WHERE id=? AND user_id=?",
            (new_status, profit, settled_at, bet_id, user_id),
        )
    return True, f"Đã set {new_status} · lãi/lỗ: {profit:+.2f}"


def update_bet(bet_id: int, user_id: int, data: dict):
    """Sửa thông tin bet (chỉ stake/odd/note nếu chưa settled)."""
    bet = get_bet(bet_id, user_id)
    if not bet:
        return False, "Bet không tồn tại"
    if bet["status"] != "pending":
        return False, "Chỉ sửa được bet đang chờ"
    updates, args = [], []
    if "stake" in data and data["stake"] is not None:
        try:
            v = float(data["stake"])
            if v > 0: updates.append("stake=?"); args.append(v)
        except: pass
    if "odd" in data and data["odd"] is not None:
        try:
            v = float(data["odd"])
            if v > 1: updates.append("odd=?"); args.append(v)
        except: pass
    if "note" in data:
        updates.append("note=?"); args.append(str(data["note"])[:200])
    if not updates:
        return False, "Không có gì để update"
    args.extend([bet_id, user_id])
    with _conn() as c:
        c.execute(f"UPDATE user_bets SET {','.join(updates)} WHERE id=? AND user_id=?", args)
    return True, "Đã cập nhật"


def delete_bet(bet_id: int, user_id: int):
    with _conn() as c:
        c.execute("DELETE FROM user_bets WHERE id=? AND user_id=?", (bet_id, user_id))
    return True


def get_bankroll(user_id: int) -> float:
    with _conn() as c:
        r = c.execute("SELECT bankroll FROM users WHERE id=?", (user_id,)).fetchone()
    return float(r[0]) if r else 0.0


def set_bankroll(user_id: int, value: float):
    value = max(0.0, float(value))
    with _conn() as c:
        c.execute("UPDATE users SET bankroll=? WHERE id=?", (value, user_id))
    return True, f"Đã set bankroll {value:,.0f}"


def kelly_stake(bankroll: float, prob: float, odd: float, fraction: float = 0.25):
    """Kelly criterion stake suggestion.
    f* = (b*p - q) / b  với b = odd-1, p = win prob, q = 1-p.
    Dùng fractional Kelly (default 1/4) để giảm rủi ro."""
    if bankroll <= 0 or prob <= 0 or prob >= 1 or odd <= 1:
        return 0
    b = odd - 1
    p = prob
    q = 1 - p
    f = (b * p - q) / b
    if f <= 0:
        return 0
    stake = bankroll * f * fraction
    return max(0, round(stake, 2))


def bets_timeline(user_id: int, limit: int = 200):
    """Trả về timeline lãi/lỗ tích lũy theo ngày — cho profit chart."""
    with _conn() as c:
        rows = c.execute(
            """SELECT settled_at, profit FROM user_bets
               WHERE user_id=? AND status!='pending' AND settled_at IS NOT NULL
               ORDER BY settled_at ASC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    points = []
    cum = 0.0
    for ts, profit in rows:
        cum += float(profit or 0)
        points.append({
            "ts": ts,
            "date": time.strftime("%d/%m", time.localtime(ts)),
            "profit": round(float(profit or 0), 2),
            "cumulative": round(cum, 2),
        })
    return points


def evaluate_bet(bet: dict, home_score: int, away_score: int) -> str:
    """Cho 1 bet + kết quả thực, trả về status: won/lost/push/half_won/half_lost."""
    bt = bet["bet_type"]; pick = bet["pick"]; line = bet.get("line")
    diff = (home_score or 0) - (away_score or 0)
    total = (home_score or 0) + (away_score or 0)

    if bt == "1x2":
        actual = "home" if diff > 0 else ("away" if diff < 0 else "draw")
        return "won" if pick == actual else "lost"

    if bt == "hdp":
        # Asian Handicap: line là handicap cho home (vd -1.5 = home phải thắng ≥2 bàn)
        # Pick = home/away
        if line is None: return "push"
        adj = diff + float(line) if pick == "away" else diff + float(line)
        # Adj cho pick home: home_diff + line; pick away: away_diff + line (=-diff+(-line))
        if pick == "home":
            adj = diff + float(line)
        else:  # away
            adj = -diff - float(line)
        # Quarter line: split half (vd 0.25 means -0/-0.5)
        f = float(line); frac = abs(f - int(f))
        if frac in (0.25, 0.75):
            # Half won/lost — simplified
            if adj > 0.25: return "won"
            if adj < -0.25: return "lost"
            if abs(adj) <= 0.25: return "half_won" if adj > 0 else "half_lost"
        if adj > 0: return "won"
        if adj < 0: return "lost"
        return "push"

    if bt == "ou":
        if line is None: return "push"
        adj = total - float(line) if pick == "over" else float(line) - total
        f = float(line); frac = abs(f - int(f))
        if frac in (0.25, 0.75):
            if adj > 0.25: return "won"
            if adj < -0.25: return "lost"
            if abs(adj) <= 0.25: return "half_won" if adj > 0 else "half_lost"
        if adj > 0: return "won"
        if adj < 0: return "lost"
        return "push"

    if bt == "btts":
        both_scored = (home_score > 0) and (away_score > 0)
        actual = "yes" if both_scored else "no"
        return "won" if pick == actual else "lost"

    return "push"  # corner, unknown — không tự settle


def auto_settle_bets(results_by_fixture: dict):
    """Cho 1 dict {fixture_id: (home_score, away_score, finished)}, tự settle các pending bet.
    Trả về list bet đã update."""
    updated = []
    fids = [int(k) for k in results_by_fixture.keys()]
    if not fids:
        return updated
    placeholders = ",".join("?" * len(fids))
    with _conn() as c:
        rows = c.execute(
            f"SELECT id,user_id,fixture_id,home_team,away_team,league,bet_type,pick,line,stake,odd,status,profit,note,placed_at,settled_at FROM user_bets WHERE status='pending' AND fixture_id IN ({placeholders})",
            fids,
        ).fetchall()
    for r in rows:
        bet = _bet_row_to_dict(r)
        fid = bet["fixture_id"]
        result = results_by_fixture.get(fid) or results_by_fixture.get(str(fid))
        if not result:
            continue
        home_score, away_score, finished = result
        if not finished:
            continue
        new_status = evaluate_bet(bet, home_score, away_score)
        settle_bet(bet["id"], bet["user_id"], new_status)
        updated.append({"bet_id": bet["id"], "status": new_status, "score": f"{home_score}-{away_score}"})
    return updated


def bets_export_csv(user_id: int):
    """Xuất sổ cược ra CSV string."""
    import csv as _csv
    import io as _io
    bets = list_bets(user_id, limit=10000)
    out = _io.StringIO()
    w = _csv.writer(out)
    w.writerow(["ID","Ngày đặt","Trận","Giải","Loại kèo","Pick","Line","Stake","Odd","Status","Profit","Ngày settle","Ghi chú"])
    for b in bets:
        placed = time.strftime("%Y-%m-%d %H:%M", time.localtime(b["placed_at"])) if b["placed_at"] else ""
        settled = time.strftime("%Y-%m-%d %H:%M", time.localtime(b["settled_at"])) if b["settled_at"] else ""
        w.writerow([
            b["id"], placed,
            f"{b['home_team']} vs {b['away_team']}",
            b["league"], b["bet_type"], b["pick"], b["line"] if b["line"] is not None else "",
            b["stake"], b["odd"], b["status"], b["profit"], settled, b["note"]
        ])
    return out.getvalue()


def bets_stats(user_id: int):
    """Tính ROI, win rate, lãi/lỗ tổng + theo league + theo bet_type."""
    with _conn() as c:
        rows = c.execute(
            "SELECT bet_type, status, stake, profit, league FROM user_bets WHERE user_id=?",
            (user_id,),
        ).fetchall()
    total = len(rows)
    if total == 0:
        return {
            "total": 0, "pending": 0, "settled": 0,
            "won": 0, "lost": 0, "push": 0,
            "win_rate": 0, "total_stake": 0, "total_profit": 0,
            "roi": 0,
            "by_league": [], "by_type": [],
        }
    pending = sum(1 for r in rows if r[1] == "pending")
    won = sum(1 for r in rows if r[1] in ("won", "half_won"))
    lost = sum(1 for r in rows if r[1] in ("lost", "half_lost"))
    push = sum(1 for r in rows if r[1] == "push")
    settled = total - pending
    total_stake = sum(r[2] for r in rows if r[1] != "pending")
    total_profit = sum(r[3] for r in rows)
    win_rate = (won / settled) if settled > 0 else 0
    roi = (total_profit / total_stake) if total_stake > 0 else 0

    # By league
    league_map = {}
    for r in rows:
        if r[1] == "pending":
            continue
        lg = (r[4] or "Khác")
        d = league_map.setdefault(lg, {"league": lg, "total": 0, "won": 0, "stake": 0, "profit": 0})
        d["total"] += 1
        if r[1] in ("won", "half_won"):
            d["won"] += 1
        d["stake"] += r[2]
        d["profit"] += r[3]
    by_league = sorted(league_map.values(), key=lambda x: -x["total"])[:10]
    for d in by_league:
        d["win_rate"] = (d["won"] / d["total"]) if d["total"] > 0 else 0
        d["roi"] = (d["profit"] / d["stake"]) if d["stake"] > 0 else 0

    # By bet type
    type_map = {}
    for r in rows:
        if r[1] == "pending":
            continue
        bt = r[0]
        d = type_map.setdefault(bt, {"bet_type": bt, "total": 0, "won": 0, "stake": 0, "profit": 0})
        d["total"] += 1
        if r[1] in ("won", "half_won"):
            d["won"] += 1
        d["stake"] += r[2]
        d["profit"] += r[3]
    by_type = sorted(type_map.values(), key=lambda x: -x["total"])
    for d in by_type:
        d["win_rate"] = (d["won"] / d["total"]) if d["total"] > 0 else 0
        d["roi"] = (d["profit"] / d["stake"]) if d["stake"] > 0 else 0

    # Bankroll info
    initial_bankroll = get_bankroll(user_id)
    current_bankroll = initial_bankroll + total_profit
    bankroll_change_pct = (total_profit / initial_bankroll) if initial_bankroll > 0 else 0
    return {
        "total": total, "pending": pending, "settled": settled,
        "won": won, "lost": lost, "push": push,
        "win_rate": round(win_rate, 4),
        "total_stake": round(total_stake, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(roi, 4),
        "initial_bankroll": round(initial_bankroll, 2),
        "current_bankroll": round(current_bankroll, 2),
        "bankroll_change_pct": round(bankroll_change_pct, 4),
        "by_league": by_league,
        "by_type": by_type,
    }


# ==================== MATCH SESSIONS ====================
import json as _json


def allocate_kelly(capital: float, picks: list[dict], fraction: float = 0.25,
                   max_per_pick: float = 0.5) -> list[dict]:
    """Chia vốn theo Kelly cho nhiều pick.
    picks: [{type, side, line, prob, odd, ...}]
    Trả về picks với thêm field 'stake' (số điểm gợi ý đặt).

    Kelly: f* = (p*odd - 1) / (odd - 1).  Dùng fractional Kelly (0.25) để safer.
    max_per_pick: trần stake mỗi pick (0.5 = 50% capital).
    """
    if capital <= 0 or not picks:
        return picks
    # Tính raw Kelly cho từng pick
    kellies = []
    for pk in picks:
        p = max(0.0, min(1.0, float(pk.get("prob") or 0)))
        odd = float(pk.get("odd") or 2.0)
        if odd <= 1:
            f = 0
        else:
            b = odd - 1
            f_full = (p * odd - 1) / b
            f = max(0.0, f_full * fraction)
        f = min(f, max_per_pick)
        kellies.append(f)
    total_f = sum(kellies)
    # Nếu tổng > 1 → normalize để tổng stake không vượt vốn
    if total_f > 1:
        kellies = [f / total_f for f in kellies]
    # Gắn stake vào pick
    out = []
    for pk, f in zip(picks, kellies):
        new_pk = dict(pk)
        new_pk["stake"] = round(capital * f, 2)
        new_pk["kelly_fraction"] = round(f, 4)
        new_pk["status"] = "pending"
        new_pk["pnl"] = 0.0
        out.append(new_pk)
    return out


def evaluate_pick(pick: dict, home_score: int, away_score: int,
                  total_corners: int = None) -> tuple[str, float]:
    """Đánh giá 1 pick theo tỉ số cuối + góc.
    pick = {type, side, line, odd, stake, ...}
    Trả (status, pnl).
    Status: 'won' | 'lost' | 'push' | 'half_won' | 'half_lost'
    PnL: lãi/lỗ tính từ stake & odd."""
    bt = (pick.get("type") or "").lower()
    side = (pick.get("side") or "").lower()
    line = float(pick.get("line") or 0)
    odd = float(pick.get("odd") or 2.0)
    stake = float(pick.get("stake") or 0)
    diff = home_score - away_score
    total_goals = home_score + away_score

    if bt in ("ou", "over_under", "tài_xỉu"):
        actual = total_goals
        if actual > line:
            won = side in ("over", "tài", "tai", "t")
        elif actual < line:
            won = side in ("under", "xỉu", "xiu", "u")
        else:   # push (line nguyên)
            return ("push", 0.0)
        if won:
            return ("won", round(stake * (odd - 1), 2))
        return ("lost", round(-stake, 2))

    if bt in ("ah", "hdp", "handicap"):
        # line cho ĐỘI NHÀ. side='home' nghĩa pick nhà.
        adj = diff + line if side == "home" else -diff - line
        if adj > 0.01:
            return ("won", round(stake * (odd - 1), 2))
        if adj < -0.01:
            return ("lost", round(-stake, 2))
        return ("push", 0.0)

    if bt == "corner":
        if total_corners is None:
            return ("pending", 0.0)
        if total_corners > line:
            won = side in ("tài", "tai", "over", "t")
        elif total_corners < line:
            won = side in ("xỉu", "xiu", "under", "u")
        else:
            return ("push", 0.0)
        if won:
            return ("won", round(stake * (odd - 1), 2))
        return ("lost", round(-stake, 2))

    if bt == "1x2":
        if diff > 0: actual = "home"
        elif diff < 0: actual = "away"
        else: actual = "draw"
        if side == actual:
            return ("won", round(stake * (odd - 1), 2))
        return ("lost", round(-stake, 2))

    return ("pending", 0.0)


def msess_create(user_id: int, fixture_id: int, capital: float,
                 picks: list[dict], home: str = "", away: str = "", league: str = "") -> int:
    """Tạo session mới hoặc update nếu đã có cho fixture đang open."""
    now = int(time.time())
    allocated = allocate_kelly(capital, picks)
    with _conn() as c:
        # Nếu đã có session open cho fixture này → update
        existing = c.execute(
            "SELECT id FROM match_sessions WHERE user_id=? AND fixture_id=? AND status='open'",
            (user_id, fixture_id),
        ).fetchone()
        if existing:
            sid = existing[0]
            c.execute(
                "UPDATE match_sessions SET capital=?, allocations_json=?, home_team=?, away_team=?, league=? WHERE id=?",
                (capital, _json.dumps(allocated, ensure_ascii=False), home, away, league, sid),
            )
            return sid
        cur = c.execute(
            "INSERT INTO match_sessions(user_id,fixture_id,home_team,away_team,league,capital,allocations_json,total_pnl,status,created_at) "
            "VALUES (?,?,?,?,?,?,?,0,'open',?)",
            (user_id, fixture_id, home, away, league, capital, _json.dumps(allocated, ensure_ascii=False), now),
        )
        return cur.lastrowid


def msess_get(user_id: int, fixture_id: int = None, session_id: int = None) -> dict | None:
    """Lấy 1 session theo fixture (đang open) hoặc theo session_id."""
    with _conn() as c:
        if session_id:
            row = c.execute(
                "SELECT id,user_id,fixture_id,home_team,away_team,league,capital,allocations_json,total_pnl,status,created_at,settled_at "
                "FROM match_sessions WHERE id=? AND user_id=?",
                (session_id, user_id),
            ).fetchone()
        else:
            row = c.execute(
                "SELECT id,user_id,fixture_id,home_team,away_team,league,capital,allocations_json,total_pnl,status,created_at,settled_at "
                "FROM match_sessions WHERE user_id=? AND fixture_id=? AND status='open' ORDER BY id DESC LIMIT 1",
                (user_id, fixture_id),
            ).fetchone()
    if not row:
        return None
    return _msess_row_to_dict(row)


def _msess_row_to_dict(row) -> dict:
    return {
        "id": row[0], "user_id": row[1], "fixture_id": row[2],
        "home_team": row[3], "away_team": row[4], "league": row[5],
        "capital": row[6],
        "picks": _json.loads(row[7] or "[]"),
        "total_pnl": row[8],
        "status": row[9],
        "created_at": row[10], "settled_at": row[11],
    }


def msess_settle(user_id: int, session_id: int, home_score: int, away_score: int,
                 total_corners: int = None) -> dict:
    """Settle 1 session khi trận FT: tính PnL từng pick + total → close session."""
    sess = msess_get(user_id, session_id=session_id)
    if not sess:
        return {"ok": False, "msg": "Không tìm thấy session"}
    if sess["status"] != "open":
        return {"ok": False, "msg": "Session đã đóng"}
    new_picks = []
    total_pnl = 0.0
    for pk in sess["picks"]:
        status, pnl = evaluate_pick(pk, home_score, away_score, total_corners)
        pk2 = dict(pk)
        pk2["status"] = status
        pk2["pnl"] = pnl
        total_pnl += pnl
        new_picks.append(pk2)
    now = int(time.time())
    with _conn() as c:
        c.execute(
            "UPDATE match_sessions SET allocations_json=?, total_pnl=?, status='closed', settled_at=? WHERE id=?",
            (_json.dumps(new_picks, ensure_ascii=False), round(total_pnl, 2), now, session_id),
        )
    return {"ok": True, "total_pnl": round(total_pnl, 2), "picks": new_picks,
            "score": f"{home_score}-{away_score}", "total_corners": total_corners}


def msess_list(user_id: int, status: str = "", limit: int = 100) -> list[dict]:
    """List session của user."""
    q = "SELECT id,user_id,fixture_id,home_team,away_team,league,capital,allocations_json,total_pnl,status,created_at,settled_at FROM match_sessions WHERE user_id=?"
    params = [user_id]
    if status:
        q += " AND status=?"
        params.append(status)
    q += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with _conn() as c:
        rows = c.execute(q, params).fetchall()
    return [_msess_row_to_dict(r) for r in rows]


def msess_stats(user_id: int) -> dict:
    """Tổng hợp PnL toàn bộ session."""
    sessions = msess_list(user_id, status="closed", limit=10000)
    total_capital = sum(s["capital"] for s in sessions)
    total_pnl = sum(s["total_pnl"] for s in sessions)
    wins = sum(1 for s in sessions if s["total_pnl"] > 0)
    losses = sum(1 for s in sessions if s["total_pnl"] < 0)
    flat = sum(1 for s in sessions if s["total_pnl"] == 0)
    open_n = len(msess_list(user_id, status="open"))
    roi = (total_pnl / total_capital) if total_capital > 0 else 0
    return {
        "total_sessions": len(sessions),
        "open_sessions": open_n,
        "wins": wins, "losses": losses, "flat": flat,
        "win_rate": round(wins / len(sessions), 4) if sessions else 0,
        "total_capital": round(total_capital, 2),
        "total_pnl": round(total_pnl, 2),
        "roi": round(roi, 4),
    }


def msess_delete(user_id: int, session_id: int) -> bool:
    with _conn() as c:
        c.execute("DELETE FROM match_sessions WHERE id=? AND user_id=?", (session_id, user_id))
    return True
