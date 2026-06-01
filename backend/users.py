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
        """)
        # Migration: thêm cột points nếu users đã tồn tại từ trước nhưng chưa có
        try:
            c.execute("ALTER TABLE users ADD COLUMN points INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # column đã tồn tại


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
