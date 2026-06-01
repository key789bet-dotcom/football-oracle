# Đưa tool lên goal789.site (Render + Hostinger DNS)

## Bước 1 — Đẩy code lên GitHub
1. Tạo tài khoản github.com (nếu chưa có) → tạo repo mới (Private), vd `football-oracle`.
2. Trên máy, mở PowerShell tại thư mục project rồi chạy:
   ```
   git init
   git add .
   git commit -m "football oracle"
   git branch -M main
   git remote add origin https://github.com/<TEN_CUA_BAN>/football-oracle.git
   git push -u origin main
   ```
   (Nếu chưa có git: tải tại https://git-scm.com)
   File `.env` và `track_log.json` đã được `.gitignore` loại trừ — KHÔNG bị lộ token.

## Bước 2 — Tạo Web Service trên Render (miễn phí)
1. Vào https://render.com → đăng ký → **New +** → **Web Service**.
2. Kết nối GitHub, chọn repo `football-oracle`.
3. Render tự đọc `render.yaml`. Nếu hỏi thủ công thì điền:
   - **Runtime**: Python
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Mục **Environment** → thêm biến bí mật (KHÔNG commit):
   - `FOOTBALLDATA_TOKEN` = token football-data của bạn
   - `ODDS_API_KEY` = key the-odds-api của bạn
   - `DATA_PROVIDER` = `footballdata`
5. Bấm **Create Web Service**. Đợi build xong → được link kiểu `https://football-oracle.onrender.com`. Mở thử, tool chạy.

## Bước 3 — Trỏ tên miền goal789.site về Render
1. Trong Render: **Settings → Custom Domains → Add** → nhập `goal789.site` (và `www.goal789.site`). Render sẽ cho bạn một giá trị CNAME/A.
2. Sang Hostinger (trang DNS đang mở) → **Quản lý bản ghi DNS** → Thêm bản ghi:
   - Với `www`: Loại `CNAME`, Tên `www`, Giá trị = `football-oracle.onrender.com` (giá trị Render đưa).
   - Với gốc `@`: Render thường yêu cầu `A` trỏ về IP của Render, hoặc dùng ALIAS/redirect. Làm theo đúng giá trị Render hiển thị ở bước Custom Domains.
3. Đợi 15 phút–24 giờ để DNS cập nhật. Render tự cấp HTTPS.

## Lưu ý
- **Gói free Render** ngủ sau ~15 phút không truy cập → lần mở đầu hơi chậm (~30s khởi động lại). Muốn chạy 24/7 mượt thì nâng gói trả phí.
- `track_log.json` (sổ kèo) trên Render free sẽ **mất khi service restart** (ổ đĩa tạm). Muốn lưu vĩnh viễn cần gắn Disk (trả phí) hoặc dùng DB ngoài.
- Bảo mật: token chỉ đặt ở Environment của Render, không bao giờ commit vào repo.
