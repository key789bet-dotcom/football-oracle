# Deploy lên VPS Hostinger (Ubuntu 24.04) — goal789.site

VPS: Ubuntu 24.04, IP `187.127.112.134`. Mọi lệnh chạy bằng tài khoản root qua SSH.

## 1. Đăng nhập VPS
Mở PowerShell (máy bạn) và SSH vào (nhập mật khẩu gốc của VPS):
```
ssh root@187.127.112.134
```
(Hoặc bấm nút **Terminal** ở trang VPS trên Hostinger.)

## 2. Cài công cụ cần thiết
```
apt update && apt -y upgrade
apt -y install python3 python3-pip python3-venv git nginx
```

## 3. Lấy code từ GitHub
Repo của bạn là Private nên cần xác thực. Cách dễ:
- Vào github.com → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate → tích quyền `repo` → copy token.
- Rồi clone (khi hỏi Username = `key789bet-dotcom`, Password = **dán token vừa tạo**):
```
cd /opt
git clone https://github.com/key789bet-dotcom/football-oracle.git
cd football-oracle
```

## 4. Cài thư viện
```
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
```

## 5. Tạo file .env (chứa token)
```
nano .env
```
Dán vào (rồi Ctrl+O, Enter, Ctrl+X để lưu):
```
DATA_PROVIDER=footballdata
FOOTBALLDATA_TOKEN=token_cua_ban
ODDS_API_KEY=key_cua_ban
CACHE_TTL=60
```

## 6. Chạy thử
```
./venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Mở trình duyệt: `http://187.127.112.134:8000` → thấy tool chạy là OK. Bấm Ctrl+C để dừng.

## 7. Chạy nền 24/7 bằng systemd
```
nano /etc/systemd/system/oracle.service
```
Dán:
```
[Unit]
Description=Football Oracle
After=network.target

[Service]
WorkingDirectory=/opt/football-oracle
ExecStart=/opt/football-oracle/venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000
Restart=always
EnvironmentFile=/opt/football-oracle/.env

[Install]
WantedBy=multi-user.target
```
Bật dịch vụ:
```
systemctl daemon-reload
systemctl enable --now oracle
systemctl status oracle      # thấy active (running) là OK
```

## 8. Nginx + tên miền goal789.site
```
nano /etc/nginx/sites-available/oracle
```
Dán:
```
server {
    listen 80;
    server_name goal789.site www.goal789.site;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $remote_addr;
    }
}
```
Kích hoạt:
```
ln -s /etc/nginx/sites-available/oracle /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
```

## 9. Trỏ DNS (trên Hostinger → Trình quản lý DNS của goal789.site)
- Bản ghi `A`, Tên `@`, Giá trị `187.127.112.134`
- Bản ghi `A`, Tên `www`, Giá trị `187.127.112.134`
Đợi 15ph–vài giờ.

## 10. Bật HTTPS miễn phí (sau khi DNS đã trỏ)
```
apt -y install certbot python3-certbot-nginx
certbot --nginx -d goal789.site -d www.goal789.site
```
Làm theo hướng dẫn → tự cấp SSL. Xong: mở https://goal789.site

## Cập nhật code về sau
```
cd /opt/football-oracle && git pull && systemctl restart oracle
```
