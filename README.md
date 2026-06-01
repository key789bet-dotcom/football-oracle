# ⚽ Tool dự đoán kết quả bóng đá realtime

Tool dự đoán thắng/hòa/thua dùng dữ liệu **realtime**, hỗ trợ 2 nguồn: **API-Football** hoặc **SofaScore** (đổi bằng biến `DATA_PROVIDER`). Kết hợp tối đa 4 tín hiệu:

1. **Poisson** – mô hình xác suất dựa trên phong độ ghi/thủng lưới gần đây.
2. **Odds** – xác suất ẩn trong tỉ lệ kèo (đã khử biên lợi nhuận nhà cái). *(API-Football)*
3. **Machine Learning** – (tùy chọn) model train trên dữ liệu lịch sử.
4. **H2H** – lịch sử đối đầu trực tiếp giữa 2 đội. *(SofaScore)*

Kết quả cuối = trung bình có trọng số của các nguồn có sẵn.

## Cấu trúc

```
toll bóng đá/
├── backend/
│   ├── api_client.py     # gọi API-Football (có cache)
│   ├── predictor.py      # Poisson + Odds + ML, hàm predict()
│   ├── train_model.py    # (tùy chọn) train model ML từ history.csv
│   └── main.py           # FastAPI: /api/matches, /api/predict
├── frontend/
│   └── index.html        # giao diện web realtime, tự refresh 30s
├── requirements.txt
└── .env.example
```

## Cài đặt

1. **Lấy API key** (miễn phí, ~100 request/ngày):
   - Đăng ký tại https://rapidapi.com/api-sports/api/api-football
   - Copy `X-RapidAPI-Key`.

2. **Tạo file `.env`** từ mẫu rồi điền key:
   ```
   cp .env.example .env      # Windows: copy .env.example .env
   ```

3. **Cài thư viện:**
   ```
   pip install -r requirements.txt
   ```

## Chạy

Từ thư mục gốc project:
```
uvicorn backend.main:app --reload
```
Mở trình duyệt: **http://127.0.0.1:8000**

- Tab "🔴 Đang đá" – trận live, tự làm mới mỗi 30s.
- Tab "📅 Hôm nay" hoặc chọn ngày.
- Bấm **Dự đoán** ở mỗi trận để xem xác suất.

API docs tự động: http://127.0.0.1:8000/docs

## Dùng SofaScore thay vì API-Football

1. Trong `.env` đặt:
   ```
   DATA_PROVIDER=sofascore
   SOFA_HOST=sofascore.p.rapidapi.com
   RAPIDAPI_KEY=key_sofascore_cua_ban
   ```
2. **Quan trọng:** mở tab *Endpoints* trên trang RapidAPI của bạn và đối chiếu tên
   đường dẫn + tham số với các hằng `PATH_*` ở đầu `backend/sofascore_client.py`
   (mỗi provider đặt tên hơi khác). Phần parse đã tự dò field nên ít phải sửa.
3. Chạy như bình thường. Nút "Dự đoán" sẽ tự dùng thêm tín hiệu **H2H** khi có `customId`.

> Mẹo bảo mật: đừng chia sẻ API key công khai. Nếu đã lỡ lộ, vào RapidAPI bấm
> **Regenerate** để tạo key mới.

## (Tùy chọn) Bật ML

1. Chuẩn bị `backend/history.csv` với cột: `home_xg, away_xg, result` (result = `home`/`draw`/`away`).
   Thu thập bằng cách lặp gọi API-Football cho các trận đã đá, tính xG từ phong độ và nhãn kết quả thật.
2. Train:
   ```
   python backend/train_model.py
   ```
   Tạo `ml_model.joblib`. Sau đó `predictor.py` tự dùng model này khi dự đoán.

## Tinh chỉnh

Trong `backend/predictor.py`:
- `W_POISSON`, `W_ODDS`, `W_ML` – trọng số trộn các nguồn.
- Hệ số lợi thế sân nhà `1.15` / `0.95` trong `expected_goals_from_form`.

## Lưu ý

- Free tier có giới hạn request → đã có cache (`CACHE_TTL`) để tiết kiệm.
- Dự đoán mang tính tham khảo, **không** dùng để cá cược. Bóng đá có yếu tố ngẫu nhiên cao.
