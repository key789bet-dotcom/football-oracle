"""
(Tùy chọn) Train model ML dự đoán kết quả trên dữ liệu lịch sử.
Yêu cầu file CSV: history.csv với các cột:
    home_xg, away_xg, result   (result = home/draw/away)

Bạn có thể tự thu thập history.csv bằng cách lặp gọi API-Football
cho các trận đã đá, tính home_xg/away_xg và nhãn kết quả thật.

Chạy:  python train_model.py
Kết quả: tạo file ml_model.joblib để predictor.py tự dùng.
"""
import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

CSV_PATH = os.path.join(os.path.dirname(__file__), "history.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml_model.joblib")


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Không tìm thấy {CSV_PATH}. Hãy chuẩn bị dữ liệu lịch sử trước.")
        return

    df = pd.read_csv(CSV_PATH)
    X = df[["home_xg", "away_xg"]].values
    y = df["result"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000, multi_class="multinomial")
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    print("Accuracy:", round(accuracy_score(y_te, pred), 3))
    try:
        print("Log loss:", round(log_loss(y_te, model.predict_proba(X_te)), 3))
    except Exception:
        pass

    joblib.dump(model, MODEL_PATH)
    print(f"Đã lưu model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
