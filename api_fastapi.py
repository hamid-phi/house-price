# api_fastapi.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# ---------- بارگذاری مدل‌ها و ابزارها ----------
model = joblib.load("house_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")  # فرض می‌کنیم دارای attribute classes_

app = FastAPI(title="House Price Predictor")

# ---------- مدل داده ورودی ----------
class HouseIn(BaseModel):
    Area: float
    Room: int
    Parking: int    # 0/1
    Warehouse: int  # 0/1
    Elevator: int   # 0/1
    Address: str

# ---------- endpoint پیش‌بینی ----------
@app.post("/predict")
def predict(h: HouseIn):
    # تبدیل Address به عدد
    try:
        addr_enc = int(le.transform([h.Address])[0])
    except Exception:
        addr_enc = 0  # اگر آدرس جدید بود، عدد پیش‌فرض بگذار

    # تمام ستون‌ها: 6 ستون ورودی + 1 ستون Address encoded
    # اگر مدل شما 7 ستون می‌خواهد، ستون اضافی را صفر قرار دهید
    X = np.array([[h.Area, h.Room, h.Parking, h.Warehouse, h.Elevator, addr_enc, 0]], dtype=float)

    # استانداردسازی
    X_scaled = scaler.transform(X)

    # پیش‌بینی
    pred = model.predict(X_scaled)[0]
    return {"predicted_price": float(pred)}
