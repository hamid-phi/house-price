import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------- مرحله 1: بارگذاری دیتاست ----------
df = pd.read_csv("housePrice.csv")  # فایل CSV خودت رو اینجا بذار

# ---------- مرحله 2: پاک‌سازی ستون قیمت ----------
# تبدیل ستون Price به string
df['Price'] = df['Price'].astype(str)

# حذف کاراکترهای غیر عددی (مثل کاما)
df['Price'] = df['Price'].str.replace(',', '').str.replace(' ', '')

# تبدیل به عدد
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# جایگزینی NaN با میانگین
df['Price'] = df['Price'].fillna(df['Price'].mean())

# ---------- مرحله 3: پیش‌پردازش ستون‌های متنی ----------
# اگر ستون‌های متنی داریم، LabelEncoder استفاده می‌کنیم
text_columns = df.select_dtypes(include=['object']).columns
for col in text_columns:
    if col != 'Price':  # ستون قیمت رو تغییر نده
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# ---------- مرحله 4: جداسازی ویژگی‌ها و هدف ----------
X = df.drop('Price', axis=1)
y = df['Price']

# ---------- مرحله 5: تقسیم داده به آموزش و تست ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- مرحله 6: نرمال‌سازی (اختیاری برای RandomForest) ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- مرحله 7: ساخت و آموزش مدل ----------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ---------- مرحله 8: پیش‌بینی و ارزیابی ----------
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")


plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()