# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# -------------------------------
# 0. تنظیمات اولیه
# -------------------------------
CSV_FILE = "housePrice.csv"  # نام فایل CSV خودت را اینجا بذار
TARGET_COL = "Price"         # ستون هدف شما (مثلاً 'Price')
ADDRESS_COL = "Address"      # ستون آدرس (اگر نامش فرق داره تغییر بده)

# -------------------------------
# 1. خواندن داده‌ها
# -------------------------------
df = pd.read_csv(CSV_FILE)
print("ابتدا: dtypes\n", df.dtypes)
print("نمونه داده‌ها:\n", df.head())

# -------------------------------
# 2. تابع پاکسازی اعداد (حذف فاصله، کاما و حروف)
# -------------------------------
def clean_numeric_series(s):
    # تبدیل به str، حذف فاصله و کاما و هر چیز غیر عدد و نقطه، سپس to_numeric
    s_clean = s.astype(str).str.replace(r'[^\d\.\-]', '', regex=True).str.strip()
    return pd.to_numeric(s_clean, errors='coerce')

# فهرست ستون‌هایی که معمولا عددی هستند؛ اگر ستون‌ عددی دیگری داری اضافه کن
possible_numeric = ['Price', 'Price(USD)', 'Area', 'Room']
# فقط ستون‌هایی که واقعا در df هستن انتخاب میشن
numeric_cols = [c for c in possible_numeric if c in df.columns]

for col in numeric_cols:
    df[col] = clean_numeric_series(df[col])

# اگر ستون‌های بولی داری تبدیل کن به عدد
bool_cols = list(df.select_dtypes(include=['bool']).columns)
for c in bool_cols:
    df[c] = df[c].astype(int)

# نمایش وضعیت بعد از پاک‌سازی
print("\nبعد از پاکسازی اولیه:")
print(df[numeric_cols].head())
print("dtypes:\n", df[numeric_cols].dtypes)

# -------------------------------
# 3. پر کردن مقادیر عددی و متنی (Imputer)
# -------------------------------
# ستون‌های عددی کامل‌تر: هر چیزی که الان عددی است
num_cols_real = list(df.select_dtypes(include=[np.number]).columns)
imputer_num = SimpleImputer(strategy="mean")
df[num_cols_real] = imputer_num.fit_transform(df[num_cols_real])

# ستون‌های متنی (object) را با most_frequent پر می‌کنیم
cat_cols = list(df.select_dtypes(include=['object']).columns)
imputer_cat = SimpleImputer(strategy="most_frequent")
if cat_cols:
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

print("\nبعد از Imputer:")
print(df.head())
print("dtypes (کامل):\n", df.dtypes)

# -------------------------------
# 4. Label encode برای Address (اگر وجود دارد)
# -------------------------------
label_encoder = None
if ADDRESS_COL in df.columns:
    le = LabelEncoder()
    # ابتدا مطمئن شو هیچ NaN متنی نداریم (Imputer بالا باید این رو پر کرده باشه)
    df[ADDRESS_COL] = df[ADDRESS_COL].astype(str)
    df[ADDRESS_COL + "_enc"] = le.fit_transform(df[ADDRESS_COL])
    label_encoder = le
    # حالا میتوانیم یا ستون متنی را حذف کنیم یا از ستون enc استفاده کنیم
    # من ستون متنی اصلی را نگه می‌دارم اما از ستون enc برای مدل استفاده می‌کنم
    print("\nنمونه Address_enc:\n", df[[ADDRESS_COL, ADDRESS_COL + "_enc"]].head())
else:
    print("\nستون Address یافت نشد؛ label_encoder ساخته نشد.")

# -------------------------------
# 5. آماده‌سازی X و y (فقط ستون‌های عددی)
# -------------------------------
# انتخاب ویژگی‌ها: همهٔ ستون‌های عددی به جز هدف
if TARGET_COL not in df.columns:
    raise ValueError(f"ستون هدف '{TARGET_COL}' در فایل وجود ندارد. نام ستون را بررسی کن.")

y = df[TARGET_COL].astype(float)

# اگر Address_enc ساخته شده، از آن استفاده کن؛ در غیر اینصورت فقط ستون‌های عددی را می‌گیریم
feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET_COL]
X = df[feature_cols]

print("\nویژگی‌ها (X) که به مدل می‌دهیم:\n", feature_cols)
print("نمونه X:\n", X.head())
print("نمونه y:\n", y.head())

# -------------------------------
# 6. نرمال‌سازی و تقسیم داده
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("\nShapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# -------------------------------
# 7. آموزش مدل
# -------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 8. ارزیابی
# -------------------------------
y_pred = model.predict(X_test)
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# -------------------------------
# 9. ذخیره مدل و ابزارها
# -------------------------------
joblib.dump(model, 'house_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
# اگر label_encoder ساخته شده ذخیره شود، در غیر اینصورت None ذخیره می‌شود
if label_encoder is not None:
    joblib.dump(label_encoder, 'label_encoder.pkl')
else:
    # برای سازگاری API بعدی یک label_encoder خالی بسازیم و ذخیره کنیم
    le_dummy = LabelEncoder()
    # fit با یک مقدار پیش‌فرض تا فایل ساخته شود
    le_dummy.classes_ = np.array(["unknown"])
    joblib.dump(le_dummy, 'label_encoder.pkl')

print("\n✅ مدل و scaler و label_encoder ذخیره شدند.")
