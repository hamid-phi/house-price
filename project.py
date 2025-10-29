import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ----------------------------
# مرحله ۱: بارگذاری دیتاست
# ----------------------------
df = pd.read_csv("housePrice.csv")
print("نمونه داده‌ها:")
print(df.head())

# ----------------------------
# مرحله ۲: پاکسازی ستون‌های عددی با کاما، فاصله و مقادیر عجیب
# ----------------------------
def clean_numeric(col):
    return pd.to_numeric(
        col.astype(str).str.replace(',', '').str.replace(' ', '').str.strip(),
        errors='coerce'
    )

numeric_cols = ['Price', 'Price(USD)', 'Area', 'Room']  # ستون‌های عددی اصلی
for col in numeric_cols:
    df[col] = clean_numeric(df[col])

# ----------------------------
# مرحله ۳: پر کردن داده‌های خالی
# ----------------------------
# ستون‌های عددی
imputer_num = SimpleImputer(strategy="mean")
df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])

# ستون‌های متنی
categorical_cols = df.select_dtypes(include=['object']).columns
imputer_cat = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

print("بعد از پر کردن داده‌های خالی:")
print(df.head())

# ----------------------------
# مرحله ۴: مشخص کردن ستون هدف و ویژگی‌ها
# ----------------------------
target_column = 'Price'
X = df.drop(columns=[target_column, 'Address'])  # حذف ستون متنی Address
y = df[target_column]

# ----------------------------
# مرحله ۵: نرمال‌سازی داده‌ها
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# مرحله ۶: تقسیم داده‌ها به train/test
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ----------------------------
# مرحله ۷: آموزش و ارزیابی چند مدل
# ----------------------------
models = {
    "SVM": SVC(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\nModel: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("="*30)

# ----------------------------
# مرحله ۸: ذخیره بهترین مدل (مثلا RandomForest) برای استفاده بعدی
# ----------------------------
best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)
joblib.dump(best_model, "best_model.pkl")
print("مدل RandomForest ذخیره شد: best_model.pkl")




