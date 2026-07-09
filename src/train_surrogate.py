"""
train_surrogate.py
آموزش مدل‌های جانشین Gaussian Process (GP / Kriging) روی دیتاست شبیه‌سازی
احتراق.

چرا Gaussian Process به‌جای Random Forest؟
    توابع فیزیکی احتراق (سرعت شعله، دما، NOx نسبت به ترکیب سوخت و phi)
    توابعی صاف و پیوسته‌اند. GP دقیقاً برای درون‌یابی چنین توابعی طراحی
    شده و با تعداد نقاط کم هم دقت بالایی می‌دهد؛ Random Forest چون بر پایه‌ی
    تقسیم‌بندی درختی کار می‌کند برای داده‌ی کم و توابع صاف مناسب نیست.
    مزیت اضافه‌ی GP: علاوه بر پیش‌بینی مقدار میانگین، عدم قطعیت
    (uncertainty / standard deviation) هر پیش‌بینی را هم می‌دهد.

اجرا:
    python src/train_surrogate.py

خروجی:
    data/surrogate_models.joblib
    شامل: مدل‌های GP آموزش‌دیده + StandardScaler ورودی‌ها + امتیاز دقت
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern,
    ConstantKernel,
    WhiteKernel,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "combustion_dataset.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "surrogate_models.joblib")

TARGETS = ["Su_cm_s", "T_ad", "NO_ppm"]
FEATURES = ["h2_fraction", "phi"]


def load_clean_data(path=DATA_PATH):
    df = pd.read_csv(path)
    before = len(df)
    df = df[df["success"] == True].copy()  # noqa: E712
    df = df.dropna(subset=FEATURES + TARGETS)
    after = len(df)
    print(
        f"داده‌های بارگذاری‌شده: {after} از {before} رکورد "
        "(بعد از حذف رکوردهای ناموفق/ناقص)"
    )
    return df


def build_kernel():
    """
    کرنل Matern (نرم‌تر از RBF، برای توابع فیزیکی واقعی مناسب‌تر است)
    + WhiteKernel برای مدل کردن نویز عددی شبیه‌سازی.
    """
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 1e1))
    )


def train_models(df):
    X_raw = df[FEATURES].values

    # نرمال‌سازی ورودی‌ها ضروری است چون GP به مقیاس ابعاد حساس است
    # (h2_fraction در بازه 0-1 ولی phi در بازه‌ی متفاوتی است)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    models = {}
    scores = {}

    for target in TARGETS:
        y_raw = df[target].values
        # نرمال‌سازی خروجی هم به پایداری آموزش GP کمک می‌کند
        y_mean, y_std = y_raw.mean(), y_raw.std()
        y = (y_raw - y_mean) / y_std

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        gp = GaussianProcessRegressor(
            kernel=build_kernel(),
            n_restarts_optimizer=8,
            normalize_y=False,  # چون خودمان دستی نرمال کردیم
            random_state=42,
        )
        gp.fit(X_train, y_train)

        y_pred_norm = gp.predict(X_test)
        y_pred = y_pred_norm * y_std + y_mean
        y_test_orig = y_test * y_std + y_mean

        r2 = r2_score(y_test_orig, y_pred)
        mae = mean_absolute_error(y_test_orig, y_pred)
        scores[target] = {"r2": r2, "mae": mae}
        print(f"  {target}: R2={r2:.4f}  MAE={mae:.3f}")

        # آموزش نهایی روی کل داده برای استفاده در بهینه‌سازی/داشبورد
        gp_full = GaussianProcessRegressor(
            kernel=build_kernel(),
            n_restarts_optimizer=8,
            normalize_y=False,
            random_state=42,
        )
        gp_full.fit(X, y)

        models[target] = {
            "gp": gp_full,
            "y_mean": y_mean,
            "y_std": y_std,
        }

    return models, scores, scaler


def main():
    print("در حال بارگذاری دیتاست...")
    df = load_clean_data()

    if len(df) < 10:
        print(
            "هشدار: تعداد رکوردهای معتبر خیلی کم است "
            f"({len(df)}). حداقل ۳۰-۵۰ نقطه‌ی معتبر برای GP توصیه می‌شود."
        )

    print("\nآموزش مدل‌های Gaussian Process برای هر خروجی:")
    models, scores, scaler = train_models(df)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "scaler": scaler,
            "scores": scores,
            "features": FEATURES,
            "targets": TARGETS,
            "h2_range": (float(df["h2_fraction"].min()), float(df["h2_fraction"].max())),
            "phi_range": (float(df["phi"].min()), float(df["phi"].max())),
        },
        MODEL_PATH,
    )
    print(f"\nمدل‌ها ذخیره شدند: {MODEL_PATH}")
    print("\nخلاصه‌ی دقت مدل‌ها (R2 نزدیک به 1 یعنی دقت بالا):")
    for t, s in scores.items():
        print(f"  {t}: R2={s['r2']:.4f}  MAE={s['mae']:.3f}")


def predict(models_bundle, h2_fraction, phi, target):
    """
    تابع کمکی برای پیش‌بینی با مدل GP ذخیره‌شده.
    ورودی می‌تواند اسکالر یا آرایه باشد.
    خروجی: (mean, std) -> میانگین پیش‌بینی و انحراف‌معیار (عدم قطعیت)
    """
    scaler = models_bundle["scaler"]
    entry = models_bundle["models"][target]
    gp = entry["gp"]
    y_mean, y_std = entry["y_mean"], entry["y_std"]

    X_raw = np.column_stack([np.atleast_1d(h2_fraction), np.atleast_1d(phi)])
    X = scaler.transform(X_raw)

    mean_norm, std_norm = gp.predict(X, return_std=True)
    mean = mean_norm * y_std + y_mean
    std = std_norm * y_std  # عدم قطعیت هم باید مقیاس شود
    return mean, std


if __name__ == "__main__":
    main()
