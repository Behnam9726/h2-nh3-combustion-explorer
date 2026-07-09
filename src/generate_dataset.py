"""
generate_dataset.py
تولید دیتاست اصلی پروژه با استفاده از Latin Hypercube Sampling (LHS).

چرا LHS به‌جای گرید یکنواخت؟
    گرید یکنواخت (linspace x linspace) خیلی از فضای دو بعدی را با نقاط
    هم‌ردیف/هم‌ستون پر می‌کند که اطلاعات تکراری تولید می‌کند. LHS با همان
    تعداد نقطه، پوشش فضایی به‌مراتب بهتر و یکنواخت‌تری می‌دهد که برای
    آموزش مدل جانشین (به‌خصوص Gaussian Process) کیفیت بسیار بالاتری دارد.
    این روش استاندارد در Design of Experiments (DoE) صنعتی/تحقیقاتی است.

اجرا:
    python src/generate_dataset.py

خروجی:
    data/combustion_dataset.csv

نکته: هر شبیه‌سازی حدود ۵-۱۵ ثانیه طول می‌کشد. برای n_samples=90 اجرای
کامل ممکن است ۱۵-۳۰ دقیقه طول بکشد. برای تست سریع n_samples را کم کنید.
"""

import sys
import os
import time
import csv

sys.path.insert(0, os.path.dirname(__file__))
from simulate import run_flame_simulation
import numpy as np
from scipy.stats import qmc

# ریشه‌ی پروژه = یک پوشه بالاتر از src/  (صرف‌نظر از اینکه اسکریپت از کجا
# اجرا می‌شود، مسیر همیشه درست محاسبه می‌شود)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "data", "combustion_dataset.csv")


def lhs_points(h2_range, phi_range, n_samples, seed=42):
    """
    تولید n_samples نقطه با روش Latin Hypercube Sampling در فضای
    (h2_fraction, phi).
    """
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    unit_samples = sampler.random(n=n_samples)  # مقادیر بین 0 و 1
    lower = np.array([h2_range[0], phi_range[0]])
    upper = np.array([h2_range[1], phi_range[1]])
    scaled = qmc.scale(unit_samples, lower, upper)
    return scaled  # shape (n_samples, 2) -> ستون 0: h2, ستون 1: phi


def generate_dataset(
    h2_range=(0.0, 1.0),
    phi_range=(0.7, 1.3),
    n_samples=90,
    T_in=300.0,
    P=101325.0,
    output_path=None,
    seed=42,
):
    if output_path is None:
        output_path = DEFAULT_OUTPUT
    output_path = os.path.abspath(output_path)

    points = lhs_points(h2_range, phi_range, n_samples, seed=seed)

    print(f"شروع تولید دیتاست با LHS: {n_samples} نقطه شبیه‌سازی")
    print(f"بازه H2: {h2_range}, بازه phi: {phi_range}")
    print(f"مسیر مطلق فایل خروجی: {output_path}")

    rows = []
    t_start = time.time()

    for i, (h2, phi) in enumerate(points, start=1):
        t0 = time.time()
        res = run_flame_simulation(
            h2_fraction=float(h2), phi=float(phi), T_in=T_in, P=P
        )
        dt = time.time() - t0

        status = "OK" if res["success"] else "FAIL"
        print(
            f"[{i}/{n_samples}] H2={h2:.3f} phi={phi:.3f} "
            f"-> {status} ({dt:.1f}s)"
        )

        rows.append(res)

        # ذخیره‌ی تدریجی (incremental save) تا در صورت قطع اجرا داده از
        # دست نرود
        save_csv(rows, output_path)

    elapsed = time.time() - t_start
    n_ok = sum(1 for r in rows if r.get("success"))
    print(
        f"\nپایان. {len(rows)} رکورد ({n_ok} موفق) در "
        f"{elapsed/60:.1f} دقیقه ذخیره شد."
    )
    print(f"فایل خروجی: {output_path}")
    if os.path.exists(output_path):
        size_kb = os.path.getsize(output_path) / 1024
        print(f"تایید: فایل موجود است ({size_kb:.1f} کیلوبایت)")
    else:
        print("هشدار: فایل خروجی پیدا نشد! لطفاً دسترسی نوشتن پوشه را بررسی کنید.")


def save_csv(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "h2_fraction",
        "phi",
        "T_in",
        "P",
        "success",
        "Su",
        "Su_cm_s",
        "T_ad",
        "NO_ppm",
        "N2O_ppm",
        "NO2_ppm",
        "error",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    # برای تست سریع اولیه می‌توانید n_samples=12 بگذارید (~2-3 دقیقه)
    generate_dataset(
        h2_range=(0.0, 1.0),
        phi_range=(0.7, 1.3),
        n_samples=90,
        # output_path را مشخص نکنید تا خودکار در PROJECT_ROOT/data ذخیره شود
    )
