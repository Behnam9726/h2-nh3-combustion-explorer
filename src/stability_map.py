"""
stability_map.py
نقشه‌ی پایداری/اشتعال‌پذیری شعله: برای هر ترکیب H2/NH3، بازه‌ی نسبت
هم‌ارزی (phi) که در آن شعله پایدار باقی می‌ماند (بین حد رقیق/lean و
غنی/rich) را پیدا می‌کند.

نکته‌ی مهندسی: نزدیک به محدوده‌ی اشتعال‌پذیری (flammability limit)،
حل‌گر عددی Cantera گاهی پیش از تشخیص شکست، دقایقی طول می‌کشد (تلاش
مکرر برای همگرایی). برای جلوگیری از کند شدن کل اسکن، هر شبیه‌سازی در
یک پردازش (process) جداگانه با سقف زمانی (timeout) اجرا می‌شود؛ اگر از
سقف زمانی عبور کند، آن نقطه به‌عنوان «نزدیک به خاموشی/ناپایدار»
علامت‌گذاری می‌شود. این روش پایدار و مستقل از سیستم‌عامل است (ویندوز،
لینوکس، مک).

اجرا:
    python src/stability_map.py

خروجی:
    data/stability_map_dataset.csv
"""

import os
import sys
import time
import csv
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "data", "stability_map_dataset.csv")


def _worker(h2_fraction, phi, width, queue):
    """این تابع داخل پردازش جدا اجرا می‌شود."""
    from simulate import run_flame_simulation

    res = run_flame_simulation(h2_fraction=h2_fraction, phi=phi, width=width)
    queue.put(res)


def evaluate_with_timeout(h2_fraction, phi, width=0.02, timeout=25):
    """
    اجرای شبیه‌سازی شعله در یک پردازش جدا با سقف زمانی. اگر زمان تمام
    شود، پردازش به‌زور بسته می‌شود و نتیجه «timeout» (به معنای نزدیکی
    به محدوده‌ی خاموشی) برگردانده می‌شود.
    """
    queue = mp.Queue()
    p = mp.Process(target=_worker, args=(h2_fraction, phi, width, queue))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return {
            "h2_fraction": h2_fraction,
            "phi": phi,
            "success": False,
            "Su_cm_s": None,
            "error": "timeout_near_flammability_limit",
        }

    if not queue.empty():
        return queue.get()

    return {
        "h2_fraction": h2_fraction,
        "phi": phi,
        "success": False,
        "Su_cm_s": None,
        "error": "process_crashed",
    }


def find_flammability_limits(h2_fraction, phi_min=0.2, phi_max=4.0, n_points=15, timeout=25):
    """
    اسکن یکنواخت روی بازه‌ی phi برای یک ترکیب H2 مشخص و ثبت موفقیت/شکست
    هر نقطه. خروجی برای رسم منطقه‌ی پایدار روی نقشه استفاده می‌شود.
    """
    phi_values = np.linspace(phi_min, phi_max, n_points)
    rows = []
    for phi in phi_values:
        res = evaluate_with_timeout(h2_fraction, float(phi), timeout=timeout)
        rows.append(res)
    return rows


def generate_stability_map(
    h2_fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
    phi_min=0.2,
    phi_max=4.0,
    n_points=15,
    timeout=25,
    output_path=None,
):
    if output_path is None:
        output_path = DEFAULT_OUTPUT
    output_path = os.path.abspath(output_path)

    total = len(h2_fractions) * n_points
    print(f"شروع اسکن نقشه‌ی پایداری: {total} نقطه (سقف زمانی هر نقطه: {timeout}s)")
    print(f"مسیر خروجی: {output_path}")

    all_rows = []
    count = 0
    t_start = time.time()

    for h2 in h2_fractions:
        phi_values = np.linspace(phi_min, phi_max, n_points)
        for phi in phi_values:
            count += 1
            t0 = time.time()
            res = evaluate_with_timeout(h2, float(phi), timeout=timeout)
            dt = time.time() - t0
            status = "OK" if res["success"] else f"FAIL ({res.get('error', '?')})"
            print(f"[{count}/{total}] H2={h2:.2f} phi={phi:.2f} -> {status} ({dt:.1f}s)")
            all_rows.append(res)
            save_csv(all_rows, output_path)

    elapsed = time.time() - t_start
    n_ok = sum(1 for r in all_rows if r["success"])
    print(f"\nپایان. {len(all_rows)} رکورد ({n_ok} پایدار) در {elapsed/60:.1f} دقیقه.")
    print(f"فایل خروجی: {output_path}")


def save_csv(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["h2_fraction", "phi", "success", "Su_cm_s", "error"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    # نکته‌ی مهم ویندوز: multiprocessing روی ویندوز نیاز به این
    # if __name__ == "__main__" دارد، وگرنه با خطای بازتولید بی‌نهایت
    # مواجه می‌شوید. این بلوک دقیقاً برای همین منظور است.
    generate_stability_map(
        h2_fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
        phi_min=0.2,
        phi_max=4.0,
        n_points=15,
        timeout=25,
    )
