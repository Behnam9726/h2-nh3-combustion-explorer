"""
ignition_delay.py
محاسبه‌ی زمان تأخیر اشتعال (Ignition Delay Time) برای مخلوط H2/NH3.

روش: شبیه‌سازی یک راکتور همگن با فشار ثابت (Constant-Pressure Homogeneous
Reactor) که مخلوط از دمای بالا شروع می‌شود و زمان رسیدن به بیشترین
شیب افزایش دما (max dT/dt) به‌عنوان زمان تأخیر اشتعال در نظر گرفته
می‌شود. این معیار استاندارد در مقالات سینتیک شیمیایی احتراق است.

اجرا:
    python src/ignition_delay.py

خروجی:
    data/ignition_delay_dataset.csv
"""

import os
import sys
import time
import csv

import numpy as np
import cantera as ct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate import MECH_PATH

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "data", "ignition_delay_dataset.csv")


def compute_ignition_delay(h2_fraction, phi, T_in, P=101325.0, t_end=2.0):
    """
    محاسبه‌ی زمان تأخیر اشتعال (بر حسب ثانیه) برای یک شرایط مشخص.

    پارامترها
    ----------
    h2_fraction : کسر مولی H2 در سوخت (بقیه NH3)
    phi         : نسبت هم‌ارزی
    T_in        : دمای اولیه‌ی مخلوط (کلوین) — باید به‌اندازه‌ی کافی بالا
                  باشد تا احتراق خودبه‌خودی در بازه‌ی زمانی معقول رخ دهد
    P           : فشار (پاسکال)
    t_end       : حداکثر زمان شبیه‌سازی (ثانیه) قبل از توقف در صورت
                  عدم اشتعال (برای جلوگیری از حلقه‌ی بی‌نهایت)

    خروجی: دیکشنری شامل success, tau (زمان تأخیر بر ثانیه)، یا خطا
    """
    result = {
        "h2_fraction": h2_fraction,
        "phi": phi,
        "T_in": T_in,
        "P": P,
        "success": False,
    }
    try:
        gas = ct.Solution(MECH_PATH)
        fuel = {"H2": h2_fraction, "NH3": 1.0 - h2_fraction}
        gas.TP = T_in, P
        gas.set_equivalence_ratio(phi, fuel, {"O2": 1.0, "N2": 3.76})

        reactor = ct.IdealGasConstPressureReactor(gas, clone=False)
        sim = ct.ReactorNet([reactor])

        t = 0.0
        times = []
        temps = []
        # سقف تعداد گام برای جلوگیری از اجرای بی‌پایان در شرایط عدم اشتعال
        max_steps = 100_000
        steps = 0
        while t < t_end and steps < max_steps:
            t = sim.step()
            times.append(t)
            temps.append(reactor.T)
            steps += 1
            if reactor.T > T_in + 800:  # احتراق کامل رخ داده، ادامه لازم نیست
                break

        times = np.array(times)
        temps = np.array(temps)

        if len(times) < 3 or (temps[-1] - T_in) < 200:
            # اشتعالی در بازه‌ی زمانی رخ نداد
            result["success"] = False
            result["tau"] = None
            result["error"] = "no ignition within t_end"
            return result

        dTdt = np.gradient(temps, times)
        idx = int(np.argmax(dTdt))

        result.update(
            {
                "success": True,
                "tau": float(times[idx]),
                "tau_ms": float(times[idx] * 1000),
            }
        )
    except Exception as e:
        result["error"] = str(e)

    return result


def generate_ignition_dataset(
    h2_fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
    T_range=(1000, 1600),
    n_T=15,
    phi=1.0,
    P=101325.0,
    output_path=None,
):
    """
    اسکن روی چند ترکیب H2/NH3 و بازه‌ی دما برای ساخت نمودار استاندارد
    Arrhenius (log(tau) در مقابل 1000/T) که در مقالات سینتیک رایج است.
    """
    if output_path is None:
        output_path = DEFAULT_OUTPUT
    output_path = os.path.abspath(output_path)

    T_values = np.linspace(T_range[0], T_range[1], n_T)
    total = len(h2_fractions) * len(T_values)
    print(f"شروع محاسبه‌ی زمان تأخیر اشتعال: {total} نقطه")
    print(f"مسیر خروجی: {output_path}")

    rows = []
    count = 0
    t_start = time.time()

    for h2 in h2_fractions:
        for T in T_values:
            count += 1
            res = compute_ignition_delay(h2_fraction=h2, phi=phi, T_in=T, P=P)
            status = "OK" if res["success"] else "FAIL"
            tau_str = f"{res.get('tau_ms', float('nan')):.4f} ms" if res["success"] else "-"
            print(f"[{count}/{total}] H2={h2:.2f} T={T:.0f}K -> {status} ({tau_str})")
            rows.append(res)
            save_csv(rows, output_path)

    elapsed = time.time() - t_start
    n_ok = sum(1 for r in rows if r["success"])
    print(f"\nپایان. {len(rows)} رکورد ({n_ok} موفق) در {elapsed/60:.1f} دقیقه.")
    print(f"فایل خروجی: {output_path}")


def save_csv(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["h2_fraction", "phi", "T_in", "P", "success", "tau", "tau_ms", "error"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    # برای تست سریع می‌توانید n_T=5 بگذارید
    generate_ignition_dataset(
        h2_fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
        T_range=(1000, 1600),
        n_T=15,
        phi=1.0,
    )
