"""
sensitivity_analysis.py
تحلیل حساسیت واکنش‌های شیمیایی مؤثر بر تشکیل NOx در شعله‌ی H2/NH3.

روش: برای هر واکنشی که مستقیماً NO تولید یا مصرف می‌کند، ثابت نرخ
واکنش (multiplier) اندکی افزایش داده می‌شود و تأثیرش روی غلظت نهایی NO
در خروجی شعله اندازه‌گیری می‌شود (روش استاندارد تفاضل محدود / finite-
difference reaction rate sensitivity که در نرم‌افزارهای سینتیک شیمیایی
مثل Chemkin و Cantera رایج است).

ضریب حساسیت نرمال‌شده:
    S_i = [ (NO_perturbed - NO_base) / NO_base ] / [ (k_new - k_old) / k_old ]

مقدار مثبت بزرگ  -> واکنش باعث افزایش NOx می‌شود (تسریع‌کننده)
مقدار منفی بزرگ  -> واکنش باعث کاهش NOx می‌شود (مصرف‌کننده/بازدارنده)

اجرا:
    python src/sensitivity_analysis.py

خروجی:
    data/nox_sensitivity_dataset.csv
"""

import os
import sys
import time
import csv

import cantera as ct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate import MECH_PATH

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "data", "nox_sensitivity_dataset.csv")


def find_no_related_reactions(mech_path=MECH_PATH):
    """پیدا کردن شماره و معادله‌ی همه‌ی واکنش‌هایی که مستقیماً NO در آن‌ها شرکت دارد."""
    gas = ct.Solution(mech_path)
    reactions = []
    for i, rxn in enumerate(gas.reactions()):
        species_in_rxn = set(rxn.reactants.keys()) | set(rxn.products.keys())
        if "NO" in species_in_rxn:
            reactions.append((i, rxn.equation))
    return reactions


def run_flame_with_multiplier(h2_fraction, phi, reaction_index=None, multiplier=1.0,
                                T_in=300.0, P=101325.0, width=0.03):
    """
    اجرای شبیه‌سازی شعله با امکان تغییر ضریب نرخ یک واکنش مشخص
    (برای تحلیل حساسیت). اگر reaction_index=None باشد، شبیه‌سازی پایه
    (بدون تغییر) اجرا می‌شود.
    """
    gas = ct.Solution(MECH_PATH)
    fuel = {"H2": h2_fraction, "NH3": 1.0 - h2_fraction}
    gas.TP = T_in, P
    gas.set_equivalence_ratio(phi, fuel, {"O2": 1.0, "N2": 3.76})

    if reaction_index is not None:
        gas.set_multiplier(multiplier, reaction_index)

    flame = ct.FreeFlame(gas, width=width)
    flame.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
    flame.transport_model = "mixture-averaged"
    flame.solve(loglevel=0, auto=True)

    idx_no = gas.species_index("NO")
    NO_ppm = float(flame.X[idx_no, -1] * 1e6)
    return NO_ppm


def compute_nox_sensitivity(h2_fraction=0.5, phi=1.0, delta=0.1, output_path=None):
    """
    محاسبه‌ی ضریب حساسیت NOx برای همه‌ی واکنش‌های مرتبط با NO.

    delta: میزان افزایش نسبی ثابت نرخ واکنش برای هر پرتوربیشن (پیش‌فرض ۱۰٪)
    """
    if output_path is None:
        output_path = DEFAULT_OUTPUT
    output_path = os.path.abspath(output_path)

    reactions = find_no_related_reactions()
    print(f"تعداد واکنش‌های قابل بررسی: {len(reactions)}")
    print(f"شرایط پایه: H2={h2_fraction}, phi={phi}")

    print("در حال اجرای شبیه‌سازی پایه (بدون تغییر)...")
    t0 = time.time()
    NO_base = run_flame_with_multiplier(h2_fraction, phi)
    print(f"NO_base = {NO_base:.2f} ppm  ({time.time()-t0:.1f}s)")

    rows = []
    t_start = time.time()
    for count, (idx, eq) in enumerate(reactions, start=1):
        t0 = time.time()
        try:
            NO_pert = run_flame_with_multiplier(
                h2_fraction, phi, reaction_index=idx, multiplier=1.0 + delta
            )
            sensitivity = ((NO_pert - NO_base) / NO_base) / delta
            status = "OK"
        except Exception as e:
            NO_pert = None
            sensitivity = None
            status = f"FAIL ({e})"

        dt = time.time() - t0
        print(f"[{count}/{len(reactions)}] [{idx}] {eq} -> S={sensitivity} ({status}, {dt:.1f}s)")

        rows.append(
            {
                "reaction_index": idx,
                "equation": eq,
                "NO_base_ppm": NO_base,
                "NO_perturbed_ppm": NO_pert,
                "sensitivity": sensitivity,
            }
        )
        save_csv(rows, output_path)

    elapsed = time.time() - t_start
    print(f"\nپایان تحلیل حساسیت در {elapsed/60:.1f} دقیقه.")
    print(f"فایل خروجی: {output_path}")

    # نمایش ۱۰ واکنش با بیشترین تأثیر (مثبت یا منفی)
    valid_rows = [r for r in rows if r["sensitivity"] is not None]
    valid_rows.sort(key=lambda r: abs(r["sensitivity"]), reverse=True)
    print("\nبرترین ۱۰ واکنش با بیشترین تأثیر روی NOx:")
    for r in valid_rows[:10]:
        print(f"  S={r['sensitivity']:+.3f}  {r['equation']}")


def save_csv(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["reaction_index", "equation", "NO_base_ppm", "NO_perturbed_ppm", "sensitivity"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    compute_nox_sensitivity(h2_fraction=0.5, phi=1.0, delta=0.1)
