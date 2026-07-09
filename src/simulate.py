"""
simulate.py
ماژول هسته‌ی شبیه‌سازی احتراق مخلوط هیدروژن-آمونیا با استفاده از Cantera.

مکانیزم شیمیایی استفاده‌شده: Alzueta 2023 (ammonia-CO-H2)
این مکانیزم به صورت پیش‌فرض همراه نصب Cantera در example_data می‌آید.
"""

import os
import cantera as ct
import numpy as np

# ---------------------------------------------------------------------------
# پیدا کردن مسیر فایل مکانیزم شیمیایی به صورت خودکار
# ---------------------------------------------------------------------------
def find_mechanism_path():
    """مسیر فایل مکانیزم Alzueta 2023 را پیدا می‌کند."""
    cantera_dir = os.path.dirname(ct.__file__)
    candidate = os.path.join(
        cantera_dir, "data", "example_data", "ammonia-CO-H2-Alzueta-2023.yaml"
    )
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        "فایل مکانیزم شیمیایی پیدا نشد. مسیر مورد انتظار: " + candidate
    )


MECH_PATH = find_mechanism_path()


# ---------------------------------------------------------------------------
# تابع اصلی: شبیه‌سازی شعله‌ی آزاد و استخراج پارامترهای احتراقی
# ---------------------------------------------------------------------------
def run_flame_simulation(
    h2_fraction: float,
    phi: float,
    T_in: float = 300.0,
    P: float = 101325.0,
    width: float = 0.03,
    loglevel: int = 0,
    return_profile: bool = False,
):
    """
    شبیه‌سازی شعله‌ی پیش‌آمیخته‌ی آزاد (Freely Propagating Flame) برای
    مخلوط سوخت H2/NH3 در هوا.

    پارامترهای ورودی
    -----------------
    h2_fraction    : کسر مولی H2 در سوخت (بین 0 و 1). بقیه NH3 است.
    phi            : نسبت هم‌ارزی (equivalence ratio)
    T_in           : دمای ورودی مخلوط (کلوین)
    P              : فشار (پاسکال)
    width          : عرض دامنه‌ی حل عددی (متر)
    return_profile : اگر True باشد، پروفایل کامل فضایی (موقعیت، دما،
                     غلظت گونه‌های کلیدی) هم در خروجی قرار می‌گیرد —
                     برای تجسم بصری ساختار شعله استفاده می‌شود.

    خروجی (دیکشنری)
    -----------------
    success : bool
    Su      : سرعت شعله آرام (m/s)
    T_ad    : دمای آدیاباتیک شعله (K)
    NO_ppm  : غلظت NO در محصولات (ppm حجمی، خشک)
    N2O_ppm : غلظت N2O در محصولات (ppm حجمی)
    NO2_ppm : غلظت NO2 در محصولات (ppm حجمی) در صورت وجود گونه
    profile : (فقط اگر return_profile=True) دیکشنری شامل آرایه‌های
              x_mm (موقعیت بر میلی‌متر)، T (دما بر کلوین)، و کسر مولی
              گونه‌های کلیدی سوخت/محصول در طول شعله
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

        flame = ct.FreeFlame(gas, width=width)
        flame.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
        flame.transport_model = "mixture-averaged"

        flame.solve(loglevel=loglevel, auto=True)

        Su = flame.velocity[0]
        T_ad = flame.T[-1]

        def species_ppm(name):
            if name in gas.species_names:
                idx = gas.species_index(name)
                return float(flame.X[idx, -1] * 1e6)
            return None

        result.update(
            {
                "success": True,
                "Su": float(Su),
                "Su_cm_s": float(Su * 100),
                "T_ad": float(T_ad),
                "NO_ppm": species_ppm("NO"),
                "N2O_ppm": species_ppm("N2O"),
                "NO2_ppm": species_ppm("NO2"),
            }
        )

        if return_profile:
            profile_species = ["H2", "NH3", "O2", "H2O", "NO", "N2"]
            species_profiles = {}
            for sp in profile_species:
                if sp in gas.species_names:
                    idx = gas.species_index(sp)
                    species_profiles[sp] = flame.X[idx, :].tolist()

            result["profile"] = {
                "x_mm": (flame.grid * 1000).tolist(),  # متر -> میلی‌متر
                "T": flame.T.tolist(),
                "species": species_profiles,
            }
    except Exception as e:
        result["error"] = str(e)

    return result


if __name__ == "__main__":
    # تست سریع ماژول
    r = run_flame_simulation(h2_fraction=0.3, phi=1.0)
    print(r)
