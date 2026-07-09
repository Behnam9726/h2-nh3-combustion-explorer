"""
diffusion_flame.py
شبیه‌سازی شعله‌ی نفوذی مخالف‌جریان (Counterflow Diffusion Flame) برای
مخلوط سوخت H2/NH3.

تفاوت بنیادی با شعله‌ی پیش‌آمیخته (simulate.py):
    در شعله‌ی پیش‌آمیخته، سوخت و هوا از قبل و به‌طور کامل مخلوط شده‌اند
    و شعله در سراسر مخلوط منتشر می‌شود. در شعله‌ی نفوذی، سوخت خالص از
    یک طرف و اکسیدکننده (هوا) از طرف مقابل به سمت هم جریان می‌یابند؛
    شعله دقیقاً در ناحیه‌ای که این دو جریان به‌وسیله‌ی نفوذ مولکولی به
    نسبت سوخت/هوای مناسب می‌رسند (نه در کل حجم) تشکیل می‌شود. این
    پیکربندی نماینده‌ی طراحی بسیاری از مشعل‌های صنعتی و برخی از
    راهکارهای پیشنهادی برای پایدارسازی شعله‌ی آمونیا (که در حالت
    پیش‌آمیخته پایداری کمتری دارد) است.

اجرا:
    python src/diffusion_flame.py
"""

import os
import sys

import numpy as np
import cantera as ct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate import MECH_PATH

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_diffusion_flame(
    h2_fraction: float,
    mdot_fuel: float = 0.5,
    mdot_ox: float = 0.5,
    width: float = 0.02,
    T_fuel: float = 300.0,
    T_ox: float = 300.0,
    P: float = 101325.0,
    loglevel: int = 0,
    return_profile: bool = False,
):
    """
    شبیه‌سازی شعله‌ی نفوذی مخالف‌جریان.

    پارامترهای ورودی
    -----------------
    h2_fraction : کسر مولی H2 در جریان سوخت (بقیه NH3، بدون رقیق‌سازی
                  با بی‌اثر — سوخت خالص از یک سمت وارد می‌شود)
    mdot_fuel   : شار جرمی جریان سوخت (kg/m^2/s) — کنترل‌کننده‌ی اصلی
                  «شدت کشش» (strain rate) شعله
    mdot_ox     : شار جرمی جریان اکسیدکننده (هوا) (kg/m^2/s)
    width       : فاصله‌ی بین دو نازل (متر)
    T_fuel      : دمای جریان سوخت (کلوین)
    T_ox        : دمای جریان اکسیدکننده (کلوین)
    P           : فشار (پاسکال)
    return_profile : اگر True، پروفایل کامل فضایی هم برگردانده می‌شود

    خروجی (دیکشنری)
    -----------------
    success    : bool
    T_max      : بیشینه‌ی دما در سراسر شعله (K)
    x_flame_mm : موقعیت بیشینه‌ی دما نسبت به نازل سوخت (میلی‌متر)
    NO_ppm_max : بیشینه‌ی غلظت NO در سراسر شعله (ppm)
    strain_rate_1_s : نرخ کشش تخمینی شعله (1/s) — معیار «شدت آشفتگی»
    profile    : (فقط اگر return_profile=True) دیکشنری پروفایل فضایی
    """
    result = {
        "h2_fraction": h2_fraction,
        "mdot_fuel": mdot_fuel,
        "mdot_ox": mdot_ox,
        "width": width,
        "success": False,
    }

    try:
        gas = ct.Solution(MECH_PATH)
        gas.TP = 300.0, P

        f = ct.CounterflowDiffusionFlame(gas, width=width)

        fuel_comp = {"H2": h2_fraction, "NH3": 1.0 - h2_fraction}
        ox_comp = {"O2": 1.0, "N2": 3.76}

        f.fuel_inlet.mdot = mdot_fuel
        f.fuel_inlet.X = fuel_comp
        f.fuel_inlet.T = T_fuel

        f.oxidizer_inlet.mdot = mdot_ox
        f.oxidizer_inlet.X = ox_comp
        f.oxidizer_inlet.T = T_ox

        f.set_refine_criteria(ratio=3, slope=0.1, curve=0.2)
        f.solve(loglevel=loglevel, auto=True)

        T = f.T
        x = f.grid
        idx_max = int(np.argmax(T))
        T_max = float(T[idx_max])
        x_flame_mm = float(x[idx_max] * 1000)

        # نکته‌ی مهم: گاهی حل‌گر عددی «همگرا» می‌شود ولی به یک جواب
        # بی‌شعله (مخلوط سرد، بدون احتراق) می‌رسد — این هم یک نوع
        # خاموشی واقعی است و نباید به‌عنوان موفقیت گزارش شود
        T_inlet_max = max(T_fuel, T_ox)
        if T_max < T_inlet_max + 100:
            result["success"] = False
            result["error"] = "no_ignition_or_extinguished"
            result["T_max"] = T_max
            return result

        NO_ppm_max = None
        if "NO" in gas.species_names:
            idx_no = gas.species_index("NO")
            NO_ppm_max = float(f.X[idx_no, :].max() * 1e6)

        # تخمین نرخ کشش کلی (global strain rate) بر اساس سرعت‌های ورودی
        rho_fuel = f.fuel_inlet.density if hasattr(f.fuel_inlet, "density") else None
        strain_rate = (mdot_fuel + mdot_ox) / width  # تخمین ساده و مرسوم

        result.update(
            {
                "success": True,
                "T_max": T_max,
                "x_flame_mm": x_flame_mm,
                "NO_ppm_max": NO_ppm_max,
                "strain_rate_1_s": float(strain_rate),
            }
        )

        if return_profile:
            profile_species = ["H2", "NH3", "O2", "H2O", "NO", "N2"]
            species_profiles = {}
            for sp in profile_species:
                if sp in gas.species_names:
                    idx = gas.species_index(sp)
                    species_profiles[sp] = f.X[idx, :].tolist()

            result["profile"] = {
                "x_mm": (x * 1000).tolist(),
                "T": T.tolist(),
                "species": species_profiles,
            }
    except Exception as e:
        result["error"] = str(e)

    return result


def find_extinction_strain_rate(h2_fraction, width=0.02, mdot_start=0.3, mdot_step=0.15, max_mdot=3.0):
    """
    افزایش تدریجی شار جرمی (و در نتیجه نرخ کشش) تا نقطه‌ای که شعله
    خاموش می‌شود (extinction) — یک معیار مهم برای پایداری شعله‌ی نفوذی
    در برابر آشفتگی (مشابه محدوده‌ی اشتعال‌پذیری در شعله‌ی پیش‌آمیخته).
    """
    mdot = mdot_start
    last_success = None
    while mdot <= max_mdot:
        res = run_diffusion_flame(h2_fraction, mdot_fuel=mdot, mdot_ox=mdot, width=width)
        if res["success"]:
            last_success = res
            mdot += mdot_step
        else:
            return {
                "h2_fraction": h2_fraction,
                "extinction_strain_rate_1_s": last_success["strain_rate_1_s"] if last_success else None,
                "extinction_mdot": mdot - mdot_step if last_success else None,
            }
    return {
        "h2_fraction": h2_fraction,
        "extinction_strain_rate_1_s": last_success["strain_rate_1_s"] if last_success else None,
        "extinction_mdot": None,
        "note": "خاموشی در بازه‌ی جستجو رخ نداد",
    }


if __name__ == "__main__":
    print("تست شعله‌ی نفوذی مخالف‌جریان...")
    res = run_diffusion_flame(h2_fraction=0.5, mdot_fuel=0.5, mdot_ox=0.5)
    for k, v in res.items():
        if k != "profile":
            print(f"  {k}: {v}")
