"""
reactor_network.py
مدل شبکه‌ی راکتور Rich-Quench-Lean (RQL) — یک مدل ساده‌شده (reduced-order)
برای تخمین رفتار احتراق در محفظه‌ی احتراق توربین گاز، بدون نیاز به CFD
سه‌بعدی.

چرا RQL؟
    این دقیقاً همان استراتژی‌ای است که در مقالات و پروژه‌های واقعی صنعتی
    برای احتراق آمونیا/هیدروژن در توربین گاز پیشنهاد می‌شود:
    ۱) ناحیه‌ی اول (Primary Zone): احتراق در حالت غنی (rich) که تشکیل NOx
       حرارتی را به‌شدت کاهش می‌دهد (کمبود O2 مانع تشکیل NO می‌شود).
    ۲) ناحیه‌ی خاموشی سریع (Quench): رقیق‌سازی سریع با هوای ثانویه تا از
       عبور از ناحیه‌ی نزدیک‌استوکیومتریک (که بیشترین NOx را تولید
       می‌کند) با سرعت بگذریم.
    ۳) ناحیه‌ی دوم (Secondary/Lean Zone): احتراق کامل در حالت رقیق (lean)
       با دمای پایین‌تر برای تکمیل احتراق با حداقل آلایندگی.

مدل‌سازی:
    ناحیه‌ی اول با یک راکتور کاملاً مخلوط (Perfectly Stirred Reactor,
    PSR) با زمان اقامت مشخص شبیه‌سازی می‌شود. ناحیه‌ی دوم با فرض جریان
    تراکمی (Plug Flow Reactor) که معادل انتگرال‌گیری زمانی یک راکتور
    بسته روی زمان اقامت است (تقریب استاندارد و رایج در مدل‌های
    reduced-order احتراق).

    شرایط ورودی (دما و فشار) به‌طور پیش‌فرض معادل شرایط واقعی خروجی
    کمپرسور یک توربین گاز است (نه هوای اتاق!) — این مهم‌ترین تفاوت این
    ماژول با شبیه‌سازی‌های قبلی (که در فشار و دمای محیط انجام می‌شدند).

اجرا:
    python src/reactor_network.py
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
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "data", "rql_network_dataset.csv")

# شرایط پیش‌فرض معادل خروجی کمپرسور یک توربین گاز صنعتی متوسط
DEFAULT_T_COMPRESSOR_OUT = 700.0   # کلوین (دمای معمول خروجی کمپرسور)
DEFAULT_P_TURBINE = 101325.0 * 15  # پاسکال (نسبت فشار ~15، معمول توربین‌های متوسط)


def run_two_inlet_psr_steady_state(
    gas_a, mdot_a_ratio, gas_b, mdot_b_ratio, residence_time, max_time=1.0
):
    """
    راکتور کاملاً مخلوط با دو جریان ورودی مستقل (خروجی ناحیه‌ی غنی +
    هوای تازه‌ی رقیق‌سازی)، با یک زمان اقامت مشخص برای کل مخلوط.

    این تابع دقیقاً پارامتری را می‌سازد که واقعاً «سرعت خاموشی» را
    کنترل می‌کند: با کاهش residence_time، گاز مدت کمتری را در ناحیه‌ی
    گذار (نزدیک‌استوکیومتریک، مستعد بیشترین تشکیل NOx) می‌گذراند.
    """
    gas_a_res = ct.Solution(MECH_PATH)
    gas_a_res.TPX = gas_a.T, gas_a.P, gas_a.X
    reservoir_a = ct.Reservoir(gas_a_res)

    gas_b_res = ct.Solution(MECH_PATH)
    gas_b_res.TPX = gas_b.T, gas_b.P, gas_b.X
    reservoir_b = ct.Reservoir(gas_b_res)

    # حالت اولیه‌ی راکتور: تخمین مخلوط آنی برای همگرایی سریع‌تر عددی
    qa = ct.Quantity(gas_a, constant="HP")
    qa.mass = mdot_a_ratio
    qb = ct.Quantity(gas_b, constant="HP")
    qb.mass = mdot_b_ratio
    mix0 = qa + qb

    gas_init = ct.Solution(MECH_PATH)
    gas_init.TPX = mix0.T, mix0.P, mix0.X

    reactor = ct.IdealGasReactor(gas_init, volume=1e-5, energy="on")

    total_mdot = reactor.mass / residence_time
    ratio_sum = mdot_a_ratio + mdot_b_ratio
    mdot_a = total_mdot * (mdot_a_ratio / ratio_sum)
    mdot_b = total_mdot * (mdot_b_ratio / ratio_sum)

    exhaust_gas = ct.Solution(MECH_PATH)
    exhaust_gas.TPX = gas_init.T, gas_init.P, gas_init.X
    exhaust = ct.Reservoir(exhaust_gas)

    mfc_a = ct.MassFlowController(reservoir_a, reactor, mdot=mdot_a)
    mfc_b = ct.MassFlowController(reservoir_b, reactor, mdot=mdot_b)
    ct.PressureController(reactor, exhaust, primary=mfc_a, K=1e-3)

    sim = ct.ReactorNet([reactor])
    sim.rtol = 1e-8
    sim.atol = 1e-18

    t = 0.0
    dt = residence_time / 20.0
    T_prev = reactor.T
    n_stable = 0
    while t < max_time:
        t += dt
        sim.advance(t)
        if abs(reactor.T - T_prev) < 1e-3:
            n_stable += 1
            if n_stable > 10:
                break
        else:
            n_stable = 0
        T_prev = reactor.T

    return reactor.thermo, t


def run_psr_steady_state(gas, residence_time, volume=1e-5, max_time=2.0):
    """
    شبیه‌سازی راکتور کاملاً مخلوط (PSR) با زمان اقامت مشخص تا رسیدن
    به حالت پایا (steady state).

    نکته‌ی مهم فیزیکی: یک PSR واقعی که به‌طور مداوم کار می‌کند، همیشه
    محتوای داغ و سوخته دارد (محصولات چرخشی است که مخلوط تازه را
    مشتعل می‌کنند)، نه اینکه هر بار از یک مخلوط سرد شروع به احتراق
    خودبه‌خودی کند. برای همین، راکتور با حالت *تعادل شیمیایی* (سوخته و
    داغ) از همان ترکیب مقداردهی اولیه می‌شود، سپس با ورودی مداوم
    مخلوط تازه (سرد) به سمت نقطه‌ی پایای واقعی خودش تکامل می‌یابد. اگر
    زمان اقامت خیلی کوتاه باشد (کمتر از حد خاموشی/blow-out)، دما در
    طول شبیه‌سازی به‌تدریج افت می‌کند و به سمت مخلوط نسوخته می‌رود —
    که خودش یک نتیجه‌ی فیزیکی معتبر (خاموشی ناحیه‌ی اول) است.
    """
    inlet_gas = ct.Solution(MECH_PATH)
    inlet_gas.TPX = gas.T, gas.P, gas.X
    inlet = ct.Reservoir(inlet_gas)

    # حالت اولیه‌ی راکتور: تعادل شیمیایی (داغ/سوخته) از همان مخلوط،
    # نه مخلوط سرد ورودی — این دقیقاً همان چیزی است که یک PSR واقعی و
    # پیوسته در حال کار را از یک شبیه‌سازی احتراق خودبه‌خودی متمایز می‌کند
    gas_hot = ct.Solution(MECH_PATH)
    gas_hot.TPX = gas.T, gas.P, gas.X
    gas_hot.equilibrate("HP")

    reactor = ct.IdealGasReactor(gas_hot, volume=volume, energy="on")

    exhaust_gas = ct.Solution(MECH_PATH)
    exhaust_gas.TPX = gas.T, gas.P, gas.X
    exhaust = ct.Reservoir(exhaust_gas)

    mass_flow_rate = reactor.mass / residence_time

    mfc = ct.MassFlowController(inlet, reactor, mdot=mass_flow_rate)
    pc = ct.PressureController(reactor, exhaust, primary=mfc, K=1e-3)

    sim = ct.ReactorNet([reactor])
    sim.rtol = 1e-8
    sim.atol = 1e-18

    t = 0.0
    dt = residence_time / 20.0
    T_prev = reactor.T
    n_stable = 0
    while t < max_time:
        t += dt
        sim.advance(t)
        # معیار همگرایی: وقتی دما دیگر تغییر محسوسی نمی‌کند، یعنی به
        # حالت پایا رسیده‌ایم (چه حالت سوخته پایدار، چه خاموشی کامل)
        if abs(reactor.T - T_prev) < 1e-3:
            n_stable += 1
            if n_stable > 10:
                break
        else:
            n_stable = 0
        T_prev = reactor.T

    return reactor.thermo, t


def run_pfr_as_batch(gas, residence_time, n_steps=200):
    """
    شبیه‌سازی ناحیه‌ی دوم (PFR) با فرض معادل‌بودن جریان تراکمی با
    انتگرال‌گیری زمانی یک راکتور بسته روی زمان اقامت — تقریب استاندارد
    وقتی نفوذ محوری نادیده گرفته می‌شود.
    """
    reactor = ct.IdealGasConstPressureReactor(gas, energy="on")
    sim = ct.ReactorNet([reactor])
    sim.rtol = 1e-9
    sim.atol = 1e-20

    times = np.linspace(0, residence_time, n_steps)
    T_history = []
    for t in times[1:]:
        sim.advance(t)
        T_history.append(reactor.T)

    return reactor.thermo, T_history


def run_rql_network(
    h2_fraction,
    phi_primary=1.4,
    phi_overall=0.6,
    tau_primary=2e-3,
    tau_quench=1e-3,
    tau_secondary=8e-3,
    T_in=DEFAULT_T_COMPRESSOR_OUT,
    P=DEFAULT_P_TURBINE,
):
    """
    اجرای کامل زنجیره‌ی RQL: ناحیه‌ی غنی -> خاموشی (با سرعت مشخص) -> ناحیه‌ی رقیق.

    پارامترها
    ----------
    h2_fraction   : کسر مولی H2 در سوخت
    phi_primary   : نسبت هم‌ارزی ناحیه‌ی اول (باید >1، یعنی غنی)
    phi_overall   : نسبت هم‌ارزی کلی سیستم پس از رقیق‌سازی (باید <1، رقیق)
    tau_primary   : زمان اقامت ناحیه‌ی اول (ثانیه)
    tau_quench    : زمان اقامت ناحیه‌ی خاموشی/مخلوط‌سازی (ثانیه) — این
                    پارامتر «سرعت خاموشی» واقعی است؛ هرچه کوچک‌تر باشد،
                    گاز سریع‌تر از ناحیه‌ی خطرناک نزدیک‌استوکیومتریک
                    عبور می‌کند و NOx کمتری تشکیل می‌شود
    tau_secondary : زمان اقامت ناحیه‌ی رقیق نهایی برای تکمیل احتراق (ثانیه)
    T_in          : دمای ورودی (خروجی کمپرسور)
    P             : فشار کاری (فشار محفظه‌ی احتراق توربین)

    خروجی: دیکشنری شامل دما و غلظت NOx در هر مرحله
    """
    result = {
        "h2_fraction": h2_fraction,
        "phi_primary": phi_primary,
        "phi_overall": phi_overall,
        "tau_primary": tau_primary,
        "tau_quench": tau_quench,
        "tau_secondary": tau_secondary,
        "T_in": T_in,
        "P": P,
        "success": False,
    }

    try:
        fuel = {"H2": h2_fraction, "NH3": 1.0 - h2_fraction}

        # --- ناحیه‌ی اول: PSR غنی ---
        gas_pz = ct.Solution(MECH_PATH)
        gas_pz.TP = T_in, P
        gas_pz.set_equivalence_ratio(phi_primary, fuel, {"O2": 1.0, "N2": 3.76})

        pz_out, t_pz = run_psr_steady_state(gas_pz, tau_primary)
        T_pz = pz_out.T
        NO_pz = pz_out["NO"].X[0] * 1e6 if "NO" in pz_out.species_names else None

        # --- ناحیه‌ی خاموشی: PSR دو-ورودی (خروجی ناحیه‌ی غنی + هوای تازه) ---
        air_gas = ct.Solution(MECH_PATH)
        air_gas.TPX = T_in, P, "O2:1.0, N2:3.76"

        # نسبت جرمی هوای ثانویه لازم برای رساندن phi کلی از phi_primary
        # به phi_overall (تخمین بر اساس موازنه‌ی مولی سوخت)
        mix_ratio = max(phi_primary / phi_overall - 1.0, 0.05)

        quench_out, t_q = run_two_inlet_psr_steady_state(
            pz_out, 1.0, air_gas, mix_ratio, tau_quench
        )
        T_quench = quench_out.T

        # --- ناحیه‌ی دوم: PFR رقیق (تکمیل احتراق) ---
        gas_sz = ct.Solution(MECH_PATH)
        gas_sz.TPX = quench_out.T, quench_out.P, quench_out.X

        sz_out, T_history = run_pfr_as_batch(gas_sz, tau_secondary)
        T_sz = sz_out.T
        NO_sz = sz_out["NO"].X[0] * 1e6 if "NO" in sz_out.species_names else None
        N2O_sz = sz_out["N2O"].X[0] * 1e6 if "N2O" in sz_out.species_names else None

        result.update(
            {
                "success": True,
                "T_primary_zone": float(T_pz),
                "NO_ppm_primary_zone": float(NO_pz) if NO_pz is not None else None,
                "T_after_quench": float(T_quench),
                "T_secondary_zone_out": float(T_sz),
                "NO_ppm_final": float(NO_sz) if NO_sz is not None else None,
                "N2O_ppm_final": float(N2O_sz) if N2O_sz is not None else None,
            }
        )
    except Exception as e:
        result["error"] = str(e)

    return result


if __name__ == "__main__":
    print("تست سریع شبکه‌ی راکتور RQL...")
    t0 = time.time()
    res = run_rql_network(h2_fraction=0.3, phi_primary=1.4, phi_overall=0.6)
    print(f"زمان اجرا: {time.time()-t0:.1f} ثانیه")
    for k, v in res.items():
        print(f"  {k}: {v}")
