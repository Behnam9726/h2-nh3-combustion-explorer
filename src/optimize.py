"""
optimize.py
بهینه‌سازی چندهدفه (NSGA-II) روی مدل‌های جانشین برای یافتن مرز Pareto
بین دو هدف متضاد:
    - حداکثر کردن سرعت شعله آرام (Su_cm_s)  -> احتراق پایدارتر/کارآمدتر
    - حداقل کردن غلظت NO (NO_ppm)           -> کاهش آلایندگی

متغیرهای تصمیم:
    x0 = h2_fraction  (کسر مولی H2 در سوخت، بین h2_range)
    x1 = phi          (نسبت هم‌ارزی، بین phi_range)

اجرا:
    python src/optimize.py

خروجی:
    data/pareto_front.csv   (نقاط بهینه‌ی مرز Pareto)
"""

import os
import joblib
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_surrogate import predict as gp_predict

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "surrogate_models.joblib")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "pareto_front.csv")


class CombustionProblem(Problem):
    """
    مسئله‌ی بهینه‌سازی دو هدفه:
        هدف ۱: حداقل کردن (-Su_cm_s)  یعنی حداکثر کردن سرعت شعله
        هدف ۲: حداقل کردن NO_ppm
    pymoo به صورت پیش‌فرض «حداقل‌سازی» انجام می‌دهد، برای همین هدف اول
    را منفی می‌کنیم تا معادل حداکثرسازی شود.
    """

    def __init__(self, models_bundle, h2_range, phi_range):
        self.models_bundle = models_bundle
        super().__init__(
            n_var=2,
            n_obj=2,
            n_constr=0,
            xl=np.array([h2_range[0], phi_range[0]]),
            xu=np.array([h2_range[1], phi_range[1]]),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        h2 = X[:, 0]
        phi = X[:, 1]
        Su_pred, _ = gp_predict(self.models_bundle, h2, phi, "Su_cm_s")
        NO_pred, _ = gp_predict(self.models_bundle, h2, phi, "NO_ppm")
        f1 = -Su_pred  # حداکثرسازی سرعت شعله
        f2 = NO_pred   # حداقل‌سازی NOx
        out["F"] = np.column_stack([f1, f2])


def run_optimization(pop_size=100, n_gen=80, seed=42):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"فایل مدل جانشین پیدا نشد: {MODEL_PATH}\n"
            "ابتدا 'python src/train_surrogate.py' را اجرا کنید."
        )

    bundle = joblib.load(MODEL_PATH)
    h2_range = bundle["h2_range"]
    phi_range = bundle["phi_range"]

    print(f"بازه‌ی جستجو -> H2: {h2_range}, phi: {phi_range}")
    print(f"اجرای NSGA-II با جمعیت={pop_size}, نسل={n_gen} ...")

    problem = CombustionProblem(bundle, h2_range, phi_range)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        seed=seed,
        verbose=True,
    )

    X = res.X  # متغیرهای تصمیم بهینه (h2_fraction, phi)
    F = res.F  # مقادیر هدف (منفیِ Su, NO)

    df = pd.DataFrame(
        {
            "h2_fraction": X[:, 0],
            "phi": X[:, 1],
            "Su_cm_s_pred": -F[:, 0],
            "NO_ppm_pred": F[:, 1],
        }
    )
    # مرتب‌سازی بر اساس سرعت شعله برای خوانایی بهتر
    df = df.sort_values("Su_cm_s_pred", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nمرز Pareto با {len(df)} نقطه یافت و ذخیره شد: {OUTPUT_PATH}")
    print("\nنمونه‌ای از نقاط بهینه (۵ نقطه اول):")
    print(df.head())

    return df


if __name__ == "__main__":
    run_optimization()
