"""
dashboard.py
داشبورد تعاملی برای اکتشاف فضای طراحی احتراق H2/NH3.

اجرا:
    streamlit run src/dashboard.py

پیش‌نیاز: باید قبلاً این دو اسکریپت اجرا شده باشند:
    python src/generate_dataset.py
    python src/train_surrogate.py
    python src/optimize.py   (اختیاری، برای نمایش Pareto front)
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_surrogate import predict as gp_predict
from simulate import run_flame_simulation
from reactor_network import run_rql_network
from diffusion_flame import run_diffusion_flame

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "surrogate_models.joblib")
PARETO_PATH = os.path.join(PROJECT_ROOT, "data", "pareto_front.csv")
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "combustion_dataset.csv")
IGNITION_PATH = os.path.join(PROJECT_ROOT, "data", "ignition_delay_dataset.csv")
STABILITY_PATH = os.path.join(PROJECT_ROOT, "data", "stability_map_dataset.csv")
SENSITIVITY_PATH = os.path.join(PROJECT_ROOT, "data", "nox_sensitivity_dataset.csv")

st.set_page_config(
    page_title="H2/NH3 Combustion Explorer",
    page_icon="🔥",
    layout="wide",
)

# ---------------------------------------------------------------------------
# رنگ‌بندی تم (آبی سرد -> نارنجی/قرمز داغ، متناسب با موضوع شعله)
# ---------------------------------------------------------------------------
FLAME_COLORSCALE = [
    [0.0, "#0d1b2a"],
    [0.15, "#1b4965"],
    [0.35, "#5fa8d3"],
    [0.55, "#f4a261"],
    [0.75, "#e76f51"],
    [1.0, "#d62828"],
]
ACCENT = "#e76f51"
BG = "#0d1b2a"


@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_pareto():
    if not os.path.exists(PARETO_PATH):
        return None
    return pd.read_csv(PARETO_PATH)


@st.cache_data
def load_raw_dataset():
    if not os.path.exists(DATASET_PATH):
        return None
    df = pd.read_csv(DATASET_PATH)
    return df[df["success"] == True]  # noqa: E712


@st.cache_data
def load_ignition_delay():
    if not os.path.exists(IGNITION_PATH):
        return None
    df = pd.read_csv(IGNITION_PATH)
    return df[df["success"] == True]  # noqa: E712


@st.cache_data(show_spinner=False)
def get_flame_profile(h2_fraction, phi):
    """
    اجرای زنده‌ی شبیه‌سازی Cantera برای گرفتن پروفایل کامل شعله.
    نتیجه کش می‌شود تا با تغییر اسلایدر به همان مقدار، دوباره اجرا نشود.
    این تابع ۵-۱۵ ثانیه طول می‌کشد (شبیه‌سازی واقعی فیزیکی، نه مدل جانشین).
    """
    res = run_flame_simulation(
        h2_fraction=float(h2_fraction), phi=float(phi), return_profile=True
    )
    return res


@st.cache_data
def load_stability_map():
    if not os.path.exists(STABILITY_PATH):
        return None
    return pd.read_csv(STABILITY_PATH)


@st.cache_data
def load_sensitivity_data():
    if not os.path.exists(SENSITIVITY_PATH):
        return None
    df = pd.read_csv(SENSITIVITY_PATH)
    return df.dropna(subset=["sensitivity"])


def sensitivity_tornado_plot(df, top_n=15):
    """نمودار Tornado: واکنش‌های با بیشترین تأثیر (مثبت یا منفی) روی NOx."""
    df = df.copy()
    df["abs_sensitivity"] = df["sensitivity"].abs()
    df = df.sort_values("abs_sensitivity", ascending=False).head(top_n)
    df = df.sort_values("sensitivity")  # برای نمایش صعودی در نمودار افقی

    colors = ["#5fa8d3" if s < 0 else "#d62828" for s in df["sensitivity"]]

    fig = go.Figure(
        data=go.Bar(
            x=df["sensitivity"],
            y=df["equation"],
            orientation="h",
            marker=dict(color=colors),
        )
    )
    fig.update_layout(
        title="واکنش‌های با بیشترین تأثیر بر تشکیل NOx (نمودار Tornado)",
        xaxis_title="ضریب حساسیت نرمال‌شده (آبی = کاهنده‌ی NOx، قرمز = افزاینده‌ی NOx)",
        height=max(450, 28 * len(df)),
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,0.3)",
        font={"color": "white", "size": 11},
    )
    return fig


@st.cache_data
def compute_grid(_bundle, h2_range, phi_range, resolution=40):
    """محاسبه‌ی گرید متراکم پیش‌بینی‌شده برای نقشه‌ی حرارتی."""
    h2_grid = np.linspace(h2_range[0], h2_range[1], resolution)
    phi_grid = np.linspace(phi_range[0], phi_range[1], resolution)
    H2, PHI = np.meshgrid(h2_grid, phi_grid)
    h2_flat = H2.ravel()
    phi_flat = PHI.ravel()

    Su, _ = gp_predict(_bundle, h2_flat, phi_flat, "Su_cm_s")
    Tad, _ = gp_predict(_bundle, h2_flat, phi_flat, "T_ad")
    NO, _ = gp_predict(_bundle, h2_flat, phi_flat, "NO_ppm")

    return {
        "h2_grid": h2_grid,
        "phi_grid": phi_grid,
        "Su": Su.reshape(H2.shape),
        "Tad": Tad.reshape(H2.shape),
        "NO": NO.reshape(H2.shape),
    }


def make_gauge(value, title, value_range, suffix="", color=ACCENT):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 16, "color": "white"}},
            number={"suffix": suffix, "font": {"color": "white"}},
            gauge={
                "axis": {"range": value_range, "tickcolor": "white"},
                "bar": {"color": color},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1,
                "bordercolor": "gray",
                "steps": [
                    {"range": value_range, "color": "rgba(255,255,255,0.05)"}
                ],
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


def flame_heatmap(grid, target_key, title, colorbar_title, h2_sel, phi_sel):
    fig = go.Figure(
        data=go.Contour(
            x=grid["h2_grid"],
            y=grid["phi_grid"],
            z=grid[target_key],
            colorscale=FLAME_COLORSCALE,
            contours=dict(showlabels=False),
            colorbar=dict(
                title=dict(text=colorbar_title, font=dict(color="white")),
                tickfont={"color": "white"},
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[h2_sel],
            y=[phi_sel],
            mode="markers",
            marker=dict(size=16, color="white", symbol="x", line=dict(width=2, color="black")),
            name="انتخاب فعلی",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="کسر مولی H2 در سوخت",
        yaxis_title="نسبت هم‌ارزی (φ)",
        height=430,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


def pareto_plot(pareto_df, current_point=None):
    fig = go.Figure(
        data=go.Scatter(
            x=pareto_df["Su_cm_s_pred"],
            y=pareto_df["NO_ppm_pred"],
            mode="markers",
            marker=dict(
                size=10,
                color=pareto_df["h2_fraction"],
                colorscale=FLAME_COLORSCALE,
                colorbar=dict(
                    title=dict(text="کسر H2", font=dict(color="white")),
                    tickfont={"color": "white"},
                ),
                line=dict(width=1, color="white"),
            ),
            text=[
                f"H2={h:.2f}, φ={p:.2f}"
                for h, p in zip(pareto_df["h2_fraction"], pareto_df["phi"])
            ],
            hoverinfo="text+x+y",
            name="مرز بهینه Pareto",
        )
    )
    if current_point is not None:
        fig.add_trace(
            go.Scatter(
                x=[current_point[0]],
                y=[current_point[1]],
                mode="markers",
                marker=dict(size=18, color="white", symbol="star", line=dict(width=2, color=ACCENT)),
                name="انتخاب فعلی",
            )
        )
    fig.update_layout(
        title="مرز بهینه‌ی Pareto: سرعت شعله در مقابل NOx",
        xaxis_title="سرعت شعله آرام (cm/s) — بیشتر بهتر",
        yaxis_title="غلظت NO (ppm) — کمتر بهتر",
        height=430,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,0.3)",
        font={"color": "white"},
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def radar_comparison(bundle, h2_sel, phi_sel):
    categories = ["سرعت شعله", "دما", "NOx (معکوس)"]

    def get_metrics(h2, phi):
        su, _ = gp_predict(bundle, h2, phi, "Su_cm_s")
        tad, _ = gp_predict(bundle, h2, phi, "T_ad")
        no, _ = gp_predict(bundle, h2, phi, "NO_ppm")
        return float(su[0]), float(tad[0]), float(no[0])

    su_h2, tad_h2, no_h2 = get_metrics(0.999, phi_sel)
    su_nh3, tad_nh3, no_nh3 = get_metrics(0.001, phi_sel)
    su_cur, tad_cur, no_cur = get_metrics(h2_sel, phi_sel)

    # نرمال‌سازی برای مقایسه‌ی بصری روی مقیاس یکسان (0-100)
    su_max = max(su_h2, su_nh3, su_cur, 1e-6)
    tad_max = max(tad_h2, tad_nh3, tad_cur, 1e-6)
    no_max = max(no_h2, no_nh3, no_cur, 1e-6)

    def normalize(su, tad, no):
        return [
            100 * su / su_max,
            100 * tad / tad_max,
            100 * (1 - no / no_max),  # معکوس چون کمتر بهتر است
        ]

    fig = go.Figure()
    for name, (su, tad, no), color in [
        ("H2 خالص", (su_h2, tad_h2, no_h2), "#5fa8d3"),
        ("NH3 خالص", (su_nh3, tad_nh3, no_nh3), "#e76f51"),
        ("انتخاب فعلی", (su_cur, tad_cur, no_cur), "#ffffff"),
    ]:
        vals = normalize(su, tad, no)
        fig.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=name,
                line=dict(color=color),
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="white"),
            bgcolor="rgba(0,0,0,0)",
            angularaxis=dict(color="white"),
        ),
        showlegend=True,
        height=430,
        margin=dict(l=40, r=40, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def ignition_delay_plot(df):
    """نمودار استاندارد Arrhenius: log10(tau) در مقابل 1000/T برای هر ترکیب H2."""
    fig = go.Figure()
    h2_values = sorted(df["h2_fraction"].unique())
    colors = ["#5fa8d3", "#7fb069", "#f4a261", "#e76f51", "#d62828"]

    for i, h2 in enumerate(h2_values):
        sub = df[df["h2_fraction"] == h2].sort_values("T_in")
        if len(sub) == 0:
            continue
        inv_T = 1000.0 / sub["T_in"]
        log_tau = np.log10(sub["tau_ms"])
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=inv_T,
                y=log_tau,
                mode="lines+markers",
                name=f"H2={h2*100:.0f}٪",
                line=dict(color=color, width=2),
                marker=dict(size=7),
            )
        )

    fig.update_layout(
        title="زمان تأخیر اشتعال (نمودار Arrhenius)",
        xaxis_title="1000 / دما (1/K)",
        yaxis_title="log10(زمان تأخیر بر میلی‌ثانیه)",
        height=450,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,0.3)",
        font={"color": "white"},
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def flammability_envelope_plot(df, h2_sel=None, phi_sel=None):
    """
    نمایش پراکندگی نقاط پایدار/ناپایدار + منحنی پوش (envelope) تخمینی
    محدوده‌ی اشتعال‌پذیری، بر اساس حداقل/حداکثر phi موفق در هر H2.
    """
    stable = df[df["success"] == True]  # noqa: E712
    unstable = df[df["success"] == False]  # noqa: E712

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=stable["h2_fraction"],
            y=stable["phi"],
            mode="markers",
            marker=dict(size=10, color="#7fb069", symbol="circle"),
            name="پایدار (شعله برقرار)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=unstable["h2_fraction"],
            y=unstable["phi"],
            mode="markers",
            marker=dict(size=10, color="#d62828", symbol="x"),
            name="ناپایدار / نزدیک خاموشی",
        )
    )

    # منحنی پوش تقریبی: برای هر H2، حداقل و حداکثر phi موفق
    envelope_rows = []
    for h2 in sorted(stable["h2_fraction"].unique()):
        sub = stable[stable["h2_fraction"] == h2]
        envelope_rows.append((h2, sub["phi"].min(), sub["phi"].max()))
    if envelope_rows:
        env_df = pd.DataFrame(envelope_rows, columns=["h2", "phi_lean", "phi_rich"])
        fig.add_trace(
            go.Scatter(
                x=env_df["h2"], y=env_df["phi_lean"],
                mode="lines", line=dict(color="#7fb069", dash="dash"),
                name="حد رقیق تخمینی (lean)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=env_df["h2"], y=env_df["phi_rich"],
                mode="lines", line=dict(color="#e76f51", dash="dash"),
                name="حد غنی تخمینی (rich)",
            )
        )

    if h2_sel is not None and phi_sel is not None:
        fig.add_trace(
            go.Scatter(
                x=[h2_sel], y=[phi_sel],
                mode="markers",
                marker=dict(size=18, color="white", symbol="star", line=dict(width=2, color=ACCENT)),
                name="انتخاب فعلی",
            )
        )

    fig.update_layout(
        title="نقشه‌ی پایداری / محدوده‌ی اشتعال‌پذیری",
        xaxis_title="کسر مولی H2 در سوخت",
        yaxis_title="نسبت هم‌ارزی (φ)",
        height=450,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,0.3)",
        font={"color": "white"},
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def flame_structure_profile_plot(profile):
    """نمودار علمی دما و غلظت گونه‌های کلیدی در طول شعله."""
    x = profile["x_mm"]
    T = profile["T"]
    species = profile["species"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=T, mode="lines", name="دما (K)",
            line=dict(color="#e76f51", width=3),
            yaxis="y1",
        )
    )

    species_colors = {
        "H2": "#5fa8d3", "NH3": "#7fb069", "O2": "#f4a261",
        "H2O": "#a8dadc", "NO": "#d62828", "N2": "#adb5bd",
    }
    for sp, color in species_colors.items():
        if sp in species:
            fig.add_trace(
                go.Scatter(
                    x=x, y=species[sp], mode="lines",
                    name=f"کسر مولی {sp}",
                    line=dict(color=color, width=1.5, dash="dot"),
                    yaxis="y2",
                )
            )

    fig.update_layout(
        title="پروفایل دما و گونه‌های شیمیایی در طول شعله",
        xaxis_title="موقعیت در طول شعله (میلی‌متر)",
        yaxis=dict(title=dict(text="دما (K)", font=dict(color="#e76f51"))),
        yaxis2=dict(
            title=dict(text="کسر مولی گونه‌ها", font=dict(color="#5fa8d3")),
            overlaying="y", side="right",
        ),
        height=430,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,0.3)",
        font={"color": "white"},
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.2),
    )
    return fig


FLAME_IMAGE_COLORSCALE = [
    [0.0, "#05030a"],
    [0.15, "#1b1440"],
    [0.35, "#7b2d8e"],
    [0.55, "#e63946"],
    [0.75, "#f77f00"],
    [0.9, "#fcbf49"],
    [1.0, "#fffbea"],
]


def build_flame_silhouette(profile, height_res=260, width_res=180):
    """
    ساخت یک تصویر دوبعدیِ شبیه‌سازی‌شده از شکل شعله با استفاده از
    پروفایل واقعیِ دما. رنگ هر نقطه از تصویر مستقیماً از دمای واقعیِ
    محاسبه‌شده توسط Cantera در آن موقعیت گرفته می‌شود — یعنی رنگ‌ها
    واقعی‌اند، فقط شکل (سیلوئت مخروطی) برای شبیه‌سازی ظاهر شعله‌ی
    یک مشعل ساخته شده است.
    """
    x_mm = np.array(profile["x_mm"])
    T = np.array(profile["T"])

    T_max = T.max()
    # کوتاه کردن دنباله‌ی طولانی و صافِ محصولات پس از رسیدن به نزدیک
    # دمای آدیاباتیک، تا شکل شعله بیش‌ازحد کشیده نشود
    idx_end = int(np.argmax(T >= 0.985 * T_max))
    if idx_end < 5:
        idx_end = len(T) - 1
    idx_end = min(len(T) - 1, idx_end + 8)

    x_crop = x_mm[: idx_end + 1]
    T_crop = T[: idx_end + 1]

    x_new = np.linspace(x_crop.min(), x_crop.max(), height_res)
    T_new = np.interp(x_new, x_crop, T_crop)

    h = np.linspace(0, 1, height_res)
    # شکل سیلوئت: پهن و گرد در پایه (ورودی سرد)، با انحنای نرم به سمت
    # نوک باریک شعله (شبیه مخروط داخلی شعله‌ی برنر بونزن)
    radius = np.cos(h * np.pi / 2) ** 0.7
    radius = radius / radius.max()
    radius = np.clip(radius, 0.03, 1.0)

    xs = np.linspace(-1, 1, width_res)
    z = np.full((height_res, width_res), np.nan)
    for i in range(height_res):
        mask = np.abs(xs) <= radius[i]
        z[i, mask] = T_new[i]

    return z, T_new


def flame_image_plot(profile):
    z, T_new = build_flame_silhouette(profile)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            zsmooth="best",
            colorscale=FLAME_IMAGE_COLORSCALE,
            zmin=float(min(profile["T"])),
            zmax=float(max(profile["T"])),
            showscale=True,
            colorbar=dict(
                title=dict(text="دما (K)", font=dict(color="white")),
                tickfont={"color": "white"},
            ),
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        title="تصویر شبیه‌سازی‌شده‌ی شعله (رنگ = دمای واقعی محاسبه‌شده)",
        height=480,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


@st.cache_data(show_spinner=False)
def get_rql_result(h2_fraction, phi_primary, phi_overall, tau_primary, tau_quench, tau_secondary):
    return run_rql_network(
        h2_fraction=h2_fraction,
        phi_primary=phi_primary,
        phi_overall=phi_overall,
        tau_primary=tau_primary,
        tau_quench=tau_quench,
        tau_secondary=tau_secondary,
    )


def rql_stage_plot(res):
    """نمودار پیشرفت دما و NOx در سه مرحله‌ی شبکه‌ی راکتور RQL."""
    stages = ["ورودی (کمپرسور)", "ناحیه‌ی غنی (Primary)", "بعد از خاموشی (Quench)", "ناحیه‌ی رقیق (خروجی)"]
    temps = [res["T_in"], res["T_primary_zone"], res["T_after_quench"], res["T_secondary_zone_out"]]
    nox = [0, res["NO_ppm_primary_zone"], None, res["NO_ppm_final"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=stages, y=temps, mode="lines+markers", name="دما (K)",
            line=dict(color="#e76f51", width=3), marker=dict(size=12),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[stages[1], stages[3]], y=[nox[1], nox[3]], mode="lines+markers",
            name="NOx (ppm)", line=dict(color="#d62828", width=3, dash="dot"),
            marker=dict(size=12, symbol="diamond"), yaxis="y2",
        )
    )
    fig.update_layout(
        title="پیشرفت دما و NOx در طول شبکه‌ی راکتور RQL",
        yaxis=dict(title=dict(text="دما (K)", font=dict(color="#e76f51"))),
        yaxis2=dict(
            title=dict(text="NOx (ppm)", font=dict(color="#d62828")),
            overlaying="y", side="right",
        ),
        height=450,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,0.3)",
        font={"color": "white"},
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


@st.cache_data(show_spinner=False)
def get_diffusion_flame(h2_fraction, mdot_fuel, mdot_ox):
    return run_diffusion_flame(
        h2_fraction=float(h2_fraction),
        mdot_fuel=float(mdot_fuel),
        mdot_ox=float(mdot_ox),
        return_profile=True,
    )


def diffusion_profile_plot(profile):
    """نمودار دما و گونه‌های کلیدی در طول شعله‌ی نفوذی (از نازل سوخت تا نازل اکسیدکننده)."""
    x = profile["x_mm"]
    T = profile["T"]
    species = profile["species"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=T, mode="lines", name="دما (K)",
            line=dict(color="#e76f51", width=3), yaxis="y1",
        )
    )
    species_colors = {
        "H2": "#5fa8d3", "NH3": "#7fb069", "O2": "#f4a261",
        "H2O": "#a8dadc", "NO": "#d62828",
    }
    for sp, color in species_colors.items():
        if sp in species:
            fig.add_trace(
                go.Scatter(
                    x=x, y=species[sp], mode="lines", name=f"کسر مولی {sp}",
                    line=dict(color=color, width=1.5, dash="dot"), yaxis="y2",
                )
            )

    fig.update_layout(
        title="پروفایل شعله‌ی نفوذی: از نازل سوخت (چپ) تا نازل اکسیدکننده (راست)",
        xaxis_title="موقعیت (میلی‌متر)",
        yaxis=dict(title=dict(text="دما (K)", font=dict(color="#e76f51"))),
        yaxis2=dict(
            title=dict(text="کسر مولی گونه‌ها", font=dict(color="#5fa8d3")),
            overlaying="y", side="right",
        ),
        height=430,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,0.3)",
        font={"color": "white"},
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.2),
    )
    return fig


def build_diffusion_flame_image(profile, height_res=180, width_res=260):
    """
    تصویرسازی شعله‌ی نفوذی به‌شکل یک «صفحه‌ی نازک نورانی» بین دو نازل —
    برخلاف شعله‌ی پیش‌آمیخته (که مخروطی است)، شعله‌ی نفوذی یک دیسک نازک
    و درخشان در محل تلاقی دو جریان است. رنگ از دمای واقعی محاسبه‌شده
    گرفته می‌شود.
    """
    x_mm = np.array(profile["x_mm"])
    T = np.array(profile["T"])

    x_new = np.linspace(x_mm.min(), x_mm.max(), width_res)
    T_new = np.interp(x_new, x_mm, T)

    # عرض عمودی نمادین: باریک در دو انتها (نازل‌ها)، پهن‌تر و درخشان در
    # محل شعله (بیشینه‌ی دما)
    idx_flame = np.argmax(T_new)
    pos = np.arange(width_res)
    sigma = width_res * 0.05
    bulge = np.exp(-0.5 * ((pos - idx_flame) / sigma) ** 2)
    half_height = (0.15 + 0.85 * bulge) * (height_res / 2)

    ys = np.linspace(-height_res / 2, height_res / 2, height_res)
    z = np.full((height_res, width_res), np.nan)
    for j in range(width_res):
        mask = np.abs(ys) <= half_height[j]
        z[mask, j] = T_new[j]

    return z, T_new


def diffusion_flame_image_plot(profile):
    z, T_new = build_diffusion_flame_image(profile)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            zsmooth="best",
            colorscale=FLAME_IMAGE_COLORSCALE,
            zmin=float(min(profile["T"])),
            zmax=float(max(profile["T"])),
            showscale=True,
            colorbar=dict(
                title=dict(text="دما (K)", font=dict(color="white")),
                tickfont={"color": "white"},
            ),
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        title="تصویر شبیه‌سازی‌شده‌ی شعله‌ی نفوذی (سوخت از چپ، هوا از راست)",
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


# ---------------------------------------------------------------------------
# رابط کاربری اصلی
# ---------------------------------------------------------------------------
def main():
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {BG}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🔥 H2/NH3 Combustion Explorer")
    st.caption(
        "شبیه‌سازی احتراق سوخت‌های بدون کربن (هیدروژن-آمونیا) با "
        "Cantera + مدل جانشین Gaussian Process + بهینه‌سازی چندهدفه NSGA-II"
    )

    bundle = load_models()
    if bundle is None:
        st.error(
            "مدل جانشین پیدا نشد. ابتدا اجرا کنید:\n\n"
            "```\npython src/generate_dataset.py\npython src/train_surrogate.py\n```"
        )
        return

    h2_range = bundle["h2_range"]
    phi_range = bundle["phi_range"]

    # --- کنترل‌های ورودی (سایدبار) ---
    st.sidebar.header("پارامترهای ورودی")
    h2_sel = st.sidebar.slider(
        "کسر مولی H2 در سوخت", float(h2_range[0]), float(h2_range[1]), 0.5, 0.01
    )
    phi_sel = st.sidebar.slider(
        "نسبت هم‌ارزی (φ)", float(phi_range[0]), float(phi_range[1]), 1.0, 0.01
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**ترکیب فعلی سوخت:**\n\n"
        f"- H2: {h2_sel*100:.1f}٪\n"
        f"- NH3: {(1-h2_sel)*100:.1f}٪\n"
        f"- نسبت هم‌ارزی: {phi_sel:.2f}"
    )

    # --- پیش‌بینی برای انتخاب فعلی ---
    su_pred, su_std = gp_predict(bundle, h2_sel, phi_sel, "Su_cm_s")
    tad_pred, _ = gp_predict(bundle, h2_sel, phi_sel, "T_ad")
    no_pred, _ = gp_predict(bundle, h2_sel, phi_sel, "NO_ppm")

    # --- ردیف گیج‌ها ---
    st.subheader("پیش‌بینی زنده")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(
            make_gauge(float(su_pred[0]), "سرعت شعله (cm/s)", [0, 300], color="#5fa8d3"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            make_gauge(float(tad_pred[0]), "دمای آدیاباتیک (K)", [1200, 2400], color="#e76f51"),
            use_container_width=True,
        )
    with c3:
        st.plotly_chart(
            make_gauge(float(no_pred[0]), "غلظت NO (ppm)", [0, 10000], color="#d62828"),
            use_container_width=True,
        )

    st.caption(f"عدم قطعیت مدل برای سرعت شعله: ± {float(su_std[0]):.2f} cm/s")

    # --- نقشه حرارتی + Pareto + Radar ---
    grid = compute_grid(bundle, h2_range, phi_range)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        [
            "🌡️ نقشه حرارتی شعله",
            "⚖️ مرز بهینه Pareto",
            "📊 مقایسه رادار",
            "⏱️ زمان تأخیر اشتعال",
            "🧯 نقشه پایداری",
            "🔥 تصویر واقعی شعله",
            "✈️ شبکه راکتور RQL (توربین)",
            "🌬️ شعله نفوذی مخالف‌جریان",
            "🌪️ حساسیت واکنش‌های NOx",
        ]
    )

    with tab1:
        target = st.radio(
            "نمایش کدام کمیت روی نقشه؟",
            ["سرعت شعله (Su)", "دمای آدیاباتیک", "غلظت NOx"],
            horizontal=True,
        )
        key_map = {
            "سرعت شعله (Su)": ("Su", "cm/s"),
            "دمای آدیاباتیک": ("Tad", "K"),
            "غلظت NOx": ("NO", "ppm"),
        }
        key, unit = key_map[target]
        st.plotly_chart(
            flame_heatmap(grid, key, f"نقشه {target} در فضای طراحی", unit, h2_sel, phi_sel),
            use_container_width=True,
        )

    with tab2:
        pareto_df = load_pareto()
        if pareto_df is None:
            st.info(
                "فایل مرز Pareto پیدا نشد. اجرا کنید:\n\n"
                "```\npython src/optimize.py\n```"
            )
        else:
            st.plotly_chart(
                pareto_plot(pareto_df, current_point=(float(su_pred[0]), float(no_pred[0]))),
                use_container_width=True,
            )

    with tab3:
        st.plotly_chart(radar_comparison(bundle, h2_sel, phi_sel), use_container_width=True)

    with tab4:
        ign_df = load_ignition_delay()
        if ign_df is None or len(ign_df) == 0:
            st.info(
                "دیتاست زمان تأخیر اشتعال پیدا نشد. اجرا کنید:\n\n"
                "```\npython src/ignition_delay.py\n```"
            )
        else:
            st.plotly_chart(ignition_delay_plot(ign_df), use_container_width=True)
            st.caption(
                "نمودار Arrhenius: شیب هر خط نشان‌دهنده‌ی انرژی فعال‌سازی "
                "(activation energy) واکنش‌های کلیدی احتراق برای آن ترکیب سوخت است."
            )

    with tab5:
        stab_df = load_stability_map()
        if stab_df is None or len(stab_df) == 0:
            st.info(
                "دیتاست نقشه‌ی پایداری پیدا نشد. اجرا کنید:\n\n"
                "```\npython src/stability_map.py\n```"
            )
        else:
            st.plotly_chart(
                flammability_envelope_plot(stab_df, h2_sel=h2_sel, phi_sel=phi_sel),
                use_container_width=True,
            )
            st.caption(
                "نقاط قرمز به معنای عدم همگرایی حل عددی نزدیک محدوده‌ی خاموشی "
                "شعله است (blow-off/extinction limit)."
            )

    with tab6:
        st.markdown(
            "این تب برخلاف بقیه، از مدل جانشین سریع استفاده نمی‌کند و "
            "شبیه‌سازی **واقعی** شعله (Cantera) را برای ترکیب انتخاب‌شده در "
            "سایدبار اجرا می‌کند. چون هر اجرا ۵ تا ۱۵ ثانیه طول می‌کشد، "
            "به‌جای اجرای خودکار روی هر حرکت اسلایدر، با دکمه‌ی زیر اجرا می‌شود."
        )
        run_clicked = st.button("🔥 شبیه‌سازی کن")

        if run_clicked:
            with st.spinner("در حال حل معادلات شعله با Cantera... (۵-۱۵ ثانیه)"):
                profile_result = get_flame_profile(h2_sel, phi_sel)
            st.session_state["flame_profile"] = profile_result

        profile_result = st.session_state.get("flame_profile")
        if profile_result is None:
            st.info("برای دیدن تصویر واقعی شعله، دکمه‌ی بالا را بزنید.")
        elif not profile_result.get("success"):
            st.error(f"شبیه‌سازی ناموفق شد: {profile_result.get('error')}")
        else:
            profile = profile_result["profile"]
            st.plotly_chart(flame_image_plot(profile), use_container_width=True)
            st.plotly_chart(flame_structure_profile_plot(profile), use_container_width=True)
            st.caption(
                "محور y در تصویر شعله صرفاً برای زیبایی بصری است و کمیت "
                "فیزیکی مستقیمی نیست؛ اطلاعات علمی (دما) با رنگ در طول محور x "
                "نمایش داده شده است."
            )

    with tab7:
        st.markdown(
            "#### مدل ساده‌شده‌ی محفظه‌ی احتراق توربین گاز (Rich-Quench-Lean)\n"
            "بر خلاف تب‌های قبلی (که یک شعله‌ی آزمایشگاهی ساده را مدل می‌کردند)، "
            "این بخش یک **مدل کاهش‌مرتبه (reduced-order)** از محفظه‌ی احتراق "
            "واقعی توربین گاز است: ناحیه‌ی غنی برای سرکوب NOx، خاموشی سریع با "
            "هوای ثانویه، و ناحیه‌ی رقیق برای تکمیل احتراق. دما و فشار ورودی "
            "معادل شرایط واقعی خروجی کمپرسور است."
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            phi_primary = st.slider("نسبت هم‌ارزی ناحیه‌ی غنی (φ_primary)", 1.0, 2.2, 1.4, 0.05)
        with c2:
            phi_overall = st.slider("نسبت هم‌ارزی کلی رقیق (φ_overall)", 0.3, 0.9, 0.6, 0.05)
        with c3:
            tau_quench_ms = st.slider(
                "⚡ سرعت خاموشی/مخلوط‌سازی (ms)", 0.1, 8.0, 1.0, 0.1,
                help="زمان اقامت گاز در ناحیه‌ی خاموشی؛ کوچک‌تر = خاموشی سریع‌تر. "
                "این پارامتر اصلی برای کنترل NOx در طراحی واقعی است.",
            )
        with c4:
            tau_secondary_ms = st.slider(
                "زمان اقامت ناحیه‌ی رقیق نهایی (ms)", 2.0, 20.0, 8.0, 0.5,
                help="زمان تکمیل احتراق پس از خاموشی (برای burnout کامل CO/سوخت باقی‌مانده).",
            )

        rql_res = get_rql_result(
            h2_fraction=h2_sel,
            phi_primary=phi_primary,
            phi_overall=phi_overall,
            tau_primary=2e-3,
            tau_quench=tau_quench_ms / 1000.0,
            tau_secondary=tau_secondary_ms / 1000.0,
        )

        if not rql_res.get("success"):
            st.error(f"شبیه‌سازی ناموفق شد: {rql_res.get('error')}")
        else:
            st.plotly_chart(rql_stage_plot(rql_res), use_container_width=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("NOx ناحیه‌ی غنی", f"{rql_res['NO_ppm_primary_zone']:.0f} ppm")
            m2.metric("NOx نهایی خروجی", f"{rql_res['NO_ppm_final']:.0f} ppm")
            m3.metric("دمای نهایی خروجی", f"{rql_res['T_secondary_zone_out']:.0f} K")
            st.caption(
                "اسلایدر «سرعت خاموشی» را تغییر بده و ببین NOx نهایی چطور تغییر "
                "می‌کند — این حساسیت واقعی به سرعت مخلوط‌سازی، دقیقاً همان چالش "
                "طراحی محفظه‌های احتراق RQL واقعی است."
            )

    with tab8:
        st.markdown(
            "#### شعله‌ی نفوذی مخالف‌جریان (Counterflow Diffusion Flame)\n"
            "برخلاف تب‌های قبلی، اینجا سوخت و هوا از قبل مخلوط نیستند — سوخت "
            "خالص از یک نازل و هوا از نازل مقابل به سمت هم جریان دارند و شعله "
            "دقیقاً در محل تلاقی (به‌واسطه‌ی نفوذ مولکولی) شکل می‌گیرد. این "
            "پیکربندی نماینده‌ی بسیاری از مشعل‌های صنعتی واقعی است."
        )

        d1, d2, d3 = st.columns(3)
        with d1:
            mdot_fuel = st.slider("شار جرمی سوخت (kg/m²/s)", 0.2, 1.5, 0.5, 0.05)
        with d2:
            mdot_ox = st.slider("شار جرمی اکسیدکننده (kg/m²/s)", 0.2, 1.5, 0.5, 0.05)
        with d3:
            st.metric("نرخ کشش تخمینی", f"{(mdot_fuel+mdot_ox)/0.02:.0f} 1/s")

        run_diff_clicked = st.button("🌬️ شبیه‌سازی شعله‌ی نفوذی کن", key="run_diffusion")

        if run_diff_clicked:
            with st.spinner("در حال حل معادلات شعله‌ی نفوذی با Cantera... (چند ثانیه)"):
                diff_result = get_diffusion_flame(h2_sel, mdot_fuel, mdot_ox)
            st.session_state["diffusion_flame"] = diff_result

        diff_result = st.session_state.get("diffusion_flame")
        if diff_result is None:
            st.info("برای دیدن شعله‌ی نفوذی، دکمه‌ی بالا را بزن.")
        elif not diff_result.get("success"):
            st.warning(
                f"شعله مشتعل نشد یا خاموش شد (extinguished): "
                f"{diff_result.get('error', 'نامشخص')}. سرعت‌های جرمی را کاهش بده "
                "و دوباره امتحان کن."
            )
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("بیشینه‌ی دما", f"{diff_result['T_max']:.0f} K")
            m2.metric("موقعیت شعله", f"{diff_result['x_flame_mm']:.1f} mm")
            m3.metric("بیشینه‌ی NOx", f"{diff_result['NO_ppm_max']:.0f} ppm")
            st.plotly_chart(diffusion_flame_image_plot(diff_result["profile"]), use_container_width=True)
            st.plotly_chart(diffusion_profile_plot(diff_result["profile"]), use_container_width=True)
            st.caption(
                "با افزایش هر دو شار جرمی (نرخ کشش بالاتر)، شعله به تدریج به "
                "سمت خاموشی (extinction) می‌رود — این معیار پایداری شعله‌ی نفوذی "
                "در برابر آشفتگی جریان است."
            )

    with tab9:
        sens_df = load_sensitivity_data()
        if sens_df is None or len(sens_df) == 0:
            st.info(
                "دیتاست تحلیل حساسیت پیدا نشد. اجرا کنید (حدود ۵ دقیقه طول می‌کشد):\n\n"
                "```\npython src/sensitivity_analysis.py\n```"
            )
        else:
            st.markdown(
                "#### کدام واکنش‌های شیمیایی بیشترین تأثیر را روی NOx دارند؟\n"
                "هر میله نشان می‌دهد که افزایش ۱۰٪ ثابت نرخ آن واکنش، چقدر "
                "غلظت نهایی NO را تغییر می‌دهد (در شرایط H2=50٪, φ=1.0)."
            )
            st.plotly_chart(sensitivity_tornado_plot(sens_df), use_container_width=True)
            st.caption(
                "میله‌های قرمز: واکنش‌هایی که تشکیل NOx را تشدید می‌کنند. "
                "میله‌های آبی: واکنش‌هایی که NOx موجود را مصرف/احیا می‌کنند "
                "(معمولاً از طریق رادیکال‌های NHi باقی‌مانده از سوخت آمونیا — "
                "همان پدیده‌ی «ری‌برن» که در تب شبکه‌ی راکتور RQL هم دیدیم)."
            )


if __name__ == "__main__":
    main()
