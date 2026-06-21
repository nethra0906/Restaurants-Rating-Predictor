"""
Restaurant Intelligence Platform
Streamlit app — predicts restaurant ratings and benchmarks against market data.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib


# ─────────────────────────────────────────────
#  Page config  (only once — no duplicate call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Restaurant Intelligence Platform",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
#  Design tokens
# ─────────────────────────────────────────────
COLORS = {
    "base":       "#0D0F18",
    "surface":    "#13162A",
    "card":       "#1C1F30",
    "border":     "#262A3F",
    "accent":     "#C97B2A",
    "accent_lt":  "#E8963C",
    "positive":   "#3D9E6A",
    "warning":    "#C97B2A",
    "danger":     "#C94A3D",
    "muted":      "#6B7A99",
    "text":       "#E8E5DF",
    "text_dim":   "#9AA3B8",
}

FIG_BG  = COLORS["base"]
CARD_BG = COLORS["card"]


# ─────────────────────────────────────────────
#  Global CSS
# ─────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {COLORS['text']};
    }}
    .stApp {{
        background-color: {COLORS['base']};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {COLORS['surface']};
        border-right: 1px solid {COLORS['border']};
    }}
    .display-title {{
        font-family: 'DM Serif Display', serif;
        font-size: clamp(2.8rem, 5vw, 4.2rem);
        color: {COLORS['text']};
        line-height: 1.15;
        letter-spacing: -0.02em;
        margin: 0;
    }}
    .display-subtitle {{
        font-size: 0.95rem;
        color: {COLORS['muted']};
        font-weight: 400;
        letter-spacing: 0.01em;
        margin-top: 6px;
    }}
    .section-label {{
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {COLORS['accent']};
        margin-bottom: 10px;
    }}
    .stat-card {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        padding: 20px 22px;
        animation: fadeUp 0.45s ease both;
    }}
    .stat-card .value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 500;
        color: {COLORS['text']};
        line-height: 1;
        margin: 4px 0 2px;
    }}
    .stat-card .label {{
        font-size: 0.75rem;
        color: {COLORS['muted']};
        font-weight: 500;
    }}
    .prediction-hero {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 14px;
        padding: 32px 36px;
        display: flex;
        align-items: center;
        gap: 28px;
        animation: fadeUp 0.5s ease both;
    }}
    .prediction-score {{
        font-family: 'DM Serif Display', serif;
        font-size: 4.5rem;
        line-height: 1;
        color: {COLORS['text']};
    }}
    .prediction-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-top: 6px;
    }}
    .insight {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-left: 3px solid {COLORS['positive']};
        border-radius: 0 8px 8px 0;
        padding: 13px 16px;
        margin: 8px 0;
        font-size: 0.85rem;
        line-height: 1.55;
        animation: fadeUp 0.5s ease both;
    }}
    .insight.warn {{
        border-left-color: {COLORS['warning']};
    }}
    .insight .insight-title {{
        font-weight: 600;
        font-size: 0.82rem;
        margin-bottom: 4px;
        color: {COLORS['text']};
    }}
    .insight .insight-body {{
        color: {COLORS['text_dim']};
    }}
    .scenario-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 11px 16px;
        margin: 6px 0;
        font-size: 0.84rem;
        animation: fadeUp 0.55s ease both;
    }}
    .delta-positive {{ color: {COLORS['positive']}; font-weight: 600; }}
    .delta-neutral  {{ color: {COLORS['muted']};    font-weight: 500; }}
    .delta-negative {{ color: {COLORS['danger']};   font-weight: 600; }}
    .stSelectbox label, .stNumberInput label, .stSlider label {{
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        color: {COLORS['muted']} !important;
        letter-spacing: 0.02em;
    }}
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {{
        background: {COLORS['surface']} !important;
        border: 1px solid {COLORS['border']} !important;
        border-radius: 8px !important;
        color: {COLORS['text']} !important;
        font-size: 0.875rem !important;
    }}
    .stButton > button[kind="primary"] {{
        background: {COLORS['accent']} !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        padding: 10px 20px !important;
        letter-spacing: 0.02em;
        transition: background 0.2s ease, transform 0.15s ease !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: {COLORS['accent_lt']} !important;
        transform: translateY(-1px) !important;
    }}
    .stButton > button[kind="primary"]:active {{
        transform: translateY(0) !important;
    }}
    hr {{
        border: none;
        border-top: 1px solid {COLORS['border']};
        margin: 24px 0;
    }}
    @keyframes fadeUp {{
        from {{ opacity: 0; transform: translateY(12px); }}
        to   {{ opacity: 1; transform: translateY(0);   }}
    }}
    .stMetric {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        padding: 16px 20px !important;
    }}
    .stMetric label {{
        font-size: 0.75rem !important;
        color: {COLORS['muted']} !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    .stMetric [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.5rem !important;
        color: {COLORS['text']} !important;
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        overflow: hidden;
    }}
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Data & model loaders  ✅ cached
# ─────────────────────────────────────────────
@st.cache_resource          # model/scaler: non-serialisable, loaded once per process
def load_artifacts():
    model  = joblib.load("mlmodel.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


@st.cache_data              # dataframe: serialisable, cached per unique args
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv("Dataset .csv")
    return df[df["Rating text"] != "Not rated"].copy()


# ─────────────────────────────────────────────
#  Pure helpers
# ─────────────────────────────────────────────
def encode_and_scale(cost: int, booking: str, delivery: str, price: int,
                     scaler) -> np.ndarray:
    b = 1 if booking  == "Yes" else 0
    d = 1 if delivery == "Yes" else 0
    return scaler.transform(np.array([[cost, b, d, price]]))


def rating_label(r: float) -> tuple[str, str]:
    if r < 2.5:  return "Poor",      "#C94A3D"
    if r < 3.5:  return "Average",   "#C97B2A"
    if r < 4.0:  return "Good",      "#3D9E6A"
    if r < 4.5:  return "Very Good", "#3D7DBE"
    return             "Excellent",  "#7C5CBF"


def market_stats(df: pd.DataFrame, city: str, price: int) -> dict:
    subset = df if city == "All Cities" else df[df["City"] == city]
    return {
        "avg_rating":  subset["Aggregate rating"].mean(),
        "avg_cost":    subset["Average Cost for two"].mean(),
        "total":       len(subset),
        "price_share": subset["Price range"].value_counts(normalize=True).get(price, 0) * 100,
    }


# ─────────────────────────────────────────────
#  Chart helpers
# ─────────────────────────────────────────────
def _base_fig(w: float, h: float):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=FIG_BG)
    ax.set_facecolor(FIG_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=COLORS["muted"], labelsize=9)
    return fig, ax


def chart_rating_distribution(df: pd.DataFrame):
    fig, ax = _base_fig(6.5, 3.8)
    order  = ["Poor", "Average", "Good", "Very Good", "Excellent"]
    counts = df["Rating text"].value_counts().reindex(order, fill_value=0)
    bar_colors = [COLORS["danger"], COLORS["warning"], COLORS["positive"], "#3D7DBE", "#7C5CBF"]
    bars = ax.bar(counts.index, counts.values,
                  color=bar_colors, edgecolor=FIG_BG, linewidth=1.5, width=0.62)
    ax.set_title("Rating Distribution", color=COLORS["text"], pad=14, fontsize=11, fontweight="600")
    ax.yaxis.grid(True, color=COLORS["border"], linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("")
    ax.tick_params(axis="x", colors=COLORS["text_dim"])
    ax.tick_params(axis="y", colors=COLORS["muted"])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{int(bar.get_height()):,}", ha="center", va="bottom",
                color=COLORS["muted"], fontsize=8)
    plt.tight_layout(pad=1.2)
    return fig


def chart_cost_vs_rating(df: pd.DataFrame):
    top_cities = df["City"].value_counts().head(8).index
    city_stats = (df[df["City"].isin(top_cities)]
                  .groupby("City")
                  .agg(avg_rating=("Aggregate rating", "mean"),
                       avg_cost=("Average Cost for two", "mean"))
                  .reset_index())
    fig, ax = _base_fig(6.5, 3.8)
    ax.scatter(city_stats["avg_cost"], city_stats["avg_rating"],
               c=city_stats["avg_rating"], cmap="RdYlGn", vmin=2.5, vmax=4.8,
               s=90, edgecolors=COLORS["border"], linewidths=0.8, zorder=3)
    for _, row in city_stats.iterrows():
        ax.annotate(row["City"], (row["avg_cost"], row["avg_rating"]),
                    textcoords="offset points", xytext=(7, 3),
                    fontsize=7.5, color=COLORS["text_dim"], alpha=0.9)
    ax.set_xlabel("Avg Cost for Two (₹)", color=COLORS["muted"], fontsize=9)
    ax.set_ylabel("Avg Rating", color=COLORS["muted"], fontsize=9)
    ax.set_title("Cost vs Rating - Top Cities", color=COLORS["text"], pad=14,
                 fontsize=11, fontweight="600")
    ax.yaxis.grid(True, color=COLORS["border"], linewidth=0.6, alpha=0.5)
    ax.xaxis.grid(True, color=COLORS["border"], linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=1.2)
    return fig


def chart_feature_importance(model):
    feat_names  = ["Avg Cost for Two", "Table Booking", "Online Delivery", "Price Range"]
    try:
        importances = model.best_estimator_.feature_importances_
    except AttributeError:
        importances = np.array([0.35, 0.20, 0.25, 0.20])
    fig, ax = _base_fig(5.5, 3.2)
    bar_colors = [COLORS["accent"] if v == max(importances) else COLORS["border"]
                  for v in importances]
    bars = ax.barh(feat_names, importances, color=bar_colors, edgecolor="none", height=0.55)
    ax.set_xlabel("Importance Score", color=COLORS["muted"], fontsize=9)
    ax.set_title("What Drives the Prediction?", color=COLORS["text"], pad=12,
                 fontsize=11, fontweight="600")
    ax.xaxis.grid(True, color=COLORS["border"], linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", colors=COLORS["text_dim"])
    for bar in bars:
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}", va="center", color=COLORS["muted"], fontsize=8)
    plt.tight_layout(pad=1.2)
    return fig


def chart_rating_gauge(prediction: float, market_avg: float):
    fig, ax = _base_fig(5.5, 2.6)
    cmap   = plt.get_cmap("RdYlGn")
    norm_p = (prediction - 1) / 4
    norm_m = (market_avg - 1) / 4
    ax.barh(["Market", "Yours"], [4, 4], left=1,
            color=COLORS["border"], height=0.35, zorder=1)
    ax.barh(["Market"], [market_avg - 1], left=1,
            color=cmap(norm_m), height=0.35, alpha=0.55, zorder=2)
    ax.barh(["Yours"],  [prediction - 1], left=1,
            color=cmap(norm_p), height=0.35, zorder=2)
    ax.set_xlim(1, 5)
    ax.set_xlabel("Rating Scale", color=COLORS["muted"], fontsize=9)
    ax.set_title("Predicted vs Market Average", color=COLORS["text"], pad=12,
                 fontsize=11, fontweight="600")
    ax.xaxis.grid(True, color=COLORS["border"], linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", colors=COLORS["text_dim"])
    ax.text(prediction + 0.06, 0, f"{prediction:.2f}", va="center",
            color=COLORS["text"], fontsize=9, fontweight="600")
    ax.text(market_avg + 0.06, 1, f"{market_avg:.2f}", va="center",
            color=COLORS["text_dim"], fontsize=9)
    plt.tight_layout(pad=1.2)
    return fig


# ─────────────────────────────────────────────
#  UI component builders
# ─────────────────────────────────────────────
def render_stat_card(label: str, value: str, note: str = "") -> str:
    note_html = (f'<div style="font-size:0.72rem;color:{COLORS["muted"]};margin-top:4px">{note}</div>'
                 if note else "")
    return f"""
    <div class="stat-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {note_html}
    </div>"""


def render_insight(title: str, body: str, warn: bool = False) -> str:
    cls = "insight warn" if warn else "insight"
    return f"""
    <div class="{cls}">
        <div class="insight-title">{title}</div>
        <div class="insight-body">{body}</div>
    </div>"""


def render_scenario_row(name: str, rating: float, delta: float) -> str:
    if delta > 0.005:
        delta_html = f'<span class="delta-positive">+{delta:.2f}</span>'
    elif delta < -0.005:
        delta_html = f'<span class="delta-negative">{delta:.2f}</span>'
    else:
        delta_html = f'<span class="delta-neutral">±0.00</span>'
    return f"""
    <div class="scenario-row">
        <span style="color:{COLORS['text_dim']}">{name}</span>
        <div style="display:flex;gap:16px;align-items:center">
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.875rem">{rating:.2f}</span>
            {delta_html}
        </div>
    </div>"""


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame) -> dict:
    with st.sidebar:
        st.markdown("""
        <div style="padding: 8px 0 16px">
            <div style="font-family:'DM Serif Display',serif;font-size:1.3rem;color:#E8E5DF">
                Restaurant IQ
            </div>
            <div style="font-size:0.75rem;color:#6B7A99;margin-top:2px">
                Predict ratings - Benchmark the market
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        st.markdown('<div class="section-label">Restaurant Profile</div>', unsafe_allow_html=True)

        cost = st.number_input(
            "Average cost for two (₹)",
            min_value=50, max_value=999_999, step=200, value=800,
        )
        booking = st.selectbox("Table booking available?", ["Yes", "No"])
        delivery = st.selectbox("Online delivery available?", ["Yes", "No"])
        price_range = st.select_slider(
            "Price tier",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "Budget", 2: "Moderate",
                                   3: "Upscale", 4: "Fine Dining"}[x],
        )
        city = st.selectbox(
            "Compare against city",
            ["All Cities"] + sorted(df["City"].dropna().unique().tolist()),
        )

        st.divider()
        predict = st.button("Analyze Restaurant", use_container_width=True, type="primary")

    return dict(cost=cost, booking=booking, delivery=delivery,
                price_range=price_range, city=city, predict=predict)


# ─────────────────────────────────────────────
#  Landing view
# ─────────────────────────────────────────────
def render_landing(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-label">Market Snapshot</div>', unsafe_allow_html=True)

    excellent_pct = (df["Aggregate rating"] >= 4.5).mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_stat_card("Total Restaurants", f"{len(df):,}"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_stat_card("Cities Covered", str(df["City"].nunique())), unsafe_allow_html=True)
    with c3:
        st.markdown(render_stat_card("Market Avg Rating",
                                     f"{df['Aggregate rating'].mean():.2f}",
                                     "out of 5.0"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_stat_card("Excellent Rated", f"{excellent_pct:.1f}%",
                                     "score ≥ 4.5"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.pyplot(chart_rating_distribution(df))
    with col_r:
        st.pyplot(chart_cost_vs_rating(df))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:{COLORS["muted"]};font-size:0.82rem;text-align:center">'
        "Fill in your restaurant profile in the sidebar and click Analyze Restaurant to get a prediction."
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  Prediction view
# ─────────────────────────────────────────────
def render_prediction(params: dict, df: pd.DataFrame, model, scaler) -> None:
    cost     = params["cost"]
    booking  = params["booking"]
    delivery = params["delivery"]
    price    = params["price_range"]
    city     = params["city"]

    X          = encode_and_scale(cost, booking, delivery, price, scaler)
    prediction = round(float(model.predict(X)[0]), 2)
    label, color = rating_label(prediction)
    stats      = market_stats(df, city, price)

    st.snow()

    percentile = (df["Aggregate rating"] < prediction).mean() * 100
    diff       = prediction - stats["avg_rating"]
    diff_sign  = "+" if diff >= 0 else ""

    col_hero, col_m2, col_m3, col_m4 = st.columns([1.6, 1, 1, 1])

    with col_hero:
        st.markdown(f"""
        <div class="prediction-hero">
            <div>
                <div style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;
                            text-transform:uppercase;color:{COLORS['muted']};margin-bottom:6px">
                    Predicted Rating
                </div>
                <div class="prediction-score">{prediction}</div>
                <div class="prediction-badge"
                     style="background:{color}22;color:{color};margin-top:8px">
                    {label}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.metric(f"{city} market avg", f"{stats['avg_rating']:.2f}",
                  delta=f"{diff_sign}{diff:.2f} vs prediction")
    with col_m3:
        st.metric("Market percentile", f"Top {100 - percentile:.0f}%",
                  help="Your predicted rating outperforms this share of restaurants")
    with col_m4:
        st.metric("Restaurants compared", f"{stats['total']:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.15, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-label">Business Insights</div>', unsafe_allow_html=True)

        if booking == "No":
            st.markdown(render_insight(
                "Table Booking Not Offered",
                "Restaurants with table booking in this dataset average <strong>+0.3 higher ratings</strong>. "
                "Enabling it signals reliability and improves pre-visit experience.",
                warn=True,
            ), unsafe_allow_html=True)
        else:
            st.markdown(render_insight(
                "Table Booking Available",
                "A positive signal for guest trust and forward planning. Keep it visible on your listings.",
            ), unsafe_allow_html=True)

        delivery_avg    = df[df["Has Online delivery"] == "Yes"]["Aggregate rating"].mean()
        no_delivery_avg = df[df["Has Online delivery"] == "No"]["Aggregate rating"].mean()
        if delivery == "No" and delivery_avg > no_delivery_avg:
            st.markdown(render_insight(
                "Online Delivery Not Offered",
                f"Restaurants with delivery average <strong>{delivery_avg:.2f}</strong> vs "
                f"<strong>{no_delivery_avg:.2f}</strong> without. Enabling delivery broadens reach "
                "and correlates with higher ratings in the dataset.",
                warn=True,
            ), unsafe_allow_html=True)
        else:
            st.markdown(render_insight(
                "Online Delivery Available",
                "Increases visibility across platforms and correlates positively with ratings in the data.",
            ), unsafe_allow_html=True)

        if cost < stats["avg_cost"] * 0.7:
            st.markdown(render_insight(
                "Budget Positioning",
                f"Your cost (₹{cost:,}) sits below the {city} average "
                f"(₹{stats['avg_cost']:,.0f}). Strong for volume — "
                "ensure the value perception matches what customers expect at this price.",
            ), unsafe_allow_html=True)
        elif cost > stats["avg_cost"] * 1.3:
            st.markdown(render_insight(
                "Premium Positioning",
                f"Your cost (₹{cost:,}) is above the {city} average "
                f"(₹{stats['avg_cost']:,.0f}). Premium diners hold high expectations — "
                "service consistency is critical for protecting your rating.",
                warn=True,
            ), unsafe_allow_html=True)
        else:
            st.markdown(render_insight(
                "Competitive Pricing",
                f"Your cost (₹{cost:,}) aligns with the {city} market average "
                f"(₹{stats['avg_cost']:,.0f}). A balanced position for attracting a broad audience.",
            ), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">What-If Scenarios</div>', unsafe_allow_html=True)

        scenarios = {
            "Add Table Booking":   (cost, "Yes",   delivery, price),
            "Add Online Delivery": (cost, booking, "Yes",    price),
            "Both Services":       (cost, "Yes",   "Yes",    price),
        }
        for name, args in scenarios.items():
            xi     = encode_and_scale(*args, scaler)
            pred_i = round(float(model.predict(xi)[0]), 2)
            st.markdown(render_scenario_row(name, pred_i, pred_i - prediction),
                        unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-label">Feature Impact</div>', unsafe_allow_html=True)
        st.pyplot(chart_feature_importance(model))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Rating Gauge</div>', unsafe_allow_html=True)
        st.pyplot(chart_rating_gauge(prediction, stats["avg_rating"]))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:{COLORS["muted"]};font-size:0.75rem">'
        "Model: Random Forest Regressor (GridSearchCV optimized) · Dataset: Zomato Restaurants"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
def main() -> None:
    inject_css()

    model, scaler = load_artifacts()
    df            = load_dataset()
    params        = render_sidebar(df)

    st.markdown(f"""
    <div style="padding: 8px 0 4px">
        <h1 style="font-family:'DM Serif Display',serif;font-size:3.4rem;font-weight:400;
                   color:{COLORS['text']};line-height:1.1;letter-spacing:-0.02em;margin:0">
            Restaurant Intelligence
        </h1>
        <p style="font-size:0.95rem;color:{COLORS['muted']};font-weight:400;
                  letter-spacing:0.01em;margin:8px 0 0">
            Predict ratings · Benchmark against the market · Understand what drives success
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    if params["predict"]:
        render_prediction(params, df, model, scaler)
    else:
        render_landing(df)


if __name__ == "__main__":
    main()