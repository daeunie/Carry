"""
Carry tab -- vol-targeted carry strategy on US Treasury ETFs (long-only, T-bill residual)
-------------------------------------------------------------------------------------------
Signal (each month-end t, positions held over t+1)
    carry_i  = matched Treasury yield (DGS2/5/10/20) - 3M T-bill (DGS3MO)
    z_i      = carry z-score vs its own last WINDOW months
    score_i  = max(z_i - Z_ESCAPE, 0)          -> z <= -2.0 : ETF escapes to T-bill (weight 0)
    weights  : each ETF's share of RISK ∝ score (risk budgeting) — or of CAPITAL if scheme="capital"
    vol target: ETF basket scaled to TARGET_VOL ex-ante (max 100% invested); rest = 3M T-bill
    covariance: OAD_i · OAD_j · cov(Δyield) over the last COV_WINDOW months

Live Signal sub-tab : yields from FRED on page load (cached 6h), falls back to data/yields_monthly.csv;
                      durations from data/oad_monthly.csv (latest available month).
                      Daily tracker: z-scores and "if rebalanced today" weights from daily yields,
                      plus actual drift since the last rebalance (daily ETF prices via yfinance, if reachable).
Historical sub-tab  : data/carry_backtest.csv (monthly) and data/daily_backtest.csv (daily, weights drift
                      between monthly rebalances) — exported by step 7 of the notebook.
Parameters          : data/strategy_config.json (exported by step 7) with defaults below.
"""

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize

# ---------------- Settings ----------------
DATA_DIR = Path(__file__).resolve().parent / "data"
ETFS = ["SHY", "IEI", "IEF", "TLT"]
YIELD_MAP = {"SHY": "DGS2", "IEI": "DGS5", "IEF": "DGS10", "TLT": "DGS20"}
TBILL = "DGS3MO"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

_DEFAULTS = {"window": 60, "scheme": "risk", "target_vol": 0.05, "z_escape": -2.0,
             "cov_window": 36, "max_gross": 1.0}
try:
    CFG = {**_DEFAULTS, **json.loads((DATA_DIR / "strategy_config.json").read_text())}
except Exception:
    CFG = _DEFAULTS
WINDOW, SCHEME = int(CFG["window"]), CFG["scheme"]
TARGET_VOL, Z_ESCAPE = float(CFG["target_vol"]), float(CFG["z_escape"])
COV_WINDOW, MAX_GROSS = int(CFG["cov_window"]), float(CFG["max_gross"])

PALETTE = {"TBILL": "#b5b2ab", "SHY": "#a4c0ec", "IEI": "#6b95dc", "IEF": "#3f6bc4", "TLT": "#22397a"}
LABELS = {"TBILL": "T-bill (cash)", "SHY": "SHY", "IEI": "IEI", "IEF": "IEF", "TLT": "TLT"}
LINE_COLORS = {"Strategy": "#3f6bc4", "Equal weight (25% each)": "#222222",
               "Bloomberg US Treasury Index": "#e8843a"}
SCHEME_TEXT = "share of risk" if SCHEME == "risk" else "share of capital"

METHOD_SUMMARY = (
    f"Each month, every maturity's carry (its Treasury yield minus the 3-month T-bill) is compared with its "
    f"own last {WINDOW} months as a z-score. An ETF with z ≤ {Z_ESCAPE:.1f} is dropped; the others get a "
    f"{SCHEME_TEXT} proportional to (z + {abs(Z_ESCAPE):.0f}). The ETF basket is then scaled to "
    f"{TARGET_VOL:.0%} expected volatility (never above 100% invested), and the rest sits in 3-month T-bills."
)


# ---------------- Data ----------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_fred_daily() -> pd.DataFrame:
    out = {}
    for sid in list(YIELD_MAP.values()) + [TBILL]:
        s = pd.read_csv(FRED_CSV.format(sid), index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
        out[sid] = s.astype(float).dropna()
    return pd.DataFrame(out).sort_index()


def _to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    m = daily.resample("ME").last()
    m.index = m.index.to_period("M")
    return m


def _read_monthly(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / name, index_col="ym")
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df


def load_yields() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """(monthly yields for completed months, daily yields, source)."""
    try:
        daily = fetch_fred_daily()
        monthly, source = _to_monthly(daily), "FRED (live)"
    except Exception:
        monthly = _read_monthly("yields_monthly.csv")
        daily = pd.read_csv(DATA_DIR / "yields_daily.csv", index_col="date", parse_dates=True)
        source = "stored file (FRED unreachable)"
    current = pd.Timestamp.today().to_period("M")
    monthly = monthly.loc[monthly.index < current]
    # the latest month only counts as completed if daily data reaches its last business day
    last = monthly.index[-1]
    if daily.index.max() < (last.to_timestamp(how="end") - pd.offsets.BDay(1)).normalize():
        monthly = monthly.iloc[:-1]
    return monthly, daily, source


@st.cache_data(show_spinner=False)
def load_daily_backtest() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "daily_backtest.csv", index_col="date", parse_dates=True)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_etf_prices(start: str) -> pd.DataFrame:
    """Daily total-return prices (dividend-adjusted) from Yahoo Finance."""
    import yfinance as yf
    px = yf.download(ETFS, start=start, auto_adjust=True, progress=False)["Close"]
    return px[ETFS].dropna(how="all")


@st.cache_data(show_spinner=False)
def load_oad() -> pd.DataFrame:
    return _read_monthly("oad_monthly.csv")


@st.cache_data(show_spinner=False)
def load_backtest() -> pd.DataFrame:
    return _read_monthly("carry_backtest.csv")


# ---------------- Signal ----------------
def risk_budget_weights(cov: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    n = len(budgets)
    res = minimize(lambda w: 0.5 * w @ cov @ w - budgets @ np.log(w), np.full(n, 1 / n),
                   method="L-BFGS-B", bounds=[(1e-10, None)] * n)
    return res.x / res.x.sum()


def weights_at(t, z: pd.DataFrame, dy: pd.DataFrame, oad: pd.DataFrame) -> tuple[pd.Series, float]:
    """Weights decided at month-end t, and the ex-ante portfolio vol."""
    score = (z.loc[t, ETFS] - Z_ESCAPE).clip(lower=0).fillna(0)
    active = [e for e in ETFS if score[e] > 0]
    w = pd.Series(0.0, index=ETFS + ["TBILL"])
    if not active:
        w["TBILL"] = 1.0
        return w, 0.0
    D = oad.loc[:t, active].ffill().iloc[-1].values            # latest duration available at t
    C = dy.loc[t - COV_WINDOW + 1:t, active].cov().values
    cov = np.outer(D, D) * C * 12
    b = score[active].values / score[active].sum()
    w_rel = risk_budget_weights(cov, b) if SCHEME == "risk" else b
    vol_rel = np.sqrt(w_rel @ cov @ w_rel)
    scale = min(MAX_GROSS, TARGET_VOL / vol_rel)
    w[active] = w_rel * scale
    w["TBILL"] = 1 - w[active].sum()
    return w, vol_rel * scale


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def compute_signal(ylds: pd.DataFrame, oad: pd.DataFrame, n_months: int = 8) -> dict:
    carry = pd.DataFrame({e: ylds[y] - ylds[TBILL] for e, y in YIELD_MAP.items()})
    z = (carry - carry.rolling(WINDOW, min_periods=WINDOW).mean()) / carry.rolling(WINDOW, min_periods=WINDOW).std()
    dy = pd.DataFrame({e: ylds[y] for e, y in YIELD_MAP.items()}).diff() / 100
    months = z.dropna(how="all").index[-n_months:]
    W, vols = {}, {}
    for t in months:
        W[t], vols[t] = weights_at(t, z, dy, oad)
    return {"carry": carry, "z": z, "W": pd.DataFrame(W).T, "vol": pd.Series(vols)}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def daily_tracker(monthly: pd.DataFrame, daily: pd.DataFrame, oad: pd.DataFrame, n_months: int = 3) -> dict:
    """Daily z-scores and 'if rebalanced today' weights. Each day d is treated as if it were a month-end:
    its carry joins the previous WINDOW-1 month-end values; covariance uses completed months only."""
    carry_m = pd.DataFrame({e: monthly[y] - monthly[TBILL] for e, y in YIELD_MAP.items()})
    dy_m = pd.DataFrame({e: monthly[y] for e, y in YIELD_MAP.items()}).diff() / 100
    carry_d = pd.DataFrame({e: daily[y] - daily[TBILL] for e, y in YIELD_MAP.items()}).dropna()
    first_month = monthly.index[-1] - (n_months - 2)
    days = carry_d.loc[carry_d.index >= first_month.start_time]
    Z, W = {}, {}
    for d, c in days.iterrows():
        m = pd.Period(d, "M")
        hist = carry_m.loc[:m - 1].tail(WINDOW - 1)
        if len(hist) < WINDOW - 1:
            continue
        full = pd.concat([hist, c.to_frame().T])
        z_d = (c - full.mean()) / full.std()
        zt = pd.DataFrame([z_d[ETFS].values], index=[m - 1], columns=ETFS)
        w, _ = weights_at(m - 1, zt, dy_m, oad)                # covariance / OAD as of last month-end
        Z[d], W[d] = z_d, w
    return {"z": pd.DataFrame(Z).T, "w": pd.DataFrame(W).T}


def build_live_table(sig: dict, ylds: pd.DataFrame) -> pd.DataFrame:
    W, z, carry = sig["W"], sig["z"], sig["carry"]
    t = W.index[-1]
    now, prev = W.iloc[-1], W.iloc[-2]
    rows = []
    for e in ETFS:
        status = "Escaped (T-bill)" if z.at[t, e] <= Z_ESCAPE else "Held"
        rows.append({"Instrument": e, "Yield used": YIELD_MAP[e], "Carry (%p)": carry.at[t, e],
                     "Z-score": z.at[t, e], "Status": status,
                     "Weight (%)": now[e] * 100, "Change (pp)": (now[e] - prev[e]) * 100})
    rows.append({"Instrument": "3-Month T-Bill", "Yield used": f"{TBILL} = {ylds.at[t, TBILL]:.2f}%",
                 "Carry (%p)": np.nan, "Z-score": np.nan, "Status": "Residual",
                 "Weight (%)": now["TBILL"] * 100, "Change (pp)": (now["TBILL"] - prev["TBILL"]) * 100})
    return pd.DataFrame(rows)


def build_recent_history(W: pd.DataFrame, n_months: int = 6, flag_pp: float = 5.0):
    """Weights held in each of the last n months (row = month held), with cells flagged when they moved ≥ flag_pp."""
    recent = W.iloc[-(n_months + 1):]
    rows, changed = [], []
    for i in range(1, len(recent)):
        now, prev = recent.iloc[i], recent.iloc[i - 1]
        rows.append({"Held in": str(recent.index[i] + 1),
                     **{LABELS[a] if a == "TBILL" else a: f"{now[a]*100:.1f}%" for a in ETFS + ["TBILL"]}})
        changed.append({"Held in": False,
                        **{LABELS[a] if a == "TBILL" else a: abs(now[a] - prev[a]) * 100 >= flag_pp
                           for a in ETFS + ["TBILL"]}})
    return pd.DataFrame(rows[::-1]), pd.DataFrame(changed[::-1])


def highlight_changes(display_df: pd.DataFrame, changed_df: pd.DataFrame):
    def style_func(_):
        styles = pd.DataFrame("", index=display_df.index, columns=display_df.columns)
        for col in changed_df.columns:
            styles[col] = changed_df[col].map(lambda c: "background-color: #FCEFC7; font-weight: 600;" if c else "")
        return styles
    return display_df.style.apply(style_func, axis=None)


# ---------------- Stats ----------------
def perf_stats(r: pd.Series, rf: pd.Series) -> dict:
    r = r.dropna()
    rf = rf.reindex(r.index).fillna(0)
    wealth = (1 + r).cumprod()
    cagr = wealth.iloc[-1] ** (12 / len(r)) - 1
    ex = r - rf
    dd = wealth / wealth.cummax() - 1
    return {"CAGR (%)": cagr * 100, "Vol (%)": r.std() * np.sqrt(12) * 100,
            "Sharpe": ex.mean() / ex.std() * np.sqrt(12), "Sortino": ex.mean() / ex[ex < 0].std() * np.sqrt(12),
            "Max Drawdown (%)": dd.min() * 100, "Calmar": cagr / abs(dd.min())}


def relative_stats(r: pd.Series, b: pd.Series) -> dict:
    act = r - b
    te = act.std() * np.sqrt(12)
    up, dn = b > 0, b < 0
    return {"Excess return (%/yr)": act.mean() * 1200, "Tracking error (%)": te * 100,
            "Information ratio": act.mean() * 12 / te, "t-stat": act.mean() / act.std() * np.sqrt(len(act)),
            "Beta": np.cov(r, b)[0, 1] / b.var(),
            "Up capture (%)": r[up].mean() / b[up].mean() * 100, "Down capture (%)": r[dn].mean() / b[dn].mean() * 100}


# ---------------- Charts ----------------
FONT = "IBM Plex Sans"


def _x():
    return alt.X("date:T", title=None,
                 axis=alt.Axis(format="%Y", tickCount={"interval": "year", "step": 2}, labelAngle=0))


def _ts(idx: pd.PeriodIndex) -> pd.DatetimeIndex:
    return idx.to_timestamp()


def _style(chart, height: int = 320):
    return (chart.properties(height=height)
            .configure_view(strokeWidth=0)
            .configure_axis(labelFont=FONT, titleFont=FONT, labelColor="#6B7488", titleColor="#6B7488",
                            gridColor="#E9EDF3", domainColor="#C9D1DE", tickColor="#C9D1DE",
                            labelFontSize=11, titleFontSize=11, titleFontWeight="normal")
            .configure_legend(labelFont=FONT, labelFontSize=12, labelColor="#1B2540",
                              symbolStrokeWidth=3, orient="top", title=None)
            .configure_title(font=FONT))


def line_chart(df: pd.DataFrame, y_title, fmt: str = ".2f", colors: dict | None = None, height=320,
               zero: bool = True, interpolate: str = "linear"):
    long = df.reset_index(names="date").melt("date", var_name="series", value_name="value")
    colors = colors or LINE_COLORS
    chart = (alt.Chart(long).mark_line(strokeWidth=2, interpolate=interpolate)
             .encode(x=_x(),
                     y=alt.Y("value:Q", title=y_title, axis=alt.Axis(format=fmt), scale=alt.Scale(zero=zero)),
                     color=alt.Color("series:N", title=None,
                                     scale=alt.Scale(domain=list(colors), range=list(colors.values()))),
                     strokeDash=alt.condition(alt.datum.series == "Strategy", alt.value([1, 0]), alt.value([5, 3])),
                     tooltip=["date:T", "series:N", alt.Tooltip("value:Q", format=fmt)]))
    return _style(chart, height)


def weights_chart(w: pd.DataFrame, daily: bool = False):
    order = ETFS + ["TBILL"]
    long = (w[order].rename(columns=LABELS).reset_index(names="date")
            .melt("date", var_name="asset", value_name="weight"))
    long["order"] = long["asset"].map({LABELS[a]: i for i, a in enumerate(order)})
    legend_order = [LABELS[a] for a in ["TBILL"] + ETFS]
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0)) if daily else _x()
    chart = (alt.Chart(long).mark_area(interpolate="step-after")
             .encode(x=x,
                     y=alt.Y("weight:Q", stack="zero", title=None,
                             axis=alt.Axis(format=".0%", tickCount=5), scale=alt.Scale(domain=[0, 1])),
                     color=alt.Color("asset:N", title=None,
                                     scale=alt.Scale(domain=legend_order,
                                                     range=[PALETTE[a] for a in ["TBILL"] + ETFS])),
                     order=alt.Order("order:Q"),
                     tooltip=["date:T", "asset:N", alt.Tooltip("weight:Q", format=".1%")]))
    return _style(chart, 240 if daily else 320)


def daily_z_chart(zd: pd.DataFrame, month_index: pd.PeriodIndex):
    long = zd.reset_index(names="date").melt("date", var_name="ETF", value_name="z")
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0))
    lines = alt.Chart(long).mark_line(strokeWidth=1.8).encode(
        x=x, y=alt.Y("z:Q", title="Z-score (if rebalanced that day)", scale=alt.Scale(zero=False)),
        color=alt.Color("ETF:N", scale=alt.Scale(domain=ETFS, range=[PALETTE[e] for e in ETFS])),
        tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), "ETF:N", alt.Tooltip("z:Q", format=".2f")])
    rebal = [m.to_timestamp(how="end").normalize() for m in month_index
             if m.to_timestamp(how="end") >= zd.index.min()]
    rules = alt.Chart(pd.DataFrame({"date": rebal})).mark_rule(strokeDash=[3, 3], color="#9AA3B5").encode(x="date:T")
    layers = [rules, lines]
    if zd.min().min() < Z_ESCAPE + 0.5:
        layers.insert(0, alt.Chart(pd.DataFrame({"y": [Z_ESCAPE]})).mark_rule(color="#b23a3a",
                                                                               strokeDash=[4, 4]).encode(y="y:Q"))
    return _style(alt.layer(*layers), 260)


def zscore_chart(z: pd.DataFrame):
    long = z.reset_index(names="date").melt("date", var_name="ETF", value_name="z")
    lo = min(-4.0, float(np.nanmin(z.values)) - 0.2)
    band = alt.Chart(pd.DataFrame({"lo": [lo], "hi": [Z_ESCAPE]})).mark_rect(
        color="#FBECEC").encode(y="lo:Q", y2="hi:Q")
    lines = alt.Chart(long).mark_line(strokeWidth=1.6).encode(
        x=_x(), y=alt.Y("z:Q", title="Z-score"),
        color=alt.Color("ETF:N", scale=alt.Scale(domain=ETFS, range=[PALETTE[e] for e in ETFS])),
        tooltip=["date:T", "ETF:N", alt.Tooltip("z:Q", format=".2f")])
    rule = alt.Chart(pd.DataFrame({"y": [Z_ESCAPE]})).mark_rule(strokeDash=[4, 4], color="#b23a3a").encode(y="y:Q")
    return _style(band + rule + lines, 280)


# ---------------- Table styling ----------------
def style_signal_table(df: pd.DataFrame):
    df = df.copy()
    dash = "\u2013"
    df["Carry (%p)"] = df["Carry (%p)"].map(lambda v: dash if pd.isna(v) else f"{v:.2f}")
    df["Z-score"] = df["Z-score"].map(lambda v: dash if pd.isna(v) else f"{v:+.2f}")
    df["Weight (%)"] = df["Weight (%)"].map(lambda v: f"{v:.1f}")
    df["Change (pp)"] = df["Change (pp)"].map(lambda v: f"{v:+.1f}")

    def status_color(v):
        return {"Held": "color: #22397a; font-weight: 600;", "Escaped (T-bill)": "color: #b23a3a; font-weight: 600;"
                }.get(v, "color: #6B7488;")
    return df.style.map(status_color, subset=["Status"])


def style_perf_table(df: pd.DataFrame):
    return (df.style.format("{:.2f}")
            .apply(lambda r: ["background-color: #EEF3FC; font-weight: 600;" if r.name == "Strategy" else ""
                              for _ in r], axis=1))


# ---------------- Layout ----------------
def render(show_header: bool = True):
    """show_header=False when the page already has its own title (standalone carry app)."""
    if show_header:
        st.header("Carry")
        st.caption(METHOD_SUMMARY)

    subtab1, subtab2 = st.tabs(["This month's signal", "Historical backtest"])

    # ---------- Live signal ----------
    with subtab1:
        with st.spinner("Fetching yields and computing this month's weights..."):
            try:
                ylds, ydaily, source = load_yields()
                oad = load_oad()
                sig = compute_signal(ylds, oad)
                table = build_live_table(sig, ylds)
                t = sig["W"].index[-1]
                w_now = sig["W"].iloc[-1]
            except Exception as e:
                st.error("The live signal couldn't be computed. Refresh the page in a few minutes.")
                st.exception(e)
                return

        st.subheader("This Month's Recommended Positioning")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Positioning for", (t + 1).strftime("%B %Y"))
        c2.metric("Signal as of (month-end)", t.strftime("%b %Y"))
        c3.metric("Invested in ETFs", f"{(1 - w_now['TBILL']):.0%}")
        c4.metric(f"Expected vol (target {TARGET_VOL:.0%})", f"{sig['vol'].iloc[-1]:.1%}")

        st.dataframe(style_signal_table(table), width="stretch", hide_index=True)
        oad_month = oad.dropna(how="all").index[-1]
        st.caption(f"Yields from {source}. Durations: Bloomberg index OAD as of {oad_month.strftime('%b %Y')}. "
                   f"Change = vs last month's weights.")

        st.subheader("Recent History")
        st.caption("Weights held in each of the last 6 months, most recent first. "
                   "Highlighted cells moved by 5 percentage points or more from the month before.")
        recent_display, recent_changed = build_recent_history(sig["W"], n_months=6)
        st.dataframe(highlight_changes(recent_display, recent_changed), width="stretch", hide_index=True)

        st.subheader("Daily Signal Tracker")
        st.caption("Rebalancing is monthly, but the signal can be followed every day. Each day is scored as if it "
                   "were a month-end: what the z-scores and weights would be if the portfolio were rebalanced "
                   "that day. Dashed vertical lines mark the monthly rebalance dates.")
        try:
            trk = daily_tracker(ylds, ydaily, oad)
            zd, wd = trk["z"], trk["w"]
            st.altair_chart(daily_z_chart(zd, ylds.index), width="stretch")

            last = wd.tail(10).iloc[::-1]
            tbl = pd.DataFrame({"Date": last.index.strftime("%Y-%m-%d")})
            for e in ETFS:
                tbl[f"z {e}"] = zd.loc[last.index, e].map("{:+.2f}".format).values
            for a in ETFS + ["TBILL"]:
                tbl[f"w {LABELS[a] if a != 'TBILL' else 'T-bill'}"] = (last[a] * 100).map("{:.1f}%".format).values
            st.dataframe(tbl, width="stretch", hide_index=True)
            st.caption("Last 10 business days. w = weight if rebalanced that day (not a trade instruction; "
                       f"trades happen at month-end). Yield data through {ydaily.index.max():%Y-%m-%d}.")
        except Exception as e:
            st.info(f"Daily tracker unavailable right now ({type(e).__name__}).")

        st.markdown("**Actual weights since the last rebalance**")
        try:
            reb = t.to_timestamp(how="end").normalize()
            px = fetch_etf_prices((reb - pd.Timedelta(days=7)).strftime("%Y-%m-%d"))
            px = px.loc[px.index >= px.index[px.index <= reb][-1]]
            rel = px / px.iloc[0]
            pos = rel.mul(w_now[ETFS].values, axis=1)
            pos["TBILL"] = w_now["TBILL"]                         # T-bill accrual ignored intra-month
            drift = pos.div(pos.sum(axis=1), axis=0)
            drift.index = pd.DatetimeIndex(drift.index)
            st.altair_chart(weights_chart(drift, daily=True), width="stretch")
            st.caption(f"Month's target weights drifting with daily ETF prices since the {reb:%d %b %Y} rebalance "
                       "(Yahoo Finance, dividend-adjusted).")
        except Exception:
            st.caption("Daily ETF prices are unavailable right now, so intra-month drift can't be shown.")

        st.subheader("Carry Z-score, Last 10 Years")
        z_recent = sig["z"].loc[t - 119:t]
        z_recent.index = _ts(z_recent.index)
        st.altair_chart(zscore_chart(z_recent), width="stretch")
        st.caption(f"Red zone = escape level (z ≤ {Z_ESCAPE:.1f}): an ETF in this zone is dropped to T-bills. "
                   "Higher z = larger weight.")

    # ---------- Historical backtest ----------
    with subtab2:
        try:
            bt = load_backtest()
        except FileNotFoundError:
            st.info("Backtest results are missing. Run step 7 of the notebook and commit carry/data/.")
            return

        cmp = bt.dropna(subset=["ret_bbg"])
        rets = pd.DataFrame({"Strategy": cmp["ret_strategy"], "Equal weight (25% each)": cmp["ret_ew"],
                             "Bloomberg US Treasury Index": cmp["ret_bbg"]})
        rf = cmp["ret_tbill"]

        st.subheader("Backtest Period")
        c1, c2, c3 = st.columns(3)
        c1.metric("Start", rets.index[0].strftime("%b %Y"))
        c2.metric("End", rets.index[-1].strftime("%b %Y"))
        c3.metric("Length", f"{len(rets)} months ({len(rets) / 12:.1f} yrs)")
        st.caption(
            f"Starts in the first month all 4 ETFs have returns (IEI launched Jan 2007) and ends with the last month "
            f"of Bloomberg index data, so all three series cover the same period. Weights, duration and turnover run "
            f"to {bt.index[-1].strftime('%b %Y')}. Settings: {WINDOW}-month z-score, {SCHEME} weighting, "
            f"{TARGET_VOL:.0%} vol target, escape at z ≤ {Z_ESCAPE:.1f}, {COV_WINDOW}-month covariance. "
            "Monthly rebalancing, ETF total returns (NAV + distributions), no transaction costs."
        )

        try:
            dbt = load_daily_backtest()
        except FileNotFoundError:
            dbt = None
        freq = st.radio("Chart frequency", ["Daily", "Monthly"], horizontal=True,
                        index=0 if dbt is not None else 1, disabled=dbt is None,
                        help="Rebalancing is monthly in both views. Daily view shows how the portfolio "
                             "moves between rebalances (Bloomberg index is monthly only).")
        daily_view = freq == "Daily" and dbt is not None

        st.subheader("Growth of $1")
        wealth = (1 + rets).cumprod()
        wealth.index = _ts(wealth.index)
        if daily_view:
            wd_ = dbt[["v_strategy", "v_ew"]].rename(columns={"v_strategy": "Strategy",
                                                              "v_ew": "Equal weight (25% each)"})
            st.altair_chart(line_chart(wd_, None, fmt="$.2f", zero=False,
                                       colors={k: LINE_COLORS[k] for k in wd_.columns}), width="stretch")
            st.caption("Daily view compares the strategy with equal weight; the Bloomberg index is only "
                       "available monthly (switch to Monthly to include it).")
        else:
            st.altair_chart(line_chart(wealth, None, fmt="$.2f", zero=False), width="stretch")

        st.subheader("Performance Summary")
        st.dataframe(style_perf_table(pd.DataFrame({c: perf_stats(rets[c], rf) for c in rets}).T), width="stretch")
        rel = pd.DataFrame({f"vs {b}": relative_stats(rets["Strategy"], rets[b])
                            for b in ["Equal weight (25% each)", "Bloomberg US Treasury Index"]}).T
        st.dataframe(rel.style.format("{:.2f}"), width="stretch")
        st.caption("Sharpe and Sortino use the 3-month T-bill as the risk-free rate.")

        st.subheader("Drawdown")
        base = wd_ if daily_view else wealth
        dd = base / base.cummax() - 1
        st.altair_chart(line_chart(dd, None, fmt=".0%", height=240,
                                   colors={k: LINE_COLORS[k] for k in dd.columns}), width="stretch")
        if daily_view:
            st.caption(f"Daily max drawdown: Strategy {dd['Strategy'].min():.1%}, equal weight "
                       f"{dd['Equal weight (25% each)'].min():.1%} (deeper than month-end figures in the table).")

        st.subheader("Realised Volatility (rolling 12 months)")
        rv = rets.rolling(12).std() * np.sqrt(12)
        rv.index = _ts(rv.index)
        st.altair_chart(line_chart(rv.dropna(), None, fmt=".0%", height=240), width="stretch")
        st.caption(f"Target = {TARGET_VOL:.0%}. With risk weighting the ETF basket is often below the target even "
                   "when fully invested, so realised volatility can sit under 5%.")

        st.subheader("Asset Weights")
        if daily_view:
            w = dbt[[f"w_{a}" for a in ETFS + ["TBILL"]]].rename(columns=lambda c: c[2:])
            st.altair_chart(weights_chart(w), width="stretch")
            st.caption("Daily weights: reset to target at each month-end rebalance, then drift with prices.")
        else:
            w = bt[[f"w_{a}" for a in ETFS + ["TBILL"]]].rename(columns=lambda c: c[2:])
            w.index = _ts(w.index)
            st.altair_chart(weights_chart(w), width="stretch")

        st.subheader("Portfolio Duration")
        d = bt[["dur_strategy", "dur_ew", "dur_bbg"]].rename(columns={
            "dur_strategy": "Strategy", "dur_ew": "Equal weight (25% each)", "dur_bbg": "Bloomberg US Treasury Index"})
        d.index = _ts(d.index)
        st.altair_chart(line_chart(d, "Years (OAD)", fmt=".0f", height=280, interpolate="step-after"), width="stretch")
        st.caption("Weighted Bloomberg index option-adjusted duration of each held ETF; T-bill = 0.25 years.")

        st.subheader("Turnover and Escape Events")
        wts = bt[[f"w_{e}" for e in ETFS]]
        wts.columns = ETFS
        held = (wts > 1e-6).astype(int)
        chg = held.diff().fillna(0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Annual turnover (one-way)", f"{bt['turnover'].mean() * 12:.0%}")
        c2.metric("Median monthly turnover", f"{bt['turnover'].median():.1%}")
        c3.metric("Months 100% in T-bills", f"{(bt['w_TBILL'] > 0.999).mean():.1%}")
        tbl = pd.DataFrame({"Avg weight (%)": wts.mean() * 100, "Max weight (%)": wts.max() * 100,
                            "% months held": held.mean() * 100, "Escape exits": (chg == -1).sum(),
                            "Re-entries": (chg == 1).sum()})
        st.dataframe(tbl.style.format({"Avg weight (%)": "{:.1f}", "Max weight (%)": "{:.1f}",
                                       "% months held": "{:.1f}"}), width="stretch")
