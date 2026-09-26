"""
Carry tab -- carry timing on US Treasury ETFs (long-only, T-bill fallback)
---------------------------------------------------------------------------
Signal
    carry_i  = matched Treasury yield (DGS2/5/10/20) - 3M T-bill (DGS3MO), month-end
    z_i      = (carry_i - rolling mean) / rolling std over the last WINDOW months
    Long if z > ENTER_Z, sell to T-bill if z < EXIT_Z, otherwise hold (hysteresis)
    Held ETFs equal-weighted; none held -> 100% 3M T-bill
Timing
    Signal at month-end t -> positioning held over month t+1

Live Signal sub-tab : yields fetched from FRED on page load (cached 6h);
                      falls back to data/yields_monthly.csv if FRED is unreachable.
Historical sub-tab  : reads data/carry_backtest.csv (exported by step 7 of the notebook).
"""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Settings ----------------
DATA_DIR = Path(__file__).resolve().parent / "data"      # works regardless of working directory
ETFS = ["SHY", "IEI", "IEF", "TLT"]
YIELD_MAP = {"SHY": "DGS2", "IEI": "DGS5", "IEF": "DGS10", "TLT": "DGS20"}
TBILL = "DGS3MO"
WINDOW = 72            # z-score look-back (months)
ENTER_Z = 1.0          # long above
EXIT_Z = -1.0          # sell below
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

PALETTE = {"TBILL": "#b5b2ab", "SHY": "#a4c0ec", "IEI": "#6b95dc", "IEF": "#3f6bc4", "TLT": "#22397a"}
LABELS = {"TBILL": "T-bill (cash)", "SHY": "SHY", "IEI": "IEI", "IEF": "IEF", "TLT": "TLT"}
LINE_COLORS = {"Strategy": "#3f6bc4", "Equal weight (25% each)": "#222222",
               "Bloomberg US Treasury Index": "#e8843a"}


# ---------------- Data ----------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_fred_yields() -> pd.DataFrame:
    """Month-end yields from FRED (no API key needed)."""
    out = {}
    for sid in list(YIELD_MAP.values()) + [TBILL]:
        s = pd.read_csv(FRED_CSV.format(sid), index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
        out[sid] = s.astype(float).dropna().resample("ME").last()
    df = pd.DataFrame(out)
    df.index = df.index.to_period("M")
    return df


def load_yields() -> tuple[pd.DataFrame, str]:
    try:
        df, source = fetch_fred_yields(), "FRED (live)"
    except Exception:
        df = pd.read_csv(DATA_DIR / "yields_monthly.csv", index_col="ym")
        df.index = pd.PeriodIndex(df.index, freq="M")
        source = "stored file (FRED unreachable)"
    current = pd.Timestamp.today().to_period("M")
    return df.loc[df.index < current], source          # use completed months only


@st.cache_data(show_spinner=False)
def load_backtest() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "carry_backtest.csv", index_col="ym")
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df


# ---------------- Signal ----------------
def compute_signal(ylds: pd.DataFrame) -> dict:
    carry = pd.DataFrame({etf: ylds[y] - ylds[TBILL] for etf, y in YIELD_MAP.items()})
    mu = carry.rolling(WINDOW, min_periods=WINDOW).mean()
    sd = carry.rolling(WINDOW, min_periods=WINDOW).std()
    z = (carry - mu) / sd

    pos = pd.DataFrame(0, index=z.index, columns=ETFS)
    for etf in ETFS:
        state = 0
        for t, v in z[etf].items():
            if np.isnan(v):
                state = 0
            elif v > ENTER_Z:
                state = 1
            elif v < EXIT_Z:
                state = 0
            pos.at[t, etf] = state
    return {"carry": carry, "z": z, "pos": pos}


def build_live_table(sig: dict, ylds: pd.DataFrame) -> pd.DataFrame:
    pos, z, carry = sig["pos"], sig["z"], sig["carry"]
    t = pos.index[-1]
    now, prev = pos.loc[t], pos.iloc[-2]
    n_long = int(now.sum())
    rows = []
    for etf in ETFS:
        if now[etf] and not prev[etf]:
            action = "Buy (Long)"
        elif now[etf]:
            action = "Hold (Long)"
        elif prev[etf]:
            action = "Sell (to Cash)"
        else:
            action = "Stay out (Cash)"
        rows.append({
            "Instrument": etf,
            "Yield used": YIELD_MAP[etf],
            "Carry (%p)": round(carry.at[t, etf], 3),
            "Z-score": round(z.at[t, etf], 2),
            "Action": action,
            "Weight (%)": round(100 / n_long, 1) if (now[etf] and n_long) else 0.0,
        })
    rows.append({
        "Instrument": "3-Month T-Bill",
        "Yield used": f"{TBILL} = {ylds.at[t, TBILL]:.2f}%",
        "Carry (%p)": np.nan,
        "Z-score": np.nan,
        "Action": "Fallback (cash)" if n_long == 0 else "Not used",
        "Weight (%)": 100.0 if n_long == 0 else 0.0,
    })
    return pd.DataFrame(rows)


def build_recent_history(pos: pd.DataFrame, n_months: int = 6):
    """Last n months of positioning (most recent first) + a mask of cells that changed vs prior month.
    Row label = month the position is HELD (signal month-end + 1)."""
    recent = pos.iloc[-(n_months + 1):]
    rows, changed = [], []
    for i in range(1, len(recent)):
        now, prev = recent.iloc[i], recent.iloc[i - 1]
        n_now, n_prev = int(now.sum()), int(prev.sum())
        row = {"Held in": str(recent.index[i] + 1)}
        chg = {"Held in": False}
        for etf in ETFS:
            row[etf] = f"Long ({100 / n_now:.1f}%)" if now[etf] else "Cash (0%)"
            chg[etf] = bool(now[etf] != prev[etf])
        row["3-Month T-Bill"] = "100.0%" if n_now == 0 else "0.0%"
        chg["3-Month T-Bill"] = (n_now == 0) != (n_prev == 0)
        rows.append(row)
        changed.append(chg)
    return pd.DataFrame(rows[::-1]), pd.DataFrame(changed[::-1])


def highlight_changes(display_df: pd.DataFrame, changed_df: pd.DataFrame):
    def style_func(_):
        styles = pd.DataFrame("", index=display_df.index, columns=display_df.columns)
        for col in changed_df.columns:
            styles[col] = changed_df[col].map(
                lambda c: "background-color: #FCEFC7; font-weight: 600;" if c else "")
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
    return {
        "CAGR (%)": round(cagr * 100, 2),
        "Vol (%)": round(r.std() * np.sqrt(12) * 100, 2),
        "Sharpe": round(ex.mean() / ex.std() * np.sqrt(12), 2),
        "Sortino": round(ex.mean() / ex[ex < 0].std() * np.sqrt(12), 2),
        "Max Drawdown (%)": round(dd.min() * 100, 2),
        "Calmar": round(cagr / abs(dd.min()), 2),
    }


def relative_stats(r: pd.Series, b: pd.Series) -> dict:
    act = r - b
    te = act.std() * np.sqrt(12)
    up, dn = b > 0, b < 0
    return {
        "Excess return (%/yr)": round(act.mean() * 1200, 2),
        "Tracking error (%)": round(te * 100, 2),
        "Information ratio": round(act.mean() * 12 / te, 2),
        "t-stat": round(act.mean() / act.std() * np.sqrt(len(act)), 2),
        "Beta": round(np.cov(r, b)[0, 1] / b.var(), 2),
        "Up capture (%)": round(r[up].mean() / b[up].mean() * 100, 1),
        "Down capture (%)": round(r[dn].mean() / b[dn].mean() * 100, 1),
    }


# ---------------- Charts ----------------
FONT = "IBM Plex Sans"


def _x():
    return alt.X("date:T", title=None,
                 axis=alt.Axis(format="%Y", tickCount={"interval": "year", "step": 2}, labelAngle=0))


def _ts(idx: pd.PeriodIndex) -> pd.DatetimeIndex:
    return idx.to_timestamp()


def _style(chart, height: int = 320):
    """Shared chart look: no frame, light grid, site fonts."""
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


def area_chart(s: pd.Series, y_title: str, fmt: str, color: str, height=220):
    df = s.rename("value").reset_index(names="date")
    chart = (alt.Chart(df).mark_area(interpolate="step-after", color=color, opacity=0.25,
                                     line={"color": color, "strokeWidth": 1.5})
             .encode(x=_x(),
                     y=alt.Y("value:Q", title=y_title, axis=alt.Axis(format=fmt)),
                     tooltip=["date:T", alt.Tooltip("value:Q", format=fmt)]))
    return _style(chart, height)


def weights_chart(w: pd.DataFrame):
    order = ETFS + ["TBILL"]                                  # SHY bottom -> TLT -> T-bill on top
    long = (w[order].rename(columns=LABELS).reset_index(names="date")
            .melt("date", var_name="asset", value_name="weight"))
    long["order"] = long["asset"].map({LABELS[a]: i for i, a in enumerate(order)})
    legend_order = [LABELS[a] for a in ["TBILL"] + ETFS]
    chart = (alt.Chart(long).mark_area(interpolate="step-after")
             .encode(x=_x(),
                     y=alt.Y("weight:Q", stack="zero", title=None,
                             axis=alt.Axis(format=".0%", tickCount=5), scale=alt.Scale(domain=[0, 1])),
                     color=alt.Color("asset:N", title=None,
                                     scale=alt.Scale(domain=legend_order,
                                                     range=[PALETTE[a] for a in ["TBILL"] + ETFS])),
                     order=alt.Order("order:Q"),
                     tooltip=["date:T", "asset:N", alt.Tooltip("weight:Q", format=".1%")]))
    return _style(chart, 320)


def zscore_chart(z: pd.DataFrame):
    long = z.reset_index(names="date").melt("date", var_name="ETF", value_name="z")
    band = alt.Chart(pd.DataFrame({"lo": [EXIT_Z], "hi": [ENTER_Z]})).mark_rect(
        color="#F4F6FA").encode(y="lo:Q", y2="hi:Q")
    lines = alt.Chart(long).mark_line(strokeWidth=1.6).encode(
        x=_x(),
        y=alt.Y("z:Q", title="Z-score"),
        color=alt.Color("ETF:N", scale=alt.Scale(domain=ETFS, range=[PALETTE[e] for e in ETFS])),
        tooltip=["date:T", "ETF:N", alt.Tooltip("z:Q", format=".2f")])
    rules = alt.Chart(pd.DataFrame({"y": [ENTER_Z, EXIT_Z]})).mark_rule(
        strokeDash=[4, 4], color="#9AA3B5").encode(y="y:Q")
    return _style(band + rules + lines, 280)


# ---------------- Table styling ----------------
def style_signal_table(df: pd.DataFrame):
    def action_color(v):
        if isinstance(v, str) and v.startswith(("Buy", "Hold")):
            return "color: #22397a; font-weight: 600;"
        if isinstance(v, str) and v.startswith("Sell"):
            return "color: #b23a3a; font-weight: 600;"
        if v == "Fallback (cash)":
            return "color: #1B2540; font-weight: 600;"
        return "color: #6B7488;"
    df = df.copy()
    df["Carry (%p)"] = df["Carry (%p)"].map(lambda v: "\u2013" if pd.isna(v) else f"{v:.2f}")
    df["Z-score"] = df["Z-score"].map(lambda v: "\u2013" if pd.isna(v) else f"{v:+.2f}")
    df["Weight (%)"] = df["Weight (%)"].map(lambda v: f"{v:.1f}")
    return df.style.map(action_color, subset=["Action"])


def style_perf_table(df: pd.DataFrame, pct_cols: list, dec_cols: list):
    fmt = {c: "{:.2f}" for c in pct_cols + dec_cols}
    return (df.style.format(fmt)
            .apply(lambda r: ["background-color: #EEF3FC; font-weight: 600;" if r.name == "Strategy" else ""
                              for _ in r], axis=1))


# ---------------- Layout ----------------
METHOD_SUMMARY = (
    f"Each month, compare every maturity's carry (its Treasury yield minus the 3-month T-bill) "
    f"with its own last {WINDOW} months. Buy when the z-score rises above +{ENTER_Z}, sell when it "
    f"falls below {EXIT_Z}, hold otherwise. Held ETFs are equal-weighted; with none held, the "
    "portfolio sits 100% in 3-month T-bills."
)


def render(show_header: bool = True):
    """show_header=False when the page already has its own title (standalone carry app)."""
    if show_header:
        st.header("Carry")
        st.caption(METHOD_SUMMARY)

    subtab1, subtab2 = st.tabs(["This month's signal", "Historical backtest"])

    # ---------- Live signal ----------
    with subtab1:
        with st.spinner("Fetching yields and computing this month's signal..."):
            try:
                ylds, source = load_yields()
                sig = compute_signal(ylds)
                table = build_live_table(sig, ylds)
                t = sig["pos"].index[-1]
                n_long = int(sig["pos"].loc[t].sum())
            except Exception as e:
                st.error("The live signal couldn't be computed. Refresh the page in a few minutes.")
                st.exception(e)
                return

        st.subheader("This Month's Recommended Positioning")
        c1, c2, c3 = st.columns(3)
        c1.metric("Positioning for", (t + 1).strftime("%B %Y"))
        c2.metric("Signal as of", f"{t.strftime('%b %Y')} month-end")
        c3.metric("ETFs held", f"{n_long} of 4" if n_long else "None \u2014 100% T-bill")

        st.dataframe(style_signal_table(table), width="stretch", hide_index=True)
        st.caption(f"Yields from {source}. Carry and z-score use the last completed month-end.")

        st.subheader("Recent History")
        st.caption("What the strategy held over the last 6 months, most recent first. "
                   "Highlighted cells mark a flip from the month before.")
        recent_display, recent_changed = build_recent_history(sig["pos"], n_months=6)
        st.dataframe(highlight_changes(recent_display, recent_changed), width="stretch", hide_index=True)

        st.subheader(f"Carry Z-score, Last 10 Years")
        z_recent = sig["z"].loc[t - 119:t]
        z_recent.index = _ts(z_recent.index)
        st.altair_chart(zscore_chart(z_recent), width="stretch")
        st.caption(f"Shaded band = hold zone between {EXIT_Z} and +{ENTER_Z}. "
                   "Above it the ETF is bought, below it the ETF is sold.")

    # ---------- Historical backtest ----------
    with subtab2:
        try:
            bt = load_backtest()
        except FileNotFoundError:
            st.info("Backtest results are missing. Run step 7 of the notebook and commit "
                    "carry/data/carry_backtest.csv.")
            return

        cmp = bt.dropna(subset=["ret_bbg"])                   # common period with the Bloomberg index
        rets = pd.DataFrame({
            "Strategy": cmp["ret_strategy"],
            "Equal weight (25% each)": cmp["ret_ew"],
            "Bloomberg US Treasury Index": cmp["ret_bbg"],
        })
        rf = cmp["ret_tbill"]

        st.subheader("Backtest Period")
        c1, c2, c3 = st.columns(3)
        c1.metric("Start", rets.index[0].strftime("%b %Y"))
        c2.metric("End", rets.index[-1].strftime("%b %Y"))
        c3.metric("Length", f"{len(rets)} months ({len(rets) / 12:.1f} yrs)")
        st.caption(
            f"Starts in the first month all 4 ETFs have returns (IEI launched Jan 2007) and ends with the "
            f"last month of Bloomberg index data, so all three series cover the same period. Weights, "
            f"duration and turnover run to {bt.index[-1].strftime('%b %Y')}. Monthly rebalancing, "
            "ETF total returns (NAV + distributions), no transaction costs."
        )

        st.subheader("Growth of $1")
        wealth = (1 + rets).cumprod()
        wealth.index = _ts(wealth.index)
        st.altair_chart(line_chart(wealth, None, fmt="$.2f", zero=False), width="stretch")

        st.subheader("Performance Summary")
        perf = pd.DataFrame({c: perf_stats(rets[c], rf) for c in rets}).T
        st.dataframe(style_perf_table(perf, ["CAGR (%)", "Vol (%)", "Max Drawdown (%)"],
                                      ["Sharpe", "Sortino", "Calmar"]), width="stretch")
        rel = pd.DataFrame({
            f"vs {b}": relative_stats(rets["Strategy"], rets[b])
            for b in ["Equal weight (25% each)", "Bloomberg US Treasury Index"]
        }).T
        st.dataframe(rel.style.format("{:.2f}"), width="stretch")
        st.caption("Sharpe and Sortino use the 3-month T-bill as the risk-free rate. "
                   "Excess return, tracking error and capture ratios compare the strategy with each benchmark.")

        st.subheader("Drawdown")
        dd = wealth / wealth.cummax() - 1
        st.altair_chart(line_chart(dd, None, fmt=".0%", height=240), width="stretch")

        st.subheader("Asset Weights")
        w = bt[[f"w_{a}" for a in ETFS + ["TBILL"]]].rename(columns=lambda c: c[2:])
        w.index = _ts(w.index)
        st.altair_chart(weights_chart(w), width="stretch")

        st.subheader("Portfolio Duration")
        d = bt[["dur_strategy", "dur_ew", "dur_bbg"]].rename(columns={
            "dur_strategy": "Strategy", "dur_ew": "Equal weight (25% each)",
            "dur_bbg": "Bloomberg US Treasury Index"})
        d.index = _ts(d.index)
        st.altair_chart(line_chart(d, "Years (OAD)", fmt=".0f", height=280, interpolate="step-after"), width="stretch")
        st.caption("Weighted Bloomberg index option-adjusted duration of each held ETF; T-bill = 0.25 years.")

        st.subheader("Turnover and Position Flips")
        pos = (bt[[f"w_{e}" for e in ETFS]] > 0).astype(int)
        pos.columns = ETFS
        chg = pos.diff().fillna(0)
        n_years = len(bt) / 12
        c1, c2, c3 = st.columns(3)
        c1.metric("Annual turnover (one-way)", f"{bt['turnover'].mean() * 12:.0%}")
        c2.metric("Months with a trade", f"{(bt['turnover'] > 0.01).mean():.0%}")
        c3.metric("Total flips", f"{int((chg != 0).sum().sum())}")
        flips = pd.DataFrame({
            "Buys": (chg == 1).sum(),
            "Sells": (chg == -1).sum(),
            "Flips / yr": (chg != 0).sum() / n_years,
            "% months held": pos.mean() * 100,
        })
        st.dataframe(flips.style.format({"Flips / yr": "{:.2f}", "% months held": "{:.1f}"}), width="stretch")
