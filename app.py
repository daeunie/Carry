"""
Carry Timing -- standalone app
--------------------------------
A carry-only version of the Style Investing site. Point a second Streamlit
Cloud app's "Main file path" at this file (same repo). It reuses
carry/carry_tab.py, so both sites always show the same signal and backtest.
"""

import streamlit as st

st.set_page_config(page_title="Carry Timing \u2014 US Treasury ETFs", page_icon="\U0001F4C8", layout="wide")

from site_style import apply_style, hero  # noqa: E402
from carry.carry_tab import METHOD_SUMMARY, render  # noqa: E402

apply_style()
hero("Carry Timing", "Long-only carry timing on US Treasury ETFs (SHY, IEI, IEF, TLT). " + METHOD_SUMMARY)
render(show_header=False)
st.caption("KSIF Strategic Asset Allocation Team. Adapted from Brooks, Palhares & Richardson (2018), "
           "Style Investing in Fixed Income.")
