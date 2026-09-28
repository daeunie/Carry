"""
Shared look for the whole Style Investing site.
Call apply_style() once at the top of app.py, then hero() for the page header.
Colours match the chart palette in the factor tabs (SHY -> TLT = light -> dark blue).
"""

import streamlit as st

INK = "#1B2540"        # body text / headings
NAVY = "#22397a"       # TLT
BLUE = "#3f6bc4"       # IEF, primary
SKY = "#a4c0ec"        # SHY
PANEL = "#F4F6FA"      # section background
LINE = "#E3E8F0"       # borders / grid
MUTED = "#6B7488"      # captions


_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');

html, body, [class*="st-"], .stMarkdown, .stCaption, .stDataFrame, button, input {{
    font-family: 'IBM Plex Sans', Arial, sans-serif;
}}
h1, h2, h3, .hero-title {{
    font-family: 'Source Serif 4', Georgia, serif !important;
    color: {INK};
    letter-spacing: -0.01em;
}}
h2 {{ font-weight: 600 !important; }}
h3 {{ font-weight: 500 !important; font-size: 1.3rem !important; }}

.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: {MUTED}; line-height: 1.55; }}

/* hero */
.hero {{ display: flex; align-items: flex-end; justify-content: space-between;
         gap: 2rem; padding: 0.4rem 0 1.4rem 0; border-bottom: 1px solid {LINE};
         margin-bottom: 0.6rem; }}
.hero .hero-title {{ font-size: 2.9rem !important; font-weight: 600; line-height: 1.05 !important; margin: 0 !important; }}
.hero .hero-sub {{ color: {MUTED}; font-size: 1rem !important; margin: 0.55rem 0 0 0 !important; max-width: 62ch; }}
.hero svg {{ flex-shrink: 0; }}
@media (max-width: 760px) {{ .hero svg {{ display: none; }} .hero-title {{ font-size: 2.2rem; }} }}

/* top-level tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ padding: 0.6rem 0.1rem; font-weight: 500; color: {MUTED}; }}
.stTabs [aria-selected="true"] {{ color: {INK} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {BLUE}; height: 2px; }}

/* metrics */
[data-testid="stMetric"] {{ background: {PANEL}; border-radius: 10px; padding: 0.9rem 1.1rem; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.85rem; }}
[data-testid="stMetricValue"] {{ font-family: 'Source Serif 4', Georgia, serif; color: {INK};
                                 font-size: 1.75rem; }}

/* bordered containers */
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: {LINE} !important; border-radius: 12px; }}

/* tables */
[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 10px; }}

a {{ color: {BLUE}; }}
</style>
"""

# small yield-curve mark: four points on the curve, coloured like the ETFs they stand for
_CURVE_SVG = f"""
<svg width="190" height="70" viewBox="0 0 190 70" role="img" aria-label="Treasury yield curve">
  <path d="M8 58 C 45 30, 90 20, 182 14" fill="none" stroke="{LINE}" stroke-width="2.5"/>
  <circle cx="20" cy="50" r="6" fill="{SKY}"/>
  <circle cx="60" cy="31" r="6" fill="#6b95dc"/>
  <circle cx="104" cy="22" r="6" fill="{BLUE}"/>
  <circle cx="170" cy="15" r="6" fill="{NAVY}"/>
  <text x="20" y="68" font-size="9" fill="{MUTED}" text-anchor="middle" font-family="IBM Plex Sans, Arial">SHY</text>
  <text x="60" y="49" font-size="9" fill="{MUTED}" text-anchor="middle" font-family="IBM Plex Sans, Arial">IEI</text>
  <text x="104" y="40" font-size="9" fill="{MUTED}" text-anchor="middle" font-family="IBM Plex Sans, Arial">IEF</text>
  <text x="170" y="33" font-size="9" fill="{MUTED}" text-anchor="middle" font-family="IBM Plex Sans, Arial">TLT</text>
</svg>
"""


def apply_style():
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str):
    st.markdown(
        f'<div class="hero"><div><div class="hero-title">{title}</div>'
        f'<p class="hero-sub">{subtitle}</p></div>{_CURVE_SVG}</div>',
        unsafe_allow_html=True,
    )
