"""
app.py — Smart BI Assistant
============================
Handles ONLY UI rendering and user interaction.
All backend calls are delegated to api_client.py.

Flow:
  1. User uploads CSV             → upload_file()
  2. Columns fetched dynamically  → get_columns()
  3. User picks target column     → st.selectbox
  4. Visualizations fetched       → get_visualizations()
  5. Charts rendered on right     → render_chart()
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Import API layer (no HTTP logic lives in this file) ─────────────────────
from api_client import upload_file, get_columns, get_visualizations


# ─────────────────────────────────────────
# ⚙️ Page configuration (must be first call)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Smart BI Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────
# 🎨 Custom CSS — clean, minimal dark theme
# ─────────────────────────────────────────
st.markdown("""
<style>
    body, .stApp {
        background-color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }

    .block-container {
        padding: 2rem 2.5rem;
        max-width: 1400px;
    }

    /* Header */
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    .hero-subtitle {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }

    /* Card */
    .panel-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }

    /* Section */
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* Button */
    .stButton > button {
        background: #3b82f6;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        width: 100%;
        padding: 0.6rem;
    }

    .stButton > button:hover {
        background: #2563eb;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #f1f5f9;
        border-radius: 10px;
        border: 1px dashed #cbd5f5;
    }
            
     /* Selectbox */
    [data-testid="stSelectbox"] {
        background: white !important;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 🔁 Session state initialisation
# ─────────────────────────────────────────
def init_session_state():
    """Initialise all session state keys on first load."""
    defaults = {
        "file_uploaded": False,   # True after successful upload
        "file_id":       None,    # ID returned by backend
        "columns":       [],      # Column list from backend
        "charts":        [],      # Chart data from backend
        "target_column": None,    # User-selected target column
        "upload_message": None,   # Latest status message from upload
        "last_error":    None,    # Last error string (any step)
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ─────────────────────────────────────────
# 📊 Chart rendering helpers
# ─────────────────────────────────────────

# Colour palette consistent with the dark theme
CHART_COLORS = [
    "#4f8ef7", "#38d9a9", "#9b72f7", "#f7a94f",
    "#f75090", "#72c7f7", "#f7f072", "#72f7b8",
]

def _apply_dark_style(ax, fig):
    """Apply unified dark-theme styling to a Matplotlib axis."""
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="#8890a4", labelsize=9)
    ax.xaxis.label.set_color("#8890a4")
    ax.yaxis.label.set_color("#8890a4")
    ax.title.set_color("#e8eaf0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#242738")
    ax.grid(axis="y", color="#242738", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)


def render_bar_chart(chart: dict):
    """Render a horizontal or vertical bar chart from chart data dict."""
    labels = chart.get("labels", [])
    values = chart.get("values", [])
    title  = chart.get("title", "Bar Chart")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(
        labels, values,
        color=CHART_COLORS[:len(labels)],
        width=0.55,
        edgecolor="none",
        linewidth=0,
    )
    # Value annotations on top of each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.012,
            f"{val:,.0f}",
            ha="center", va="bottom",
            fontsize=8.5, color="#8890a4",
            fontfamily="Space Grotesk",
        )
    ax.set_title(title, fontsize=11, fontweight="600", pad=12)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    _apply_dark_style(ax, fig)
    plt.tight_layout()
    return fig


def render_pie_chart(chart: dict):
    """Render a donut-style pie chart from chart data dict."""
    labels = chart.get("labels", [])
    values = chart.get("values", [])
    title  = chart.get("title", "Pie Chart")

    fig, ax = plt.subplots(figsize=(6, 4))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        autopct="%1.1f%%",
        pctdistance=0.82,
        colors=CHART_COLORS[:len(labels)],
        startangle=140,
        wedgeprops={"linewidth": 2, "edgecolor": "#1a1d27", "width": 0.55},  # donut
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("#e8eaf0")

    # Legend
    legend_patches = [
        mpatches.Patch(color=CHART_COLORS[i], label=labels[i])
        for i in range(len(labels))
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(len(labels), 4),
        fontsize=8,
        frameon=False,
        labelcolor="#8890a4",
    )
    ax.set_title(title, fontsize=11, fontweight="600", color="#e8eaf0", pad=14)
    fig.patch.set_facecolor("#1a1d27")
    plt.tight_layout()
    return fig


def render_line_chart(chart: dict):
    """Render a smooth line chart from chart data dict."""
    labels = chart.get("labels", [])
    values = chart.get("values", [])
    title  = chart.get("title", "Line Chart")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(labels))

    ax.plot(
        x, values,
        color="#4f8ef7", linewidth=2.5,
        marker="o", markersize=5,
        markerfacecolor="#1a1d27", markeredgecolor="#4f8ef7", markeredgewidth=2,
        solid_capstyle="round",
    )
    # Gradient fill under line
    ax.fill_between(x, values, alpha=0.12, color="#4f8ef7")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    ax.set_title(title, fontsize=11, fontweight="600", pad=12)
    _apply_dark_style(ax, fig)
    plt.tight_layout()
    return fig


# Chart-type dispatcher
CHART_RENDERERS = {
    "bar":  render_bar_chart,
    "pie":  render_pie_chart,
    "line": render_line_chart,
}

def render_chart(chart: dict):
    """
    Dispatch chart rendering based on the 'type' key in chart dict.
    Renders inside a styled card wrapper.
    """
    chart_type = chart.get("type", "bar").lower()
    renderer   = CHART_RENDERERS.get(chart_type, render_bar_chart)

    st.markdown(f'<div class="chart-card"><div class="chart-title">● {chart.get("title", chart_type.title())}</div>', unsafe_allow_html=True)
    try:
        fig = renderer(chart)
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.error(f"Could not render chart: {e}")
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# 🖥️  UI — Left Panel
# ─────────────────────────────────────────

def render_left_panel():
    """
    Left column UI:
      1. File upload widget
      2. Upload button → calls API
      3. Column dropdown (shown after upload)
      4. Visualize button → calls API
    """

    # ── Header ──────────────────────────────────
    st.markdown('<div class="section-label">Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Dataset</div>', unsafe_allow_html=True)

    # ── 1. File uploader ────────────────────────
    uploaded_file = st.file_uploader(
        label="",
        type=["csv"],
        help="Upload a CSV file to begin analysis",
        label_visibility="collapsed",
    )

    # ── 2. Upload button ────────────────────────
    if st.button("⬆ Upload & Analyse", disabled=(uploaded_file is None)):
        if uploaded_file is None:
            st.warning("Please select a CSV file first.")
        else:
            with st.spinner("Uploading file…"):
                # ── API call: upload_file ────────
                result = upload_file(uploaded_file)

            if result["success"]:
                st.session_state.file_uploaded = True
                st.session_state.file_id       = result.get("file_id")
                st.session_state.charts        = []   # clear old charts
                st.session_state.target_column = None

                # ── API call: get_columns ────────
                with st.spinner("Fetching columns…"):
                    col_result = get_columns(st.session_state.file_id)

                if col_result["success"]:
                    st.session_state.columns = col_result["columns"]
                    st.markdown(
                        f'<span class="badge badge-success">✓ {result["message"]}</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.session_state.columns = []
                    st.error(col_result["message"])
            else:
                st.session_state.file_uploaded = False
                st.error(result["message"])

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── 3. Column selector (visible only after upload) ──
if st.session_state.file_uploaded and st.session_state.columns:

    options = ["Select target column"] + st.session_state.columns

    selected = st.selectbox(
        "Select Target Column",
        options=options,
        index=0
    )

    if selected == "Select target column":
        st.session_state.target_column = None
    else:
        st.session_state.target_column = selected
        # ── 4. Visualize button ─────────────────
        if st.button("📊 Generate Visualizations"):
            if not st.session_state.target_column:
                st.warning("Please select a target column.")
            else:
                with st.spinner("Generating charts…"):
                    # ── API call: get_visualizations ──
                    viz_result = get_visualizations(
                        target_column=st.session_state.target_column,
                        file_id=st.session_state.file_id,
                    )

                if viz_result["success"]:
                    st.session_state.charts = viz_result["charts"]
                    st.markdown(
                        f'<span class="badge badge-info">📈 {viz_result["message"]}</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.session_state.charts = []
                    st.error(viz_result["message"])

        elif st.session_state.file_uploaded and not st.session_state.columns:
            st.markdown(
            '<span class="badge badge-warn">⚠ No columns returned by the server.</span>',
            unsafe_allow_html=True,
        )

    # ── Bottom info blurb ───────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color:#4a5068; line-height:1.7;">
        <strong style="color:#8890a4;">How it works</strong><br>
        1. Upload a CSV file<br>
        2. Select your target column<br>
        3. Click <em>Generate Visualizations</em><br>
        4. Explore bar, pie & line charts →
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# 🖥️  UI — Right Panel
# ─────────────────────────────────────────

def render_right_panel():
    """
    Right column UI:
      - Shows visualizations once chart data is available in session state
      - Shows an empty state prompt when no charts are loaded
    """

    st.markdown('<div class="section-label">Output</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Visualizations</div>', unsafe_allow_html=True)

    if not st.session_state.charts:
        # ── Empty state ──────────────────────────
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📉</div>
            <p>
                No visualizations yet.<br>
                Upload a dataset, pick a target column,<br>
                and click <strong>Generate Visualizations</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Render each chart ────────────────────────
    target = st.session_state.target_column
    if target:
        st.markdown(
            f'Analysing target: <span class="col-pill">{target}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

    charts = st.session_state.charts

    # Lay charts in a 2-column grid if there are multiple
    if len(charts) == 1:
        render_chart(charts[0])
    else:
        # Pair charts side-by-side when possible
        for i in range(0, len(charts), 2):
            pair = charts[i : i + 2]
            if len(pair) == 2:
                c1, c2 = st.columns(2, gap="medium")
                with c1:
                    render_chart(pair[0])
                with c2:
                    render_chart(pair[1])
            else:
                render_chart(pair[0])


# ─────────────────────────────────────────
# 🚀  Main — Entry point
# ─────────────────────────────────────────

def main():
    # ── App header ──────────────────────────────
    st.markdown(
        '<div class="hero-title">Smart BI Assistant</div>'
        '<div class="hero-subtitle">Upload · Select · Visualise — in seconds</div>',
        unsafe_allow_html=True,
    )

    # ── 2-column layout ─────────────────────────
    # Left panel: narrower control panel
    # Right panel: wider visualization area
    left_col, right_col = st.columns([1, 2.6], gap="large")

    with left_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        render_left_panel()
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        render_right_panel()


if __name__ == "__main__":
    main()