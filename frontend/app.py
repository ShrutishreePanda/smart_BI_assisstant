from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import (
    get_eda_correlation,
    get_eda_summary,
    get_visualization,
    run_ml,
    upload_file,
)


st.set_page_config(
    page_title="Smart BI Assistant",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
    .stApp {
        background: #f7f9fc;
        color: #182033;
    }

    .block-container {
        max-width: 1320px;
        padding: 4.5rem 2.25rem 3rem;
    }

    .app-title {
        color: #14213d;
        font-size: 2rem;
        font-weight: 760;
        line-height: 1.1;
    }

    .app-subtitle {
        color: #64748b;
        margin: 0.35rem 0 1.35rem;
    }

    .metric-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.5rem 0 1.25rem;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #dde5f0;
        border-radius: 8px;
        padding: 0.85rem 1rem;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.78rem;
        margin-bottom: 0.2rem;
    }

    .metric-value {
        color: #14213d;
        font-size: 1.35rem;
        font-weight: 720;
    }

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #b8c4d6;
        border-radius: 8px;
        padding: 0.5rem;
    }

    .stButton > button {
        background: #2563eb;
        border: 0;
        border-radius: 8px;
        color: #ffffff;
        font-weight: 650;
    }

    .stButton > button:hover {
        background: #1d4ed8;
        color: #ffffff;
    }
</style>
""",
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    defaults = {
        "file_uploaded": False,
        "file_id": None,
        "columns": [],
        "rows": 0,
        "columns_count": 0,
        "preview": [],
        "target_column": None,
        "eda": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header() -> None:
    st.markdown(
        '<div class="app-title">Smart BI Assistant</div>'
        '<div class="app-subtitle">Upload, explore, visualize, and train ML models from one dashboard.</div>',
        unsafe_allow_html=True,
    )


def render_upload_panel() -> None:
    uploaded_file = st.file_uploader(
        "Upload CSV",
        help="Choose one CSV file.",
    )
    selected_file = uploaded_file if uploaded_file and uploaded_file.name.lower().endswith(".csv") else None

    if uploaded_file and not selected_file:
        st.warning("Please choose a file whose name ends with .csv.")

    if st.button("Upload and Analyze", disabled=selected_file is None, use_container_width=True):
        with st.spinner("Uploading dataset..."):
            result = upload_file(selected_file)

        if not result["success"]:
            st.session_state.file_uploaded = False
            st.error(result["message"])
            return

        st.session_state.file_uploaded = True
        st.session_state.file_id = result.get("file_id")
        st.session_state.columns = result.get("columns", [])
        st.session_state.rows = result.get("rows", 0)
        st.session_state.columns_count = result.get("columns_count", len(st.session_state.columns))
        st.session_state.preview = result.get("preview", [])
        st.session_state.target_column = None

        with st.spinner("Loading EDA summary..."):
            eda_result = get_eda_summary()

        if eda_result["success"]:
            st.session_state.eda = eda_result["data"]
        else:
            st.session_state.eda = {"columns": st.session_state.columns}
            st.warning(eda_result["message"])

        st.success(result["message"])


def render_dataset_metrics() -> None:
    if not st.session_state.file_uploaded:
        st.info("Upload a CSV to unlock EDA, ML, and visualization tabs.")
        return

    duplicate_rows = st.session_state.eda.get("duplicate_rows", 0)
    st.markdown(
        f"""
        <div class="metric-strip">
            <div class="metric-card"><div class="metric-label">Rows</div><div class="metric-value">{st.session_state.rows:,}</div></div>
            <div class="metric-card"><div class="metric-label">Columns</div><div class="metric-value">{st.session_state.columns_count:,}</div></div>
            <div class="metric-card"><div class="metric-label">Duplicate Rows</div><div class="metric-value">{duplicate_rows:,}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_eda_section() -> None:
    st.subheader("EDA Summary")
    if not st.session_state.file_uploaded:
        st.info("Upload a dataset first.")
        return

    if st.session_state.preview:
        st.write("### Preview")
        st.dataframe(pd.DataFrame(st.session_state.preview), use_container_width=True)

    eda = st.session_state.eda
    if eda.get("dtypes"):
        st.write("### Columns")
        column_df = pd.DataFrame(
            {
                "column": list(eda["dtypes"].keys()),
                "dtype": list(eda["dtypes"].values()),
                "missing": [eda.get("null_counts", {}).get(col, 0) for col in eda["dtypes"]],
                "unique": [eda.get("cardinality", {}).get(col, 0) for col in eda["dtypes"]],
            }
        )
        st.dataframe(column_df, use_container_width=True, hide_index=True)


def render_top_values_section() -> None:
    st.subheader("Top Values")
    if not st.session_state.file_uploaded:
        st.info("Upload a dataset first.")
        return

    value_counts = st.session_state.eda.get("value_counts", {})
    if not value_counts:
        st.info("No low-cardinality columns available for top-value charts.")
        return

    selected_column = st.selectbox("Column", list(value_counts.keys()))
    values = value_counts[selected_column]
    chart_df = pd.DataFrame({"value": list(values.keys()), "count": list(values.values())})
    st.bar_chart(chart_df.set_index("value"))


def render_correlation_section() -> None:
    st.subheader("Correlation")
    if not st.session_state.file_uploaded:
        st.info("Upload a dataset first.")
        return

    result = get_eda_correlation()
    if not result["success"]:
        st.error(result["message"])
        return

    data = result["data"]
    columns = data.get("columns", [])
    matrix = data.get("matrix", [])
    if not columns or not matrix:
        st.info("Need at least two numeric columns to show correlation.")
        return

    corr_df = pd.DataFrame(matrix, columns=columns, index=columns)
    fig = px.imshow(corr_df, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)


def render_ml_section() -> None:
    st.subheader("Machine Learning")
    if not st.session_state.file_uploaded:
        st.info("Upload a dataset first.")
        return

    columns = st.session_state.eda.get("columns", st.session_state.columns)
    fraud_targets = {"isfraud", "is_fraud", "fraud"}
    default_target = next((column for column in columns if column.lower() in fraud_targets), "")
    default_index = ([""] + columns).index(default_target) if default_target in columns else 0

    target = st.selectbox(
        "Select Target Column (leave empty for clustering)",
        [""] + columns,
        index=default_index,
    )
    k = st.slider("Number of clusters (K-Means)", 2, 6, 3)

    if st.button("Run Model"):
        with st.spinner("Training model..."):
            result = run_ml(target if target else None, k)

        if not result["success"]:
            st.error(result["message"])
            return

        data = result["data"]
        st.success(f"{data['model']} completed")

        model_result = data.get("result", {})
        metric_cols = st.columns(min(len(model_result), 3) or 1)
        for idx, (name, value) in enumerate(model_result.items()):
            display_value = round(value, 4) if isinstance(value, float) else value
            metric_cols[idx % len(metric_cols)].metric(name.replace("_", " ").title(), display_value)

        st.write("### Result")
        st.json(model_result)

        st.write("### Insights")
        for insight in data.get("insights", []):
            st.write(f"- {insight}")


def render_visualization_section() -> None:
    st.subheader("Quick Visualizations")
    if not st.session_state.file_uploaded:
        st.info("Upload a dataset first.")
        return

    result = get_visualization()
    if not result["success"]:
        st.error(result["message"])
        return

    data = result["data"]

    st.write("### Bar Chart")
    bar_df = pd.DataFrame(
        {"value": list(data.get("bar", {}).keys()), "count": list(data.get("bar", {}).values())}
    )
    if not bar_df.empty:
        st.bar_chart(bar_df.set_index("value"))
    else:
        st.info("No bar chart data returned.")

    st.write("### Line Chart")
    line_df = pd.DataFrame(
        {"index": list(data.get("line", {}).keys()), "value": list(data.get("line", {}).values())}
    )
    if not line_df.empty:
        st.line_chart(line_df.set_index("index"))
    else:
        st.info("No line chart data returned.")

    st.write("### Correlation Heatmap")
    heatmap = data.get("heatmap", {})
    if heatmap:
        heatmap_df = pd.DataFrame(heatmap)
        fig = px.imshow(heatmap_df, text_auto=True, color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No heatmap data returned.")


def render_dashboard() -> None:
    render_dataset_metrics()
    tabs = st.tabs(["EDA", "Top Values", "Correlation", "ML", "Visualize"])

    with tabs[0]:
        render_eda_section()
    with tabs[1]:
        render_top_values_section()
    with tabs[2]:
        render_correlation_section()
    with tabs[3]:
        render_ml_section()
    with tabs[4]:
        render_visualization_section()


def main() -> None:
    init_session_state()
    render_header()
    render_upload_panel()
    render_dashboard()


if __name__ == "__main__":
    main()
