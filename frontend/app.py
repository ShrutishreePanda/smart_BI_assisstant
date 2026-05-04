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


HUMAN_LABELS = {
    "IsFraud": "Fraud Status",
    "is_fraud": "Fraud Status",
    "fraud": "Fraud Status",
    "TransactionAmount": "Transaction Amount",
    "TransactionHour": "Transaction Hour",
    "TransactionDayOfWeek": "Transaction Day of Week",
    "IsWeekend": "Weekend Transaction",
    "TransactionDistanceKm": "Transaction Distance (km)",
    "CustomerAge": "Customer Age (Years)",
    "CityPopulation": "City Population",
    "TransactionCategoryEncoded": "Transaction Category",
}

DEMO_USE_CASES = {
    "Fraud Detection": {
        "model": "Logistic Regression",
        "target": "IsFraud",
        "features": [
            "TransactionAmount",
            "TransactionHour",
            "TransactionDayOfWeek",
            "IsWeekend",
            "TransactionDistanceKm",
            "CustomerAge",
            "CityPopulation",
            "TransactionCategoryEncoded",
        ],
        "description": "We use behavioral and transactional features to detect anomalies indicative of fraud.",
    },
    "Predictive Modeling": {
        "model": "Linear Regression",
        "target": "TransactionAmount",
        "features": [
            "TransactionHour",
            "TransactionDayOfWeek",
            "IsWeekend",
            "CustomerAge",
            "CityPopulation",
            "TransactionCategoryEncoded",
        ],
        "description": "We use historical and behavioral features to predict transaction values.",
    },
    "Behavioral Analysis": {
        "model": "K-Means",
        "target": "",
        "features": [
            "TransactionAmount",
            "TransactionHour",
            "TransactionDistanceKm",
            "CustomerAge",
            "CityPopulation",
            "TransactionCategoryEncoded",
        ],
        "description": "We identify customer segments such as high spenders, low spenders, and irregular users.",
    },
}


def humanize_column(column: str) -> str:
    if column in HUMAN_LABELS:
        return HUMAN_LABELS[column]

    label = ""
    previous = ""
    for char in column.replace("_", " "):
        if label and char.isupper() and previous and previous.islower():
            label += " "
        label += char
        previous = char
    return label.strip().title()


def humanize_metric(name: str) -> str:
    labels = {
        "accuracy": "Overall Correct Predictions",
        "precision": "Fraud Alert Precision",
        "recall": "Fraud Capture Rate",
        "mae": "Average Prediction Error",
        "mse": "Large Error Penalty",
        "clusters": "Customer Segments Found",
    }
    return labels.get(name, name.replace("_", " ").title())


def relabel_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={column: humanize_column(str(column)) for column in df.columns})


def existing_columns(columns: list[str], requested_columns: list[str]) -> list[str]:
    return [column for column in requested_columns if column in columns]


def render_model_visualizations(task: str, visualizations: dict) -> None:
    if task == "classification":
        fraud_distribution = visualizations.get("fraud_distribution", {})
        if fraud_distribution:
            fraud_df = pd.DataFrame(
                {"Fraud Status": list(fraud_distribution.keys()), "Transactions": list(fraud_distribution.values())}
            )
            fig = px.pie(
                fraud_df,
                names="Fraud Status",
                values="Transactions",
                title="Fraud vs Normal Transactions in the Test Data",
                hole=0.35,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        prediction_summary = visualizations.get("prediction_summary", {})
        if prediction_summary:
            summary_df = pd.DataFrame(
                {"Prediction": list(prediction_summary.keys()), "Transactions": list(prediction_summary.values())}
            )
            fig = px.bar(
                summary_df,
                x="Prediction",
                y="Transactions",
                title="Model Prediction Summary",
                labels={"Prediction": "Predicted Fraud Status", "Transactions": "Number of Transactions"},
            )
            st.plotly_chart(fig, use_container_width=True)

    elif task == "regression":
        actual_vs_predicted = visualizations.get("actual_vs_predicted", [])
        if actual_vs_predicted:
            comparison_df = pd.DataFrame(actual_vs_predicted)
            fig = px.scatter(
                comparison_df,
                x="Actual Transaction Amount",
                y="Predicted Transaction Amount",
                title="Actual vs Predicted Transaction Amount",
                labels={
                    "Actual Transaction Amount": "Actual Transaction Amount",
                    "Predicted Transaction Amount": "Predicted Transaction Amount",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        error_distribution = visualizations.get("error_distribution", [])
        if error_distribution:
            error_df = pd.DataFrame(error_distribution)
            fig = px.histogram(
                error_df,
                x="Prediction Error",
                nbins=30,
                title="Prediction Error Distribution",
                labels={"Prediction Error": "Actual Amount Minus Predicted Amount"},
            )
            st.plotly_chart(fig, use_container_width=True)

    elif task == "clustering":
        cluster_distribution = visualizations.get("cluster_distribution", {})
        if cluster_distribution:
            cluster_df = pd.DataFrame(
                {"Customer Segment": list(cluster_distribution.keys()), "Transactions": list(cluster_distribution.values())}
            )
            fig = px.pie(
                cluster_df,
                names="Customer Segment",
                values="Transactions",
                title="Customer Segment Distribution",
                hole=0.35,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        cluster_scatter = visualizations.get("cluster_scatter", [])
        if cluster_scatter:
            scatter_df = pd.DataFrame(cluster_scatter)
            fig = px.scatter(
                scatter_df,
                x="Transaction Amount",
                y="Transaction Distance (km)",
                color="Customer Segment",
                title="Customer Segments by Spending and Distance",
            )
            st.plotly_chart(fig, use_container_width=True)


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
        "raw_rows": 0,
        "raw_columns": 0,
        "generated_file": "",
        "feature_mappings": {},
        "raw_eda": {},
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
        "Upload Raw CSV",
        help="Upload the raw transaction CSV. The backend will run EDA and generate model-ready features automatically.",
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
        st.session_state.raw_rows = result.get("raw_rows", 0)
        st.session_state.raw_columns = result.get("raw_columns", 0)
        st.session_state.generated_file = result.get("generated_file", "")
        st.session_state.feature_mappings = result.get("feature_mappings", {})
        st.session_state.raw_eda = result.get("raw_eda", {})

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
            <div class="metric-card"><div class="metric-label">Raw Rows</div><div class="metric-value">{st.session_state.raw_rows:,}</div></div>
            <div class="metric-card"><div class="metric-label">Feature Columns</div><div class="metric-value">{st.session_state.columns_count:,}</div></div>
            <div class="metric-card"><div class="metric-label">Duplicate Rows</div><div class="metric-value">{duplicate_rows:,}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Raw CSV -> Automatic EDA -> Feature Engineering -> "
        f"{st.session_state.generated_file or 'UnifiedTransactionFeatures.csv'} -> ML and visualizations"
    )

    mappings = st.session_state.feature_mappings
    if mappings:
        with st.expander("Feature Engineering Mapping"):
            mapping_df = pd.DataFrame(
                {
                    "Generated Feature": [humanize_column(column) for column in mappings.keys()],
                    "Source / Derivation": list(mappings.values()),
                }
            )
            st.dataframe(mapping_df, use_container_width=True, hide_index=True)


def render_eda_section() -> None:
    st.subheader("EDA Summary")
    if not st.session_state.file_uploaded:
        st.info("Upload a dataset first.")
        return

    if st.session_state.preview:
        st.write("### Generated Feature Dataset Preview")
        st.dataframe(relabel_dataframe(pd.DataFrame(st.session_state.preview)), use_container_width=True)

    raw_eda = st.session_state.raw_eda
    if raw_eda:
        with st.expander("Raw Dataset EDA Used for Feature Engineering"):
            raw_summary_df = pd.DataFrame(
                {
                    "Column": [humanize_column(col) for col in raw_eda.get("dtypes", {})],
                    "Data Type": list(raw_eda.get("dtypes", {}).values()),
                    "Missing Values": [
                        raw_eda.get("null_counts", {}).get(col, 0)
                        for col in raw_eda.get("dtypes", {})
                    ],
                    "Unique Values": [
                        raw_eda.get("cardinality", {}).get(col, 0)
                        for col in raw_eda.get("dtypes", {})
                    ],
                }
            )
            st.dataframe(raw_summary_df, use_container_width=True, hide_index=True)

    eda = st.session_state.eda
    if eda.get("dtypes"):
        st.write("### Unified Feature Columns")
        column_df = pd.DataFrame(
            {
                "Column": [humanize_column(col) for col in eda["dtypes"]],
                "Data Type": list(eda["dtypes"].values()),
                "Missing Values": [eda.get("null_counts", {}).get(col, 0) for col in eda["dtypes"]],
                "Unique Values": [eda.get("cardinality", {}).get(col, 0) for col in eda["dtypes"]],
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

    selected_column = st.selectbox("Column", list(value_counts.keys()), format_func=humanize_column)
    values = value_counts[selected_column]
    chart_df = pd.DataFrame({"Value": list(values.keys()), "Rows": list(values.values())})
    chart_title = f"Most Common Values in {humanize_column(selected_column)}"

    if len(chart_df) <= 6:
        fig = px.pie(
            chart_df,
            names="Value",
            values="Rows",
            title=f"{humanize_column(selected_column)} Breakdown",
            hole=0.35,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
    else:
        fig = px.bar(
            chart_df,
            x="Value",
            y="Rows",
            title=chart_title,
            labels={"Value": humanize_column(selected_column), "Rows": "Number of Rows"},
        )

    fig.update_layout(legend_title_text=humanize_column(selected_column))
    st.plotly_chart(fig, use_container_width=True)


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

    readable_columns = [humanize_column(column) for column in columns]
    corr_df = pd.DataFrame(matrix, columns=readable_columns, index=readable_columns)
    fig = px.imshow(
        corr_df,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Relationship Strength Between Numeric Columns",
    )
    fig.update_layout(coloraxis_colorbar_title="Correlation")
    st.plotly_chart(fig, use_container_width=True)


def render_ml_section() -> None:
    st.subheader("Machine Learning")
    if not st.session_state.file_uploaded:
        st.info("Upload a dataset first.")
        return

    columns = st.session_state.eda.get("columns", st.session_state.columns)
    use_case = st.selectbox(
        "Demo Use Case",
        list(DEMO_USE_CASES.keys()),
        help="Each use case uses the same unified feature dataset with a different target and feature subset.",
    )
    config = DEMO_USE_CASES[use_case]
    st.caption(config["description"])

    model_choice = config["model"]
    target_options = [""] if use_case == "Behavioral Analysis" else [config["target"]]

    target = st.selectbox(
        "Target Column",
        target_options,
        index=0,
        format_func=lambda value: "No target - group similar rows" if value == "" else humanize_column(value),
        disabled=True,
        help="The demo uses a fixed target for each use case.",
    )

    effective_target = target or None
    feature_options = existing_columns(columns, config["features"])
    default_features = [] if use_case == "Behavioral Analysis" else feature_options
    selected_features = st.multiselect(
        "Input Features",
        feature_options,
        default=default_features,
        format_func=humanize_column,
        help="Supervised models use selected features. Leave this empty for clustering to trigger behavioral analysis automatically.",
    )

    st.info(
        "All models use the same unified feature dataset, but each model dynamically selects "
        "a relevant subset of features based on the selected target and use case."
    )

    if use_case == "Fraud Detection":
        st.write("**Model:** Logistic Regression")
        st.write("**Target:** Fraud Status")
    elif use_case == "Predictive Modeling":
        st.write("**Model:** Linear Regression")
        st.write("**Target:** Transaction Amount")
    else:
        st.write("**Model:** K-Means Clustering")
        st.write("**Target:** None")
        st.warning(
            "When no target or features are selected, the system switches to unsupervised learning "
            "and performs behavioral clustering."
        )

    k = st.slider(
        "Number of groups",
        2,
        6,
        3,
        disabled=use_case != "Behavioral Analysis",
        help="Used only when running K-Means clustering.",
    )

    if st.button("Run Model"):
        if use_case != "Behavioral Analysis" and not selected_features:
            st.error("Please select at least one input feature for supervised learning.")
            return

        request_model = "K-Means" if not effective_target and not selected_features else model_choice
        request_features = selected_features or None

        with st.spinner("Training model..."):
            result = run_ml(
                target=effective_target,
                k=k,
                features=request_features,
                model=request_model,
            )

        if not result["success"]:
            st.error(result["message"])
            return

        data = result["data"]
        st.success(f"{data.get('use_case', use_case)} completed with {data['model']}")

        model_result = data.get("result", {})
        metric_cols = st.columns(min(len(model_result), 3) or 1)
        metric_index = 0
        for idx, (name, value) in enumerate(model_result.items()):
            if isinstance(value, dict) or name == "optional_r2":
                continue
            display_value = round(value, 4) if isinstance(value, float) else value
            metric_cols[metric_index % len(metric_cols)].metric(humanize_metric(name), display_value)
            metric_index += 1

        if "optional_r2" in model_result:
            st.caption(
                f"Optional model-fit reference: R-squared = {model_result['optional_r2']:.4f}. "
                "This is shown as context only, not as the primary demo metric."
            )

        render_model_visualizations(data.get("task", ""), data.get("visualizations", {}))

        st.write("### Result")
        st.json(model_result)

        st.write("### Features Used")
        used_features = data.get("features", selected_features)
        st.write(", ".join(humanize_column(feature) for feature in used_features))

        st.write("### Insights")
        for insight in data.get("insights", []):
            st.write(f"- {insight}")

        st.success(
            "We built a unified feature dataset, and based on the selected target-or the absence "
            "of one-the system automatically applies the most appropriate machine learning model "
            "and visualizes the results."
        )


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

    distribution = data.get("distribution", {})
    if distribution:
        column = distribution.get("column", "")
        values = distribution.get("values", {})
        chart_df = pd.DataFrame({"Value": list(values.keys()), "Rows": list(values.values())})
        fig = px.pie(
            chart_df,
            names="Value",
            values="Rows",
            title=f"{humanize_column(column)} Breakdown",
            hole=0.35,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(legend_title_text=humanize_column(column))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No category breakdown is available for this dataset.")

    comparison = data.get("comparison", {})
    if comparison:
        column = comparison.get("column", "")
        values = comparison.get("values", {})
        chart_df = pd.DataFrame({"Value": list(values.keys()), "Rows": list(values.values())})
        fig = px.bar(
            chart_df,
            x="Value",
            y="Rows",
            title=f"Top Values for {humanize_column(column)}",
            labels={"Value": humanize_column(column), "Rows": "Number of Rows"},
        )
        fig.update_traces(
            hovertemplate=f"{humanize_column(column)}: %{{x}}<br>Rows: %{{y}}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No comparison chart data returned.")

    time_series = data.get("time_series", {})
    if time_series:
        values = time_series.get("values", {})
        line_df = pd.DataFrame({"Date": list(values.keys()), "Average Value": list(values.values())})
        fig = px.line(
            line_df,
            x="Date",
            y="Average Value",
            title=(
                f"{humanize_column(time_series.get('value_column', 'Value'))} "
                f"Over {humanize_column(time_series.get('column', 'Time'))}"
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.write("### Correlation Heatmap")
    heatmap = data.get("heatmap", {})
    if heatmap:
        heatmap_df = pd.DataFrame(heatmap)
        heatmap_df = heatmap_df.rename(index=humanize_column, columns=humanize_column)
        fig = px.imshow(
            heatmap_df,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Relationship Strength Between Numeric Columns",
        )
        fig.update_layout(coloraxis_colorbar_title="Correlation")
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
