from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.cluster import KMeans

from backend.routes.preprocessing import preprocess


def train_behavioral_clustering(df: pd.DataFrame, features: list[str], k: int) -> dict[str, Any]:
    X, _, feature_names = preprocess(df, None, features)
    if X.shape[0] <= k:
        raise ValueError("K-Means needs more rows than the number of clusters")

    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    cluster_names = friendly_cluster_names(df, features, labels)
    distribution = cluster_distribution(labels, cluster_names)

    return {
        "task": "clustering",
        "model": "K-Means",
        "result": {
            "clusters": int(k),
            "cluster_distribution": distribution,
        },
        "visualizations": {
            "cluster_distribution": distribution,
            "cluster_scatter": cluster_scatter_points(df, features, labels, cluster_names),
        },
        "insights": [
            "We identify customer segments such as high spenders, low spenders, and irregular users.",
            "Cluster size distribution shows how transactions are grouped by similar behavior.",
        ],
        "features": list(feature_names),
        "use_case": "Behavioral Analysis",
    }


def friendly_cluster_names(df: pd.DataFrame, features: list[str], labels) -> dict[int, str]:
    amount_column = "TransactionAmount" if "TransactionAmount" in features else features[0]
    profile = pd.DataFrame(
        {
            "cluster": labels,
            "amount": pd.to_numeric(df[amount_column], errors="coerce").fillna(0),
        }
    )
    ordered_clusters = (
        profile.groupby("cluster")["amount"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )

    base_names = ["Low Spenders", "Moderate Spenders", "High Spenders"]
    names: dict[int, str] = {}
    for index, cluster_id in enumerate(ordered_clusters):
        names[int(cluster_id)] = base_names[index] if index < len(base_names) else f"Segment {index + 1}"
    return names


def cluster_distribution(labels, cluster_names: dict[int, str]) -> dict[str, int]:
    counts = pd.Series(labels).value_counts().sort_index()
    return {
        cluster_names.get(int(cluster_id), f"Segment {int(cluster_id) + 1}"): int(count)
        for cluster_id, count in counts.items()
    }


def cluster_scatter_points(
    df: pd.DataFrame,
    features: list[str],
    labels,
    cluster_names: dict[int, str],
) -> list[dict[str, Any]]:
    x_column = "TransactionAmount" if "TransactionAmount" in features else features[0]
    y_column = "TransactionDistanceKm" if "TransactionDistanceKm" in features else features[min(1, len(features) - 1)]
    frame = pd.DataFrame(
        {
            "Transaction Amount": pd.to_numeric(df[x_column], errors="coerce").fillna(0),
            "Transaction Distance (km)": pd.to_numeric(df[y_column], errors="coerce").fillna(0),
            "Customer Segment": [cluster_names.get(int(label), f"Segment {int(label) + 1}") for label in labels],
        }
    )
    return frame.head(500).to_dict(orient="records")
