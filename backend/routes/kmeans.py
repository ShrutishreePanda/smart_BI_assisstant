from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from backend.routes.preprocessing import preprocess


def run_kmeans(df, k=3):
    X, _ = preprocess(df)

    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)

    return {
        "clusters": k,
        "silhouette": float(score)
    }