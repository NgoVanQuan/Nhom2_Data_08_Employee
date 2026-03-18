from sklearn.cluster import KMeans
import pandas as pd

def perform_clustering(df_scaled, n_clusters=3):
    print(f"Tiến hành phân cụm K-Means với K={n_clusters}...")
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(df_scaled)
    return labels, model

def calculate_wcss(df_scaled, k_range=range(2, 8)):
    wcss = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(df_scaled)
        wcss.append(km.inertia_)
    return wcss