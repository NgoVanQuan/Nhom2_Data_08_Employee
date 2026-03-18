import joblib
import os
from .association import run_fp_growth
from .clustering import perform_clustering, calculate_wcss
from sklearn.preprocessing import StandardScaler

class Miner:
    def __init__(self, min_support=0.15, max_len=3, n_clusters=3):
        self.min_support = min_support
        self.max_len = max_len
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans_model = None

    def mine_association_rules(self, df):
        return run_fp_growth(df, self.min_support, self.max_len)

    def run_kmeans(self, df_encoded):
        features = df_encoded.drop(columns=[c for c in df_encoded.columns if 'Attrition' in c], errors='ignore')
        X_scaled = self.scaler.fit_transform(features)
        labels, self.kmeans_model = perform_clustering(X_scaled, self.n_clusters)
        return labels

    def save_models(self):
        os.makedirs('../outputs/models', exist_ok=True)
        joblib.dump(self.scaler, '../outputs/models/scaler_kmeans.pkl')
        joblib.dump(self.kmeans_model, '../outputs/models/kmeans_model.pkl')
        print("✅ Đã lưu mô hình thành công!")