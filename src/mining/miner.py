import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

class Miner:
    def __init__(self, min_support=0.15, max_len=3, n_clusters=3):
        self.min_support = min_support
        self.max_len = max_len
        self.n_clusters = n_clusters
        self.kmeans_model = None
        self.scaler = StandardScaler()

    def mine_association_rules(self, df_transactions):
        """Chạy thuật toán FP-Growth tìm luật kết hợp"""
        print("Đang chuẩn bị ma trận nhị phân (Boolean) cho FP-Growth...")
        df_bool = pd.get_dummies(df_transactions).astype(bool)
        
        print(f"Đang chạy FP-Growth với min_support = {self.min_support}...")
        frequent_itemsets = fpgrowth(
            df_bool, 
            min_support=self.min_support, 
            use_colnames=True, 
            max_len=self.max_len
        )
        
        print("Đang trích xuất các luật kết hợp...")
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        return rules

    def run_kmeans(self, df_encoded):
        """Chạy K-Means phân cụm (Tự động chuẩn hóa dữ liệu)"""
        print("Đang chuẩn hóa dữ liệu cho K-Means (StandardScaler)...")
        # Bỏ cột Attrition nếu có để không thiên lệch khoảng cách
        features = df_encoded.drop(columns=[c for c in df_encoded.columns if 'Attrition' in c], errors='ignore')
        X_scaled = self.scaler.fit_transform(features)
        
        print(f"Tiến hành phân cụm với K = {self.n_clusters}...")
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters, 
            random_state=42, 
            n_init=10
        )
        cluster_labels = self.kmeans_model.fit_predict(X_scaled)
        return cluster_labels

    def save_models(self):
        """Lưu mô hình KMeans và Scaler"""
        os.makedirs('../outputs/models', exist_ok=True)
        joblib.dump(self.scaler, '../outputs/models/scaler_kmeans.pkl')
        joblib.dump(self.kmeans_model, '../outputs/models/kmeans_model.pkl')
        print("✅ Đã lưu file mô hình (kmeans_model.pkl & scaler_kmeans.pkl) vào thư mục outputs/models/")