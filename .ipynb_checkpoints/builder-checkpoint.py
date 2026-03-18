import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeatureBuilder:
    def __init__(self):
        self.scaler = StandardScaler()

    def build_rules_data(self, df):
        """Tiền xử lý riêng cho FP-Growth (Giữ nguyên chữ, Binning, không dùng LabelEncoder)"""
        df_binned = df.copy()
        
        # Tự động dọn rác cơ bản
        cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours', 'EmpID']
        df_binned = df_binned.drop(columns=[c for c in cols_to_drop if c in df_binned.columns], errors='ignore')
        df_binned = df_binned.dropna()

        # Binning (Rời rạc hóa) Độ tuổi và Mức lương
        if 'Age' in df_binned.columns:
            df_binned['AgeGroup'] = pd.cut(df_binned['Age'], bins=[0, 30, 45, 100], labels=['Tre', 'Trung_nien', 'Lon_tuoi'])
        if 'MonthlyIncome' in df_binned.columns:
            df_binned['SalarySlab'] = pd.cut(df_binned['MonthlyIncome'], bins=[0, 5000, 10000, 15000, 100000], labels=['Upto_5k', '5k_to_10k', '10k_to_15k', '15k_plus'])

        return df_binned

    def build_ml_data(self, df_rules):
        """Tiền xử lý riêng cho K-Means (One-hot Encoding và StandardScaler)"""
        # Áp dụng One-hot Encoding chuẩn chỉnh
        df_encoded = pd.get_dummies(df_rules).astype(int)

        # Áp dụng chuẩn hóa StandardScaler (Bỏ qua cột biến mục tiêu nếu có)
        features_to_scale = df_encoded.drop(columns=[c for c in df_encoded.columns if 'Attrition' in c], errors='ignore')
        scaled_data = self.scaler.fit_transform(features_to_scale)
        df_scaled = pd.DataFrame(scaled_data, columns=features_to_scale.columns)

        return df_encoded, df_scaled, self.scaler