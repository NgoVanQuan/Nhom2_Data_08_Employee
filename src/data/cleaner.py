import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_hr_data(df):
    """
    Hàm làm sạch dữ liệu HR và mã hóa các biến phân loại.
    """
    print("⏳ Bắt đầu làm sạch dữ liệu...")
    df_clean = df.copy()

    # 1. Loại bỏ các cột không có giá trị phân tích (ID hoặc hằng số)
    cols_to_drop = ['EmpID', 'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns], errors='ignore')
    
    # 2. Xử lý giá trị thiếu (nếu có)
    # Bộ dữ liệu của IBM thường khá sạch, nhưng cứ thêm lệnh dropna cho chắc cốp
    df_clean = df_clean.dropna()

    # 3. Mã hóa các biến nhị phân (Yes/No -> 1/0)
    binary_mapping = {'Yes': 1, 'No': 0}
    if 'Attrition' in df_clean.columns:
        df_clean['Attrition'] = df_clean['Attrition'].map(binary_mapping)
    if 'OverTime' in df_clean.columns:
        df_clean['OverTime'] = df_clean['OverTime'].map(binary_mapping)
        
    # Xử lý riêng giới tính
    if 'Gender' in df_clean.columns:
        df_clean['Gender'] = df_clean['Gender'].map({'Male': 1, 'Female': 0})

    # 4. Mã hóa các biến phân loại nhiều nhãn (Categorical -> Numeric)
    # Dùng LabelEncoder sẽ rất tốt cho các mô hình cây như Random Forest/XGBoost sau này
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    
    for col in categorical_cols:
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    print(f"✅ Đã làm sạch xong! Kích thước dữ liệu hiện tại: {df_clean.shape}")
    return df_clean

def save_clean_data(df, output_path):
    """Lưu dữ liệu đã làm sạch ra file CSV."""
    df.to_csv(output_path, index=False)
    print(f"💾 Đã lưu dữ liệu sạch tại: {output_path}")