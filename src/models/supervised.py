import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, auc

def load_config(config_path="configs/params.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def train_and_evaluate(df, target_col='Attrition', config_path="configs/params.yaml"):
    """
    Huấn luyện mô hình phân lớp dự đoán nghỉ việc và đánh giá.
    """
    print("Bắt đầu quá trình Mô hình hóa (Supervised Learning)...")
    config = load_config(config_path)
    test_size = config['split']['test_size']
    random_state = config['split']['random_state']

    # 1. Tách X, y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Kích thước tập Train: {X_train.shape[0]} mẫu, Tập Test: {X_test.shape[0]} mẫu.")

    # Tính toán tỷ lệ mất cân bằng để gán trọng số cho XGBoost (scale_pos_weight)
    pos_cases = sum(y_train == 1)
    neg_cases = sum(y_train == 0)
    scale_pos_weight = neg_cases / pos_cases if pos_cases > 0 else 1.0

    # 3. Khởi tạo mô hình
    # Baseline: Random Forest với tham số cơ bản
    rf_model = RandomForestClassifier(random_state=random_state, class_weight='balanced')
    
    # Advanced: XGBoost xử lý Imbalance cực tốt
    xgb_model = XGBClassifier(
        random_state=random_state, 
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )

    models = {'Random Forest (Baseline)': rf_model, 'XGBoost (Advanced)': xgb_model}
    results = {}

    # 4. Huấn luyện và Đánh giá
    for name, model in models.items():
        print(f"\n Đang huấn luyện {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]

        # Tính PR-AUC (Chuẩn rubric cho dữ liệu Imbalance)
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        pr_auc = auc(recall, precision)
        
        # Tính F1-Score
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            'model': model,
            'F1-Score': f1,
            'PR-AUC': pr_auc,
            'report': classification_report(y_test, y_pred)
        }
        
        print(f"{name} - F1-Score: {f1:.4f} | PR-AUC: {pr_auc:.4f}")

    return results, X_test, y_test