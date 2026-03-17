import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import classification_report, f1_score

def train_semi_supervised(df, target_col='Attrition', test_size=0.2, unlabelled_ratio=0.8, random_state=42):
    """
    Huấn luyện mô hình Bán giám sát (Self-Training) bằng cách giả lập xóa đi 80% nhãn.
    """
    print("🚀 Bắt đầu Mô hình Bán giám sát (Semi-Supervised Learning)...")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 1. Chia Train/Test (Tập Test giữ nguyên để đánh giá công bằng)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2. Giả lập "Mất nhãn" trên tập Train
    # Chọn ngẫu nhiên 'unlabelled_ratio' (VD: 80%) dữ liệu để xóa nhãn (gán thành -1 theo chuẩn của Sklearn)
    rng = np.random.RandomState(random_state)
    random_unlabeled_points = rng.rand(y_train.shape[0]) < unlabelled_ratio
    
    y_train_semi = np.copy(y_train)
    y_train_semi[random_unlabeled_points] = -1

    labeled_count = sum(y_train_semi != -1)
    unlabeled_count = sum(y_train_semi == -1)
    print(f"📉 Đã giả lập mất nhãn: Giữ lại {labeled_count} mẫu CÓ nhãn, {unlabeled_count} mẫu KHÔNG CÓ nhãn.")

    # 3. Khởi tạo mô hình Self-Training
    # Dùng Random Forest làm mô hình cơ sở
    base_model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
    
    # Mô hình sẽ tự học, nếu độ tự tin > 0.75 thì nó sẽ gán nhãn giả (pseudo-label) cho tập không nhãn
    self_training_model = SelfTrainingClassifier(base_model, threshold=0.75, max_iter=10)
    
    print("⏳ Đang huấn luyện mô hình cho máy tự học...")
    self_training_model.fit(X_train, y_train_semi)

    # 4. Đánh giá trên tập Test
    y_pred = self_training_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    
    print(f"✅ Huấn luyện xong! F1-Score trên tập Test: {f1:.4f}")
    
    return self_training_model, classification_report(y_test, y_pred)