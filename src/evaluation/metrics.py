import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    average_precision_score, # Dùng cho PR-AUC
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    silhouette_score,
    davies_bouldin_score
)

def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Tính toán các chỉ số đánh giá cho mô hình phân lớp.
    Đặc biệt tập trung vào F1-macro và PR-AUC cho dữ liệu mất cân bằng.
    """
    metrics = {}
    
    # F1 Score (ưu tiên macro cho bài toán imbalanced)
    metrics['F1 Score'] = f1_score(y_true, y_pred, average='macro')
    
    # Tính PR-AUC và ROC-AUC nếu có xác suất dự đoán (y_prob)
    if y_prob is not None:
        # Giả sử y_prob là xác suất của class positive (class 1 - Leave)
        metrics['PR-AUC'] = average_precision_score(y_true, y_prob)
        try:
            metrics['ROC-AUC'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['ROC-AUC'] = np.nan # Xử lý trường hợp test set chỉ có 1 class
            
    return metrics

def calculate_regression_metrics(y_true, y_pred):
    """
    Tính toán các chỉ số cho bài toán hồi quy (Dự đoán điểm hiệu suất/hài lòng).
    """
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
    }
    return metrics

def calculate_clustering_metrics(X, labels):
    """
    Tính toán các chỉ số đánh giá chất lượng phân cụm (Clustering).
    """
    # Nếu chỉ có 1 cụm thì không tính được các chỉ số này
    if len(np.unique(labels)) > 1:
        metrics = {
            'Silhouette Score': silhouette_score(X, labels),
            'Davies-Bouldin Index': davies_bouldin_score(X, labels)
        }
    else:
        metrics = {
            'Silhouette Score': -1.0,
            'Davies-Bouldin Index': -1.0
        }
    return metrics

def get_confusion_matrix_df(y_true, y_pred, labels=['Stay', 'Leave']):
    """
    Trả về Confusion Matrix dưới dạng DataFrame để dễ dàng in và vẽ biểu đồ.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=[f'True {l}' for l in labels], 
                         columns=[f'Pred {l}' for l in labels])
    return cm_df

def print_full_classification_report(y_true, y_pred):
    """
    In ra báo cáo phân lớp chi tiết (Precision, Recall, F1 cho từng class).
    """
    print(classification_report(y_true, y_pred, target_names=['Stay', 'Leave']))