import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_learning_curve(fractions, supervised_scores, semi_scores, metric_name='F1-Score', save_path=None):
    """
    Vẽ biểu đồ Learning Curve so sánh Supervised và Semi-supervised.
    """
    plt.figure(figsize=(8, 5))
    labeled_percentages = [f"{int(f*100)}%" for f in fractions]
    
    plt.plot(labeled_percentages, supervised_scores, marker='o', linestyle='--', linewidth=2, label='Supervised Only')
    plt.plot(labeled_percentages, semi_scores, marker='s', linestyle='-', linewidth=2, color='green', label='Self-Training')

    plt.title(f'Learning Curve: Hiệu suất {metric_name} theo tỷ lệ nhãn', fontsize=14)
    plt.xlabel('Tỷ lệ dữ liệu có nhãn', fontsize=12)
    plt.ylabel(f'Điểm {metric_name}', fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.5)

    # Hiện số liệu trên biểu đồ
    for i, txt in enumerate(semi_scores):
        plt.annotate(f"{txt:.3f}", (labeled_percentages[i], semi_scores[i] + 0.02), ha='center')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Đã lưu biểu đồ tại: {save_path}")
        
    plt.show()

def plot_model_comparison(results_dict, metrics=['F1 Score', 'PR-AUC'], title='So sánh mô hình', save_path=None):
    """
    Vẽ biểu đồ cột so sánh các metric giữa các mô hình.
    """
    df_results = pd.DataFrame(results_dict).T
    
    fig, ax = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5))
    if len(metrics) == 1:
        ax = [ax] # Đảm bảo ax luôn có thể lặp (iterable)
        
    for i, metric in enumerate(metrics):
        if metric in df_results.columns:
            df_results[metric].plot(kind='bar', color=sns.color_palette("husl", len(df_results)), ax=ax[i])
            ax[i].set_title(f'So sánh {metric}')
            ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=45, ha='right')
            ax[i].grid(axis='y', linestyle='--', alpha=0.7)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Đã lưu biểu đồ tại: {save_path}")
        
    plt.show()
    return df_results