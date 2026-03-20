import papermill as pm
import os
import sys

# ---------------------------------------------------------
# TỰ ĐỘNG ĐỊNH VỊ THƯ MỤC GỐC CỦA DỰ ÁN
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)
print(f"✅ Đã set thư mục làm việc tại: {os.getcwd()}\n")

os.makedirs("notebooks/runs", exist_ok=True)

# ---------------------------------------------------------
# CẤU HÌNH PIPELINE 6 BƯỚC CHUẨN MLOPS
# ---------------------------------------------------------
PIPELINE = [
    {
        "step": "Bước 1: Khám phá dữ liệu (EDA)",
        "input_path": "notebooks/01_eda.ipynb",
        "output_path": "notebooks/runs/01_eda_run.ipynb",
        "params": {"DATA_PATH": "data/raw/HR_Analytics.csv"}
    },
    {
        "step": "Bước 2: Tiền xử lý & Trích xuất đặc trưng",
        "input_path": "notebooks/02_preprocess_feature.ipynb",
        "output_path": "notebooks/runs/02_preprocess_feature_run.ipynb",
        "params": {
            "RAW_DATA_PATH": "data/raw/HR_Analytics.csv",
            "OUTPUT_RULES_PATH": "data/processed/hr_rules_binned.csv",
            "OUTPUT_ML_PATH": "data/processed/hr_ml_encoded.csv"
        }
    },
    {
        "step": "Bước 3: Khai phá Dữ liệu (FP-Growth & K-Means)",
        "input_path": "notebooks/03_mining_or_clustering.ipynb", 
        "output_path": "notebooks/runs/03_mining_run.ipynb",
        "params": {
            "RULES_DATA_PATH": "data/processed/hr_rules_binned.csv",
            "ML_DATA_PATH": "data/processed/hr_ml_encoded.csv",
            "MIN_SUPPORT": 0.15,
            "N_CLUSTERS": 3
        }
    },
    {
        "step": "Bước 4a: Học máy Có giám sát (Supervised Learning)",
        "input_path": "notebooks/04_modeling.ipynb",
        "output_path": "notebooks/runs/04a_supervised_learning_run.ipynb",
        "params": {
            "ML_DATA_PATH": "data/processed/hr_ml_encoded.csv",
            "TARGET_COL": "Attrition_Yes",
        }
    },
    {
        "step": "Bước 4b: Thử nghiệm Học máy Bán giám sát (Semi-supervised)",
        "input_path": "notebooks/04b_semi_supervised.ipynb",
        "output_path": "notebooks/runs/04b_semi_supervised_run.ipynb",
        "params": {
            "ML_DATA_PATH": "data/processed/hr_ml_encoded.csv",
            "TARGET_COL": "Attrition_Yes",
            "UNLABELLED_RATIO": 0.8
        }
    },
    {
        "step": "Bước 5: Đánh giá & Tổng hợp Insight",
        "input_path": "notebooks/05_evaluation_report.ipynb", # Sửa dòng này nếu tên file 05 của bác khác nhé
        "output_path": "notebooks/runs/05_evaluation_run.ipynb",
        "params": {}
    }
]

print("🚀 BẮT ĐẦU CHẠY TỰ ĐỘNG PIPELINE HR ANALYTICS BẰNG PAPERMILL...\n")

for task in PIPELINE:
    print(f"⏳ Đang chạy {task['step']}...")
    
    # Tính năng quét lỗi thông minh: Báo ngay nếu sai tên file
    if not os.path.exists(task['input_path']):
        print(f"\n🚨 LỖI TÌM FILE: Không tìm thấy file '{task['input_path']}'")
        print("👉 Hệ thống phát hiện các file Notebook sau đang có mặt trong thư mục 'notebooks/':")
        for file in os.listdir("notebooks"):
            if file.endswith(".ipynb") and file != "runs":
                print(f"   - {file}")
        print("\n💡 Cách khắc phục: Bác nhìn danh sách file thực tế ở trên xem có bị dư dấu cách, viết hoa hay sai số nào không, đổi tên lại cho chuẩn là chạy được ngay!")
        sys.exit(1)
        
    try:
        pm.execute_notebook(
            task['input_path'],
            task['output_path'],
            parameters=task['params'],
            kernel_name="python3"
        )
        print(f"✅ Xong {task['step']}!\n")
    except Exception as e:
        print(f"🚨 CÓ LỖI XẢY RA BÊN TRONG FILE {task['input_path']}:")
        print(e)
        sys.exit(1)

print("🎉 ĐÃ CHẠY THÀNH CÔNG TOÀN BỘ PIPELINE DỰ ÁN!")