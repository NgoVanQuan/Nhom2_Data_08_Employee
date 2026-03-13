import pandas as pd
import yaml

def load_config(config_path="configs/params.yaml"):
    """Đọc file cấu hình YAML."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def load_raw_data(config_path="configs/params.yaml"):
    """Đọc dữ liệu thô HR_Analytics từ đường dẫn trong config."""
    config = load_config(config_path)
    data_path = config["data"]["raw_path"]
    
    df = pd.read_csv(data_path)
    print(f"Đã load dữ liệu thành công! Kích thước dữ liệu: {df.shape}")
    return df