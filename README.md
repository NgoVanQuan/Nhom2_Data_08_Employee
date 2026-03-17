# ĐỒ ÁN KHAI PHÁ DỮ LIỆU: PHÂN TÍCH HIỆU SUẤT VÀ RỦI RO NGHỈ VIỆC CỦA NHÂN VIÊN (HR ANALYTICS)

## 1. Giới thiệu Dự án
Dự án này ứng dụng phương pháp Data-Driven vào Quản trị Nhân sự (HR), nhằm giải quyết bài toán chảy máu chất xám trong doanh nghiệp. Chúng tôi kết hợp các kỹ thuật Khai phá dữ liệu và Học máy để:
- Trực quan hóa và tìm hiểu nguyên nhân nhân viên rời tổ chức (EDA).
- Khai phá các luật kết hợp (FP-Growth) kích hoạt quyết định nghỉ việc.
- Gom cụm nhân sự (K-Means) để xây dựng hồ sơ rủi ro.
- Xây dựng mô hình Học máy (Random Forest, XGBoost) và thử nghiệm Học bán giám sát (Semi-supervised) để dự đoán sớm nhân viên có ý định nghỉ.

## 2. Cấu trúc Thư mục (Chuẩn MLOps)
- `configs/`: Chứa file `params.yaml` quản lý tham số tập trung (đường dẫn, test_size, min_support...).
- `data/`: Lưu trữ dữ liệu.
  - `raw/`: Dữ liệu thô từ Kaggle (IBM HR Analytics).
  - `processed/`: Dữ liệu đã làm sạch và số hóa.
- `notebooks/`: Các file Jupyter Notebook chạy từng bước pipeline (EDA, Feature Engineering).
- `src/`: Mã nguồn lõi (Core logic).
  - `data/`: `loader.py`, `cleaner.py` (Đọc và tiền xử lý).
  - `mining/`: Thuật toán K-Means và FP-Growth.
  - `models/`: Huấn luyện mô hình Supervised và Semi-supervised.
- `outputs/`: Chứa các mô hình đã được trích xuất (`.pkl`) và biểu đồ kết quả.

## 3. Hướng dẫn Cài đặt & Sử dụng
**Bước 1: Cài đặt thư viện**
Dự án yêu cầu Python 3.10+. Khởi tạo môi trường và chạy lệnh:
`pip install -r requirements.txt`

**Bước 2: Tiền xử lý & Khám phá dữ liệu**
Chạy lần lượt các file trong thư mục `notebooks/`:
- `01_eda.ipynb`: Trực quan hóa dữ liệu.
- `02_preprocess_feature.ipynb`: Làm sạch và tạo đặc trưng.

**Bước 3: Chạy Huấn luyện Mô hình**
Thực thi trực tiếp mã nguồn khai phá và mô hình học máy:
- Khai phá luật kết hợp: `python src/mining/association.py`
- Dự đoán Học có giám sát: `python src/models/supervised.py`
- Thử nghiệm Bán giám sát: `python src/models/semi_supervised.py`

## 4. Insight Kỹ thuật Nổi bật
- **Imbalanced Data:** Dữ liệu mất cân bằng nặng (84% Ở lại - 16% Nghỉ việc), mô hình XGBoost cho thấy F1-score vượt trội hơn so với Random Forest.
- **Actionable Insights:** Phát hiện "Nhóm Lính mới" và nhóm "Lương thấp + Làm thêm giờ" là các điểm nóng rủi ro cần HR can thiệp ngay lập tức bằng các chính sách luân chuyển và phúc lợi cổ phiếu (ESOP).
- **Semi-Supervised Failure:** Thực nghiệm chứng minh thuật toán Bán giám sát (Pseudo-labeling) không hiệu quả và sụp đổ (F1 = 0.0) nếu áp dụng trực tiếp lên dữ liệu mất cân bằng chưa qua xử lý SMOTE.