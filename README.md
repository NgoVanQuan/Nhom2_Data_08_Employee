# MINI PROJECT: PHÂN CỤM KHÁCH HÀNG DỰA TRÊN LUẬT KẾT HỢP

## 1. Giới thiệu
Dự án này thực hiện **phân cụm khách hàng dựa trên hành vi mua sắm**, sử dụng kết hợp giữa:

- **Luật kết hợp (Association Rules – Apriori/FP-Growth)** để khai thác các mẫu sản phẩm thường được mua cùng nhau  
- **Thuật toán K-Means** để phân nhóm khách hàng có hành vi tương đồng

Khác với cách tiếp cận truyền thống chỉ dựa trên RFM (Recency – Frequency – Monetary), dự án sử dụng **các luật kết hợp làm đặc trưng đầu vào cho mô hình phân cụm**, giúp phản ánh rõ hơn hành vi mua theo combo, bundle hoặc cross-sell của khách hàng.

Mục tiêu cuối cùng là **xây dựng hồ sơ khách hàng (Customer Profiling)** và đề xuất **chiến lược marketing phù hợp cho từng nhóm**.

---

## 2. Mục tiêu dự án
- Hiểu và thực hành quy trình khai phá dữ liệu trong Data Mining  
- Áp dụng luật kết hợp để phân tích hành vi mua sắm  
- Thực hiện Feature Engineering từ các luật kết hợp  
- Áp dụng thuật toán **K-Means** để phân cụm khách hàng  
- Trực quan hóa kết quả và diễn giải đặc điểm từng cụm  
- Đề xuất chiến lược marketing cho từng phân khúc khách hàng  

---

## 3. Quy trình thực hiện (Pipeline)

### Bước 1: Tiền xử lý & Khai phá luật
- Làm sạch dữ liệu giao dịch  
- Áp dụng thuật toán **Apriori hoặc FP-Growth**  
- Lựa chọn các luật có giá trị **Support, Confidence và Lift cao**

### Bước 2: Trích xuất đặc trưng
- Mỗi luật mạnh được xem như một đặc trưng hành vi  
- Với mỗi khách hàng, xây dựng vector đặc trưng dựa trên mức độ xuất hiện của các luật liên quan  
- Kết quả là ma trận đặc trưng khách hàng – luật kết hợp  

### Bước 3: Phân cụm khách hàng
- Sử dụng thuật toán **K-Means**  
- Xác định số cụm phù hợp bằng **Elbow Method** hoặc **Silhouette Score**  
- Trong dự án này, số cụm tối ưu được chọn là **K = 2**

### Bước 4: Phân tích & Trực quan hóa
- Phân tích các luật chiếm ưu thế trong từng cụm  
- Trực quan hóa kết quả bằng biểu đồ (PCA, biểu đồ phân bố)  
- Diễn giải đặc điểm hành vi mua sắm của từng nhóm khách hàng  

---

## 4. Cấu trúc thư mục

- `data/`: Chứa dữ liệu thô và dữ liệu đã qua xử lý (CSV, Parquet).
- `notebooks/`: Chứa các Jupyter Notebook thực hiện từng bước của dự án.
- `src/`: Chứa thư viện `cluster_library.py` - lõi xử lý của dự án.
- `app.py`: Ứng dụng Streamlit Dashboard.
- `report.html`: Báo cáo kết quả dự án dưới dạng HTML tĩnh (không cần chạy server).
- `BLOG.md`: Bài viết Blog tóm tắt quá trình và kết quả dự án.
- `run_papermill.py`: Script tự động chạy toàn bộ pipeline notebook.
- `final_analysis.py`: Script thực hiện so sánh hệ thống và profiling cuối cùng.


## 5. Thành phần chính của Project

| Thành phần | Mô tả |
|-----------|------|
| `cluster_library.py` | Chuyển luật kết hợp thành vector đặc trưng và thực hiện K-Means |
| `clustering_from_rules.ipynb` | Notebook chạy toàn bộ pipeline: từ luật → phân cụm → trực quan |
| `app.py` | Dashboard Streamlit hiển thị kết quả phân cụm và gợi ý marketing |

---

## 6. Hướng dẫn cài đặt & sử dụng

### 1. Cài đặt môi trường
Khuyên dùng Python 3.10+.
```bash
pip install -r requirements.txt
```

### 2. Chạy toàn bộ Pipeline
Bạn có thể chạy toàn bộ các bước phân tích tự động qua Papermill:
```bash
python run_papermill.py
```

### 3. Chạy Dashboard
Để xem kết quả phân cụm và gợi ý Marketing tương tác:
```bash
python -m streamlit run app.py
```


## 7. Kết quả phân cụm & Chiến lược Marketing

Dữ liệu khách hàng được chia thành 2 cụm chính:

🔹 Cụm 1 – Khách hàng thân thiết

- Mua hàng thường xuyên

- Giá trị chi tiêu cao

- Có xu hướng mua theo combo hoặc sản phẩm liên quan

- Chiến lược đề xuất:

- Bundle sản phẩm

- Upsell sản phẩm cao cấp

- Chương trình khách hàng thân thiết

🔹 Cụm 0 – Khách hàng tiềm năng / ít hoạt động

- Ít mua hoặc đã lâu chưa quay lại

- Giá trị đơn hàng thấp

- Chiến lược đề xuất:

- Gửi mã giảm giá cá nhân hóa

- Email re-engagement

- Gợi ý lại các sản phẩm từng quan tâm

## 8. Kết luận

Dự án cho thấy việc kết hợp Association Rules và Clustering giúp phân tích hành vi khách hàng sâu hơn so với phương pháp truyền thống. Kết quả phân cụm có ý nghĩa thực tế và có thể ứng dụng trực tiếp trong các chiến dịch marketing.