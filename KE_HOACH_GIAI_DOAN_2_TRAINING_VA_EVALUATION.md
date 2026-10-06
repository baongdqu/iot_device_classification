# KẾ HOẠCH GIAI ĐOẠN 2: HUẤN LUYỆN, ĐÁNH GIÁ & SO SÁNH MÔ HÌNH
## Đề tài: Phân loại thiết bị IoT trong mạng Smart Home bằng Machine Learning
**Môn học:** NT120 — Nhập môn Trí tuệ Nhân tạo cho Mạng máy tính và An toàn thông tin  
**Trạng thái đầu vào:** Đã hoàn thành Giai đoạn 1 (Bộ dữ liệu `train.csv` 160k dòng, `test.csv` 52k dòng theo Time-based Split).

---

## 1. Mục tiêu của Giai đoạn 2

1. **Xây dựng và so sánh đa dạng các mô hình Machine Learning:**
   - Huấn luyện 5 thuật toán tiêu biểu đại diện cho các trường phái học máy khác nhau.
   - So sánh định lượng để tìm ra mô hình tối ưu nhất cho bài toán phân loại thiết bị IoT.
2. **Thực nghiệm trên 2 cấp độ phân loại:**
   - **Nhiệm vụ 1 (Chính):** Phân loại chi tiết 20 dòng thiết bị IoT (`device_label`).
   - **Nhiệm vụ 2 (Mở rộng):** Phân loại 8 nhóm danh mục chức năng (`category_label`: *Camera, Smart Plug, Sensor, Audio, Non-IoT...*).
3. **Đánh giá đa chiều & Phân tích chuyên sâu:**
   - Đo lường không chỉ độ chính xác (Accuracy, F1-score) mà cả **thời gian huấn luyện** và **tốc độ suy diễn (Inference latency)** — tiêu chí sống còn khi triển khai thực tế trên Router/IoT Gateway.
   - Phân tích độ quan trọng của đặc trưng (Feature Importance): Xác định đặc tính lưu lượng nào có giá trị phân biệt thiết bị cao nhất.
   - Phân tích ma trận nhầm lẫn (Confusion Matrix) và giải thích nguyên nhân lỗi mạng.

---

## 2. Danh sách Mô hình Thử nghiệm (từ Baseline đến Nâng cao)

| STT | Mô hình | Nhóm thuật toán | Vai trò trong nghiên cứu |
| :---: | :--- | :--- | :--- |
| **0** | **DummyClassifier** | Trivial Baseline | **Mốc sàn tối thiểu (Floor Baseline)**: Luôn đoán lớp đa số (`most_frequent`). Nếu mô hình ML nào không vượt qua mốc này thì không có giá trị học. |
| **1** | **Logistic Regression** | Linear ML Baseline | **Đường cơ sở học máy (Linear Baseline)**: Phân loại tuyến tính đa lớp (Softmax), đo đạc hiệu năng nền tảng trước khi áp dụng mô hình phức tạp. |
| **2** | **Decision Tree (CART)** | Single Tree (Non-linear) | Mô hình phi tuyến cơ sở, tốc độ suy diễn tức thời, có thể trích xuất thành tập luật IF-THEN cho Router. |
| **3** | **Random Forest** | Ensemble Bagging | **Ứng viên sáng giá nhất**: Đạt hiệu năng hàng đầu trên dữ liệu dạng bảng/lưu lượng mạng, chống overfitting tốt, tính toán Feature Importance. |
| **4** | **XGBoost** | Ensemble Gradient Boosting | Tối ưu hóa hàm mất mát từng bước, xử lý tốt các quan hệ phi tuyến phức tạp giữa các đặc trưng mạng. |
| **5** | **K-Nearest Neighbors (KNN)** | Instance-based | Kiểm chứng giả thuyết các thiết bị cùng nhóm có footprint lưu lượng cụm lại trong không gian đặc trưng. |
| **6** | **Multilayer Perceptron (MLP)** | Neural Network (Deep Learning) | Mạng nơ-ron truyền thẳng đánh giá khả năng biểu diễn phi tuyến trên vector đặc trưng đã chuẩn hóa `StandardScaler`. |

---

## 3. Lộ trình triển khai chi tiết (6 Bước)

### Bước 2.1: Thiết lập Training & Evaluation Pipeline (`evaluate_all.py`)
- Đọc dữ liệu từ `data_preprocessed/train.csv` và `data_preprocessed/test.csv`.
- Tách ma trận đặc trưng $X$ (15 đặc trưng `*_scaled`) và nhãn mục tiêu $y$ (`device_label` hoặc `category_label`).
- Cố định `RANDOM_STATE = 42` để đảm bảo tính tái lập (Reproducibility) theo yêu cầu mục 3.1 của đề cương môn học NT120.

### Bước 2.2: Huấn luyện 5 mô hình & Đo đạc tài nguyên
Cấu hình siêu tham số ban đầu hợp lý cho từng mô hình:
- **Decision Tree:** `max_depth=15, min_samples_split=10`.
- **Random Forest:** `n_estimators=100, max_depth=20, n_jobs=-1`.
- **XGBoost:** `n_estimators=100, learning_rate=0.1, max_depth=6, eval_metric='mlogloss'`.
- **KNN:** `n_neighbors=5, metric='euclidean', n_jobs=-1`.
- **MLP:** `hidden_layer_sizes=(128, 64), activation='relu', max_iter=50, early_stopping=True`.

*Đo đạc 2 chỉ số thời gian:*
- $T_{train}$: Thời gian huấn luyện (giây).
- $T_{infer}$: Thời gian dự đoán trung bình trên 1 mẫu (microseconds/mẫu).

### Bước 2.3: Hệ thống chỉ số đánh giá (Evaluation Metrics)
Vì dữ liệu các thiết bị trong thực tế có sự chênh lệch số lượng mẫu (Imbalanced Data), việc chỉ nhìn vào Accuracy sẽ gây sai lệch. Kế hoạch sử dụng bộ chỉ số toàn diện:

1. **Accuracy (Độ chính xác tổng thể):** Tỷ lệ mẫu dự đoán đúng trên toàn bộ tập Test.
2. **Macro F1-score:** Trung bình cộng F1 của từng thiết bị (coi trọng công bằng tất cả các thiết bị, bất kể số lượng mẫu nhiều hay ít).
3. **Weighted F1-score:** Trung bình F1 có tính trọng số theo tần suất xuất hiện của thiết bị.
4. **Macro Precision & Macro Recall:** Đánh giá độ chuẩn xác và độ bao phủ của mô hình.

### Bước 2.4: Trực quan hóa Ma trận nhầm lẫn (Confusion Matrix Heatmap)
- Vẽ Heatmap ma trận nhầm lẫn chuẩn hóa (Normalized Confusion Matrix) cho mô hình tốt nhất.
- Phân tích các cặp thiết bị dễ bị nhầm lẫn nhất:
  - *Ví dụ:* Camera A và Camera B có bị nhầm lẫn khi cùng stream qua UDP port 554 (RTSP)?
  - Cảm biến môi trường và thiết bị báo cháy có bị nhầm khi cùng ngủ (sleep mode) phần lớn thời gian?

### Bước 2.5: Phân tích độ quan trọng của đặc trưng (Feature Importance)
- Trích xuất điểm quan trọng đặc trưng từ Random Forest và XGBoost.
- Xếp hạng 15 đặc trưng và vẽ biểu đồ Bar chart:
  - Nhóm đặc trưng nào quyết định nhất: Tốc độ (`byte_rate`, `pkt_rate`), Kích thước (`size_mean`, `size_std`), hay Dịch vụ mạng (`has_dns`, `has_http_https`, `has_ntp`)?
- Viết luận giải cho phần *7. Phân tích và thảo luận* của đồ án.

### Bước 2.6: Đóng gói và Lưu trữ Artifacts
1. Lưu mô hình chiến thắng có Macro F1 cao nhất vào `models/best_model.joblib`.
2. Tạo file `reports/benchmark_results.csv` và bảng tổng kết Markdown.
3. Xuất các biểu đồ đánh giá chất lượng cao (`confusion_matrix.png`, `feature_importance.png`, `model_comparison.png`) vào thư mục `reports/figures/`.

---

## 4. Dự kiến Kết quả bàn giao sau Giai đoạn 2

| Thành phần | Đường dẫn lưu trữ dự kiến | Mô tả |
| :--- | :--- | :--- |
| **Script thực thi** | `evaluate_all.py` | Script huấn luyện toàn bộ 5 mô hình và in báo cáo |
| **Mô hình tốt nhất** | `models/best_model.joblib` | Model được lưu để sẵn sàng inference trong tương lai |
| **Bảng tổng kết số liệu** | `reports/benchmark_results.csv` | Bảng so sánh 5 thuật toán theo 6 chỉ số |
| **Biểu đồ trực quan** | `reports/figures/*.png` | Confusion Matrix, Feature Importance, Model Comparison |
| **Báo cáo Giai đoạn 2** | `GIAI_DOAN_2_KET_QUA_THUC_NGHIEM.md` | Báo cáo chi tiết kèm phân tích chuyên môn cho đề tài |
