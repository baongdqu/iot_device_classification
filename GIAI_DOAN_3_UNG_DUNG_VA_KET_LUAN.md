# BÁO CÁO GIAI ĐOẠN 3: TỐI ƯU HÓA, XÂY DỰNG ỨNG DỤNG THỰC TẾ & KẾT LUẬN
## Đề tài: Phân loại thiết bị IoT trong mạng Smart Home bằng Machine Learning
**Môn học:** NT120 — Nhập môn Trí tuệ Nhân tạo cho Mạng máy tính và An toàn thông tin  
**Sản phẩm hoàn thiện:** Mô hình tối ưu hóa Edge Router & Ứng dụng Web Dashboard Giám sát IoT thời gian thực

---

## 1. Tối ưu hóa Mô hình & Rút gọn Đặc trưng (Feature Pruning)

### 1.1 Mục tiêu kỹ thuật
Để đưa mô hình Machine Learning xuống chạy trực tiếp trên các Router gia đình hoặc thiết bị Gateway (Raspberry Pi, OpenWrt Router) với phần cứng hạn chế (CPU đơn nhân 500MHz, RAM 128MB), ta cần trả lời câu hỏi:
> *Có thể cắt giảm bao nhiêu đặc trưng mà vẫn duy trì được độ chính xác >95%?*

### 1.2 Kết quả thực nghiệm rút gọn đặc trưng (Feature Pruning Benchmark)
Trích xuất từ file [reports/feature_pruning_results.csv](file:///c:/Users/s3cr3t/My%20Drive%20%28baongdqu@gmail.com%29/zzz%20k%E1%BB%B9%20n%C4%83ng%20tech%20-%20ai%20%28separator%29/%28%20project%20%29%20machine%20learning%20classify%20iot/reports/feature_pruning_results.csv):

| Bộ đặc trưng | Số lượng | Random Forest Acc | Random Forest F1 | Decision Tree Acc | Decision Tree F1 | Thời gian Train (DT) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full 15 Features** | 15 | **97.86%** | **79.51%** | **97.53%** | **76.11%** | 1.23s |
| **Top 8 Features** ⭐ | **8** | **97.66%** | **76.62%** | **97.50%** | **75.53%** | **0.49s** |
| **Top 5 Minimal Features** | 5 | 97.59% | 78.44% | 97.24% | 74.51% | 0.41s |

#### Phát hiện quan trọng:
1. Khi giảm từ **15 đặc trưng xuống Top 8 đặc trưng** (`size_max`, `size_mean`, `size_std`, `size_min`, `byte_count`, `byte_rate`, `has_http_https`, `pkt_rate`):
   - **Decision Tree** hầu như **không bị suy giảm độ chính xác** (Accuracy chỉ giảm **0.03%** từ 97.53% $\rightarrow$ 97.50%, Macro F1 chỉ giảm 0.58%).
   - Thời gian huấn luyện giảm hơn **60%** (chỉ còn **0.49 giây** trên 160,000 mẫu).
2. **Xuất bản mô hình siêu nhẹ cho Router:**
   - Đã lưu mô hình `models/edge_decision_tree_top8.joblib` với dung lượng chỉ **319 KB**, sẵn sàng nạp trực tiếp vào bộ nhớ RAM của bất kỳ Router nào.

---

## 2. Ứng dụng Thực tế: IoT Sentinel Guard Dashboard

Mã nguồn hoàn chỉnh nằm trong thư mục [application/](file:///c:/Users/s3cr3t/My%20Drive%20%28baongdqu@gmail.com%29/zzz%20k%E1%BB%B9%20n%C4%83ng%20tech%20-%20ai%20%28separator%29/%28%20project%20%29%20machine%20learning%20classify%20iot/application/):
- `app.py`: Máy chủ Flask API phân loại thời gian thực.
- `templates/index.html`: Giao diện Dashboard giao diện tối (Dark mode) chuẩn công nghiệp.
- `static/css/style.css`: Thiết kế hiệu ứng kính mờ (Glassmorphism), màu sắc tương phản cao.
- `static/js/app.js`: Tích hợp biểu đồ Chart.js và chế độ giả lập luồng gói tin tự động.

### Các tính năng chính của Ứng dụng:
1. **Lựa chọn Chế độ Triển khai Kép (Dual-Deployment Switch):**
   - ⚡ **Chế độ Edge Router (Decision Tree):** Độ trễ siêu tốc 0.36 µs, tiêu thụ RAM tối thiểu.
   - 🌲 **Chế độ Server Ensemble (Random Forest / XGBoost):** Độ chính xác tối đa, xác suất tin cậy mượt mà.
2. **Tự động Lập danh mục Thiết bị (Automatic Device Inventory):**
   - Tự động nhận diện thiết bị đang kết nối qua các cửa sổ 10 giây.
   - Gán nhãn phân nhóm danh mục: *Camera, Smart Plug, Sensor, Audio, Non-IoT*.
   - Hiển thị thanh đo mức độ tin cậy (Confidence score %).
3. **Cảnh báo Thiết bị Lạ (Rogue / Unknown Device Detection):**
   - Cho phép người dùng tùy chỉnh ngưỡng tin cậy (mặc định 60%).
   - Khi phát hiện một gói tin có hành vi bất thường hoặc mô hình chỉ dự đoán với độ tin cậy thấp $\rightarrow$ Hệ thống lập tức kích hoạt cảnh báo đỏ **"🚨 Thiết bị Lạ"**.
4. **Trực quan hóa Dòng lưu lượng (Real-time Analytics):**
   - Biểu đồ Doughnut tỷ lệ các loại thiết bị trong mạng.
   - Biểu đồ Bar chart so sánh lưu lượng Băng thông (Bytes/s) và Tốc độ gói (Pkts/s).
   - Chế độ **"Bật Giám sát Thời gian thực"** tự động cập nhật liên tục mỗi 2.5 giây.

---

## 3. Thảo luận & Hướng phát triển (Mục 8 Đề cương NT120)

### 3.1 Đóng góp của Đồ án
1. **Phương pháp luận vững chắc:** Áp dụng chiến lược **Time-based Split** (chia theo ngày thực tế), triệt tiêu hoàn toàn rủi ro rò rỉ dữ liệu (Temporal Data Leakage) so với Random Split.
2. **Khung đánh giá đa chiều:** Xây dựng đầy đủ 2 mô hình cơ sở (Floor Baseline + Linear Baseline) và 5 mô hình nâng cao, đo đạc cả chỉ số học thuật (Macro F1) lẫn chỉ số kỹ thuật mạng (Latency, Throughput, RAM).
3. **Sản phẩm hoàn chỉnh:** Không chỉ dừng lại ở file Jupyter Notebook, đồ án đã đóng gói thành **Ứng dụng Web Dashboard thực tế** phục vụ quản trị mạng Smart Home.

### 3.2 Hướng phát triển trong tương lai
- **Hỗ trợ giao thức QUIC / HTTP/3:** Khi lưu lượng chuyển dịch sang UDP-based encrypted protocols, cần bổ sung thêm các đặc trưng về chuỗi thời gian burst traffic.
- **Phát hiện tấn công giả mạo (Adversarial Robustness):** Nghiên cứu cơ chế chống lại kỹ thuật chèn gói tin rác (Traffic Padding) nhằm đánh lừa bộ phân loại IoT.
