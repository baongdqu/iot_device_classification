# KẾ HOẠCH GIAI ĐOẠN 3: TỐI ƯU HÓA, XÂY DỰNG ỨNG DỤNG & HOÀN THIỆN BÁO CÁO
## Đề tài: Phân loại thiết bị IoT trong mạng Smart Home bằng Machine Learning
**Môn học:** NT120 — Nhập môn Trí tuệ Nhân tạo cho Mạng máy tính và An toàn thông tin  
**Trạng thái:** Kế hoạch tiếp nối sau khi hoàn thành Giai đoạn 2 (Model Training & Evaluation).

---

## 1. Mục tiêu cốt lõi của Giai đoạn 3

Giai đoạn 3 là giai đoạn **đưa mô hình vào thực tế** và **hoàn thiện sản phẩm đồ án** để đạt điểm xuất sắc:
1. **Tối ưu hóa mô hình:** Tinh chỉnh siêu tham số (Hyperparameter Tuning) và rút gọn đặc trưng (Feature Selection) để mô hình có thể chạy được trên thiết bị phần cứng hạn chế (Router Gateway / Raspberry Pi).
2. **Xây dựng ứng dụng thực tế (Application Demo):** Xây dựng một Dashboard kiểm kê và giám sát thiết bị IoT tự động trong thư mục `application/`.
3. **Phân tích An toàn thông tin & Thảo luận:** Trả lời các câu hỏi về ứng dụng thực tế, độ trễ và khả năng chống chịu trước các kỹ thuật ẩn danh lưu lượng.
4. **Hoàn thiện tài liệu đồ án:** Điền toàn bộ nội dung khoa học vào notebook [NT120_DoAn_KhungMau.ipynb](file:///c:/Users/s3cr3t/My%20Drive%20%28baongdqu@gmail.com%29/zzz%20k%E1%BB%B9%20n%C4%83ng%20tech%20-%20ai%20%28separator%29/%28%20project%20%29%20machine%20learning%20classify%20iot/course/NT120_DoAn_KhungMau.ipynb) và chuẩn bị slide thuyết trình.

---

## 2. Bốn Nhiệm vụ Trọng tâm trong Giai đoạn 3

### Nhiệm vụ 3.1: Tinh chỉnh Siêu tham số & Rút gọn Đặc trưng (Optimization & Pruning)
1. **Hyperparameter Tuning:**
   - Sử dụng `RandomizedSearchCV` hoặc `GridSearchCV` với 3-fold Cross Validation trên tập Train để tìm bộ tham số tối ưu cho mô hình dẫn đầu Giai đoạn 2 (ví dụ: Random Forest / XGBoost).
2. **Feature Selection (Tìm tập đặc trưng tối thiểu):**
   - Từ 15 đặc trưng ban đầu, thực hiện phân tích mức độ suy giảm hiệu năng khi chỉ giữ lại **Top 6–8 đặc trưng quan trọng nhất** (ví dụ: `byte_rate`, `size_mean`, `out_ratio`, `tcp_ratio`, `iat_approx`, `has_dns`).
   - *Ý nghĩa thực tiễn:* Router mạng không cần tốn nhiều CPU để tính toán 15 đặc trưng, chỉ cần 6 đặc trưng vẫn đạt độ chính xác >95%.

---

### Nhiệm vụ 3.2: Xây dựng Ứng dụng Demo thực tế trong thư mục `application/`
*(Đáp ứng trực tiếp yêu cầu của Đề tài 18: "Discuss how this kind of model could support automatic device inventory on a managed network.")*

Xây dựng ứng dụng **IoT Smart Inventory & Traffic Monitor Dashboard** (bằng Streamlit hoặc Flask Web App):
- **Tính năng 1 — Automatic Device Inventory (Tự động lập danh mục):**
  - Người dùng tải lên file lưu lượng mạng (CSV hoặc luồng gói tin mô phỏng).
  - Hệ thống gom cửa sổ thời gian, tự động quét và hiển thị bảng danh sách các thiết bị IoT đang kết nối trong mạng (Tên thiết bị, Địa chỉ MAC, Danh mục thiết bị, Mức độ tin cậy/Confidence score).
- **Tính năng 2 — Rogue Device Detection (Cảnh báo thiết bị lạ):**
  - Nếu xuất hiện thiết bị có footprint lưu lượng bất thường hoặc độ tin cậy dự đoán < ngưỡng an toàn $\rightarrow$ Hệ thống cảnh báo đỏ: *Phát hiện thiết bị lạ không rõ nguồn gốc trong mạng LAN*.
- **Tính năng 3 — Live Traffic Analytics:**
  - Trực quan hóa tỷ lệ phần trăm băng thông của từng thiết bị trong mạng qua biểu đồ tròn (Pie chart) và biểu đồ cột (Bar chart).

---

### Nhiệm vụ 3.3: Phân tích An toàn Thông tin & Thảo luận (Discussion & Security Analysis)
*(Phục vụ mục 7 & 8 của đồ án)*
1. **Phân tích ca lỗi (Error Case Analysis):**
   - Đào sâu các mẫu dự đoán sai: Tại sao một số loại Camera bị nhầm sang Baby Monitor? Tại sao cảm biến môi trường đôi khi bị nhầm sang thiết bị báo cháy?
2. **Khía cạnh An toàn mạng & Quyền riêng tư:**
   - Mô hình có ưu điểm tuyệt đối là **bảo vệ quyền riêng tư người dùng** vì hoàn toàn không cần đọc Payload.
   - Bàn luận về nguy cơ tấn công đánh lừa mô hình (*Adversarial Traffic Padding*): Kẻ tấn công có thể chèn các gói tin rác (dummy packets) để giả mạo hành vi của một thiết bị khác nhằm qua mặt hệ thống kiểm kê.

---

### Nhiệm vụ 3.4: Hoàn thiện Báo cáo Đồ án & Chuẩn bị Thuyết trình
1. **Hoàn thiện Notebook mẫu [NT120_DoAn_KhungMau.ipynb](file:///c:/Users/s3cr3t/My%20Drive%20%28baongdqu@gmail.com%29/zzz%20k%E1%BB%B9%20n%C4%83ng%20tech%20-%20ai%20%28separator%29/%28%20project%20%29%20machine%20learning%20classify%20iot/course/NT120_DoAn_KhungMau.ipynb):**
   - Điền đầy đủ từ mục 1 đến mục 13 theo đúng quy chuẩn trường UIT.
   - Đảm bảo notebook chạy từ đầu đến cuối qua `Restart & Run All` mà không có lỗi.
2. **Đối chiếu Checklist tự đánh giá (Mục 12 của đề cương):**
   - [x] Có baseline rõ ràng (DummyClassifier + Logistic Regression).
   - [x] Có chiến lược chia dữ liệu hợp lý không rò rỉ (Time-based Split).
   - [x] Đánh giá đa chiều với F1-score và Confusion Matrix.
   - [x] Cố định random seed (SEED=42).
3. **Chuẩn bị Dàn ý Slide Thuyết trình (Mục 13 của đề cương):**
   - Bố cục 10-12 slides chuẩn cho buổi báo cáo đồ án trước giảng viên.

---

## 3. Tổng kết Lộ trình Toàn bộ 3 Giai đoạn của Đồ án

| Giai đoạn | Trọng tâm | Trạng thái | Sản phẩm chính |
| :---: | :--- | :---: | :--- |
| **Giai đoạn 1** | **Xác định bài toán & Tiền xử lý dữ liệu** | ✅ **ĐÃ XONG** | `preprocess.py`, `train.csv` (160k dòng), `test.csv` (52k dòng), Time-based Split, `scaler.joblib`. |
| **Giai đoạn 2** | **Huấn luyện, Đánh giá & So sánh mô hình** | ⏳ **SẴN SÀNG** | `evaluate_all.py`, 2 Baselines + 5 ML Models, Benchmark table, Confusion Matrix, Feature Importance. |
| **Giai đoạn 3** | **Tối ưu hóa, Ứng dụng Demo & Báo cáo** | 📋 **KẾ HOẠCH** | Web Dashboard kiểm kê trong `application/`, Hoàn thiện Notebook NT120, Slide báo cáo. |
