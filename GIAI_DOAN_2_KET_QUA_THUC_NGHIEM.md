# BÁO CÁO GIAI ĐOẠN 2: THỰC NGHIỆM, ĐÁNH GIÁ & SO SÁNH MÔ HÌNH
## Đề tài: Phân loại thiết bị IoT trong mạng Smart Home bằng Machine Learning
**Môn học:** NT120 — Nhập môn Trí tuệ Nhân tạo cho Mạng máy tính và An toàn thông tin  
**Chiến lược kiểm thử:** Time-based Split (Train trên 3 ngày đầu, Test trên ngày tiếp theo chưa từng xuất hiện)

---

## 1. Tổng quan Thực nghiệm Giai đoạn 2

- **Tập Train (3 ngày đầu):** 160,582 mẫu cửa sổ 10s (tương ứng 1,877,610 gói tin).
- **Tập Test (Ngày tiếp theo):** 52,509 mẫu cửa sổ 10s (tương ứng 449,646 gói tin).
- **Số lớp phân loại:** 20 dòng thiết bị IoT thực tế.
- **Số mô hình thử nghiệm:** 7 mô hình (gồm 2 Baselines và 5 giải thuật ML đại diện).
- **Cố định ngẫu nhiên:** `RANDOM_STATE = 42`.

---

## 2. Bảng Tổng kết Kết quả Benchmark Toàn diện

Số liệu được trích xuất từ file `reports/benchmark_results.csv`:

| Nhóm | Mô hình | Accuracy (%) | Balanced Acc (%) | Macro Precision (%) | Macro Recall (%) | Macro F1 (%) | Weighted F1 (%) | Train Time (s) | Độ trễ suy diễn (µs/mẫu) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trivial Baseline** | **Dummy (Majority)** | 16.45% | 5.56% | 0.91% | 5.56% | 1.57% | 4.65% | 0.08s | **0.01 µs** |
| **Linear Baseline** | **Logistic Regression** | 87.06% | 53.82% | 57.60% | 48.44% | 51.03% | 85.73% | 44.73s | 0.39 µs |
| **Single Tree** | **Decision Tree** | 97.53% | 86.56% | 80.56% | 77.91% | 76.11% | 97.65% | **0.84s** | 0.36 µs |
| **Ensemble (Bagging)**| **Random Forest** ⭐ | **97.87%** | 87.40% | **85.76%** | 78.66% | 79.55% | **97.89%** | 4.06s | 9.63 µs |
| **Ensemble (Boosting)**| **XGBoost** 🏆 | 97.80% | **88.35%** | 83.63% | **79.51%** | **79.74%** | 97.83% | 29.38s | 24.83 µs |
| **Instance-based** | **KNN (k=5)** | 96.46% | 76.81% | 67.95% | 69.13% | 68.07% | 96.58% | 0.46s | 287.89 µs |
| **Neural Network** | **MLP (Deep Net)** | 97.06% | 85.80% | 74.99% | 73.54% | 73.97% | 97.21% | 116.45s | 6.53 µs |

---

## 3. Phân tích Khoa học & Thảo luận Kết quả (Mục 7 Đề cương)

### 3.1 Ý nghĩa của 2 Mô hình Baseline
1. **DummyClassifier (Floor Baseline = 16.45% Acc, 1.57% F1):**
   - Chỉ ra rằng nếu chỉ đoán mò theo lớp chiếm đa số (`most_frequent`), mô hình hoàn toàn thất bại trên các lớp thiểu số.
2. **Logistic Regression (Linear Baseline = 87.06% Acc, 51.03% F1):**
   - Chứng minh các đặc trưng thống kê cửa sổ 10s có khả năng phân tách tuyến tính khá tốt (đạt 87%).
   - Tuy nhiên, **Macro F1 chỉ đạt 51.03%** cho thấy các ranh giới tuyến tính không thể bắt trọn mối tương quan phức tạp giữa các thiết bị có lưu lượng tương đồng.
3. **Bước nhảy vọt của Ensemble Models (+28.7% Macro F1):**
   - Khi chuyển sang **Random Forest (79.55% F1)** và **XGBoost (79.74% F1)**, hiệu năng tăng vọt rõ rệt, chứng minh tính ưu việt của cây quyết định tổ hợp trong việc học các quy luật phi tuyến của giao thức mạng.

---

### 3.2 So sánh giữa Random Forest và XGBoost
- **Về độ chính xác:** Cả hai mô hình tương đương nhau (~97.8% Accuracy, ~79.7% Macro F1). XGBoost nhỉnh hơn về Balanced Accuracy (88.35% vs 87.40%).
- **Về tốc độ huấn luyện:** **Random Forest vượt trội hoàn toàn**, chỉ mất **4.06 giây** để train trên 160k mẫu (nhờ tận dụng tối đa đa luồng `n_jobs=-1`), trong khi XGBoost mất 29.38 giây.
- **Về độ trễ suy diễn (Inference Latency):** Random Forest chỉ mất **9.63 µs/mẫu** (nhanh hơn 2.5 lần so với XGBoost: 24.83 µs/mẫu).
- **Kết luận:** **Random Forest là mô hình cân bằng tối ưu nhất**, trong khi XGBoost là đối trọng mạnh mẽ cho các tác vụ phân tích chuyên sâu tại máy chủ trung tâm.

---

### 3.3 Phân tích Đánh đổi Kỹ thuật (Engineering Trade-off): Tại sao không chỉ chọn Decision Tree?

Khi nhìn vào bảng số liệu, **Decision Tree (CART)** thể hiện thông số tốc độ cực kỳ ấn tượng:
* **Độ trễ suy diễn (Inference Latency):** Chỉ **0.36 µs/mẫu** — nhanh hơn Random Forest **27 lần** và nhanh hơn XGBoost **69 lần**.
* **Thông lượng (Throughput):** Đạt tới **~2,770,000 mẫu/giây** trên CPU đơn nhân.
* **Kích thước mô hình:** Chỉ **294 KB** (so với **62 MB** của Random Forest, nhẹ hơn **210 lần**).

#### Bảng so sánh Đánh đổi Kỹ thuật (Trade-off Matrix)

| Tiêu chí kỹ thuật | Decision Tree (Single Tree) | Random Forest / XGBoost (Ensemble) |
| :--- | :---: | :---: |
| **Độ trễ suy diễn (Latency)** | **0.36 µs / mẫu** (Cực nhanh) ⚡ | 9.63 µs – 24.83 µs / mẫu |
| **Thông lượng xử lý (Throughput)** | **~2.77 triệu mẫu/giây** 🚀 | ~40k – 103k mẫu/giây |
| **Dung lượng file trên đĩa** | **294 KB** (Siêu nhẹ) 💾 | 4.5 MB – 62 MB |
| **Mức tiêu thụ RAM khi chạy** | Rất thấp (< 2 MB) | Lớn (phải nạp 100 cây vào bộ nhớ) |
| **Khả năng nhúng (Embedded/Edge)** | **Tối ưu:** Dễ biên dịch thành mã C `if-else` trên kernel Linux / eBPF của Router. | Khó khăn, đòi hỏi thư viện Machine Learning hỗ trợ. |
| **Độ bền vững (Variance)** | **Kém (High Variance):** Dễ bị rẽ sai nhánh khi thiết bị cập nhật firmware hoặc đổi cổng mạng. | **Rất cao (Low Variance):** Nhờ cơ chế biểu quyết tổ hợp (Bagging/Boosting). |
| **Macro F1-score (Thiết bị hiếm)** | **76.11%** (Dễ đoán sót các lớp ít mẫu) | **~79.7% (+3.6% F1)** (Nhận diện tốt lớp thiểu số) |
| **Chất lượng phân bố xác suất** | Rời rạc (thường 0% hoặc 100%), khó đặt ngưỡng cảnh báo thiết bị lạ. | Mượt mà (Calibrated Probabilities), hỗ trợ tốt cho phát hiện thiết bị bất thường. |

#### Kết luận Chiến lược Triển khai Thực tế (Dual-Deployment Strategy)
Trong thực tế hệ thống mạng Smart Home, không có một mô hình nào là tốt nhất cho mọi ngữ cảnh, mà cần áp dụng **chiến lược triển khai kép (Dual-Deployment)**:
1. **Kịch bản 1 — Ultra-Edge Deployment (Trực tiếp trên Router LAN Gateway):**
   - **Lựa chọn hàng đầu: Decision Tree.**
   - Với tài nguyên nghèo nàn của Router gia đình (CPU MIPS/ARM 500MHz, RAM 128MB), Decision Tree cho phép phân loại gói tin tức thời ở tốc độ đường truyền (Line-rate speed: 0.36 µs), không gây nghẽn mạng hay tràn RAM, trong khi vẫn đạt Accuracy rất cao (**97.53%**).
2. **Kịch bản 2 — Centralized / Cloud Monitoring (Tại máy chủ quản trị mạng tập trung):**
   - **Lựa chọn hàng đầu: Random Forest / XGBoost.**
   - Khi dữ liệu lưu lượng được tổng hợp về máy chủ quản lý mạng tập trung (có CPU/GPU mạnh), việc chấp nhận độ trễ vài microsecond để đổi lấy **Macro F1 tăng thêm +3.6%** và khả năng xuất điểm tin cậy (Confidence score) chuẩn xác là hoàn toàn xứng đáng.

---

### 3.4 Phân tích Chi tiết Từng Lớp (Per-class Performance — XGBoost)

- **Các thiết bị đạt độ chính xác gần như tuyệt đối (F1 = 97% – 100%):**
  - `Smart Things` (Hub): F1 = **1.00** (Hoạt động polling định kỳ rất đặc trưng).
  - `Dropcam` (Camera): F1 = **1.00** (Lưu lượng video stream liên tục, băng thông cao ổn định).
  - `Amazon Echo` (Loa thông minh): F1 = **0.99** (Đặc trưng gói audio và NTP/DNS handshake chuẩn xác).
  - `Netatmo Welcome`, `Withings Baby Monitor`, `Samsung SmartCam`: F1 = **0.97 – 0.99**.
  - `HP Printer`: F1 = **0.98**.
- **Các thiết bị có độ khó cao (Thách thức):**
  - `Belkin Wemo switch` (F1 = 0.76) và `Belkin wemo motion sensor` (F1 = 0.83): Hai thiết bị này cùng của hãng Belkin Wemo, chia sẻ chung kiến trúc firmware, giao thức UPnP/SSDP và mẫu giao tiếp cloud, dẫn đến tỷ lệ nhầm lẫn qua lại nhẹ.
  - `NEST Protect smoke alarm` (F1 = 0.60): Thiết bị báo cháy ở trạng thái ngủ phần lớn thời gian (rất ít gói tin), chỉ phát sinh lưu lượng kiểm tra ngắt quãng (chỉ có 5 mẫu trong ngày test).

---

### 3.5 Phân tích Độ quan trọng của Đặc trưng (Feature Importance)
Biểu đồ trích xuất từ Random Forest đã chỉ ra Top các đặc trưng quyết định nhất:
1. **`size_mean` & `size_std` (Kích thước gói trung bình & độ lệch chuẩn):** Là dấu vân tay đặc trưng nhất — camera video stream có packet size lớn (~1400 bytes), cảm biến chỉ gửi gói nhỏ (~70–120 bytes).
2. **`byte_rate` & `pkt_rate` (Băng thông & tốc độ gói):** Phân biệt tức thời giữa thiết bị truyền dữ liệu đa phương tiện (loa, camera) với thiết bị điều khiển/cảm biến.
3. **`out_ratio` (Tỷ lệ chiều gửi Outbound/Inbound):** Camera chủ yếu gửi ra ngoài (`out_ratio` cao), thiết bị nghe nhạc nhận stream về (`out_ratio` thấp).
4. **`has_dns` & `tcp_ratio` / `udp_ratio`:** Nhận diện kiến trúc kết nối và dịch vụ mạng.

---

## 4. Các Tệp Artifacts đã xuất ra

1. **Bảng dữ liệu kết quả:** [reports/benchmark_results.csv](file:///c:/Users/s3cr3t/My%20Drive%20%28baongdqu@gmail.com%29/zzz%20k%E1%BB%B9%20n%C4%83ng%20tech%20-%20ai%20%28separator%29/%28%20project%20%29%20machine%20learning%20classify%20iot/reports/benchmark_results.csv)
2. **Biểu đồ so sánh mô hình:** `reports/figures/model_comparison.png`
3. **Biểu đồ Feature Importance:** `reports/figures/feature_importance.png`
4. **Biểu đồ Ma trận nhầm lẫn chuẩn hóa:** `reports/figures/confusion_matrix_best.png`
5. **Mô hình tốt nhất được lưu:** `models/best_model.joblib`
