# Phân loại Thiết bị IoT trong mạng Smart Home bằng Machine Learning
### Đồ án môn học: NT120 — Nhập môn Trí tuệ Nhân tạo cho Mạng máy tính & An toàn thông tin

---

## 📌 Tổng quan dự án (Project Overview)
Dự án tập trung vào bài toán **nhận diện và phân loại thiết bị IoT** trong hệ thống mạng Smart Home dựa trên các đặc trưng thống kê lưu lượng mạng (Packet Metadata & Traffic Statistics) mà **không cần giải mã nội dung gói tin (Payload-agnostic)**, bảo toàn tính riêng tư và hoạt động hiệu quả ngay cả khi lưu lượng được mã hóa TLS/HTTPS.

- **Dữ liệu thực nghiệm:** Bộ dữ liệu đo kiểm thực tế [UNSW Smart Home IoT Traffic Dataset (IEEE TMC 2018)](https://iotanalytics.unsw.edu.au/iottraces) thu thập qua 20 ngày liên tục.
- **Chiến lược phân chia:** *Time-based Split* (Huấn luyện trên các ngày trước, kiểm thử trên ngày tương lai độc lập) nhằm phản ánh đúng kịch bản triển khai mạng thực tế và loại trừ triệt để hiện tượng Data Leakage.
- **Rút gọn đặc trưng (Feature Pruning):** Tối ưu hóa mô hình từ 15 đặc trưng xuống **Top 8 đặc trưng cốt lõi**, cho phép mô hình Decision Tree đạt **97.50% Accuracy** với dung lượng chỉ **~319 KB**, sẵn sàng nạp trực tiếp vào bộ nhớ RAM của các thiết bị Edge Router (OpenWrt, Raspberry Pi).
- **Ứng dụng thực tế:** Hệ thống web dashboard **IoT Sentinel Guard** (Flask + JavaScript + Chart.js) hỗ trợ kiểm kê thiết bị và phân loại lưu lượng thời gian thực.

---

## 📊 Tóm tắt kết quả thực nghiệm (Benchmark Summary)

### 1. So sánh các mô hình Machine Learning (Full 15 Features)
| Mô hình | Accuracy | Macro F1 | Weighted F1 | Training Time | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** ⭐ | **97.86%** | **79.51%** | **97.68%** | 10.37s | 62.0 MB |
| **XGBoost** | 97.80% | 78.47% | 97.63% | 2.59s | 4.5 MB |
| **Decision Tree** | 97.53% | 76.11% | 97.35% | 1.23s | 294 KB |
| **MLP Neural Net** | 97.10% | 68.96% | 96.88% | 85.04s | 289 KB |
| **k-NN (k=5)** | 97.05% | 68.64% | 96.86% | 0.05s | 7.9 MB |
| **Logistic Regression** | 87.97% | 40.54% | 86.85% | 7.94s | 3.7 KB |

### 2. Rút gọn đặc trưng cho Edge Computing (Feature Pruning)
| Bộ đặc trưng | Số lượng | Random Forest Acc | Random Forest F1 | Decision Tree Acc | Decision Tree F1 | Thời gian huấn luyện (DT) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full 15 Features** | 15 | 97.86% | 79.51% | 97.53% | 76.11% | 1.23s |
| **Top 8 Features** 🚀 | **8** | **97.66%** | **76.62%** | **97.50%** | **75.53%** | **0.49s** |
| **Top 5 Features** | 5 | 97.59% | 78.44% | 97.24% | 74.51% | 0.41s |

*Top 8 đặc trưng chọn lọc:* `size_max`, `size_mean`, `size_std`, `size_min`, `byte_count`, `byte_rate`, `has_http_https`, `pkt_rate`.

---

## 📁 Cấu trúc thư mục (Directory Structure)

```
├── application/                         # Mã nguồn Web Dashboard IoT Sentinel Guard
│   ├── app.py                           # Flask backend API
│   ├── static/                          # CSS giao diện & JS xử lý biểu đồ
│   └── templates/                       # Giao diện HTML Dashboard
├── course/                              # Tài liệu môn học & Notebook phân tích
│   └── NT120_DoAn_Nhom05_PhanLoaiThietBiIoT.ipynb
├── dataset/                             # Danh sách thiết bị và 20 file nén (.zip) dữ liệu gốc
│   ├── List_Of_Devices.txt              # Ánh xạ MAC Address sang tên thiết bị
│   └── *.csv.zip                        # 20 file zip dữ liệu lưu lượng từng ngày
├── data_preprocessed/                   # Dữ liệu train/test và các encoder/scaler
│   ├── train.csv                        # Tập huấn luyện (Time-based: 23-25/09)
│   ├── test.csv                         # Tập kiểm thử (Time-based: 26/09)
│   ├── scaler.joblib                    # Bộ chuẩn hóa dữ liệu
│   ├── label_encoder.joblib             # Bộ mã hóa nhãn thiết bị
│   └── category_encoder.joblib          # Bộ mã hóa danh mục nhóm thiết bị
├── models/                              # Các mô hình đã huấn luyện (.joblib)
│   ├── edge_decision_tree_top8.joblib   # Mô hình siêu nhẹ (~319 KB) cho Edge Router
│   ├── best_model.joblib                # Mô hình Random Forest / XGBoost tốt nhất
│   └── ...
├── reports/                             # Biểu đồ đánh giá & kết quả benchmark
│   ├── benchmark_results.csv            # Bảng so sánh chỉ số các mô hình
│   ├── confusion_matrix_best.png        # Ma trận nhầm lẫn
│   ├── feature_importance.png           # Biểu đồ tầm quan trọng đặc trưng
│   └── model_comparison.png             # Biểu đồ so sánh trực quan
├── preprocess.py                        # Pipeline tiền xử lý & trích xuất đặc trưng
├── evaluate_all.py                      # Kịch bản huấn luyện & đánh giá benchmark
├── tune_and_prune.py                    # Kịch bản rút gọn đặc trưng & tối ưu Edge AI
├── requirements.txt                     # Danh sách thư viện Python cần thiết
└── README.md                            # Hướng dẫn dự án
```

---

## 🚀 Hướng dẫn cài đặt và sử dụng (Getting Started)

### 1. Cài đặt môi trường
Khuyến nghị sử dụng Python 3.9+:

```bash
git clone https://github.com/baongdqu/iot_device_classification.git
cd iot_device_classification
pip install -r requirements.txt
```

### 2. Tiền xử lý dữ liệu (Nếu muốn tái tạo lại dữ liệu)
Giải nén các file dữ liệu trong `dataset/` (nếu cần xử lý từ dữ liệu thô) và chạy:

```bash
python preprocess.py
```

### 3. Huấn luyện và Đánh giá Mô hình
```bash
python evaluate_all.py
```

### 4. Tối ưu hóa và Rút gọn Đặc trưng cho Edge Router
```bash
python tune_and_prune.py
```

### 5. Khởi chạy Web Dashboard Giám sát Realtime
```bash
cd application
python app.py
```
Truy cập vào trình duyệt tại: `http://localhost:5000`

---

## 🛡️ License
Dự án được thực hiện phục vụ mục đích học tập và nghiên cứu trong khuôn khổ môn học NT120.
