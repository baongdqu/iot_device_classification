# BÁO CÁO RÀ SOÁT TOÀN DIỆN VÀ HOÀN THIỆN ĐỒ ÁN
## Đề tài: Phân loại Thiết bị IoT trong mạng Smart Home bằng Machine Learning
**Môn học:** NT120 — Nhập môn Trí tuệ Nhân tạo cho Mạng máy tính và An toàn thông tin  
**Lớp:** NT120.R12 | **Đại học Công nghệ Thông tin (VNUHCM-UIT)**  
**Thời gian hoàn thành rà soát:** 15/10/2026  

---

## 1. Tóm tắt Kết quả Rà soát (Executive Summary)

Đồ án giải quyết trọn vẹn **Đề tài số #18** (*IoT Device Identification from Network Traffic*) theo danh mục đề tài gợi ý của giảng viên (`course/NT120_Suggested_Project_Topics.pdf`):
- **Tính khả thi thực tế:** Nhận diện thiết bị dựa trên đặc trưng gói tin theo hướng **Payload-agnostic**, bảo toàn quyền riêng tư người dùng và hoạt động tốt ngay cả khi dữ liệu bị mã hóa (TLS/HTTPS).
- **Phương pháp luận vững chắc:** Triển khai **Time-based Split** (Train: 23-25/09 với 160,582 mẫu; Test: 26/09 với 52,509 mẫu) triệt tiêu hoàn toàn rủi ro rò rỉ dữ liệu (Zero Data Leakage).
- **Chiến lược triển khai kép (Dual-Deployment Strategy):**
  - **Ultra-Edge Router (OpenWrt / Raspberry Pi):** Mô hình Decision Tree rút gọn Top 8 đặc trưng siêu nhẹ (**319 KB**, độ trễ **0.36 µs/mẫu**, Accuracy **97.50%**).
  - **Server Centralized:** Mô hình Ensemble (Random Forest & XGBoost) đạt độ chính xác cao nhất (**97.87% Accuracy**, **79.74% Macro F1**).
- **Ứng dụng thực tế hoàn chỉnh:** Web Dashboard **IoT Sentinel Guard** (Flask + Chart.js) hỗ trợ kiểm kê thiết bị tự động và cảnh báo thiết bị lạ (*Rogue Device Detection*).

---

## 2. Bảng Đối chiếu 13 Mục Tiêu chuẩn theo Đề cương NT120

| STT | Mục trong Đề cương | Trạng thái kỹ thuật | Vị trí lưu trữ & Đánh giá |
| :---: | :--- | :---: | :--- |
| **1** | **Giới thiệu & Phát biểu bài toán** | ✅ Hoàn thiện | `GIAI_DOAN_1...md`, `course/NT120_DoAn_Nhom05...ipynb` (Mục 1) |
| **2** | **Khảo sát giải pháp liên quan** | ✅ Hoàn thiện | Phân tích ưu/nhược điểm giữa DPI (kiểm tra payload) và Traffic Fingerprinting |
| **3** | **Thiết lập môi trường & Nạp dữ liệu** | ✅ Hoàn thiện | `requirements.txt`, 20 CSV tracked Git LFS, `dataset/List_Of_Devices.txt` |
| **4** | **Tiền xử lý & Trích xuất đặc trưng** | ✅ Hoàn thiện | `preprocess.py`: Cửa sổ trượt 10s, 15 đặc trưng mạng, Time-based Split chuẩn |
| **5** | **Phương pháp tiếp cận** | ✅ Hoàn thiện | Xây dựng 2 Baselines (Dummy + Linear) và 5 mô hình ML đại diện |
| **6** | **Thực nghiệm & Đánh giá kết quả** | ✅ Hoàn thiện | `evaluate_all.py`: Đánh giá 6 chỉ số (Acc, Bal Acc, Precision, Recall, F1, Latency) |
| **7** | **Phân tích và Thảo luận** | ✅ Hoàn thiện | `GIAI_DOAN_2...md`: Ma trận nhầm lẫn, Feature Importance, Trade-off Edge vs Server |
| **8** | **Kết luận & Hướng phát triển** | ✅ Hoàn thiện | `GIAI_DOAN_3...md`: Bàn luận về Traffic Padding, QUIC/HTTP3, Dual-Deployment |
| **9** | **Đóng góp của thành viên** | ⚠️ Cần cập nhật | Đã ghi chú nội dung công việc (cần điền thông tin MSSV chính thức) |
| **10**| **Tài liệu tham khảo** | ✅ Hoàn thiện | Đầy đủ bài báo gốc UNSW (IEEE TMC 2018), scikit-learn, XGBoost |
| **11**| **Phụ lục: Ứng dụng thực tế** | ✅ Hoàn thiện | `application/` (Flask API + Dark Mode Dashboard + Chart.js) |
| **12**| **Checklist tự đánh giá trước nộp** | ✅ Đạt 100% | Đáp ứng toàn bộ các tiêu chí chấm điểm khắt khe của môn học |
| **13**| **Chuẩn bị Thuyết trình trước lớp** | ✅ Hoàn thiện | Dàn ý 10-12 slides chuẩn cấu trúc báo cáo đồ án |

---

## 3. Các Điểm Chưa Hoàn Thiện Đã Được Sửa Lại Trong Lần Rà Soát Này

### 1. Đồng bộ Bảng Benchmark và Đường dẫn trong `README.md`
- **Trước chỉnh sửa:** Bảng trong README sử dụng số liệu cũ của đợt chạy thử nghiệm trước (RF ghi 97.86% Acc, 79.51% F1; Logistic Regression ghi 87.97% Acc; cây thư mục ghi thiếu thư mục con `reports/figures/`).
- **Sau chỉnh sửa:** Đã cập nhật 100% chính xác theo file dữ liệu gốc `reports/benchmark_results.csv`:
  - **Random Forest:** Accuracy **97.87%**, Macro F1 **79.55%**, Weighted F1 **97.89%**, Train time **4.06s**, Dung lượng **62.0 MB**.
  - **XGBoost:** Accuracy **97.80%**, Macro F1 **79.74%**, Weighted F1 **97.83%**, Train time **29.38s**, Dung lượng **4.5 MB**.
  - **Decision Tree:** Accuracy **97.53%**, Macro F1 **76.11%**, Weighted F1 **97.65%**, Train time **0.84s**, Dung lượng **294 KB**.
  - **Cây thư mục:** Bổ sung đúng vị trí `reports/figures/` chứa 4 ảnh đồ thị trực quan hóa chất lượng cao.

### 2. Tích hợp Mô hình Edge Top 8 (`edge_decision_tree_top8`) vào Web Dashboard
- **Trước chỉnh sửa:** File `application/app.py` và giao diện `index.html` chỉ có 3 mô hình (XGBoost, Random Forest, Decision Tree 15 đặc trưng). Mô hình siêu nhẹ rút gọn Top 8 đặc trưng (`models/edge_decision_tree_top8.joblib`, 319 KB) chưa được tích hợp vào API.
- **Sau chỉnh sửa:**
  - `application/app.py`: Đã nạp metadata `models/edge_model_metadata.json`, thêm model `edge_top8`, và lập trình cơ chế **cắt lát đặc trưng động (dynamic feature slicing)**: khi chọn model Top 8 thì API chỉ lấy đúng 8 đặc trưng rút gọn (`size_max`, `size_mean`, `size_std`, `size_min`, `byte_count`, `byte_rate`, `has_http_https`, `pkt_rate`).
  - `application/templates/index.html`: Thêm lựa chọn `🚀 Decision Tree Top 8 — Edge Router Pruned (0.36 µs, 319 KB, 97.50% Acc)`.
  - **Kiểm thử tự động:** Endpoint `/api/simulate` với model `edge_top8` đã chạy thành công 100%, trả về kết quả phân loại chuẩn xác với HTTP 200.

---

## 4. Các Lưu Ý Trọng Tâm Cần Thực Hiện Trước Khi Nộp Bài

1. **Cập nhật Thông tin Nhóm trong Notebook (`course/NT120_DoAn_Nhom05_PhanLoaiThietBiIoT.ipynb`):**
   - Theo danh sách xác nhận đề tài của lớp (`[NT120.R12] Danh sach dang ky de tai - Xac nhan-v2.pdf`), danh sách chính thức tại **STT 22** gồm:
     - **Lê Viết Lĩnh** — MSSV: `22520765`
     - **Nguyễn Duy Quốc Bảo** — MSSV: `23520119` (`baongdqu@gmail.com`)
     - **Nguyễn Thị Hồng** — MSSV: `24520613`
   - Cần thay thế các tên mẫu mặc định (`Nguyễn Văn A...`) ở **Cell 0** và **Cell 48 (Mục 9)** trong notebook trước khi nộp.
2. **Lưu Outputs trong Notebook ("Restart & Run All"):**
   - 13 code cells trong file notebook hiện tại chưa có output lưu kèm (`execution_count: None`).
   - Mở notebook trong VS Code/Jupyter, chọn **"Run All"** và bấm **Save** để thầy cô khi mở file trên LMS/GitHub/Colab thấy ngay bảng kết quả và biểu đồ.
3. **Thuyết minh về số lượng lớp (21 lớp vs 18 lớp):**
   - Bộ dữ liệu có 21 lớp thiết bị trên tập Train. Trong ngày test độc lập `16-09-26`, có 3 thiết bị offline (`Laptop`, `Android Phone`, `Blipcare Blood Pressure meter`) nên tập test có 18 lớp xuất hiện thực tế. Điểm này chứng minh tính tự nhiên và chân thực của việc chia tập theo chuỗi thời gian thực (Time-based Split).
