# BÁO CÁO GIAI ĐOẠN 1: XÁC ĐỊNH BÀI TOÁN & TIỀN XỬ LÝ DỮ LIỆU
## Đề tài: Phân loại thiết bị IoT trong mạng Smart Home bằng Machine Learning
**Môn học:** NT120 — Nhập môn Trí tuệ Nhân tạo cho Mạng máy tính và An toàn thông tin  
**Bộ dữ liệu sử dụng:** UNSW Smart Home IoT Traffic Dataset (IEEE TMC 2018)

---

## 1. Xác định và Phát biểu bài toán (Problem Formulation)

### 1.1 Bối cảnh & Động lực
Trong các mạng gia đình thông minh (Smart Home) và mạng doanh nghiệp hiện đại, sự gia tăng nhanh chóng của các thiết bị IoT (camera giám sát, loa thông minh, cảm biến, phích cắm thông minh, thiết bị y tế gia đình...) tạo ra thách thức lớn cho công tác quản trị và an toàn thông tin:
- **Khó khăn trong kiểm kê thiết bị (Device Inventory):** Quản trị viên khó nắm bắt được chính xác thiết bị nào đang kết nối vào hệ thống mạng.
- **Nguy cơ an ninh mạng:** Thiết bị IoT thường có năng lực tính toán hạn chế, firmware ít được cập nhật, dễ bị xâm nhập và biến thành botnet (như Mirai).
- **Thách thức mã hóa:** Hầu hết lưu lượng IoT hiện nay đều sử dụng TLS/HTTPS, khiến các kỹ thuật kiểm tra sâu gói tin (DPI — Deep Packet Inspection) truyền thống trở nên vô hiệu hoặc vi phạm quyền riêng tư của người dùng.

> **Mục tiêu cốt lõi:** Xây dựng mô hình Machine Learning có khả năng **nhận diện chính xác loại thiết bị IoT** đang kết nối mạng chỉ dựa trên **thống kê lưu lượng mạng (Packet Metadata & Traffic Characteristics)** mà **hoàn toàn không cần giải mã nội dung gói tin (Payload)**.

---

### 1.2 Phát biểu bài toán chi tiết
- **Dạng bài toán:** Học có giám sát (Supervised Learning) — Bài toán phân loại đa lớp (Multi-class Classification).
- **Đầu vào (Input):** Chuỗi gói tin mạng thu thập được trên cổng Gateway của mạng LAN (gồm timestamp, kích thước gói tin, địa chỉ MAC, địa chỉ IP, protocol, cổng dịch vụ).
- **Đầu ra (Output):** Nhãn nhận diện thiết bị, được thiết kế theo 3 mức độ phục vụ các ứng dụng thực tế khác nhau:
  1. **Fine-grained Device Classification:** Nhận diện chính danh từng model thiết bị (20 lớp: *Amazon Echo, Dropcam, Smart Things, Samsung SmartCam, TP-Link Cloud Camera, Withings Baby Monitor, Belkin Wemo Switch, Netatmo Weather Station...*).
  2. **Coarse-grained Category Classification:** Nhận diện nhóm chức năng của thiết bị (8 nhóm: *Camera, Smart Plug/Switch, Sensor/Alarm, Health/Wellness, Audio/Media, Smart Light, Hub, Non-IoT*).
  3. **Binary Identification:** Phân biệt thiết bị IoT thông minh với thiết bị thông thường (*IoT vs Non-IoT* như Laptop, Smartphone).

---

## 2. Khảo sát & Phân tích hiện trạng bộ dữ liệu thô (Raw Dataset)

### 2.1 Nguồn gốc dữ liệu
Bộ dữ liệu được thu thập từ mô hình Smart Home Testbed thực tế của Đại học New South Wales (UNSW Sydney, Úc) trong công trình nghiên cứu công bố tại *IEEE Transactions on Mobile Computing (TMC 2018)*.
- **Thư mục lưu trữ:** `dataset/`
- **Tập tin:** Gồm 20 file CSV (tương ứng 20 ngày thu thập liên tục từ `16-09-23.csv` đến `16-10-12.csv`) và file ánh xạ `dataset/List_Of_Devices.txt`.

### 2.2 Các hạn chế của dữ liệu thô ban đầu (Trước khi tiền xử lý)
Khi kiểm tra các file CSV thô, dữ liệu chưa thể đưa vào huấn luyện mô hình Machine Learning vì các nguyên nhân:
1. **Chưa có nhãn (Unlabeled):** Dữ liệu thô chỉ chứa MAC address (`eth.src`, `eth.dst`), chưa được map với tên thiết bị.
2. **Nhiễu mức gói tin (Packet-level noise):** Mỗi dòng là 1 gói tin đơn lẻ. Ở mức này, sự chồng lấn giữa các thiết bị là rất lớn (ví dụ mọi thiết bị đều có gói tin TCP ACK 66 bytes hoặc TLS handshake 443), khiến mô hình khó phân biệt nếu không có ngữ cảnh thời gian.
3. **Chưa chuẩn hóa đại lượng đo:** Kích thước (`Size`) dao động từ 40 đến 1500 bytes; thời gian (`TIME`) là số nguyên Unix timestamp lớn (~1.47 tỷ); cổng mạng (`port.src`, `port.dst`) là giá trị định danh dịch vụ nhưng đang ở dạng số nguyên.
4. **Địa chỉ dạng văn bản:** Địa chỉ MAC và IP là chuỗi Hex/dấu chấm, không phù hợp để làm vector đặc trưng trực tiếp.

```
+----------------------------------------------------------------------------------------------------+
|                                    DỮ LIỆU GÓI TIN THÔ BAN ĐẦU                                    |
| Packet ID | TIME       | Size | eth.src           | eth.dst           | IP.proto | port.src | port.dst|
| 1         | 1474552802 | 70   | 18:b7:9e:02:20:44 | 14:cc:20:51:33:ea | 6        | 40234    | 5228    |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼ (Quy trình Tiền xử lý)
+----------------------------------------------------------------------------------------------------+
|                                VECTOR ĐẶC TRƯNG SAU TIỀN XỬ LÝ (10s Window)                        |
| pkt_count | byte_count | size_mean | iat_approx | out_ratio | tcp_ratio | has_dns | device_label (Y)   |
| 14.0      | 1,280      | 91.4      | 0.71s      | 0.85      | 1.0       | 0       | 0 (Amazon Echo)    |
+----------------------------------------------------------------------------------------------------+
```

---

## 3. Quy trình Tiền xử lý & Trích xuất đặc trưng (Pre-processing Pipeline)

Quy trình đã được lập trình tự động hóa hoàn toàn trong file `preprocess.py`.

### Bước 1: Gán nhãn thiết bị & Xác định hướng lưu lượng (Labeling & Traffic Direction)
- Sử dụng địa chỉ MAC của Router Gateway LAN (`14:cc:20:51:33:ea`) làm mốc:
  - Nếu `eth.src` khớp với danh sách thiết bị và khác Gateway $\rightarrow$ Lưu lượng gửi đi (**Outbound**, `is_outbound = 1`). Thiết bị gửi là nhãn mục tiêu.
  - Nếu `eth.dst` khớp với danh sách thiết bị và khác Gateway $\rightarrow$ Lưu lượng nhận về (**Inbound**, `is_outbound = 0`). Thiết bị nhận là nhãn mục tiêu.
- Lọc bỏ các gói tin quảng bá (broadcast) không xác định hoặc gói nội bộ giữa các Router.

### Bước 2: Gom cụm theo cửa sổ thời gian (Time-Window Aggregation - 10s)
Thay vì học trên từng packet, dữ liệu của mỗi thiết bị được gom theo cửa sổ trượt $T = 10$ giây (`window_id = TIME // 10`). Trong mỗi cửa sổ 10 giây, tính toán **15 đặc trưng thống kê chuyên sâu**:

| STT | Tên đặc trưng | Mô tả ý nghĩa mạng |
| :---: | :--- | :--- |
| 1 | `pkt_count` | Tổng số lượng gói tin trong cửa sổ 10 giây |
| 2 | `byte_count` | Tổng dung lượng (bytes) truyền/nhận trong cửa sổ |
| 3 | `size_min` | Kích thước gói tin nhỏ nhất (bytes) |
| 4 | `size_max` | Kích thước gói tin lớn nhất (bytes) |
| 5 | `size_mean` | Kích thước gói tin trung bình (bytes) |
| 6 | `size_std` | Độ lệch chuẩn kích thước gói (độ biến thiên payload) |
| 7 | `pkt_rate` | Tốc độ truyền gói (`pkt_count / 10s`) |
| 8 | `byte_rate` | Băng thông tiêu thụ (`byte_count / 10s`) |
| 9 | `iat_approx` | Khoảng cách thời gian trung bình giữa 2 gói liên tiếp (Inter-Arrival Time) |
| 10 | `out_ratio` | Tỷ lệ gói tin gửi đi so với gói nhận về (`Outbound / Total`) |
| 11 | `tcp_ratio` | Tỷ lệ gói tin sử dụng giao thức TCP |
| 12 | `udp_ratio` | Tỷ lệ gói tin sử dụng giao thức UDP |
| 13 | `has_dns` | Cờ nhị phân: Thiết bị có phát sinh truy vấn DNS (Port 53) hay không |
| 14 | `has_http_https`| Cờ nhị phân: Thiết bị có dùng web/API traffic (Port 80/443) hay không |
| 15 | `has_ntp` | Cờ nhị phân: Thiết bị có đồng bộ thời gian mạng (Port 123) hay không |

### Bước 3: Chiến lược phân chia Time-based Split (Chia theo trục thời gian)
Thay vì sử dụng Random Sampling (vốn gây ra hiện tượng rò rỉ dữ liệu chuỗi thời gian - *Temporal Data Leakage*), đề tài lựa chọn chiến lược **Time-based Split**:
- **Tập Train (Quá khứ):** Gồm 3 ngày đầu tiên liên tiếp (`16-09-23.csv`, `16-09-24.csv`, `16-09-25.csv`).
  - Tổng số gói tin thô đã xử lý: **1,877,610 gói**.
  - Tổng số mẫu đặc trưng cửa sổ 10s: **160,582 mẫu**.
- **Tập Test (Tương lai):** Ngày kế tiếp (`16-09-26.csv`).
  - Tổng số gói tin thô đã xử lý: **449,646 gói**.
  - Tổng số mẫu đặc trưng cửa sổ 10s: **52,509 mẫu**.

> **Lý do lựa chọn Time-based Split:**
> 1. Phản ánh đúng 100% kịch bản thực tế: Huấn luyện mô hình từ dữ liệu đã qua để dự đoán cho lưu lượng tương lai.
> 2. Đánh giá tính bền vững (Generalization) của mô hình trước sự thay đổi hành vi theo ngày của người dùng và thiết bị.
> 3. Tuyệt đối không để 2 cửa sổ 10s gần kề nhau bị chia một nửa vào Train, một nửa vào Test.

### Bước 4: Chuẩn hóa đặc trưng & Mã hóa nhãn (Zero Data Leakage)
1. **Chuẩn hóa giá trị số (`StandardScaler`):**
   $$\mu_{train}, \sigma_{train} = \text{Fit}(X_{train})$$
   $$X_{train\_scaled} = \frac{X_{train} - \mu_{train}}{\sigma_{train}}, \quad X_{test\_scaled} = \frac{X_{test} - \mu_{train}}{\sigma_{train}}$$
   *Lưu ý quan trọng:* Scaler chỉ học tham số thống kê trên tập Train, sau đó áp dụng biến đổi cho cả Train và Test, đảm bảo không có bất kỳ thông tin nào từ tập Test bị rò rỉ vào quá trình chuẩn hóa.
2. **Mã hóa nhãn (`LabelEncoder`):**
   - Mã hóa 20 thiết bị thành các giá trị số nguyên $0, 1, \dots, 19$.
   - Mã hóa 8 danh mục chức năng thành các số nguyên $0, 1, \dots, 7$.

---

## 4. Tổng kết Kết quả & Cấu trúc Dữ liệu đầu ra

Sau khi thực thi `preprocess.py`, toàn bộ kết quả chuẩn hóa đã được lưu trữ hoàn chỉnh tại thư mục `data_preprocessed/`:

```
data_preprocessed/
├── train.csv                # Tập huấn luyện (160,582 dòng, gồm cả raw và scaled features)
├── test.csv                 # Tập kiểm tra tương lai (52,509 dòng)
├── scaler.joblib            # StandardScaler đã fit sẵn để tái sử dụng
├── label_encoder.joblib     # LabelEncoder chuyển đổi tên thiết bị
├── category_encoder.joblib  # LabelEncoder chuyển đổi danh mục thiết bị
└── dataset_metadata.json    # Metadata mô tả cấu trúc, thông số kỹ thuật và các lớp
```

---

## 5. Định hướng Giai đoạn 2 (Model Training & Evaluation)
Với dữ liệu đã được làm sạch và chuẩn hóa tối ưu:
1. Huấn luyện các mô hình phân loại đa lớp cơ sở: **Decision Tree, Random Forest, K-Nearest Neighbors (KNN), Multilayer Perceptron (MLP), XGBoost**.
2. Đánh giá đa chiều với các chỉ số đo lường chuẩn: **Accuracy, Macro Precision, Macro Recall, Macro F1-score**.
3. Vẽ đồ thị **Confusion Matrix** để phân tích cặp thiết bị nào dễ bị nhầm lẫn và lý do kỹ thuật (ví dụ cùng sử dụng chung máy chủ cloud hoặc cùng giao thức streaming).
