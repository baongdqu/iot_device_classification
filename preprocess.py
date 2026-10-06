"""
Preprocess UNSW Smart Home IoT Traffic Dataset
================================================
Topic: IoT Device Classification (NT120)
Split Strategy: TIME-BASED SPLIT (Chia theo trục thời gian thực tế)
- Train: Các ngày trước (16-09-23, 16-09-24, 16-09-25)
- Test : Ngày tiếp theo (16-09-26 - Ngày tương lai chưa từng xuất hiện)
"""

import os
import re
import glob
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==================== CẤU HÌNH ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_preprocessed")
DEVICE_LIST_FILE = os.path.join(DATASET_DIR, "List_Of_Devices.txt")

ROUTER_GATEWAY_MAC = "14:cc:20:51:33:ea"
WINDOW_SECONDS = 10          # Cửa sổ thời gian gom gói tin (10s)

# Phân chia theo thời gian (Time-based split)
TRAIN_FILES = ["16-09-23.csv", "16-09-24.csv", "16-09-25.csv"]
TEST_FILES = ["16-09-26.csv"]

# Phân nhóm danh mục thiết bị (Device Categories)
DEVICE_CATEGORY_MAP = {
    # Camera
    "Dropcam": "Camera",
    "Nest Dropcam": "Camera",
    "Samsung SmartCam": "Camera",
    "Netatmo Welcome": "Camera",
    "TP-Link Day Night Cloud camera": "Camera",
    "Insteon Camera": "Camera",
    "Withings Smart Baby Monitor": "Camera",
    
    # Smart Plug & Switch
    "Belkin Wemo switch": "Smart Plug/Switch",
    "TP-Link Smart plug": "Smart Plug/Switch",
    "iHome": "Smart Plug/Switch",
    
    # Sensor & Alarm
    "Belkin wemo motion sensor": "Sensor/Alarm",
    "NEST Protect smoke alarm": "Sensor/Alarm",
    "Netatmo weather station": "Sensor/Alarm",
    
    # Health & Wellness
    "Withings Smart scale": "Health/Wellness",
    "Blipcare Blood Pressure meter": "Health/Wellness",
    "Withings Aura smart sleep sensor": "Health/Wellness",
    
    # Audio / Display
    "Amazon Echo": "Audio/Media",
    "Triby Speaker": "Audio/Media",
    "PIX-STAR Photo-frame": "Audio/Media",
    
    # Appliances / Hub
    "Light Bulbs LiFX Smart Bulb": "Smart Light",
    "HP Printer": "Printer",
    "Smart Things": "Hub",
    
    # Non-IoT Devices
    "Laptop": "Non-IoT",
    "MacBook": "Non-IoT",
    "Android Phone": "Non-IoT",
    "IPhone": "Non-IoT",
    "Samsung Galaxy Tab": "Non-IoT",
    "MacBook/Iphone": "Non-IoT",
}


def load_device_mapping(device_file):
    """Đọc file List_Of_Devices.txt để map MAC address sang tên thiết bị."""
    devices = {}
    with open(device_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("List of Devices"):
                continue
            match = re.search(r"([0-9a-f]{2}(?::[0-9a-f]{2}){5})", line, re.I)
            if match:
                mac = match.group(1).lower()
                name = line[:match.start()].strip()
                if not name:
                    name = "Insteon Camera"
                devices[mac] = name
    return devices


def extract_window_features(df, devices):
    """
    Trích xuất đặc trưng thống kê theo time-window (10 giây) cho mỗi thiết bị.
    """
    src_dev = df["eth.src"].map(devices)
    dst_dev = df["eth.dst"].map(devices)

    is_out = (df["eth.src"] != ROUTER_GATEWAY_MAC) & src_dev.notna()
    is_in = (df["eth.dst"] != ROUTER_GATEWAY_MAC) & dst_dev.notna()

    device_col = np.where(is_out, src_dev, np.where(is_in, dst_dev, None))
    df["device"] = device_col
    df["is_outbound"] = np.where(is_out, 1, 0)
    
    valid_df = df.dropna(subset=["device"]).copy()
    if valid_df.empty:
        return pd.DataFrame()

    valid_df["window_id"] = valid_df["TIME"] // WINDOW_SECONDS

    valid_df["is_tcp"] = (valid_df["IP.proto"] == 6).astype(int)
    valid_df["is_udp"] = (valid_df["IP.proto"] == 17).astype(int)
    valid_df["has_dns"] = ((valid_df["port.src"] == 53) | (valid_df["port.dst"] == 53)).astype(int)
    valid_df["has_http_https"] = (
        valid_df["port.src"].isin([80, 443]) | valid_df["port.dst"].isin([80, 443])
    ).astype(int)
    valid_df["has_ntp"] = ((valid_df["port.src"] == 123) | (valid_df["port.dst"] == 123)).astype(int)

    grouped = valid_df.groupby(["device", "window_id"])
    
    agg_df = grouped.agg(
        pkt_count=("Size", "count"),
        byte_count=("Size", "sum"),
        size_min=("Size", "min"),
        size_max=("Size", "max"),
        size_mean=("Size", "mean"),
        size_std=("Size", "std"),
        time_span=("TIME", lambda x: max(x) - min(x)),
        out_ratio=("is_outbound", "mean"),
        tcp_ratio=("is_tcp", "mean"),
        udp_ratio=("is_udp", "mean"),
        has_dns=("has_dns", "max"),
        has_http_https=("has_http_https", "max"),
        has_ntp=("has_ntp", "max"),
    ).reset_index()

    agg_df["size_std"] = agg_df["size_std"].fillna(0.0)
    agg_df["pkt_rate"] = agg_df["pkt_count"] / WINDOW_SECONDS
    agg_df["byte_rate"] = agg_df["byte_count"] / WINDOW_SECONDS
    agg_df["iat_approx"] = np.where(
        agg_df["pkt_count"] > 1,
        agg_df["time_span"] / (agg_df["pkt_count"] - 1),
        WINDOW_SECONDS
    )
    
    agg_df.drop(columns=["time_span"], inplace=True)
    return agg_df


def process_files(file_list, devices, split_name):
    """Đọc và trích xuất đặc trưng cho một danh sách file CSV."""
    extracted_dfs = []
    total_packets = 0

    print(f"\n[*] Đang xử lý tập [{split_name}]: {len(file_list)} files ({', '.join(file_list)})")
    for f in file_list:
        path = os.path.join(DATASET_DIR, f)
        t0 = time.time()
        df = pd.read_csv(path)
        total_packets += len(df)
        feat_df = extract_window_features(df, devices)
        extracted_dfs.append(feat_df)
        t1 = time.time()
        print(f"    - {f}: {len(df):,} gói tin -> {len(feat_df):,} mẫu cửa sổ ({t1 - t0:.2f}s)")

    combined_df = pd.concat(extracted_dfs, ignore_index=True)
    combined_df["category"] = combined_df["device"].map(DEVICE_CATEGORY_MAP).fillna("Other")
    combined_df["is_iot"] = (combined_df["category"] != "Non-IoT").astype(int)
    print(f"    => Tổng số mẫu [{split_name}]: {len(combined_df):,} (từ {total_packets:,} gói tin)")
    return combined_df, total_packets


def main():
    print("=" * 65)
    print("TIỀN XỬ LÝ & CHUẨN HÓA DỮ LIỆU IOT — TIME-BASED SPLIT")
    print("=" * 65)
    start_total = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Nạp thiết bị
    devices = load_device_mapping(DEVICE_LIST_FILE)
    print(f"[*] Đã nạp {len(devices)} địa chỉ MAC thiết bị.")

    # 2. Xử lý tập Train theo thời gian (các ngày trước)
    train_df, train_pkts = process_files(TRAIN_FILES, devices, "TRAIN (Quá khứ)")

    # 3. Xử lý tập Test theo thời gian (ngày sau)
    test_df, test_pkts = process_files(TEST_FILES, devices, "TEST (Tương lai)")

    # 4. Lọc thiết bị: Chỉ giữ các thiết bị đã xuất hiện trong tập Train
    train_devices = set(train_df["device"].unique())
    test_df = test_df[test_df["device"].isin(train_devices)].copy()

    # 5. Danh sách đặc trưng cần chuẩn hóa
    feature_cols = [
        "pkt_count", "byte_count", "size_min", "size_max", 
        "size_mean", "size_std", "out_ratio", "tcp_ratio", 
        "udp_ratio", "has_dns", "has_http_https", "has_ntp", 
        "pkt_rate", "byte_rate", "iat_approx"
    ]

    target_col = "device"
    category_col = "category"

    # 6. Chuẩn hóa đặc trưng: CHỈ fit trên tập Train, sau đó transform cho Train & Test
    print("\n[*] Đang fit StandardScaler trên tập Train (tránh rò rỉ dữ liệu)...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])

    # 7. Mã hóa nhãn (LabelEncoder) trên Train
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df[target_col])
    test_labels = label_encoder.transform(test_df[target_col])

    category_encoder = LabelEncoder()
    train_cat_labels = category_encoder.fit_transform(train_df[category_col])
    test_cat_labels = category_encoder.transform(test_df[category_col])

    # 8. Đóng gói DataFrame hoàn chỉnh
    scaled_feature_cols = [f"{col}_scaled" for col in feature_cols]

    train_out_df = pd.DataFrame(train_scaled, columns=scaled_feature_cols)
    train_out_df["device"] = train_df["device"].values
    train_out_df["device_label"] = train_labels
    train_out_df["category"] = train_df["category"].values
    train_out_df["category_label"] = train_cat_labels
    train_out_df["is_iot"] = train_df["is_iot"].values

    test_out_df = pd.DataFrame(test_scaled, columns=scaled_feature_cols)
    test_out_df["device"] = test_df["device"].values
    test_out_df["device_label"] = test_labels
    test_out_df["category"] = test_df["category"].values
    test_out_df["category_label"] = test_cat_labels
    test_out_df["is_iot"] = test_df["is_iot"].values

    # Lưu thêm giá trị thô để đối chiếu
    for col in feature_cols:
        train_out_df[f"raw_{col}"] = train_df[col].values
        test_out_df[f"raw_{col}"] = test_df[col].values

    # 9. Lưu ra file
    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
    le_path = os.path.join(OUTPUT_DIR, "label_encoder.joblib")
    cat_le_path = os.path.join(OUTPUT_DIR, "category_encoder.joblib")
    metadata_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")

    print(f"\n[*] Đang lưu các file đầu ra vào: {OUTPUT_DIR}")
    train_out_df.to_csv(train_path, index=False)
    test_out_df.to_csv(test_path, index=False)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, le_path)
    joblib.dump(category_encoder, cat_le_path)

    metadata = {
        "split_strategy": "time-based",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "window_seconds": WINDOW_SECONDS,
        "train_files": TRAIN_FILES,
        "test_files": TEST_FILES,
        "train_raw_packets": int(train_pkts),
        "test_raw_packets": int(test_pkts),
        "train_samples": int(len(train_out_df)),
        "test_samples": int(len(test_out_df)),
        "feature_columns": feature_cols,
        "scaled_columns": scaled_feature_cols,
        "device_classes": [str(c) for c in label_encoder.classes_],
        "category_classes": [str(c) for c in category_encoder.classes_],
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("HOÀN THÀNH TIỀN XỬ LÝ TIME-BASED SPLIT THÀNH CÔNG!")
    print(f"Tổng thời gian: {time.time() - start_total:.2f}s")
    print(f"- Train ({', '.join(TRAIN_FILES)}): {len(train_out_df):,} mẫu")
    print(f"- Test  ({', '.join(TEST_FILES)}): {len(test_out_df):,} mẫu")
    print(f"- Scaler: {scaler_path}")
    print(f"- Labels: {le_path}, {cat_le_path}")
    print(f"- Metadata: {metadata_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
