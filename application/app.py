"""
IoT Device Classification & Network Monitor - Flask Application
================================================================
Stage 3: Interactive Demo & Automatic Device Inventory
Supports:
- Dual Deployment Modes: Edge Router (Decision Tree) vs Server (Random Forest / XGBoost)
- Live Traffic Simulation from UNSW Test Set
- Automatic Device Inventory & Category Mapping
- Rogue / Unknown Device Detection (< Threshold Confidence)
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data_preprocessed")

app = Flask(__name__)

# Tải trước Scaler, Encoders và Metadata
print("[*] Đang khởi tạo ứng dụng IoT Monitor...")
scaler = joblib.load(os.path.join(PREPROCESSED_DIR, "scaler.joblib"))
label_encoder = joblib.load(os.path.join(PREPROCESSED_DIR, "label_encoder.joblib"))
category_encoder = joblib.load(os.path.join(PREPROCESSED_DIR, "category_encoder.joblib"))

with open(os.path.join(PREPROCESSED_DIR, "dataset_metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

scaled_cols = metadata["scaled_columns"]
raw_cols = metadata["feature_columns"]
device_classes = metadata["device_classes"]

# Tải sẵn các mô hình
models = {
    "xgboost": {
        "name": "XGBoost (Best Accuracy - Server Mode)",
        "type": "server",
        "model": joblib.load(os.path.join(MODELS_DIR, "xgboost.joblib")),
        "latency_us": 24.8,
        "features": scaled_cols
    },
    "random_forest": {
        "name": "Random Forest (Ensemble - Server Mode)",
        "type": "server",
        "model": joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib")),
        "latency_us": 9.6,
        "features": scaled_cols
    },
    "decision_tree": {
        "name": "Decision Tree (Ultra-Edge - Router Mode)",
        "type": "edge",
        "model": joblib.load(os.path.join(MODELS_DIR, "decision_tree.joblib")),
        "latency_us": 0.36,
        "features": scaled_cols
    }
}

# Ánh xạ thiết bị sang Category
DEVICE_CATEGORY_MAP = {
    "Dropcam": "Camera",
    "Nest Dropcam": "Camera",
    "Samsung SmartCam": "Camera",
    "Netatmo Welcome": "Camera",
    "TP-Link Day Night Cloud camera": "Camera",
    "Insteon Camera": "Camera",
    "Withings Smart Baby Monitor": "Camera",
    "Belkin Wemo switch": "Smart Plug/Switch",
    "TP-Link Smart plug": "Smart Plug/Switch",
    "iHome": "Smart Plug/Switch",
    "Belkin wemo motion sensor": "Sensor/Alarm",
    "NEST Protect smoke alarm": "Sensor/Alarm",
    "Netatmo weather station": "Sensor/Alarm",
    "Withings Smart scale": "Health/Wellness",
    "Blipcare Blood Pressure meter": "Health/Wellness",
    "Withings Aura smart sleep sensor": "Health/Wellness",
    "Amazon Echo": "Audio/Media",
    "Triby Speaker": "Audio/Media",
    "PIX-STAR Photo-frame": "Audio/Media",
    "Light Bulbs LiFX Smart Bulb": "Smart Light",
    "HP Printer": "Printer",
    "Smart Things": "Hub",
    "Laptop": "Non-IoT",
    "MacBook": "Non-IoT",
    "Android Phone": "Non-IoT",
    "IPhone": "Non-IoT",
    "Samsung Galaxy Tab": "Non-IoT",
    "MacBook/Iphone": "Non-IoT",
}

# Tải một tập mẫu từ Test Set để giả lập dữ liệu mạng thực tế
test_df = pd.read_csv(os.path.join(PREPROCESSED_DIR, "test.csv"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models", methods=["GET"])
def get_available_models():
    """Trả về danh sách các mô hình khả dụng và thông số"""
    model_list = []
    for k, v in models.items():
        model_list.append({
            "id": k,
            "name": v["name"],
            "type": v["type"],
            "latency_us": v["latency_us"]
        })
    return jsonify(model_list)


@app.route("/api/simulate", methods=["POST"])
def simulate_traffic():
    """
    Giả lập luồng gói tin đến Router, phân loại bằng mô hình đã chọn.
    """
    data = request.json or {}
    model_key = data.get("model_id", "decision_tree")
    sample_size = min(int(data.get("sample_size", 40)), 200)
    confidence_threshold = float(data.get("threshold", 0.60))

    if model_key not in models:
        model_key = "decision_tree"

    selected_model_info = models[model_key]
    model = selected_model_info["model"]

    # Lấy mẫu ngẫu nhiên từ tập Test
    sample_rows = test_df.sample(n=sample_size, random_state=int(time.time()) % 1000).copy()
    X_input = sample_rows[scaled_cols].values

    # Đo thời gian suy diễn thực tế
    t0 = time.time()
    predictions = model.predict(X_input)
    
    # Tính toán xác suất tin cậy (Confidence)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_input)
        confidences = np.max(probas, axis=1)
    else:
        # Với mô hình không có predict_proba (hoặc đơn giản)
        confidences = np.ones(len(predictions)) * 0.95

    infer_time_ms = (time.time() - t0) * 1000

    results = []
    category_counts = {}
    total_bandwidth_bytes = 0
    rogue_count = 0

    for i in range(len(sample_rows)):
        pred_label_id = predictions[i]
        dev_name = device_classes[pred_label_id] if pred_label_id < len(device_classes) else "Unknown"
        conf = float(confidences[i])
        cat = DEVICE_CATEGORY_MAP.get(dev_name, "Other")
        
        # Kiểm tra thiết bị lạ (Rogue Device) nếu confidence quá thấp
        is_rogue = conf < confidence_threshold
        if is_rogue:
            rogue_count += 1
            dev_display = f"⚠️ Cảnh báo thiết bị lạ ({conf*100:.1f}%)"
        else:
            dev_display = dev_name

        pkt_rate = float(sample_rows.iloc[i].get("raw_pkt_rate", 5.0))
        byte_rate = float(sample_rows.iloc[i].get("raw_byte_rate", 500.0))
        total_bandwidth_bytes += byte_rate * 10

        category_counts[cat] = category_counts.get(cat, 0) + 1

        results.append({
            "id": i + 1,
            "device": dev_display,
            "actual": sample_rows.iloc[i]["device"],
            "category": cat,
            "confidence": round(conf * 100, 1),
            "is_rogue": is_rogue,
            "pkt_rate": round(pkt_rate, 1),
            "byte_rate": round(byte_rate, 1),
            "tcp_ratio": round(float(sample_rows.iloc[i].get("raw_tcp_ratio", 0.8)) * 100, 1),
            "has_dns": bool(sample_rows.iloc[i].get("raw_has_dns", 0)),
        })

    # Thống kê tổng hợp
    unique_devices = len(set(r["device"] for r in results if not r["is_rogue"]))

    return jsonify({
        "status": "success",
        "model_used": selected_model_info["name"],
        "model_type": selected_model_info["type"],
        "infer_time_ms": round(infer_time_ms, 2),
        "avg_latency_us": round((infer_time_ms / len(sample_rows)) * 1000, 2),
        "total_samples": len(results),
        "unique_devices": unique_devices,
        "rogue_alerts": rogue_count,
        "category_counts": category_counts,
        "total_bandwidth_kb": round(total_bandwidth_bytes / 1024, 1),
        "devices": results
    })


if __name__ == "__main__":
    print("[*] Máy chủ IoT Dashboard sẵn sàng tại: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
