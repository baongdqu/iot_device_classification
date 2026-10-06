"""
Stage 3.1: Hyperparameter Tuning & Feature Pruning
===================================================
Topic: IoT Device Classification (NT120)
Mục tiêu:
1. Rút gọn đặc trưng (Feature Pruning): So sánh hiệu năng khi giảm từ 15 đặc trưng xuống Top 8 và Top 5 đặc trưng.
2. Tinh chỉnh siêu tham số (Tuning) cho mô hình chiến lược: Random Forest và Decision Tree.
3. Đo đạc sự đánh đổi giữa Dung lượng / Tốc độ và Độ chính xác để phục vụ triển khai nhúng trên Router.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data_preprocessed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")

RANDOM_STATE = 42

def main():
    print("=" * 70)
    print("GIAI ĐOẠN 3.1: RÚT GỌN ĐẶC TRƯNG & TINH CHỈNH SIÊU THAM SỐ")
    print("=" * 70)
    
    # 1. Nạp dữ liệu
    train_df = pd.read_csv(os.path.join(PREPROCESSED_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(PREPROCESSED_DIR, "test.csv"))
    with open(os.path.join(PREPROCESSED_DIR, "dataset_metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    all_scaled_cols = meta["scaled_columns"]
    raw_feature_cols = meta["feature_columns"]
    y_train = train_df["device_label"].values
    y_test = test_df["device_label"].values

    # 2. Xác định thứ hạng quan trọng của đặc trưng từ Random Forest
    rf_base = joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib"))
    importances = rf_base.feature_importances_
    ranked_indices = np.argsort(importances)[::-1]
    
    print("\n[*] Thứ hạng 15 đặc trưng theo Random Forest:")
    for rank, idx in enumerate(ranked_indices, 1):
        print(f"    {rank:2d}. {raw_feature_cols[idx]:20s} (Score: {importances[idx]:.4f})")

    # 3. Thử nghiệm Rút gọn Đặc trưng (Feature Pruning Experiments)
    # Các bộ đặc trưng: 15 (Full), Top 8 (Medium), Top 5 (Minimal)
    feature_sets = {
        "Full 15 Features": [all_scaled_cols[i] for i in ranked_indices[:15]],
        "Top 8 Features": [all_scaled_cols[i] for i in ranked_indices[:8]],
        "Top 5 Minimal Features": [all_scaled_cols[i] for i in ranked_indices[:5]]
    }

    pruning_results = []
    print("\n[*] Đánh giá mức độ suy giảm hiệu năng khi rút gọn đặc trưng:")
    for set_name, cols in feature_sets.items():
        X_tr = train_df[cols].values
        X_te = test_df[cols].values

        # Đánh giá trên Random Forest
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_tr, y_train)
        rf_train_time = time.time() - t0
        rf_pred = rf.predict(X_te)
        rf_acc = accuracy_score(y_test, rf_pred)
        rf_f1 = f1_score(y_test, rf_pred, average="macro", zero_division=0)

        # Đánh giá trên Decision Tree (Mô hình biên)
        t0 = time.time()
        dt = DecisionTreeClassifier(max_depth=15, min_samples_split=10, random_state=RANDOM_STATE)
        dt.fit(X_tr, y_train)
        dt_train_time = time.time() - t0
        dt_pred = dt.predict(X_te)
        dt_acc = accuracy_score(y_test, dt_pred)
        dt_f1 = f1_score(y_test, dt_pred, average="macro", zero_division=0)

        print(f"\n--- {set_name} ({len(cols)} đặc trưng) ---")
        print(f"    Đặc trưng sử dụng: {[raw_feature_cols[ranked_indices[i]] for i in range(len(cols))]}")
        print(f"    Random Forest -> Accuracy: {rf_acc*100:.2f}%, Macro F1: {rf_f1*100:.2f}%, Train: {rf_train_time:.2f}s")
        print(f"    Decision Tree -> Accuracy: {dt_acc*100:.2f}%, Macro F1: {dt_f1*100:.2f}%, Train: {dt_train_time:.2f}s")

        pruning_results.append({
            "Feature Set": set_name,
            "Num Features": len(cols),
            "RF Accuracy (%)": round(rf_acc * 100, 2),
            "RF Macro F1 (%)": round(rf_f1 * 100, 2),
            "DT Accuracy (%)": round(dt_acc * 100, 2),
            "DT Macro F1 (%)": round(dt_f1 * 100, 2),
        })

    pruning_df = pd.DataFrame(pruning_results)
    pruning_csv_path = os.path.join(REPORTS_DIR, "feature_pruning_results.csv")
    pruning_df.to_csv(pruning_csv_path, index=False)
    print(f"\n[*] Đã lưu bảng kết quả rút gọn đặc trưng: {pruning_csv_path}")

    # 4. Vẽ biểu đồ so sánh rút gọn đặc trưng
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pruning_df))
    width = 0.2
    ax.bar(x - 1.5*width, pruning_df["RF Accuracy (%)"], width, label="RF Accuracy", color="#2b5c8f")
    ax.bar(x - 0.5*width, pruning_df["RF Macro F1 (%)"], width, label="RF Macro F1", color="#e27c38")
    ax.bar(x + 0.5*width, pruning_df["DT Accuracy (%)"], width, label="DT Accuracy", color="#4ba860")
    ax.bar(x + 1.5*width, pruning_df["DT Macro F1 (%)"], width, label="DT Macro F1", color="#c24d4d")

    ax.set_ylabel("Điểm phần trăm (%)", fontsize=11)
    ax.set_title("Ảnh hưởng của việc Rút gọn Đặc trưng đến Hiệu năng Mô hình", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pruning_df["Feature Set"], fontsize=11)
    ax.set_ylim(60, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(loc="lower right", fontsize=10)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}", (p.get_x() + p.get_width()/2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    chart_path = os.path.join(FIGURES_DIR, "feature_pruning_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[*] Đã lưu biểu đồ: {chart_path}")

    # 5. Lưu mô hình siêu nhẹ dành cho Edge Router (Top 8 features)
    top8_cols = feature_sets["Top 8 Features"]
    top8_dt = DecisionTreeClassifier(max_depth=15, min_samples_split=10, random_state=RANDOM_STATE)
    top8_dt.fit(train_df[top8_cols].values, y_train)
    top8_dt_path = os.path.join(MODELS_DIR, "edge_decision_tree_top8.joblib")
    joblib.dump(top8_dt, top8_dt_path)
    
    top8_meta = {
        "num_features": 8,
        "selected_features": [raw_feature_cols[ranked_indices[i]] for i in range(8)],
        "scaled_columns": top8_cols,
        "test_accuracy": float(pruning_df.loc[1, "DT Accuracy (%)"]),
        "test_macro_f1": float(pruning_df.loc[1, "DT Macro F1 (%)"]),
        "file_size_kb": os.path.getsize(top8_dt_path) / 1024
    }
    with open(os.path.join(MODELS_DIR, "edge_model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(top8_meta, f, indent=4, ensure_ascii=False)

    print(f"\n[*] Đã xuất bản mô hình siêu nhẹ Edge Router: {top8_dt_path} ({os.path.getsize(top8_dt_path)/1024:.1f} KB)")
    print("=" * 70)
    print("HOÀN THÀNH GIAI ĐOẠN 3.1 THÀNH CÔNG!")
    print("=" * 70)

if __name__ == "__main__":
    main()
