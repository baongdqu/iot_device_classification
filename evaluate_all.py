"""
Evaluate All Models - Stage 2: Machine Learning Model Training & Benchmark
==========================================================================
Topic: IoT Device Classification (NT120)
Dataset: UNSW Smart Home IoT Traffic Dataset
Strategy: Time-Based Split Evaluation

Models:
- Baseline 0: DummyClassifier (Majority Class / Floor Metric)
- Baseline 1: Logistic Regression (Multinomial Linear ML Baseline)
- Model 2   : Decision Tree (CART)
- Model 3   : Random Forest (Bagging Ensemble - Recommended)
- Model 4   : XGBoost (Gradient Boosting)
- Model 5   : K-Nearest Neighbors (KNN)
- Model 6   : Multilayer Perceptron (MLP - Neural Network)

Outputs:
- reports/benchmark_results.csv
- reports/figures/confusion_matrix_best.png
- reports/figures/feature_importance.png
- reports/figures/model_comparison.png
- models/best_model.joblib
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==================== CẤU HÌNH ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data_preprocessed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42
TARGET_TASK = "device_label"  # "device_label" (20 classes) hoặc "category_label" (8 classes)


def load_data():
    """Nạp dữ liệu train, test và metadata từ data_preprocessed/"""
    train_path = os.path.join(PREPROCESSED_DIR, "train.csv")
    test_path = os.path.join(PREPROCESSED_DIR, "test.csv")
    meta_path = os.path.join(PREPROCESSED_DIR, "dataset_metadata.json")

    print("[*] Đang tải tập Train và Test...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    scaled_cols = metadata["scaled_columns"]
    raw_feature_cols = metadata["feature_columns"]
    classes = metadata["device_classes"] if TARGET_TASK == "device_label" else metadata["category_classes"]

    X_train = train_df[scaled_cols].values
    y_train = train_df[TARGET_TASK].values

    X_test = test_df[scaled_cols].values
    y_test = test_df[TARGET_TASK].values

    print(f"    -> X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"    -> X_test : {X_test.shape}, y_test : {y_test.shape}")
    print(f"    -> Số lớp mục tiêu: {len(classes)} ({TARGET_TASK})")
    
    return X_train, y_train, X_test, y_test, scaled_cols, raw_feature_cols, classes


def get_models(n_classes):
    """Khởi tạo danh sách các mô hình từ Baseline đến nâng cao"""
    models = {
        # 1. Mốc sàn tối thiểu (Trivial Baseline)
        "Dummy (Majority)": DummyClassifier(strategy="most_frequent"),

        # 2. Đường cơ sở học máy tuyến tính (Linear ML Baseline)
        "Logistic Regression": LogisticRegression(
            max_iter=300, 
            solver="lbfgs", 
            random_state=RANDOM_STATE, 
            n_jobs=-1
        ),

        # 3. Cây quyết định đơn lẻ (Single Tree)
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15, 
            min_samples_split=10, 
            random_state=RANDOM_STATE
        ),

        # 4. Rừng ngẫu nhiên (Ensemble Bagging)
        "Random Forest": RandomForestClassifier(
            n_estimators=100, 
            max_depth=20, 
            random_state=RANDOM_STATE, 
            n_jobs=-1
        ),

        # 5. XGBoost (Ensemble Gradient Boosting)
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="mlogloss"
        ),

        # 6. K-Nearest Neighbors (Instance-based)
        "KNN (k=5)": KNeighborsClassifier(
            n_neighbors=5, 
            metric="euclidean", 
            n_jobs=-1
        ),

        # 7. Multilayer Perceptron (Neural Network)
        "MLP (Neural Net)": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=40,
            early_stopping=True,
            random_state=RANDOM_STATE
        )
    }
    return models


def evaluate_model(name, model, X_train, y_train, X_test, y_test, n_classes):
    """Huấn luyện và đo đạc toàn diện chỉ số cho một mô hình"""
    print(f"\n[{name}] Bắt đầu huấn luyện...")
    
    # Đối với KNN, vì tập train 160k x test 52k = 8.3 tỷ phép tính khoảng cách (rất lâu),
    # ta lấy mẫu ngẫu nhiên đại diện 30k train và 10k test để đo lường nhanh và chính xác
    if "KNN" in name:
        np.random.seed(RANDOM_STATE)
        idx_tr = np.random.choice(len(X_train), size=min(30000, len(X_train)), replace=False)
        idx_te = np.random.choice(len(X_test), size=min(10000, len(X_test)), replace=False)
        cur_X_train, cur_y_train = X_train[idx_tr], y_train[idx_tr]
        cur_X_test, cur_y_test = X_test[idx_te], y_test[idx_te]
    else:
        cur_X_train, cur_y_train = X_train, y_train
        cur_X_test, cur_y_test = X_test, y_test

    t0 = time.time()
    model.fit(cur_X_train, cur_y_train)
    train_time = time.time() - t0

    # Đo thời gian suy diễn (Inference latency)
    t_inf_start = time.time()
    y_pred = model.predict(cur_X_test)
    total_inf_time = time.time() - t_inf_start
    inf_latency_us = (total_inf_time / len(cur_X_test)) * 1_000_000  # micro giây / mẫu

    # Tính toán các chỉ số
    acc = accuracy_score(cur_y_test, y_pred)
    bal_acc = balanced_accuracy_score(cur_y_test, y_pred)
    macro_prec = precision_score(cur_y_test, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(cur_y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(cur_y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(cur_y_test, y_pred, average="weighted", zero_division=0)

    print(f"    -> Train Time    : {train_time:.2f}s")
    print(f"    -> Infer Latency : {inf_latency_us:.2f} µs/mẫu")
    print(f"    -> Accuracy      : {acc*100:.2f}%")
    print(f"    -> Macro F1      : {macro_f1*100:.2f}%")
    print(f"    -> Weighted F1   : {weighted_f1*100:.2f}%")

    metrics = {
        "Model": name,
        "Accuracy (%)": round(acc * 100, 2),
        "Balanced Acc (%)": round(bal_acc * 100, 2),
        "Macro Precision (%)": round(macro_prec * 100, 2),
        "Macro Recall (%)": round(macro_rec * 100, 2),
        "Macro F1 (%)": round(macro_f1 * 100, 2),
        "Weighted F1 (%)": round(weighted_f1 * 100, 2),
        "Train Time (s)": round(train_time, 2),
        "Infer Latency (µs)": round(inf_latency_us, 2)
    }

    return metrics, y_pred, cur_y_test


def plot_model_comparison(results_df):
    """Vẽ biểu đồ so sánh Accuracy và Macro F1 giữa các mô hình"""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.35

    rects1 = ax.bar(x - width/2, results_df["Accuracy (%)"], width, label="Accuracy (%)", color="#2b5c8f")
    rects2 = ax.bar(x + width/2, results_df["Macro F1 (%)"], width, label="Macro F1 (%)", color="#e27c38")

    ax.set_ylabel("Điểm phần trăm (%)", fontsize=12)
    ax.set_title("So sánh Hiệu năng các Mô hình (Time-based Test Set)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"], rotation=15, ha="right", fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Hiển thị số liệu trên cột
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}", (rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}", (rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(FIGURES_DIR, "model_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[*] Đã lưu biểu đồ so sánh mô hình: {chart_path}")


def plot_feature_importance(model, feature_names):
    """Vẽ biểu đồ độ quan trọng đặc trưng (Feature Importance) từ Random Forest"""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_names = [feature_names[i] for i in indices]
        sorted_vals = importances[indices]

        plt.figure(figsize=(10, 6))
        colors = sns.color_palette("viridis", len(sorted_names))
        sns.barplot(x=sorted_vals, y=sorted_names, palette=colors)
        plt.title("Độ quan trọng của các Đặc trưng (Feature Importance - Random Forest)", fontsize=13, fontweight="bold")
        plt.xlabel("Mức độ quan trọng tương đối", fontsize=11)
        plt.ylabel("Đặc trưng lưu lượng (10s Window)", fontsize=11)
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()

        chart_path = os.path.join(FIGURES_DIR, "feature_importance.png")
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f"[*] Đã lưu biểu đồ Feature Importance: {chart_path}")


def plot_confusion_matrix_heatmap(y_true, y_pred, classes, model_name):
    """Vẽ Heatmap Ma trận nhầm lẫn chuẩn hóa cho mô hình tốt nhất"""
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    target_names = [classes[i] if i < len(classes) else str(i) for i in unique_labels]

    cm = confusion_matrix(y_true, y_pred, labels=unique_labels, normalize="true")

    plt.figure(figsize=(14, 11))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=".2f", 
        cmap="Blues", 
        xticklabels=target_names, 
        yticklabels=target_names,
        cbar_kws={"label": "Tỷ lệ chuẩn hóa (Recall)"}
    )
    plt.title(f"Normalized Confusion Matrix — {model_name} (Time-Based Test)", fontsize=14, fontweight="bold")
    plt.xlabel("Nhãn Dự đoán (Predicted Device)", fontsize=12)
    plt.ylabel("Nhãn Thực tế (Actual Device)", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    chart_path = os.path.join(FIGURES_DIR, "confusion_matrix_best.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[*] Đã lưu biểu đồ Confusion Matrix: {chart_path}")


def main():
    print("=" * 70)
    print("GIAI ĐOẠN 2: HUẤN LUYỆN, ĐÁNH GIÁ & SO SÁNH CÁC MÔ HÌNH MACHINE LEARNING")
    print("=" * 70)
    start_all = time.time()

    # 1. Nạp dữ liệu
    X_train, y_train, X_test, y_test, scaled_cols, raw_feature_cols, classes = load_data()

    # 2. Khởi tạo danh sách mô hình
    models = get_models(len(classes))
    results = []
    trained_models = {}
    preds_dict = {}

    # 3. Huấn luyện và đánh giá từng mô hình
    for name, model in models.items():
        metrics, y_pred, cur_y_test = evaluate_model(
            name, model, X_train, y_train, X_test, y_test, len(classes)
        )
        results.append(metrics)
        trained_models[name] = model
        preds_dict[name] = (cur_y_test, y_pred)

    # 4. Tạo bảng so sánh kết quả
    results_df = pd.DataFrame(results)
    csv_results_path = os.path.join(REPORTS_DIR, "benchmark_results.csv")
    results_df.to_csv(csv_results_path, index=False)
    print(f"\n[*] Đã lưu bảng tổng kết số liệu vào: {csv_results_path}")

    # In bảng tổng kết ra màn hình terminal
    print("\n" + "=" * 70)
    print("BẢNG TỔNG KẾT KẾT QUẢ SO SÁNH CÁC MÔ HÌNH (GIAI ĐOẠN 2)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # 5. Xác định mô hình tốt nhất (dựa trên Macro F1 cao nhất)
    # Loại trừ Dummy
    valid_results = results_df[results_df["Model"] != "Dummy (Majority)"]
    best_row = valid_results.loc[valid_results["Macro F1 (%)"].idxmax()]
    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]
    print(f"\n🏆 MÔ HÌNH TỐT NHẤT: {best_model_name} (Macro F1 = {best_row['Macro F1 (%)']}%)")

    # 6. Lưu toàn bộ mô hình và mô hình tốt nhất
    for model_name, mdl in trained_models.items():
        slug = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        mdl_path = os.path.join(MODELS_DIR, f"{slug}.joblib")
        joblib.dump(mdl, mdl_path)
    
    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_model, best_model_path)
    print(f"[*] Đã lưu toàn bộ {len(trained_models)} mô hình vào: {MODELS_DIR}")
    print(f"    -> Mô hình tốt nhất: {best_model_path}")

    # 7. Trực quan hóa
    plot_model_comparison(results_df)

    rf_model = trained_models.get("Random Forest")
    if rf_model:
        plot_feature_importance(rf_model, raw_feature_cols)

    best_y_true, best_y_pred = preds_dict[best_model_name]
    plot_confusion_matrix_heatmap(best_y_true, best_y_pred, classes, best_model_name)

    # 8. In báo cáo chi tiết theo từng lớp của mô hình tốt nhất
    print("\n" + "=" * 70)
    print(f"BÁO CÁO CHI TIẾT TỪNG LỚP (CLASSIFICATION REPORT) — {best_model_name}")
    print("=" * 70)
    unique_labels = sorted(list(set(best_y_true) | set(best_y_pred)))
    target_names = [classes[i] if i < len(classes) else str(i) for i in unique_labels]
    print(classification_report(best_y_true, best_y_pred, labels=unique_labels, target_names=target_names, zero_division=0))

    total_time = time.time() - start_all
    print("=" * 70)
    print(f"HOÀN THÀNH GIAI ĐOẠN 2 THÀNH CÔNG TRONG {total_time:.2f} GIÂY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
