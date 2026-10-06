// IoT Sentinel Guard - Frontend Logic & Chart.js Integration

let categoryChart = null;
let trafficChart = null;
let isStreaming = false;
let streamInterval = null;
let currentDevicesData = [];

document.addEventListener("DOMContentLoaded", () => {
    initCharts();
    bindEvents();
    fetchSimulationData();
});

function initCharts() {
    // 1. Category Breakdown Chart (Doughnut)
    const ctxCat = document.getElementById("categoryChart").getContext("2d");
    categoryChart = new Chart(ctxCat, {
        type: "doughnut",
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: [
                    "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", 
                    "#ec4899", "#06b6d4", "#64748b", "#f97316"
                ],
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "right",
                    labels: { color: "#9ca3af", font: { family: "Inter", size: 11 } }
                }
            },
            cutout: "68%"
        }
    });

    // 2. Traffic Volume Chart (Bar)
    const ctxTraffic = document.getElementById("trafficChart").getContext("2d");
    trafficChart = new Chart(ctxTraffic, {
        type: "bar",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Băng thông (Bytes/s)",
                    data: [],
                    backgroundColor: "rgba(59, 130, 246, 0.75)",
                    borderRadius: 6,
                    yAxisID: "y"
                },
                {
                    label: "Tốc độ gói (Pkts/s)",
                    data: [],
                    backgroundColor: "rgba(16, 185, 129, 0.8)",
                    borderRadius: 6,
                    yAxisID: "y1"
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    ticks: { color: "#9ca3af", font: { family: "Inter", size: 10 } },
                    grid: { color: "rgba(255, 255, 255, 0.04)" }
                },
                y: {
                    type: "linear",
                    position: "left",
                    ticks: { color: "#93c5fd", font: { size: 10 } },
                    grid: { color: "rgba(255, 255, 255, 0.04)" }
                },
                y1: {
                    type: "linear",
                    position: "right",
                    grid: { drawOnChartArea: false },
                    ticks: { color: "#6ee7b7", font: { size: 10 } }
                }
            },
            plugins: {
                legend: {
                    labels: { color: "#9ca3af", font: { family: "Inter", size: 11 } }
                }
            }
        }
    });
}

function bindEvents() {
    const btnSimulate = document.getElementById("btn-simulate");
    const btnAutoStream = document.getElementById("btn-auto-stream");
    const modelSelect = document.getElementById("model-select");
    const sampleSlider = document.getElementById("sample-slider");
    const sampleLabel = document.getElementById("sample-count-lbl");
    const thresholdSlider = document.getElementById("threshold-slider");
    const thresholdLabel = document.getElementById("threshold-lbl");
    const searchBox = document.getElementById("search-box");

    btnSimulate.addEventListener("click", () => fetchSimulationData());

    sampleSlider.addEventListener("input", (e) => {
        sampleLabel.textContent = e.target.value;
    });

    thresholdSlider.addEventListener("input", (e) => {
        thresholdLabel.textContent = `${e.target.value}%`;
    });

    modelSelect.addEventListener("change", () => {
        fetchSimulationData();
    });

    btnAutoStream.addEventListener("click", () => {
        isStreaming = !isStreaming;
        const streamText = document.getElementById("stream-text");
        if (isStreaming) {
            streamText.textContent = "⏹ Dừng Giám sát";
            btnAutoStream.style.background = "rgba(239, 68, 68, 0.25)";
            btnAutoStream.style.borderColor = "rgba(239, 68, 68, 0.5)";
            streamInterval = setInterval(fetchSimulationData, 2500);
        } else {
            streamText.textContent = "🔄 Bật Giám sát Thời gian thực";
            btnAutoStream.style.background = "";
            btnAutoStream.style.borderColor = "";
            clearInterval(streamInterval);
        }
    });

    searchBox.addEventListener("input", (e) => {
        const query = e.target.value.toLowerCase();
        renderTable(currentDevicesData.filter(d => 
            d.device.toLowerCase().includes(query) || 
            d.category.toLowerCase().includes(query)
        ));
    });
}

async function fetchSimulationData() {
    const modelId = document.getElementById("model-select").value;
    const sampleSize = document.getElementById("sample-slider").value;
    const threshold = parseInt(document.getElementById("threshold-slider").value) / 100;

    try {
        const res = await fetch("/api/simulate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model_id: modelId,
                sample_size: sampleSize,
                threshold: threshold
            })
        });

        const data = await res.json();
        if (data.status === "success") {
            updateDashboard(data);
        }
    } catch (err) {
        console.error("Lỗi khi nạp dữ liệu mô phỏng:", err);
    }
}

function updateDashboard(data) {
    // Cập nhật thẻ KPI
    document.getElementById("total-samples").textContent = data.total_samples;
    document.getElementById("unique-devices").textContent = data.unique_devices;
    
    const rogueBadge = document.getElementById("rogue-alerts");
    rogueBadge.textContent = data.rogue_alerts;
    rogueBadge.style.color = data.rogue_alerts > 0 ? "#ef4444" : "#10b981";

    const latencyEl = document.getElementById("latency-val");
    latencyEl.textContent = `${data.avg_latency_us} µs`;
    
    const modeEl = document.getElementById("latency-mode");
    if (data.model_type === "edge") {
        modeEl.textContent = "⚡ Chế độ Edge Router (Siêu tốc)";
        latencyEl.style.color = "#10b981";
    } else {
        modeEl.textContent = "🌲 Chế độ Server Ensemble (Chính xác cao)";
        latencyEl.style.color = "#3b82f6";
    }

    // Cập nhật biểu đồ Category
    const catLabels = Object.keys(data.category_counts);
    const catValues = Object.values(data.category_counts);
    categoryChart.data.labels = catLabels;
    categoryChart.data.datasets[0].data = catValues;
    categoryChart.update();

    // Cập nhật biểu đồ Traffic (Top 10 mẫu tiêu biểu)
    const topDevices = data.devices.slice(0, 8);
    trafficChart.data.labels = topDevices.map(d => d.device.replace("⚠️ Cảnh báo thiết bị lạ", "Unknown").substring(0, 14));
    trafficChart.data.datasets[0].data = topDevices.map(d => d.byte_rate);
    trafficChart.data.datasets[1].data = topDevices.map(d => d.pkt_rate);
    trafficChart.update();

    // Cập nhật Bảng danh mục
    currentDevicesData = data.devices;
    renderTable(currentDevicesData);
}

function renderTable(devices) {
    const tbody = document.getElementById("inventory-tbody");
    tbody.innerHTML = "";

    if (!devices || devices.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" class="text-center" style="padding: 24px; color: #9ca3af;">Không tìm thấy thiết bị nào phù hợp</td></tr>`;
        return;
    }

    devices.forEach((d) => {
        const tr = document.createElement("tr");

        let catClass = "cat-other";
        if (d.category === "Camera") catClass = "cat-camera";
        else if (d.category === "Smart Plug/Switch") catClass = "cat-plug";
        else if (d.category === "Sensor/Alarm") catClass = "cat-sensor";
        else if (d.category === "Audio/Media") catClass = "cat-audio";
        else if (d.category === "Non-IoT") catClass = "cat-non-iot";

        const statusBadge = d.is_rogue 
            ? `<span class="rogue-badge">🚨 Thiết bị Lạ</span>` 
            : `<span class="normal-badge">✓ Đã xác thực</span>`;

        const confColor = d.confidence > 80 ? "#10b981" : (d.confidence > 60 ? "#f59e0b" : "#ef4444");

        tr.innerHTML = `
            <td style="color: #6b7280; font-weight: 600;">${d.id}</td>
            <td style="font-weight: 600;">${d.device}</td>
            <td><span class="category-tag ${catClass}">${d.category}</span></td>
            <td>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${d.confidence}%; background: ${confColor};"></div>
                </div>
                <strong>${d.confidence}%</strong>
            </td>
            <td>${d.pkt_rate}</td>
            <td>${d.byte_rate.toLocaleString()} B/s</td>
            <td>
                <span style="font-size: 0.75rem; color: #9ca3af;">TCP: ${d.tcp_ratio}% | DNS: ${d.has_dns ? "Yes" : "No"}</span>
            </td>
            <td>${statusBadge}</td>
        `;
        tbody.appendChild(tr);
    });
}
