#!/bin/bash
source venv/bin/activate

# Thư mục lưu lịch sử
LOG_DIR="history"
mkdir -p "$LOG_DIR"

# Danh sách các bộ dữ liệu cần chạy
# DATASETS=("BGL" "Hadoop" "HDFS" "Spirit" "Thunderbird")
DATASETS=("BGL")

echo "Starting job pipeline for all datasets..."

for DATASET in "${DATASETS[@]}"; do
    echo "========================================================="
    echo "[$DATASET] Processing Flow"
    echo "========================================================="
    
    GRAPH_SCRIPT="GraphGeneration_${DATASET}.py"
    MAIN_SCRIPT="main_${DATASET}.py"
    
    GRAPH_LOG="${LOG_DIR}/GraphGeneration_${DATASET}.log"
    MAIN_LOG="${LOG_DIR}/main_${DATASET}.log"

    # Step 1: Chạy GraphGeneration
    if [ -f "$GRAPH_SCRIPT" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 1. Running $GRAPH_SCRIPT..."
        # Chạy script và pipe output vào file log
        python "$GRAPH_SCRIPT" > "$GRAPH_LOG" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] -> GraphGeneration for $DATASET hoàn thành. Log save: $GRAPH_LOG"
        else
            echo "[!] Lỗi khi chạy $GRAPH_SCRIPT. Xem $GRAPH_LOG để biết chi tiết."
            # Nếu Graph sinh bị lỗi, skip sang dataset khác hoặc dừng tùy lựa chọn
            echo "Skipping training for $DATASET due to GraphGeneration error."
            continue
        fi
    else
        echo "[!] Không tìm thấy file $GRAPH_SCRIPT, bỏ qua dataset này."
        continue
    fi

    echo "---------------------------------------------------------"

    # Step 2: Chạy main để train mô hình
    if [ -f "$MAIN_SCRIPT" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 2. Running $MAIN_SCRIPT..."
        python "$MAIN_SCRIPT" > "$MAIN_LOG" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] -> Training cho $DATASET hoàn thành. Log save: $MAIN_LOG"
        else
            echo "[!] Lỗi khi chạy $MAIN_SCRIPT. Xem $MAIN_LOG để biết chi tiết."
        fi
    else
        echo "[!] Không tìm thấy file $MAIN_SCRIPT."
    fi

    echo -e "\n"
done

echo "Tất cả các bộ dữ liệu đã được xử lý xong!"
