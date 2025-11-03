#!/bin/bash
#
# Real-time Quantization Progress Monitor
# ========================================
#
# Monitors the NVFP4 quantization process in real-time
# Shows: GPU usage, calibration progress, memory stats, time elapsed
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
LOG_DIR="./logs"
LATEST_LOG=$(ls -t "$LOG_DIR"/nvfp4_quantization_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo -e "${RED}Error: No quantization log file found in $LOG_DIR${NC}"
    echo "Make sure quantization is running first."
    exit 1
fi

echo -e "${BOLD}========================================${NC}"
echo -e "${BOLD}NVFP4 Quantization Progress Monitor${NC}"
echo -e "${BOLD}========================================${NC}"
echo ""
echo -e "${BLUE}Monitoring log: $LATEST_LOG${NC}"
echo -e "${BLUE}Press Ctrl+C to exit${NC}"
echo ""

# Track start time
if [ -f "$LATEST_LOG" ]; then
    START_TIME=$(stat -c %Y "$LATEST_LOG" 2>/dev/null || stat -f %m "$LATEST_LOG" 2>/dev/null)
else
    START_TIME=$(date +%s)
fi

# Function to format elapsed time
format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
}

# Function to get GPU stats
get_gpu_stats() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1
    else
        echo "N/A,N/A,N/A,N/A"
    fi
}

# Function to extract progress from log
get_progress() {
    local log_file="$1"

    # Check for calibration progress
    local calib_progress=$(grep -o "Calibrated [0-9]*/[0-9]* samples" "$log_file" 2>/dev/null | tail -1)

    # Check for quantization stage
    local stage=""
    if grep -q "Loading calibration data" "$log_file" 2>/dev/null; then
        stage="Loading Data"
    fi
    if grep -q "Loading model from Hugging Face" "$log_file" 2>/dev/null; then
        stage="Loading Model"
    fi
    if grep -q "YaRN RoPE Scaling Applied" "$log_file" 2>/dev/null; then
        stage="YaRN Applied"
    fi
    if grep -q "Starting calibration forward pass" "$log_file" 2>/dev/null; then
        stage="Calibrating"
    fi
    if grep -q "Quantization complete" "$log_file" 2>/dev/null; then
        stage="Quantized"
    fi
    if grep -q "Exporting quantized model" "$log_file" 2>/dev/null; then
        stage="Exporting"
    fi
    if grep -q "QUANTIZATION COMPLETE" "$log_file" 2>/dev/null; then
        stage="COMPLETE"
    fi

    echo "$stage|$calib_progress"
}

# Main monitoring loop
while true; do
    # Clear screen for refresh
    clear

    # Header
    echo -e "${BOLD}========================================${NC}"
    echo -e "${BOLD}NVFP4 Quantization Progress Monitor${NC}"
    echo -e "${BOLD}========================================${NC}"
    echo ""

    # Time elapsed
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    echo -e "${BLUE}Time Elapsed:${NC} $(format_time $ELAPSED)"
    echo -e "${BLUE}Current Time:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # GPU Stats
    GPU_STATS=$(get_gpu_stats)
    IFS=',' read -r GPU_UTIL MEM_USED MEM_TOTAL GPU_TEMP <<< "$GPU_STATS"

    echo -e "${BOLD}GPU Statistics:${NC}"
    echo -e "  Utilization: ${GREEN}${GPU_UTIL}%${NC}"
    echo -e "  Memory:      ${GREEN}${MEM_USED} MB${NC} / ${MEM_TOTAL} MB"
    echo -e "  Temperature: ${GREEN}${GPU_TEMP}°C${NC}"
    echo ""

    # Quantization Progress
    PROGRESS_INFO=$(get_progress "$LATEST_LOG")
    IFS='|' read -r STAGE CALIB_PROGRESS <<< "$PROGRESS_INFO"

    echo -e "${BOLD}Quantization Progress:${NC}"
    if [ "$STAGE" == "COMPLETE" ]; then
        echo -e "  Stage: ${GREEN}✓ COMPLETE${NC}"
    elif [ -n "$STAGE" ]; then
        echo -e "  Stage: ${YELLOW}⧗ $STAGE${NC}"
    else
        echo -e "  Stage: ${YELLOW}⧗ Starting...${NC}"
    fi

    if [ -n "$CALIB_PROGRESS" ]; then
        echo -e "  Progress: ${YELLOW}$CALIB_PROGRESS${NC}"

        # Extract numbers for percentage
        if [[ "$CALIB_PROGRESS" =~ ([0-9]+)/([0-9]+) ]]; then
            CURRENT="${BASH_REMATCH[1]}"
            TOTAL="${BASH_REMATCH[2]}"
            PERCENT=$((CURRENT * 100 / TOTAL))

            # Progress bar
            BAR_WIDTH=40
            FILLED=$((PERCENT * BAR_WIDTH / 100))
            EMPTY=$((BAR_WIDTH - FILLED))

            printf "  ["
            printf "%${FILLED}s" | tr ' ' '='
            printf "%${EMPTY}s" | tr ' ' '-'
            printf "] ${PERCENT}%%\n"
        fi
    fi
    echo ""

    # Recent log entries (last 10 lines)
    echo -e "${BOLD}Recent Log Activity:${NC}"
    echo -e "${BLUE}────────────────────────────────────────${NC}"
    tail -10 "$LATEST_LOG" | sed 's/^/  /'
    echo -e "${BLUE}────────────────────────────────────────${NC}"
    echo ""

    # Estimated completion
    if [ "$STAGE" == "Calibrating" ] && [ -n "$CALIB_PROGRESS" ]; then
        if [[ "$CALIB_PROGRESS" =~ ([0-9]+)/([0-9]+) ]]; then
            CURRENT="${BASH_REMATCH[1]}"
            TOTAL="${BASH_REMATCH[2]}"
            if [ "$CURRENT" -gt 0 ]; then
                TIME_PER_SAMPLE=$((ELAPSED / CURRENT))
                REMAINING_SAMPLES=$((TOTAL - CURRENT))
                EST_REMAINING=$((TIME_PER_SAMPLE * REMAINING_SAMPLES))

                echo -e "${BOLD}Estimated Time Remaining:${NC} $(format_time $EST_REMAINING)"
                echo ""
            fi
        fi
    fi

    # Check if complete
    if [ "$STAGE" == "COMPLETE" ]; then
        echo -e "${GREEN}${BOLD}✓ QUANTIZATION COMPLETE!${NC}"
        echo ""
        echo "Press Ctrl+C to exit monitor."
        break
    fi

    # Footer
    echo -e "${BLUE}Refreshing every 5 seconds... (Press Ctrl+C to exit)${NC}"

    # Wait before next refresh
    sleep 5
done

# Keep showing final state
if [ "$STAGE" == "COMPLETE" ]; then
    while true; do
        sleep 10
    done
fi
