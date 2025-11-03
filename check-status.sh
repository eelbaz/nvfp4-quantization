#!/bin/bash
# NVFP4 Quantization Status Check Script
# Usage: ./check-status.sh

set -e

echo "════════════════════════════════════════════════════════════════════════════════"
echo "NVFP4 Quantization Status Check - $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Check if running on spark-alpha
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" != "spark-alpha" ]]; then
    echo "⚠️  WARNING: Not running on spark-alpha.local"
    echo "   Current host: $HOSTNAME"
    echo ""
fi

# Docker container status
echo "━━━ Docker Container Status ━━━"
CONTAINER_COUNT=$(docker ps --filter "ancestor=nvcr.io/nvidia/pytorch:25.10-py3" --format "{{.ID}}" 2>/dev/null | wc -l)
if [ "$CONTAINER_COUNT" -eq 0 ]; then
    echo "❌ No quantization containers running"
    echo "   Status: IDLE or STOPPED"
else
    echo "✅ Active containers: $CONTAINER_COUNT"
    docker ps --filter "ancestor=nvcr.io/nvidia/pytorch:25.10-py3" \
        --format "   Container: {{.ID}} | Status: {{.Status}} | Started: {{.RunningFor}}" 2>/dev/null
fi
echo ""

# GPU status
echo "━━━ GPU Status ━━━"
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null | head -1)

    if [ -n "$GPU_INFO" ]; then
        IFS=',' read -r GPU_IDX GPU_NAME GPU_TEMP GPU_UTIL GPU_MEM_UTIL GPU_MEM_USED GPU_MEM_TOTAL <<< "$GPU_INFO"
        echo "   GPU: $GPU_NAME (ID: $GPU_IDX)"
        echo "   Temperature: ${GPU_TEMP}°C"
        echo "   GPU Utilization: ${GPU_UTIL}%"
        echo "   Memory Utilization: ${GPU_MEM_UTIL}%"

        # Interpret GPU utilization
        if [ "$GPU_UTIL" -lt 5 ]; then
            echo "   📊 Status: IDLE (initialization or waiting)"
        elif [ "$GPU_UTIL" -lt 50 ]; then
            echo "   📊 Status: ACTIVE (processing)"
        else
            echo "   📊 Status: HIGH LOAD (heavy computation)"
        fi
    else
        echo "   ❌ No GPU information available"
    fi
else
    echo "   ❌ nvidia-smi not found"
fi
echo ""

# Check output files
echo "━━━ Output Files Status ━━━"
PROJECT_DIR="/home/exobit/development/sgl/nvfp4-quantization"

# Calibration data
if [ -f "$PROJECT_DIR/calibration-data/calibration.json" ]; then
    CALIB_SIZE=$(du -h "$PROJECT_DIR/calibration-data/calibration.json" 2>/dev/null | cut -f1)
    echo "   ✅ Calibration data: $CALIB_SIZE"
else
    echo "   ❌ Calibration data: NOT FOUND"
fi

# Quantized model
QUANT_DIR="$PROJECT_DIR/quantized-output/Qwen3-VLTO-32B-Instruct-NVFP4"
if [ -d "$QUANT_DIR" ] && [ -f "$QUANT_DIR/hf_quant_config.json" ]; then
    QUANT_SIZE=$(du -sh "$QUANT_DIR" 2>/dev/null | cut -f1)
    FILE_COUNT=$(find "$QUANT_DIR" -type f | wc -l)
    echo "   ✅ Quantized model: $QUANT_SIZE ($FILE_COUNT files)"

    # Check for key files
    [ -f "$QUANT_DIR/config.json" ] && echo "      ├─ config.json ✓"
    [ -f "$QUANT_DIR/hf_quant_config.json" ] && echo "      ├─ hf_quant_config.json ✓"
    ls "$QUANT_DIR"/*.safetensors &>/dev/null && echo "      ├─ safetensors files ✓"
    [ -f "$QUANT_DIR/tokenizer.json" ] && echo "      └─ tokenizer.json ✓"
else
    echo "   ⏳ Quantized model: IN PROGRESS or NOT STARTED"
fi

# Logs
echo ""
echo "━━━ Recent Logs ━━━"
LATEST_LOG=$(ls -t "$PROJECT_DIR/logs/"*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    LOG_NAME=$(basename "$LATEST_LOG")
    LOG_SIZE=$(du -h "$LATEST_LOG" 2>/dev/null | cut -f1)
    LOG_LINES=$(wc -l < "$LATEST_LOG" 2>/dev/null)
    echo "   Latest: $LOG_NAME ($LOG_SIZE, $LOG_LINES lines)"

    # Show last few lines
    echo "   Last 3 log entries:"
    tail -3 "$LATEST_LOG" 2>/dev/null | sed 's/^/      /'
else
    echo "   ❌ No log files found"
fi

echo ""
echo "━━━ Workflow Phase Status ━━━"

# Determine current phase
PHASE="Unknown"
if [ ! -f "$PROJECT_DIR/calibration-data/calibration.json" ]; then
    PHASE="Phase 4: Calibration Prep (NOT STARTED)"
elif [ ! -d "$QUANT_DIR" ]; then
    if [ "$CONTAINER_COUNT" -gt 0 ]; then
        PHASE="Phase 5: NVFP4 Quantization (RUNNING) ⚙️"
    else
        PHASE="Phase 5: NVFP4 Quantization (READY TO START)"
    fi
elif [ -f "$QUANT_DIR/hf_quant_config.json" ]; then
    PHASE="Phase 5: NVFP4 Quantization (COMPLETE) ✅"
    PHASE="$PHASE\n   Next: Phase 6 - Inference Testing"
fi

echo -e "   $PHASE"

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "Quick Commands:"
echo "  • Monitor logs:     tail -f $PROJECT_DIR/logs/*.log"
echo "  • GPU monitoring:   watch -n 1 nvidia-smi"
echo "  • Container logs:   docker logs <container_id>"
echo "════════════════════════════════════════════════════════════════════════════════"
