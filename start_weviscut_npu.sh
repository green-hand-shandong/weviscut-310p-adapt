#!/bin/bash
# WeVisCut NPU 启动脚本（vLLM 推荐配置）— Atlas 300I Duo
# 用法: bash start_weviscut_npu.sh [卡ID列表，默认 4,5] [端口，默认 8081]
set -e
CARDS="${1:-4,5}"
PORT="${2:-8081}"
docker rm -f weviscut-studio 2>/dev/null || true
docker run -d --name weviscut-studio \
  --cpus=16 --shm-size=32g \
  -e ASCEND_VISIBLE_DEVICES="$CARDS" \
  -e AICUT_STUDIO_CONFIG=/workspace/WeVisCut/studio_config_npu.yaml \
  -e AICUT_MODEL_PROFILE=weviscut_9b \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -v /data3/WeVisCut/WeVisCut:/workspace/WeVisCut \
  -v /data3/WeVisCut/checkpoints:/workspace/checkpoints:ro \
  -p "$PORT:8081" \
  weviscut-npu:310p-v5 \
  bash -c "cd /workspace/WeVisCut && export PYTHONPATH=/workspace/WeVisCut && \
    python3 /workspace/WeVisCut/npu_adapt_patch/apply_loader_patch.py /workspace/WeVisCut/server/model_loader.py && \
    python3 -m uvicorn studio.app:app --host 0.0.0.0 --port 8081 --workers 1 --ws-ping-interval 30 --ws-ping-timeout 60 --log-level info"
echo "started. URL: http://$(hostname -I | awk '{print $1}'):$PORT/"
echo "profile: weviscut_9b (vLLM fp16, 约 10 tok/s)；如需 transformers 后端改 AICUT_MODEL_PROFILE=weviscut_9b_hf"
