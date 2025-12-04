#!/bin/bash

# 定义运行次数
TOTAL_ROUNDS=10
# 定义Python脚本名称
SCRIPT_NAME="openai_llm.py"

echo "开始任务，共需运行 $TOTAL_ROUNDS 轮..."

for ((i=1; i<=TOTAL_ROUNDS; i++)); do
    echo "----------------------------------------"
    echo "正在运行第 $i / $TOTAL_ROUNDS 轮"
    echo "----------------------------------------"

    # 1. qwen3 (ner&re)
    echo "[1/6] Running qwen3 ner&re..."
    MODEL_NAME=qwen3-30b-inst LLM_API_URL=http://0.0.0.0:8001/v1 TASK="ner&re" NUM=$i python "$SCRIPT_NAME"

    # 2. qwen3 (ner)
    echo "[2/6] Running qwen3 ner..."
    MODEL_NAME=qwen3-30b-inst LLM_API_URL=http://0.0.0.0:8001/v1 TASK=ner NUM=$i python "$SCRIPT_NAME"

    echo "第 $i 轮结束。"
    echo ""
done

echo "所有 $TOTAL_ROUNDS 轮任务已全部完成。"
