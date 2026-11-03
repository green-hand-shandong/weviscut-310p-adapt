# WeVisCut NPU 适配镜像 — 华为 Atlas 300I Duo (Ascend 310P3)
# 基础镜像自带 torch 2.10 + torch_npu + vllm-ascend(仅作参考，WeVisCut 走 transformers 后端)
# 目标：补齐 WeVisCut 依赖 + 降级 transformers 到 5.15.1(权重训练版本) + NPU 补丁
FROM m.daocloud.io/quay.io/ascend/vllm-ascend:v0.30.0rc2-310p

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV HF_ENDPOINT=https://hf-mirror.com

# 系统依赖：ffmpeg(音视频处理)、git(deepagents pin 安装)、字体
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git fonts-noto-cjk curl wget \
    && rm -rf /var/lib/apt/lists/*

# transformers 5.14.1 → 5.15.1（权重由 5.15.1 训练保存，qwen3_5 架构代码需要匹配）
RUN pip install --no-cache-dir "transformers==5.15.1" "accelerate>=1.0.0"

# WeVisCut 核心依赖（deepagents 用 PyPI 等价版本，避免对 github 的依赖）
RUN pip install --no-cache-dir \
    "deepagents==0.7.7" \
    "langchain>=1.3.14,<2.0.0" \
    "langchain-core>=1.5.0,<2.0.0" \
    "langgraph>=1.2.0" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.34.0" \
    "websockets>=14.0" \
    "pydantic>=2.10.0" \
    "pyyaml>=6.0" \
    "httpx>=0.28.0" \
    "requests>=2.31.0" \
    "openai>=1.0.0" \
    "soundfile>=0.12.1" \
    "fonttools>=4.59.2" \
    "tqdm>=4.66.0" \
    "edge-tts>=6.1.0" \
    "yt-dlp>=2024.1.1"

# ms-swift：TransformersEngine 加载 qwen3_5 的推理引擎（NPU 上不装 vllm 依赖）
RUN pip install --no-cache-dir "ms-swift>=3.0.0"

WORKDIR /workspace
CMD ["/bin/bash"]

# ms-swift 的 qwen 处理器需要 qwen_vl_utils
RUN pip install --no-cache-dir "qwen_vl_utils>=0.0.14"

# 310P: vLLM EngineCore 构造模型时 nn.Embedding 的 bf16 随机初始化
# 会触发 aclnnInplaceNormal 不支持错误（权重随后被 checkpoint 覆盖，不影响数值）
COPY sitecustomize_npu.py /usr/local/python3.12.13/lib/python3.12/site-packages/sitecustomize.py
