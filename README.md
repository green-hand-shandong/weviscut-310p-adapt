# WeVisCut × 华为 Atlas 300I Duo（Ascend 310P3）适配

> 腾讯 [Tencent/WeVisCut](https://github.com/Tencent/WeVisCut)（长文成片工具智能体，9B 策略模型）在华为昇腾 **Atlas 300I Duo**（Ascend 310P3）上的容器化适配。
> 官方代码**未修改一行逻辑**，通过镜像层补丁 + NPU 专用配置完成移植。**vLLM-ascend 后端实测生成速度 9.8–10 tok/s**，已可日常使用。

[![Platform](https://img.shields.io/badge/Platform-Ascend%20310P3-blue)](https://www.hiascend.com/)
[![Backend](https://img.shields.io/badge/Inference-vLLM%20fp16-green)](https://github.com/vllm-project/vllm-ascend)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](WeVisCut/LICENSE)

---

## 这是什么

[WeVisCut](https://github.com/Tencent/WeVisCut) 是腾讯 2026-10 开源的**长文成片工具智能体**：一个 9B 策略模型（自定义 `qwen3_5` 架构，hybrid 线性注意力），读入知识类长文/口播稿，自主调用 12 个剪辑工具（切分文章 → 分镜 → TTS → 素材检索 → 模板动画 → 渲染 → 质量判别 → 反思修复 → 拼接），约 12 次工具调用产出数分钟成片。没有硬编码状态机，轨迹由策略模型自主决定。

本仓库记录把它从 NVIDIA CUDA 移植到 **昇腾 300I Duo** 的完整过程、补丁与实测数据。适合需要在国产 NPU 上部署 WeVisCut、或正在做 CUDA → Ascend 迁移的团队参考。

## 实测结果

### 环境

| 项 | 值 |
|---|---|
| 硬件 | Atlas 300I Duo ×8（Ascend 310P3，单芯片 44GB） |
| CANN / 驱动 | 9.2 / 25.5.1 |
| 基础镜像 | `quay.io/ascend/vllm-ascend:v0.30.0rc2-310p`（torch 2.10 + torch_npu） |
| 策略权重 | `tencent/WeVisCut`（18.8GB bf16 + LoRA adapter） |

### 性能对比（同一台机器、同一权重）

| 推理后端 | 生成速度 | 说明 |
|---|---|---|
| **vLLM fp16（推荐，已设默认）** | **9.8–10 tok/s** | ms-swift `VllmEngine` + vllm-ascend 融合算子 |
| transformers eager bf16 | ~0.5 tok/s | 310P 上 SDPA 全后端有 bug，只能 eager |

**提升约 20 倍。** 性能慢的根因不是"310P 不行"——同一批卡上 vLLM 正在服务 Qwen3.6-35B 生产负载——而是 eager attention 全量重算 + 部分 kernel CPU 回退。

### 功能验证

- ✅ 9B 模型 NPU 加载（vLLM：权重 30s 载入 + 引擎初始化约 5 分钟后常驻；transformers：约 70s）
- ✅ 端到端推理输出正确（think 标签格式、中文回复、`enable_thinking` 工作正常）
- ✅ Studio Web 服务（八轨道创作台，HTTP 200）

## 快速开始

### 0. 前置

宿主机已装 Ascend 驱动 + Docker，且 Docker 配置了 [ascend-docker-runtime](https://gitee.com/ascend/ascend-docker-runtime)（`default-runtime: ascend`）。权重不在本仓库，自行下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 国内
pip install -U "huggingface_hub[cli]"
hf download tencent/WeVisCut --local-dir /data3/WeVisCut/checkpoints/weviscut-9b
```

### 1. 构建镜像

```bash
cd npu-adapt
docker build -t weviscut-npu:310p .
```

### 2. 启动

```bash
# 修改 start_weviscut_npu.sh 顶部的权重/代码路径后：
bash start_weviscut_npu.sh 4,5 8081   # 参数：NPU 逻辑卡 ID 列表、端口
```

打开 `http://<host>:8081/`（创作台界面在 `/0907_newvis`）。模型在后台加载（vLLM 引擎初始化约 5 分钟，之后常驻）。

### 3.（可选）成片渲染的最后一步

HyperFrames 渲染器（Node + chrome-headless-shell，约 1.1GB）未内置，容器内联网执行：

```bash
bash scripts/setup_external_tools.sh
```

## 适配要点（CUDA → Ascend 的 5 处差异）

官方代码逻辑零改动。全部差异集中在：

| # | 差异 | 说明 |
|---|---|---|
| 1 | **推理后端** vLLM → vllm-ascend | 官方默认 profile 就是 vLLM，`vllm-ascend` 原生注册了 `Qwen3_5ForConditionalGeneration`，ms-swift `VllmEngine` 直接可用 |
| 2 | **dtype：vLLM 用 fp16** | 310P 的 CANN 9.2 对 **bf16 标量写 op**（`normal_`/`fill_`/`ones_`）支持不全，vLLM 构造模型期的随机初始化会撞 `aclnnInplaceNormal`/`Tile` 错误。fp16 在 310P 支持完整（宿主机生产也是 fp16）。补丁自动区分：vLLM→fp16，transformers→bf16 |
| 3 | **attn_impl** flash_attention_2 → eager | torch 2.10 + CANN 9.2 的 `F.scaled_dot_product_attention` 4D 输入报 `can not cast format`（flash/efficient/math 全挂，`npu_fusion_attention` 也报错）。**这是 transformers 路径慢 20 倍的根因**；vLLM 路径不受影响（自带融合算子） |
| 4 | **device_map** `cuda:0` → `npu` | transformers 路径用 |
| 5 | **dtype 判断** | `server/model_loader.py` 的 `torch.cuda.is_available()` 增加 `torch.npu.is_available()` 分支（启动时自动补丁，幂等） |

另加 `sitecustomize.py` 全局兜底：把 NPU 上 bf16 构造期初始化转 fp32（权重随后被 checkpoint 覆盖，不影响数值），解决 vLLM EngineCore 子进程的初始化报错。

## 性能优化空间

- **SDPA 修复后 transformers 路径可提速 5-10 倍**：降 CANN 到 8.x 或升 torch_npu 补丁版本。vLLM 路径不受此 bug 影响。
- **vLLM 冷启动约 5 分钟**：含 aclnn 图构建；日志里的 `GEInitialize/OpCompileProcessor init failed` 是在线编译器初始化失败的非致命告警（已编译算子不受影响），升级 CANN 时留意。
- **多卡**：`vllm_tensor_parallel_size` 可调，本适配验证的是单芯（44GB 装下 9B fp16 权重 + KV cache）。

## 已知限制

- 成片渲染最后一步（HyperFrames）未验证——模型侧分镜/口播/质量判别均正常。
- vLLM 引擎冷启动含 aclnn 编译，约 5 分钟；之后同形状请求走缓存。
- `--no-model` 模式（纯 UI）不占卡。

## 仓库结构

```
├── WeVisCut/                # 官方仓库快照（Apache-2.0）+ studio_config_npu.yaml（唯一新增文件）
├── npu-adapt/
│   ├── Dockerfile           # NPU 适配镜像构建
│   ├── sitecustomize_npu.py # bf16 构造期初始化兜底补丁（vLLM EngineCore 子进程生效）
│   ├── apply_loader_patch.py# server/model_loader.py dtype 幂等补丁
│   ├── dtype_patch.py       # 补丁源码
│   ├── npu_device_patch.md  # 补丁细节说明
│   └── docker-compose.yml   # compose 部署
├── start_weviscut_npu.sh    # 一键启动
└── README_NPU.md            # 更详细的适配记录（早期 transformers 路径全过程）
```

## 致谢

- [Tencent/WeVisCut](https://github.com/Tencent/WeVisCut)（Apache-2.0）— 原项目
- [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) — 昇腾 vLLM 后端
- [Ascend ms-swift](https://github.com/modelscope/ms-swift) — 推理引擎封装

本仓库中由适配者编写的补丁与文档以 Apache-2.0 发布；WeVisCut 上游代码与第三方运行时保留各自授权。
