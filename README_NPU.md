# WeVisCut NPU 适配 — Atlas 300I Duo (Ascend 310P3)

腾讯 WeVisCut（长文成片工具智能体，9B 策略模型 qwen3_5 架构）在华为昇腾
300I Duo 上的容器化适配。无需修改上游代码逻辑，仅一处 dtype 判断补丁 +
NPU 专用配置。

## 环境总览

| 项 | 值 |
|---|---|
| 代码 | /data3/WeVisCut/WeVisCut（官方仓库，仅加 studio_config_npu.yaml） |
| 权重 | /data3/WeVisCut/checkpoints/weviscut-9b（18.8GB bf16 + LoRA adapter） |
| 镜像 | weviscut-npu:310p-v5（基础：vllm-ascend v0.30.0rc2-310p，torch 2.10 + torch_npu + CANN 9.2） |
| 补丁 | /data3/WeVisCut/npu-adapt/（Dockerfile、npu_entrypoint.sh、docker-compose.yml） |
| 启动 | /data3/WeVisCut/start_weviscut_npu.sh |

## 关键适配点（与 GPU 版的差异）

1. **推理后端**：vLLM → transformers（`weviscut_9b_hf` profile）。310P 无
   vLLM 后端支持，ms-swift TransformersEngine 直接可用。
2. **transformers 版本**：5.14.1 → 5.15.1（权重训练保存版本，qwen3_5 架构
   代码在 5.15.x 完整内置，`Qwen3_5ForConditionalGeneration` 原生支持）。
3. **attn_impl**：flash_attention_2 → eager。310P 的 torch 2.10 SDPA 全部
   后端有 `can not cast format` bug（flash/efficient/math 全挂，
   `npu_fusion_attention` 也报 aclnn 错误），eager 路径可用且输出正确。
4. **device_map**：`cuda:0` → `npu`。
5. **dtype 补丁**：`server/model_loader.py:146` 的
   `torch.cuda.is_available()` 增加 `torch.npu.is_available()` 分支
   （启动时自动 sed，见 start 脚本内嵌补丁），否则静默降级 fp32。
6. **设备挂载**：走宿主机 `ascend` docker runtime（daemon.json 已配置），
   `ASCEND_VISIBLE_DEVICES=<逻辑ID列表>` 即可，禁止手动 `--device /dev/davinci*`。

## 已验证结果

- torch_npu bf16 matmul：OK
- 9B 模型 NPU 加载（含 LoRA adapter）：LOAD_OK，约 70s，显存 38.5GB/单芯片 44GB
- 端到端推理：输出正确（think 标签 + 中文回复），enable_thinking 工作正常
- Studio 服务：HTTP 200，UI「创作台」可达，权重后台加载

## 启动

```bash
# 默认用 davinci 逻辑卡 4,5（物理 NPU 5 卡），端口 8081
bash /data3/WeVisCut/start_weviscut_npu.sh

# 指定卡和端口
bash /data3/WeVisCut/start_weviscut_npu.sh 12,13 8082
```

打开 `http://<host>:8081/`，trustworthy-cut 界面在 `/0907_newvis`。

## 性能实测结论（vLLM vs transformers）

| 后端 | 生成速度 | 备注 |
|---|---|---|
| **vLLM fp16（推荐，已设为默认）** | **9.8-10 tok/s** | ms-swift VllmEngine + vllm-ascend，走 aclnn 融合算子 |
| transformers eager bf16 | ~0.5 tok/s | 310P 上 SDPA 全后端有 bug 只能 eager |

**提升约 20 倍。** 性能慢的根因不是"310P 不行"——宿主机同一批卡上 vLLM
正在服务 Qwen3.6-35B 生产负载——而是 eager attention 全量重算 + 部分
kernel CPU 回退。

### vLLM 路径适配要点（相对 transformers 路径的增量）

1. **dtype 用 fp16**：310P 的 CANN 9.2 对 bf16 标量写 op（fill_/normal_/
   ones_ 等）支持不全，vLLM 构造模型时的随机初始化会撞 aclnn 错误。
   fp16 在 310P 支持完整（宿主机生产也是 fp16）。已做进 dtype 补丁：
   vLLM 后端自动 fp16，transformers 后端保持 bf16。
2. **sitecustomize.py 全局补丁**：兜底把 NPU bf16 构造期初始化转 fp32
   （权重随后被 checkpoint 覆盖，不影响数值）。
3. **权重加载**：18.8GB 单 safetensors 约 30 秒载入；引擎初始化约 5 分钟
   （含 aclnn 图构建），之后常驻。
4. **enable_prefix_caching 关闭**：与宿主机生产一致，310P 上更稳。

## 已知限制

- HyperFrames 渲染器 / chrome-headless-shell / 素材库未装（成片渲染的
  最后一步不可用），模型侧的分镜/口播/质量判别均正常。装法：
  容器内 `bash scripts/setup_external_tools.sh`（需外网）。
- 首次推理含 aclnn 在线编译，约 20 分钟；之后同形状请求走缓存。
- `--no-model` 模式（纯 UI）不占卡。

## 文件清单

```
/data3/WeVisCut/
├── WeVisCut/                # 官方仓库 + studio_config_npu.yaml（唯一新增文件）
├── checkpoints/weviscut-9b/ # 官方权重（hf-mirror 下载，校验一致）
├── npu-adapt/
│   ├── Dockerfile           # 镜像构建文件
│   ├── npu_entrypoint.sh    # 启动时打 dtype 补丁
│   ├── docker-compose.yml   # compose 部署
│   └── patch/npu_device_patch.md  # 补丁细节说明
├── start_weviscut_npu.sh    # 一键启动
└── README_NPU.md            # 本文档
```
