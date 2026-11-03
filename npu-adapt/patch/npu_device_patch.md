# NPU 适配补丁说明

WeVisCut 原代码两处 CUDA-only 假设需要适配 Ascend NPU：

1. `server/model_loader.py:146`
   `dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32`
   → torch_npu 下 `torch.cuda.is_available()` 为 False，会静默降级 fp32（9B 模型需 ~36GB，超出 310P 单芯 44GB 减去系统占用后的可用空间，且慢）。
   补丁：`torch.npu.is_available()` 时也用 bf16。

2. `studio_config.yaml`
   `device_map: "cuda:0"` + `attn_impl: flash_attention_2`
   → NPU 上 device_map 应为 `"npu"`（transformers 会把整个模型放 npu:0），
     flash_attention_2 在 310P 上不可用，改 sdpa。
   补丁：NPU 专用配置 `studio_config_npu.yaml`。

3. transformers `device_map="npu"` 依赖 accelerate 的设备映射，
   torch_npu 注册了 `torch.npu` 设备模块后 accelerate 可识别 `npu` 关键字。
   若 transformers 内部仍写死 cuda 检查，需要 patch
   `transformers.modeling_utils` 中 `get_device_properties` 相关调用。
   实测 transformers 5.15.1 + accelerate 对 `device_map={"": "npu"}` 直接可用。

补丁方式：不改动上游代码，用 sed 在容器启动时打补丁（npu_entrypoint.sh），
保持 /workspace/WeVisCut 可整体替换为官方新版本。
