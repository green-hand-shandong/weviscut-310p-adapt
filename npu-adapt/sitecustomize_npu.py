# Atlas 300I Duo (310P3) + CANN 9.2 + torch 2.10:
# 模型构造期（权重加载前）的 bf16 随机/常量初始化会触发 NPU 算子不支持
# （aclnnInplaceNormal、Tile 等）。此处把构造期初始化转到 fp32 再拷回 bf16。
# 权重随后被 checkpoint 覆盖，不影响数值。
try:
    import torch
except Exception:
    torch = None

def _patch():
    if torch is None:
        return
    init = torch.nn.init
    if hasattr(init, '_orig_no_grad_normal_zc'):
        return
    _orig_normal = init._no_grad_normal_
    def _no_grad_normal(tensor, mean, std, generator=None):
        try:
            if tensor.device.type == 'npu' and tensor.dtype == torch.bfloat16 and tensor.numel() > 0:
                with torch.no_grad():
                    t32 = torch.empty(tensor.shape, dtype=torch.float32, device=tensor.device)
                    _orig_normal(t32, mean, std, generator)
                    tensor.copy_(t32.to(torch.bfloat16))
                return tensor
        except Exception:
            pass
        return _orig_normal(tensor, mean, std, generator)
    init._no_grad_normal_ = _no_grad_normal

    _orig_fill = init._no_grad_fill_
    def _no_grad_fill(tensor, value):
        try:
            if tensor.device.type == 'npu' and tensor.dtype == torch.bfloat16 and tensor.numel() > 0:
                with torch.no_grad():
                    t32 = torch.full(tensor.shape, float(value), dtype=torch.float32, device=tensor.device)
                    tensor.copy_(t32.to(torch.bfloat16))
                return tensor
        except Exception:
            pass
        return _orig_fill(tensor, value)
    init._no_grad_fill_ = _no_grad_fill

    init._orig_no_grad_normal_zc = _orig_normal

_patch()
