# 容器启动第一步：给 server/model_loader.py 打 NPU dtype 补丁（幂等）
import sys

p = sys.argv[1]
s = open(p).read()
old = "dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32"
if '_npu_ok' in s:
    print('[entrypoint] patch already present')
    sys.exit(0)
assert old in s, 'dtype line not found; upstream changed?'
new = (
    "import torch_npu  # noqa: F401\n"
    "        _npu_ok = getattr(torch, 'npu', None) is not None and torch.npu.is_available()\n"
    "        import os as _os\n"
    "        _vllm_fp16 = _os.environ.get('AICUT_VLLM_DTYPE', 'float16') == 'float16'\n"
    "        dtype = (torch.float16 if _vllm_fp16 else torch.bfloat16) if _npu_ok"
    " else (torch.bfloat16 if torch.cuda.is_available() else torch.float32)"
)
s = s.replace(old, new, 1)
open(p, 'w').write(s)
print('[entrypoint] npu dtype patch applied')
