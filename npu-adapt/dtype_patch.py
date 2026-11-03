# 启动时由 npu_entrypoint 应用到 server/model_loader.py：
# vllm 后端在 310P 用 fp16（bf16 标量 op 支持不全），transformers 后端保持 bf16
import sys

p = sys.argv[1]
s = open(p).read()
old = "dtype = torch.bfloat16 if (torch.cuda.is_available() or _npu_ok) else torch.float32"
new = ("_npu_ok = getattr(torch, 'npu', None) is not None and torch.npu.is_available()\n"
       "        import os as _os\n"
       "        _vllm_fp16 = _os.environ.get('AICUT_VLLM_DTYPE', 'float16') == 'float16'\n"
       "        dtype = (torch.float16 if _vllm_fp16 else torch.bfloat16) if _npu_ok else (torch.bfloat16 if torch.cuda.is_available() else torch.float32)")
assert old in s, 'dtype line not found'
s = s.replace(old, new, 1)
open(p, 'w').write(s)
print('dtype patch applied:', p)
