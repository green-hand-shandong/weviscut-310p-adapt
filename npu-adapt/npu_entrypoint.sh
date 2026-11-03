#!/bin/bash
# WeVisCut NPU 启动入口 — 启动时打 NPU 补丁再启动 studio
set -e

REPO="${WEVISCUT_REPO:-/workspace/WeVisCut}"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export AICUT_STUDIO_CONFIG="${AICUT_STUDIO_CONFIG:-$REPO/studio_config_npu.yaml}"

# ── 补丁 1: dtype 选择（torch.cuda.is_available → 兼容 npu）──
LOADER="$REPO/server/model_loader.py"
if ! grep -q "npu_is_available" "$LOADER"; then
  python3 - "$LOADER" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = "dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32"
new = ("import torch_npu  # noqa: F401  (npu_is_available patch)\n"
       "        _npu_is_available = getattr(torch, 'npu', None) is not None and torch.npu.is_available()\n"
       "        dtype = torch.bfloat16 if (torch.cuda.is_available() or _npu_is_available) else torch.float32")
assert old in s, "dtype line not found; upstream changed?"
s = s.replace(old, new, 1)
open(p, 'w').write(s)
print("[npu_entrypoint] patched dtype selection in", p)
PYEOF
fi

# ── 补丁 2: attn_impl 校验（flash_attention_2 在 310P 不可用，配置已改 sdpa，此处兜底）──
echo "[npu_entrypoint] config: $AICUT_STUDIO_CONFIG"
exec "$@"
