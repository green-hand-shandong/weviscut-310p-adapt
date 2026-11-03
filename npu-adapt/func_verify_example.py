"""最小功能验证：真实模型 + 一段口播稿 + 官方 schema。验证模型两大核心能力。"""
import sys, json, time
import multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()  # vLLM EngineCore spawn 需要
sys.path.insert(0, '/workspace/WeVisCut')

from studio.config import install_import_paths, load as load_cfg
import os
install_import_paths()
cfg = load_cfg('/workspace/WeVisCut/studio_config.yaml')
from studio.model_client import ModelService
mc = ModelService(cfg.infer_config, cfg.system_prompt)

def main():
    print('[load] loading model     (vLLM ~5min cold start)...', flush=True)
    t0 = time.monotonic()
    mc.load()
    print('[load] ready in %.0fs' % (time.monotonic()-t0), flush=True)
    loader = mc.loader

    SCRIPT = (
        "很多人以为做视频最难的是剪辑，其实真正难的是开始。"
        "我见过太多人收藏了几十教程，却连第一条片子都没剪出来。"
        "问题出在哪？在于把创作想成了一个大工程。"
        "正确的做法是把流程拆开：先有一篇讲稿，再把讲稿切成章节。"
        "每个章节只规划三五个镜头，镜头定了再找素材。"
        "素材齐了配音配乐，最后才拼接成片。"
        "你看，每一步都不难，难的是跳过素材去想象成片。"
        "所以别再等灵感了，先写下一篇三百字的讲稿，今天就把它做成你的第一条片子。"
    )

    # ── 1. orchestrator：工具决策 ──
    from server.model_loader import _make_agent_tools
    from server.tool_schema import TOOL_CARDS
    tools = _make_agent_tools(list(TOOL_CARDS.keys()))
    sys_prompt = open('/workspace/WeVisCut/configs/unified_v5_system_prompt.txt').read()
    user = json.dumps({
        "task": "将用户输入的一段口播稿，通过 AICutPipeline 工具链做成完整长视频。",
        "input_script": SCRIPT,
        "runtime": {"stage": "started", "next_stage": "segment_article"},
    }, ensure_ascii=False)
    t0 = time.monotonic()
    r1 = loader.predict_orchestrator(sys_prompt, user, max_tokens=1024, temperature=0.0)
    print('[step1] orchestrator %.1fs' % (time.monotonic()-t0), flush=True)
    tc = {'name': r1.tool_name, 'arguments': r1.arguments} if r1.tool_name else None
    name = tc.get('name') if tc else None
    args = tc.get('arguments') if tc else None
    if not name:
        # 宽松解析：模型可能输出 {"stage":...} / {"name":...} / <tool_name>...</tool_name> 等多种格式
        import re as _re
        cleaned_full = _re.sub(r"ardin(?:json)?\s*\n?", "", r1.cleaned_output or '').strip()
        try:
            obj = json.loads(cleaned_full[cleaned_full.index('{'):cleaned_full.rindex('}')+1])
            if isinstance(obj, dict):
                name = obj.get('name') or obj.get('tool_name') or obj.get('stage') or obj.get('next_tool')
                args = obj.get('arguments') if isinstance(obj.get('arguments'), dict) else args
        except Exception:
            pass
        if not name:
            m = _re.search(r"<tool_name>\s*(.+?)\s*</tool_name>", r1.raw_output or '', _re.S)
            if m:
                name = m.group(1).strip()
            ma = _re.search(r"<tool_arguments>\s*(\{.*?\})\s*</tool_arguments>", r1.raw_output or '', _re.S)
            if ma:
                try:
                    args = json.loads(ma.group(1))
                except Exception:
                    args = None
        m = _re.search(r"<tool_name>\s*(.+?)\s*</tool_name>", r1.raw_output or '', _re.S)
        if m:
            name = m.group(1).strip()
        ma = _re.search(r"<tool_arguments>\s*(\{.*?\})\s*</tool_arguments>", r1.raw_output or '', _re.S)
        if ma:
            try:
                args = json.loads(ma.group(1))
            except Exception:
                args = None
    print('[step1] parse_ok=%s raw_head=%s' % (r1.parse_ok, (r1.raw_output or '')[:300].replace(chr(10),' ')), flush=True)
    name = ('Tool_' + name) if name and not name.startswith('Tool_') else name
    ok1 = name == 'Tool_SegmentArticle'
    print('[step1] tool=%s args=%s -> %s' % (name, json.dumps(args, ensure_ascii=False)[:100] if args else '{}', 'PASS' if ok1 else 'FAIL'), flush=True)

    # ── 2. generator：章节切分（官方训练对齐 payload） ──
    import server.infer_graph as ig
    payload = ig.InferenceOrchestrator._build_skill_payload_segment_article(
        SCRIPT, (args or {}), (args or {}).get('num_segments_target', 4))
    t0 = time.monotonic()
    r2 = loader.predict_generator(json.dumps(payload, ensure_ascii=False), system_prompt=loader.skill_prompt, max_tokens=2048, temperature=0.0)
    dt = time.monotonic()-t0
    txt = r2.raw_output or ''
    js = None
    try:
        js = json.loads(txt[txt.index('{'):txt.rindex('}')+1])
    except Exception:
        pass
    segs = js.get('segments') if isinstance(js, dict) else None
    ok2 = False
    if segs:
        cover = ''.join(s.get('narration_script','') for s in segs)
        ratio = len(cover)/max(len(SCRIPT),1)
        ok2 = 2 <= len(segs) <= 10 and 0.5 <= ratio <= 2.5
        print('[step2] generator %.1fs segments=%d coverage=%.2f -> %s' % (dt, len(segs), ratio, 'PASS' if ok2 else 'FAIL'), flush=True)
        for s_ in segs[:4]:
            print('   %s %s | %s' % (s_.get('segment_id'), (s_.get('title') or '')[:16], (s_.get('narration_script') or '')[:36]), flush=True)
    else:
        print('[step2] generator %.1fs -> FAIL (no valid JSON)' % dt, flush=True)

    print('FUNCTION_TEST: %s' % ('PASS' if (ok1 and ok2) else 'FAIL'), flush=True)


if __name__ == '__main__':
    main()
