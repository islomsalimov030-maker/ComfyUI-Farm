"""Постановка v2, зонд MHR (ПК): исходники и сигнатуры модели тела MHR внутри SAM 3D Body — для поправки суставов.
  C:/work/ComfyUI-next/venv/Scripts/python tools/scene3d/posev2/mhr_probe.py --comfy C:/work/ComfyUI-next --out <txt>
Пишет: классы MHRRig и MHRHead (исходник целиком), сигнатуры forward, буферы/параметры MHRRig (имя, форма),
файл модуля и соседние функции, которые вызывают MHRRig (поиск 'mhr(' в исходнике пакета)."""
import argparse
import asyncio
import inspect
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--model", default="sam_3d_body_dinov3_int8_convrot.safetensors")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.comfy)
    sys.argv = [sys.argv[0]]
    import folder_paths  # noqa: F401
    from comfy_extras import nodes_sam3d_body as n

    async def run(cls, **kw):
        r = cls.execute(**kw)
        if asyncio.iscoroutine(r):
            r = await r
        return r.result if hasattr(r, "result") else r
    model = asyncio.new_event_loop().run_until_complete(run(n.SAM3DBody_Loader, model_file=a.model))[0]
    root = getattr(model, "model", model)
    L = []
    seen = set()
    for name, m in root.named_modules():
        t = type(m)
        if t.__name__ in ("MHRRig", "MHRHead") and t not in seen:
            seen.add(t)
            L += [f"===== {t.__name__} ({name}) file {inspect.getsourcefile(t)}",
                  f"forward{inspect.signature(m.forward)}", inspect.getsource(t)]
            if t.__name__ == "MHRRig":
                L.append("----- buffers/params of MHRRig")
                L += [f"{k}: {tuple(v.shape)} {v.dtype}" for k, v in list(m.named_buffers())[:80]]
                L += [f"{k}: {tuple(v.shape)} {v.dtype}" for k, v in list(m.named_parameters())[:80]]
                src_file = Path(inspect.getsourcefile(t))
    if "src_file" in locals():
        for f in sorted(src_file.parent.glob("*.py")):
            txt = f.read_text(encoding="utf-8", errors="replace")
            hits = [i for i, line in enumerate(txt.splitlines()) if "mhr(" in line or "self.mhr" in line]
            if hits:
                lines = txt.splitlines()
                L.append(f"===== calls in {f.name}")
                for i in hits[:12]:
                    L += lines[max(0, i - 6): i + 8] + ["..."]
    Path(a.out).write_text("\n".join(L), encoding="utf-8")
    print(f"[mhr] {a.out}: {len(L)} строк, классы {[t.__name__ for t in seen]}", flush=True)


if __name__ == "__main__":
    main()
