"""WS-15: моды строки таблицы → H3_REF_MODS (farm.util_nodes.ref_mods) без ComfyUI и torch."""
import pytest

from farm import util_nodes as U


def test_ref_mods_from_names(monkeypatch):
    """WS-15: моды из таблицы → H3_REF_MODS кодом загрузчика; (none) пропускается, чужого мода нет — ошибка."""
    import sys
    import types
    calls = []

    class Loader:
        FUNCTION = "load"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"mod_1": (["(none)", "farm/a_face", "farm/a_body"],)}}

        def load(self, show_info=False, max_total_tokens=0, **kw):
            calls.append(kw)
            return ([(kw[f"mod_{i}"], kw[f"strength_{i}"]) for i in range(1, 9) if kw[f"mod_{i}"] != "(none)"], "")

    monkeypatch.setitem(sys.modules, "nodes", types.SimpleNamespace(NODE_CLASS_MAPPINGS={"MiniMaxH3RefModsLoader": Loader}))
    assert U.ref_mods(["farm/a_face", "farm/a_body"]) == [("farm/a_face", 1.0), ("farm/a_body", 1.0)]
    assert U.ref_mods(["farm/a_face", "(none)"]) == [("farm/a_face", 1.0)]
    assert U.ref_mods(["(none)", ""]) == [] and len(calls) == 2
    with pytest.raises(FileNotFoundError):
        U.ref_mods(["farm/b_face"])
    assert U.face_mods("farm/a_face", 0.5) == [("farm/a_face", 0.5)]
