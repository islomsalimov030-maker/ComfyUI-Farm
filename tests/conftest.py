import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from mock_comfy import MockComfy, tiny_png  # noqa: E402

from farm.config import load_config  # noqa: E402
from farm.contracts import CheckResult  # noqa: E402
from farm.control import Control  # noqa: E402
from farm.db import DB  # noqa: E402
from farm.plugins import Plugins, StubPromptBuilder  # noqa: E402
from farm.storage import Paths  # noqa: E402
from farm.worker import Worker  # noqa: E402

FIX = Path(__file__).parent / "fixtures"


class ScriptedQA:
    """rules[(shot_id, stage)] = [коды на try 0, try 1, ...]; None — проходит. Производные (post_fix) — проходят."""

    def __init__(self, rules=None):
        self.rules = rules or {}
        self.calls = []

    def run_stage(self, stage, asset, ctx):
        sid = ctx.shot["shot_id"]
        try_no = (ctx.attempt or {}).get("try_no", 0)
        self.calls.append((sid, stage, try_no, asset.asset_id))
        seq = self.rules.get((sid, stage), [])
        code = seq[min(try_no, len(seq) - 1)] if seq else None
        is_fixed = "fix_" in Path(asset.path).name
        if code and not is_fixed:
            return [CheckResult("sharp", 0.9, True, "soft"),
                    CheckResult(code.lower(), 0.1, False, "hard", fail_code=code)]
        return [CheckResult("sharp", 0.9, True, "soft"), CheckResult("main", 0.95, True, "hard")]


class CopyFixer:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.calls = 0

    def fix(self, asset, ctx):
        self.calls += 1
        out = self.tmp / f"fixed_{asset.asset_id}.mp4"
        out.write_bytes(Path(asset.path).read_bytes() + b"|looped")
        return str(out)


class FaceLibrary:
    def __init__(self, path):
        self.path = str(path)

    def resolve_inputs(self, shot, stage):
        return {"face_ref": self.path} if stage == "frame" else {}


class Recorder:
    def __init__(self):
        self.uploads = []
        self.updates = []

    def upload_final(self, asset_path, shot):
        self.uploads.append((shot["shot_id"], asset_path))
        return f"https://drive.example/{shot['shot_id']}"

    def on_shot_update(self, shot, links, scores):
        self.updates.append((shot["shot_id"], shot["state"], list(links), dict(scores)))


@pytest.fixture
def mock():
    m = MockComfy().start()
    yield m
    m.stop()


@pytest.fixture
def farm_home(tmp_path, monkeypatch):
    home = tmp_path / "farm"
    monkeypatch.setenv("FARM_HOME", str(home))
    return home


def make_cfg(mock, **over):
    base = {"comfy": {"url": mock.url, "ws_url": mock.ws_url, "poll_s": 0.05, "request_timeout_s": 10,
                      "startup_wait_s": 5},
            "vram": {"wait_s": 2, "poll_s": 0.05},
            "timeouts": {"frame": 10, "video": 10},
            "worker": {"idle_sleep_s": 0.05}}
    from farm.config import deep_merge
    return load_config(deep_merge(base, over))


@pytest.fixture
def make_worker(mock, tmp_path, farm_home):
    face = tmp_path / "face.png"
    face.write_bytes(tiny_png(32, 32, 7))
    created = []

    def _make(qa=None, cfg_over=None, fixer=None, recorder=None, db=None, client_mock=None, rng_seed=1):
        import random
        from farm.bandit import Bandit
        m = client_mock or mock
        cfg = make_cfg(m, **(cfg_over or {}))
        paths = Paths(farm_home)
        rec = recorder or Recorder()
        plugins = Plugins(qa=qa or ScriptedQA(), prompts=StubPromptBuilder(), library=FaceLibrary(face),
                          uploader=rec, sink=rec, fixers={"post.loop": fixer or CopyFixer(tmp_path)},
                          autoload=False)
        dbi = db or DB(paths.ensure().db)
        ctrl = Control(paths.state / "control.json")
        ctrl.set_running(True)
        w = Worker(db=dbi, paths=paths, cfg=cfg, profiles_dir=FIX / "profiles",
                   workflows_dir=FIX / "workflows", plugins=plugins, control=ctrl,
                   bandit=Bandit(dbi, cfg["bandit"]["cold_start_n"], random.Random(rng_seed)))
        w.recorder = rec
        created.append(w)
        return w

    yield _make
    for w in created:
        w.stop(timeout=2)
        w.client.close()


def add_shot(db, shot_id, **kw):
    row = {"shot_id": shot_id, "profile": "test_sfw", "kind": "video", "duration_s": 6.0,
           "pose_ru": "стоит у окна", "location": "loft", "character": "stacy", "priority": 0}
    row.update(kw)
    db.add_shot(row)


def visited(db, shot_id):
    return [r["to_state"] for r in db.all("SELECT to_state FROM events WHERE shot_id=? ORDER BY event_id",
                                          (shot_id,))]
