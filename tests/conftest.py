import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def run_script():
    """Run `scripts/<name>` with the release's src/ importable; return stdout."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("AQUAFM_")}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), env.get("PYTHONPATH", "")])
    env["CUDA_VISIBLE_DEVICES"] = ""

    def run(name: str, *args) -> str:
        result = subprocess.run([sys.executable, str(REPO / "scripts" / name), *map(str, args)],
                                cwd=REPO, env=env, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
        return result.stdout

    return run
