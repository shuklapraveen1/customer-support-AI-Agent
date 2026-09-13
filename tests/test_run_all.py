import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "twcs_sample.csv"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_all import maybe_subsample_raw_csv  # noqa: E402


def test_maybe_subsample_raw_csv_leaves_small_files_untouched(tmp_path):
    src = tmp_path / "small.csv"
    df = pd.DataFrame({"a": range(5)})
    df.to_csv(src, index=False)

    result_path = maybe_subsample_raw_csv(str(src), sample_size=100, seed=42)
    assert result_path == str(src)


def test_maybe_subsample_raw_csv_subsamples_large_files(tmp_path):
    src = tmp_path / "big.csv"
    df = pd.DataFrame({"a": range(1000)})
    df.to_csv(src, index=False)

    result_path = maybe_subsample_raw_csv(str(src), sample_size=100, seed=42)
    assert result_path != str(src)
    assert Path(result_path).exists()

    sampled = pd.read_csv(result_path)
    assert len(sampled) == 100


def test_maybe_subsample_raw_csv_is_deterministic(tmp_path):
    src = tmp_path / "big.csv"
    df = pd.DataFrame({"a": range(1000)})
    df.to_csv(src, index=False)

    path1 = maybe_subsample_raw_csv(str(src), sample_size=50, seed=7)
    rows1 = sorted(pd.read_csv(path1)["a"].tolist())

    Path(path1).unlink()
    path2 = maybe_subsample_raw_csv(str(src), sample_size=50, seed=7)
    rows2 = sorted(pd.read_csv(path2)["a"].tolist())

    assert rows1 == rows2


@pytest.mark.slow
def test_run_all_fast_mode_end_to_end(tmp_path, monkeypatch):
    """Full subprocess-based smoke test: copies the fixture into an
    isolated temp working directory (with symlinked configs) and runs
    `python scripts/run_all.py --brand BrandA_Support --fast`, asserting
    it exits 0 and produces every required evaluation artifact. This is
    the closest thing to an actual reproduction check for the '--fast
    reproduces headline results quickly' requirement, run against the
    tiny synthetic fixture since the real dataset isn't available here.
    """
    work_dir = tmp_path / "workdir"
    (work_dir / "data" / "raw").mkdir(parents=True)
    (work_dir / "data" / "golden").mkdir(parents=True)
    (work_dir / "artifacts" / "index").mkdir(parents=True)
    (work_dir / "artifacts" / "evaluation").mkdir(parents=True)
    (work_dir / "configs").mkdir(parents=True)
    (work_dir / "scripts").mkdir(parents=True)
    (work_dir / "src").symlink_to(REPO_ROOT / "src")
    (work_dir / "scripts" / "run_all.py").symlink_to(REPO_ROOT / "scripts" / "run_all.py")
    (work_dir / "scripts" / "prepare_data.py").symlink_to(REPO_ROOT / "scripts" / "prepare_data.py")
    (work_dir / "scripts" / "build_index.py").symlink_to(REPO_ROOT / "scripts" / "build_index.py")
    (work_dir / "scripts" / "create_golden_template.py").symlink_to(REPO_ROOT / "scripts" / "create_golden_template.py")
    (work_dir / "configs" / "config.yaml").symlink_to(REPO_ROOT / "configs" / "config.yaml")
    (work_dir / "configs" / "intents.yaml").symlink_to(REPO_ROOT / "configs" / "intents.yaml")
    (work_dir / "configs" / "escalation.yaml").symlink_to(REPO_ROOT / "configs" / "escalation.yaml")

    import shutil

    shutil.copy(FIXTURE, work_dir / "data" / "raw" / "twcs.csv")

    result = subprocess.run(
        [sys.executable, "scripts/run_all.py", "--brand", "BrandA_Support", "--fast", "--sample-size", "100"],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    out_dir = work_dir / "artifacts" / "evaluation"
    for filename in ("predictions.jsonl", "results.json", "results.csv", "confusion_matrix.png", "report.json", "failure_analysis.json"):
        assert (out_dir / filename).exists(), f"missing {filename}\n{result.stdout}"
