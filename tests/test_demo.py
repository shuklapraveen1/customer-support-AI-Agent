import subprocess
import sys
from pathlib import Path

import pytest

from src.agent import AgentResult
from src.demo import print_result
from src.retrieval.search import RetrievalResult

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "twcs_sample.csv"


def _fake_result(reply=None, decision="auto_handle", evidence=None):
    return AgentResult(
        customer_message="my order never arrived",
        intent="delivery_delay",
        intent_confidence=0.83,
        intent_reason="test",
        retrieved_evidence=evidence or [],
        reply=reply,
        decision=decision,
        reason="all signals passed",
        signals={"intent_confidence": 0.83},
    )


def test_print_result_includes_all_required_sections(capsys):
    evidence = [
        RetrievalResult(
            resolution_id="res_1", customer_message="old message", brand_reply="old reply",
            intent="delivery_delay", similarity_score=0.9, conversation_id="c1", timestamp=None,
        )
    ]
    print_result(_fake_result(reply="Sorry about that!", evidence=evidence))
    out = capsys.readouterr().out

    assert "Customer:" in out
    assert "Intent:" in out
    assert "Confidence:" in out
    assert "Historical evidence:" in out
    assert "Draft reply:" in out
    assert "Decision:" in out
    assert "Reason:" in out
    assert "AUTO_HANDLE" in out
    assert "Sorry about that!" in out


def test_print_result_handles_escalate_with_no_reply(capsys):
    print_result(_fake_result(reply=None, decision="escalate"))
    out = capsys.readouterr().out
    assert "ESCALATE" in out
    assert "no reply drafted" in out.lower()


def test_print_result_handles_no_evidence(capsys):
    print_result(_fake_result(reply="hi", evidence=[]))
    out = capsys.readouterr().out
    assert "none" in out.lower()


@pytest.mark.slow
def test_demo_cli_one_shot_message(tmp_path):
    """Subprocess smoke test of `python -m src.demo --message ...` against
    an isolated copy of the repo pointed at the tiny fixture — no retrieval
    corpus is built, so this exercises the 'no corpus found, continuing
    with empty searcher' path specifically."""
    work_dir = tmp_path / "workdir"
    (work_dir / "configs").mkdir(parents=True)
    (work_dir / "src").symlink_to(REPO_ROOT / "src")
    (work_dir / "configs" / "intents.yaml").symlink_to(REPO_ROOT / "configs" / "intents.yaml")
    (work_dir / "configs" / "escalation.yaml").symlink_to(REPO_ROOT / "configs" / "escalation.yaml")

    result = subprocess.run(
        [sys.executable, "-m", "src.demo", "--message", "My refund still hasn't arrived"],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Customer:" in result.stdout
    assert "Decision:" in result.stdout
    assert "mock" in result.stdout
