"""The OpenClaw-RL candidate gate: publish the climb, reject the collapse.

The stream's failure mode is that a few steps past adaptation the policy stops
answering; the default ``AlwaysSelect`` publishes those steps anyway. This gate
probes each candidate on a fixed set and rejects a regression, so serving holds
the last good weights. These tests pin that behaviour without a model: a fake
runtime returns canned probe replies, so the gate's scoring and its
best-checkpoint selection are exercised directly.
"""

from __future__ import annotations

import pytest

from recipes.openclawrl.candidate_evaluator import build
from reef.train.evaluation.contracts import UpdateCandidate

_CLEAN = (
    "She sold 48 clips in April. In May she sold half as many, so 48 divided by 2 "
    "equals 24. Altogether that is 48 plus 24 equals 72 clips."
)
_MARKDOWN = "**Answer:** 72\n- April: 48\n- May: 24\n"
_EMPTY = ""  # a no-reply / tool-loop turn


class _FakeEngine:
    def render_prompt(self, messages):
        # The gate only forwards these to probe_candidate, which the fake
        # overrides, so any stable token stand-in is fine.
        return tuple(messages[0]["content"])


class _FakeRuntime:
    """A runtime whose candidate probe returns whatever the test scripts."""

    def __init__(self) -> None:
        self.engine = _FakeEngine()
        self.replies: list[str] = []

    def probe_candidate(self, candidate_id, prompts, *, max_tokens):
        return list(self.replies)


def _gate(margin: float = 0.17):
    runtime = _FakeRuntime()
    gate = build(
        {"probe_size": 6, "max_tokens": 64, "regression_margin": margin},
        runtime=runtime,
        scenario="gsm8k",
        environ={},
    )
    return gate, runtime


def _decide(gate, runtime, replies):
    runtime.replies = replies
    candidate = UpdateCandidate(candidate_id="c")
    evaluation = gate.evaluate(candidate)
    decision = gate.decide(candidate, evaluation)
    return decision, evaluation


def test_the_criterion_scores_clean_markdown_and_empty_replies() -> None:
    gate, runtime = _gate()
    # Six probes: three clean, one markdown, one empty, one clean -> 4/6 clean.
    _, evaluation = _decide(gate, runtime, [_CLEAN, _CLEAN, _MARKDOWN, _EMPTY, _CLEAN, _CLEAN])
    assert evaluation.metrics["n_clean"] == 4
    assert evaluation.metrics["n_total"] == 6
    assert evaluation.metrics["clean_rate"] == pytest.approx(4 / 6)
    # The empty turn is not even an answer; markdown answered but not clean.
    assert evaluation.metrics["answered_rate"] == pytest.approx(5 / 6)


def test_it_admits_the_adaptation_climb() -> None:
    gate, runtime = _gate()
    for replies, expected_best in (
        ([_CLEAN, _CLEAN, _EMPTY, _EMPTY, _EMPTY, _EMPTY], 2 / 6),
        ([_CLEAN, _CLEAN, _CLEAN, _CLEAN, _EMPTY, _EMPTY], 4 / 6),
        ([_CLEAN] * 6, 1.0),
    ):
        decision, _ = _decide(gate, runtime, replies)
        assert decision.selected
        assert gate._best == pytest.approx(expected_best)


def test_it_rejects_the_post_adaptation_collapse() -> None:
    gate, runtime = _gate()
    _decide(gate, runtime, [_CLEAN] * 6)  # peak: best -> 1.0
    # The policy stops answering: every probe is empty.
    collapse, _ = _decide(gate, runtime, [_EMPTY] * 6)
    assert not collapse.selected
    assert "holding" in collapse.reason
    # A near-empty step is still below the bar (1.0 - 0.17 = 0.83).
    barely, _ = _decide(gate, runtime, [_CLEAN, _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY])
    assert not barely.selected
    # Rejection does not move the bar, so recovery is still measured against 1.0.
    assert gate._best == pytest.approx(1.0)


def test_a_within_margin_dip_is_still_admitted() -> None:
    gate, runtime = _gate(margin=0.17)
    _decide(gate, runtime, [_CLEAN] * 6)  # best -> 1.0
    dip, _ = _decide(gate, runtime, [_CLEAN] * 5 + [_EMPTY])  # 5/6 = 0.83 >= 0.83
    assert dip.selected


def test_build_refuses_a_runtime_that_cannot_probe_a_candidate() -> None:
    class _NoProbe:
        engine = _FakeEngine()

    with pytest.raises(ValueError, match="probe_candidate"):
        build({}, runtime=_NoProbe(), scenario="gsm8k", environ={})
