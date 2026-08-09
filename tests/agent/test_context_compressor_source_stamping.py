"""Regression tests for the compress() ``_source='compaction-replay'`` stamping contract.

The context-compressor's documented contract (context_compressor.py:197-198) is that
*every* message in the compressed output — head + summary + tail — carries
``_source='compaction-replay'`` so downstream persistence can tag it as a compaction
replay copy, distinct from genuine ``real`` conversation turns (which
``--no-compaction`` in session_search must keep).

Upstream merge ``c656bec9e`` (2026-07-23) dropped the stamp from the **tail loop only**,
so tail replicas persisted as ``source=NULL`` (live input) or inherited the original's
``real`` label (cold-resume input). No existing test asserted the stamping invariant, so
the regression slipped through. These tests close that gap:

- ``test_compress_stamps_every_surviving_message_as_compaction_replay`` — the core
  contract: every surviving message (head + summary + tail) must carry the stamp.
  This test FAILS on the current (buggy) code because the tail block omits it.
- ``test_compress_output_always_contains_a_stamped_message`` — structural guard that at
  least the summary/head blocks stamp correctly (so the failure above is attributable to
  the tail, not to the whole stamping feature being absent).
"""

from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    REPLAY_SOURCE_METADATA_KEY,
    REPLAY_SOURCE_VALUE,
    ContextCompressor,
)


def _compressor(protect_last_n: int = 3) -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=1,
            protect_last_n=protect_last_n,
            quiet_mode=True,
        )


def _response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _long_turns(count: int) -> list:
    """A conversation long enough to force real compaction with a surviving tail."""
    turns = [{"role": "system", "content": "system prompt"}]
    for i in range(count):
        turns.append({"role": "user", "content": f"user turn {i} " + "x" * 60})
        turns.append({"role": "assistant", "content": f"assistant turn {i} " + "y" * 60})
    return turns


def test_compress_stamps_every_surviving_message_as_compaction_replay():
    """Core contract: every compressed message (head + summary + tail) must carry
    ``_source='compaction-replay'``.

    Fails on current code because the tail loop appends tail replicas without the
    stamp — replicas then persist as NULL or inherit the original's `real` label.
    """
    compressor = _compressor(protect_last_n=3)
    messages = _long_turns(12)

    with patch(
        "agent.context_compressor.call_llm", return_value=_response("middle was summarized")
    ):
        compressed = compressor.compress(messages, current_tokens=90_000)

    # Sanity: compression actually ran and produced survivors (head + tail).
    assert len(compressed) < len(messages), "compress() did not shrink the transcript"
    assert len(compressed) >= 2, "compress() produced no surviving messages to check"

    unstamped = [
        msg
        for msg in compressed
        if isinstance(msg, dict)
        and msg.get(REPLAY_SOURCE_METADATA_KEY) != REPLAY_SOURCE_VALUE
    ]
    assert unstamped == [], (
        f"{len(unstamped)} surviving compressed message(s) lack "
        f"{REPLAY_SOURCE_METADATA_KEY}={REPLAY_SOURCE_VALUE!r}: "
        f"{[(m.get('role'), m.get('content', '')[:24]) for m in unstamped]}"
    )


def test_compress_output_always_contains_a_stamped_message():
    """Structural guard: when a tail exists, the output must contain *at least one*
    stamped message (head/summary). Isolates the tail-only regression from a
    hypothetical whole-feature absence."""
    compressor = _compressor(protect_last_n=3)
    messages = _long_turns(12)

    with patch(
        "agent.context_compressor.call_llm", return_value=_response("middle was summarized")
    ):
        compressed = compressor.compress(messages, current_tokens=90_000)

    assert any(
        isinstance(m, dict) and m.get(REPLAY_SOURCE_METADATA_KEY) == REPLAY_SOURCE_VALUE
        for m in compressed
    ), "no message in the compressed output carries _source='compaction-replay'"