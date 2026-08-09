"""Tests for the ``messages.source`` field persistence contract.

The ``source`` column has three states:

- ``'real'``             — a genuine conversation turn.
- ``'compaction-replay'`` — a replica copy produced by context compression.
- ``NULL``                — unknown/legacy, or a strict-passthrough miss.

These tests pin the WRITE-path contract so regressions like the 2026-08-09
tail-stamp miss are caught at the persistence layer (not just in the
compressor unit test). That incident: upstream merge ``c656bec9e`` dropped
``_source='compaction-replay'`` from the compressor's tail loop, so tail
replicas persisted as ``source=NULL`` (live input) or inherited ``'real'``
(cold-resume input) — 350 polluted rows in state.db.

Covered here:
  - ``archive_and_compact`` → ``_insert_message_rows`` strict pass-through:
    a present ``_source`` is written verbatim; a missing one is ``NULL``
    (NOT ``'real'``) — the anti-laundering contract.
  - ``get_messages()`` surfaces each persisted source as in-memory ``_source``.
  - ``run_agent._flush_messages_to_session_db`` rotation-path fallback:
    missing ``_source`` ⇒ ``'real'`` (the ``or "real"`` at run_agent.py:2045);
    present ``_source='compaction-replay'`` survives verbatim.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_state import SessionDB

REPLAY = "compaction-replay"


# ---------------------------------------------------------------------------
# DB layer: _insert_message_rows strict pass-through (via archive_and_compact)
# ---------------------------------------------------------------------------


def test_archive_and_compact_writes_source_verbatim_and_null_when_missing(tmp_path):
    """``_insert_message_rows`` must pass ``_source`` through strictly.

    A message carrying ``_source='compaction-replay'`` persists as such; a
    message WITHOUT ``_source`` persists as ``NULL`` — never ``'real'``. This
    is the anti-laundering contract (:5795) that keeps /retry /undo /branch and
    compaction replays honest.
    """
    db = SessionDB(tmp_path / "state.db")
    sid = "s1"
    db.create_session(sid, source="cli", model="test/model")

    db.archive_and_compact(sid, [
        {"role": "user", "content": "summary marker", "_source": REPLAY},
        {"role": "assistant", "content": "tail copy, no _source"},
    ])

    rows = sorted(db.get_messages(sid), key=lambda r: r["id"])
    by_content = {r["content"]: r["source"] for r in rows}
    assert by_content["summary marker"] == REPLAY
    assert by_content["tail copy, no _source"] is None, (
        "missing _source must persist as NULL, not 'real' (strict pass-through)"
    )


def test_archive_and_compact_does_not_invent_real_for_missing_source(tmp_path):
    """Guard against a future ``or 'real'`` fallback sneaking into the in-place
    compaction path — the exact mislabeling the tail-stamp bug produced."""
    db = SessionDB(tmp_path / "state.db")
    sid = "s2"
    db.create_session(sid, source="cli", model="test/model")

    db.archive_and_compact(sid, [{"role": "user", "content": "no source key"}])

    rows = db.get_messages(sid)
    assert len(rows) == 1
    assert rows[0]["source"] is None


# ---------------------------------------------------------------------------
# DB read layer: get_messages() surfaces persisted source as _source
# ---------------------------------------------------------------------------


def test_get_messages_surfaces_three_source_states(tmp_path):
    """Persisted ``source`` must round-trip into the in-memory ``_source`` key,
    and ``NULL`` must surface as an ABSENT ``_source`` (not a literal null)."""
    db = SessionDB(tmp_path / "state.db")
    sid = "s3"
    db.create_session(sid, source="cli", model="test/model")
    db.append_message(session_id=sid, role="user", content="real one", source="real")
    db.append_message(session_id=sid, role="user", content="replay one", source=REPLAY)
    db.append_message(session_id=sid, role="user", content="null one")  # source defaults None

    by_content = {m["content"]: m for m in db.get_messages(sid)}
    assert by_content["real one"].get("_source") == "real"
    assert by_content["replay one"].get("_source") == REPLAY
    assert "_source" not in by_content["null one"], (
        "NULL source must stay absent in memory, not collapse to a string"
    )


# ---------------------------------------------------------------------------
# run_agent rotation-path fallback: _flush_messages_to_session_db 'or "real"'
# ---------------------------------------------------------------------------


def _make_agent(session_db, session_id):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )


def test_flush_rotation_falls_back_to_real_when_no_source(tmp_path):
    """Rotation-path flush (run_agent.py:2045) must map a missing ``_source`` to
    ``'real'`` — this is the documented fallback for the per-session rotation
    re-append, and it must NOT be removed even though strict paths refuse to."""
    db = SessionDB(tmp_path / "state.db")
    sid = "rot"
    db.create_session(session_id=sid, source="test")
    agent = _make_agent(db, sid)
    agent._last_flushed_db_idx = 0

    agent._flush_messages_to_session_db(
        [
            {"role": "user", "content": "genuine turn"},  # no _source
            {"role": "assistant", "content": "replay", "_source": REPLAY},
        ],
        None,
    )

    rows = sorted(db.get_messages(sid), key=lambda r: r["id"])
    by_content = {r["content"]: r["source"] for r in rows}
    assert by_content["genuine turn"] == "real", "no _source ⇒ 'real' (rotation fallback)"
    assert by_content["replay"] == REPLAY, "_source survives verbatim"


def test_flush_rotation_preserves_compaction_replay_source(tmp_path):
    """The rotation-path fallback must not clobber a present ``'compaction-replay'``
    marker with ``'real'`` — otherwise compressed replays get laundered as real."""
    db = SessionDB(tmp_path / "state.db")
    sid = "rot2"
    db.create_session(session_id=sid, source="test")
    agent = _make_agent(db, sid)
    agent._last_flushed_db_idx = 0

    agent._flush_messages_to_session_db(
        [{"role": "user", "content": "replay copy", "_source": REPLAY}],
        None,
    )

    rows = db.get_messages(sid)
    assert rows[0]["source"] == REPLAY