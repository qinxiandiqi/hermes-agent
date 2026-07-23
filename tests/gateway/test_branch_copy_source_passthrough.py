"""Tests for /branch copy loops preserving ``source=`` and ``timestamp=``.

Regression tests for the anti-laundering fix that adds explicit
``source=`` and ``timestamp=`` kwargs to all four production /branch
copy loops plus the underlying ``SessionStore.append_to_transcript``
kwarg surface:

    - ``gateway/slash_commands.py``     : async gateway RPC
    - ``hermes_cli/cli_commands_mixin.py`` : CLI in-memory history (sync)
    - ``tui_gateway/server.py``         : ``_persist_branch_seed`` (sync)
    - ``tui_gateway/server.py``         : ``@method("session.branch")`` RPC (sync)
    - ``gateway/session.py``            : ``SessionStore.append_to_transcript`` kwarg passthrough

Before the fix all 4 branch callers wrote rows with ``source=NULL`` and
an auto-generated ``time.time()`` timestamp, even when the parent row
carried ``source="compaction-replay"`` or a stable historical timestamp.

Strictly pass-through semantics: ``_source=None`` (or missing) on the
parent means ``NULL`` on the branch — NOT ``"real"``. CLI's
``conversation_history`` may drop ``_source`` because the agent's flush
only writes the DB column (not the live dict); in that case branch copy
honestly writes NULL and lets the next agent flush back-fill on the new
session's first turn.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from hermes_state import AsyncSessionDB, SessionDB


# ── Helpers ─────────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    """Standalone in-tmpdir SessionDB for low-level assertions."""
    return SessionDB(tmp_path / "state.db")


# ── Underlying DB layer contract ─────────────────────────────────────
#
# The four branch callers all call SessionDB.append_message(source=...,
# timestamp=...). These tests pin the contract that contract depends on.


@pytest.mark.asyncio
async def test_append_message_persists_explicit_source(db: SessionDB) -> None:
    """``source='compaction-replay'`` round-trips through DB read."""
    sdb = AsyncSessionDB(db)
    await sdb.create_session(session_id="parent", source="t")
    await sdb.create_session(session_id="branch", source="t")

    await sdb.append_message(
        session_id="parent",
        role="user",
        content="hi",
        source="compaction-replay",
        timestamp=1234567890.0,
    )

    history = db.get_messages_as_conversation("parent", repair_alternation=False)
    assert len(history) == 1
    assert history[0]["_source"] == "compaction-replay", (
        f"_source should round-trip; got {history[0].get('_source')!r}"
    )
    assert abs(history[0]["timestamp"] - 1234567890.0) < 1e-3


@pytest.mark.asyncio
async def test_branch_copy_preserves_compaction_replay_source(db: SessionDB) -> None:
    """The full /branch copy loop preserves parent ``source='compaction-replay'``."""
    sdb = AsyncSessionDB(db)
    await sdb.create_session(session_id="parent", source="t")
    await sdb.create_session(session_id="branch", source="t")

    await sdb.append_message(
        session_id="parent",
        role="user",
        content="hi",
        source="compaction-replay",
        timestamp=1234567890.0,
    )
    history = db.get_messages_as_conversation("parent", repair_alternation=False)

    # Mirror the production /branch copy loop exactly.
    for msg in history:
        await sdb.append_message(
            session_id="branch",
            role=msg.get("role", "user"),
            content=msg.get("content"),
            source=msg.get("_source"),
            timestamp=msg.get("timestamp"),
        )

    branch_rows = db.get_messages("branch")
    assert len(branch_rows) == 1
    assert branch_rows[0]["source"] == "compaction-replay"
    assert abs(branch_rows[0]["timestamp"] - 1234567890.0) < 1e-3


@pytest.mark.asyncio
async def test_branch_copy_strict_passthrough_no_or_real_fallback(
    db: SessionDB,
) -> None:
    """History rows with no ``_source`` ⇒ branch row has ``source=NULL``.

    Anti-laundering contract: ``or "real"`` would falsely mark metadata
    rows (session_meta, system recall prompts) as live conversation.
    Strict pass-through keeps the branch honest.
    """
    sdb = AsyncSessionDB(db)
    await sdb.create_session(session_id="branch", source="t")

    history = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "y"},
    ]
    for msg in history:
        await sdb.append_message(
            session_id="branch",
            role=msg.get("role", "user"),
            content=msg.get("content"),
            source=msg.get("_source"),  # None
            timestamp=msg.get("timestamp"),  # None
        )

    rows = db.get_messages("branch")
    assert [r["source"] for r in rows] == [None, None], (
        f"strict pass-through: expected [None, None], "
        f"got {[r['source'] for r in rows]!r}"
    )


# ── Static regression guard ─────────────────────────────────────────
#
# The four call sites are pinned by line-window inspection so future
# refactors that drop the kwargs trip the test before they hit
# production.


# (file_relative_to_repo, inclusive_start_line, inclusive_end_line)
# Line windows anchored on ``source=msg.get("_source")`` after the
# upstream-main merge (1198 commits absorbed). Each window is 16 lines
# wide to comfortably contain the call site + surrounding kwargs.
_BRANCH_COPY_LINES = {
    "gateway/slash_commands.py": (4164, 4180),
    "hermes_cli/cli_commands_mixin.py": (965, 981),
    "tui_gateway/server.py": (2120, 2144),
    "tui_gateway/server.py": (9505, 9520),
}


def test_all_branch_copy_loops_pass_source_and_timestamp() -> None:
    """Each /branch copy site must pass ``source=msg.get("_source")`` and
    ``timestamp=msg.get("timestamp")`` to ``append_message``.
    """
    project_root = Path(__file__).parent.parent.parent
    for rel_path, (start, end) in _BRANCH_COPY_LINES.items():
        path = project_root / rel_path
        lines = path.read_text(encoding="utf-8").splitlines()
        snippet = "\n".join(lines[start - 1:end])
        assert 'source=msg.get("_source")' in snippet, (
            f"{rel_path}:L{start}-{end} must pass "
            f'source=msg.get("_source") to append_message.\nSnippet:\n{snippet}'
        )
        assert 'timestamp=msg.get("timestamp")' in snippet, (
            f"{rel_path}:L{start}-{end} must pass "
            f'timestamp=msg.get("timestamp") to append_message.\nSnippet:\n{snippet}'
        )


def test_session_store_append_to_transcript_accepts_source_kwarg() -> None:
    """``SessionStore.append_to_transcript`` must declare ``source=`` (canonical name).

    Located in ``gateway/session.py`` ~L2503. ``AsyncSessionStore``
    forwards attribute access via ``__getattr__`` so the underlying
    ``SessionStore`` signature is what callers actually invoke.
    """
    from gateway import session as gateway_session

    sig = inspect.signature(gateway_session.SessionStore.append_to_transcript)
    assert "source" in sig.parameters, (
        "SessionStore.append_to_transcript must declare a source= kwarg so "
        "gateway callers can mark explicit anti-laundering intent."
    )
    assert sig.parameters["source"].default is None
