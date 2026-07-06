"""Tests for session_db propagation to temporary compression agents.

When Gateway hygiene compression or the /compress command creates a temporary
AIAgent for context compression, the agent MUST receive ``session_db`` so that
``compress_context()`` can perform session rotation (end old → create new with
parent_session_id).  Without ``session_db``, rotation is skipped but
``rewrite_transcript()`` still runs unconditionally, overwriting the original
session's messages with a handful of compressed summaries — irreversible data
loss.

Regression test for the local fix that adds ``session_db=self._session_db`` to
both the hygiene agent (gateway/run.py) and the /compress temp agent
(gateway/slash_commands.py).
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from gateway import run as gateway_run
from gateway import slash_commands


# ── Helpers ──────────────────────────────────────────────────────────


def _find_aigent_calls(source: str, agent_var: str) -> list[dict]:
    """Find all ``AIAgent(...)`` calls that assign to *agent_var* in *source*.

    Returns a list of dicts with keys:
        lineno    — line number of the call
        has_session_db — True iff ``session_db=`` is among the keyword args
        has_session_id — True iff ``session_id=`` is among the keyword args
    """
    tree = ast.parse(textwrap.dedent(source))
    results: list[dict] = []

    class _Visitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            # Match: _hyg_agent = AIAgent(...) or tmp_agent = AIAgent(...)
            if len(node.targets) != 1:
                return
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id != agent_var:
                return
            call = node.value
            if not isinstance(call, ast.Call):
                return
            # Check if it's AIAgent(...)
            func = call.func
            if isinstance(func, ast.Name) and func.id == "AIAgent":
                keywords = {kw.arg for kw in call.keywords if kw.arg}
                results.append({
                    "lineno": node.lineno,
                    "has_session_db": "session_db" in keywords,
                    "has_session_id": "session_id" in keywords,
                })
            self.generic_visit(node)

    _Visitor().visit(tree)
    return results


# ── Tests: gateway/run.py hygiene agent ──────────────────────────────


class TestHygieneAgentSessionDB:
    """Verify that the hygiene compression agent in gateway/run.py receives
    ``session_db=self._session_db``."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = inspect.getsource(gateway_run)

    def test_hygiene_agent_receives_session_db(self):
        """The hygiene agent AIAgent() call must include session_db."""
        calls = _find_aigent_calls(self.source, "_hyg_agent")
        assert calls, (
            "No ``_hyg_agent = AIAgent(...)`` call found in gateway/run.py. "
            "Either the variable was renamed or the hygiene compression logic "
            "was restructured."
        )
        # There should be exactly one hygiene agent creation
        assert len(calls) == 1, (
            f"Expected exactly 1 ``_hyg_agent = AIAgent(...)`` call, found {len(calls)}"
        )
        call = calls[0]
        assert call["has_session_id"], (
            f"Hygiene agent at line {call['lineno']} is missing ``session_id=`` parameter"
        )
        assert call["has_session_db"], (
            f"Hygiene agent at line {call['lineno']} is missing ``session_db=`` parameter. "
            "Without session_db, compress_context() skips session rotation and "
            "rewrite_transcript() overwrites the original session's messages — "
            "causing irreversible data loss."
        )


# ── Tests: gateway/slash_commands.py /compress agent ─────────────────


class TestCompressAgentSessionDB:
    """Verify that the /compress temporary agent in slash_commands.py receives
    ``session_db=self._session_db``."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = inspect.getsource(slash_commands)

    def test_compress_agent_receives_session_db(self):
        """The /compress temp agent AIAgent() call must include session_db."""
        calls = _find_aigent_calls(self.source, "tmp_agent")
        assert calls, (
            "No ``tmp_agent = AIAgent(...)`` call found in slash_commands.py. "
            "Either the variable was renamed or the /compress logic was restructured."
        )
        # There should be at least one tmp_agent creation for /compress
        assert len(calls) >= 1, (
            f"Expected at least 1 ``tmp_agent = AIAgent(...)`` call, found {len(calls)}"
        )
        for call in calls:
            assert call["has_session_id"], (
                f"/compress agent at line {call['lineno']} is missing ``session_id=`` parameter"
            )
            assert call["has_session_db"], (
                f"/compress agent at line {call['lineno']} is missing ``session_db=`` parameter. "
                "Without session_db, compress_context() skips session rotation and "
                "rewrite_transcript() overwrites the original session's messages — "
                "causing irreversible data loss."
            )


# ── Tests: behavioral verification with state.db ─────────────────────


class TestSessionRotationBehavior:
    """Simulate the compress_context session rotation logic with a real
    SQLite database to verify that original messages are preserved."""

    @pytest.fixture
    def db_path(self, tmp_path):
        """Create a temporary database with a session and messages."""
        import sqlite3

        db_file = str(tmp_path / "test_state.db")
        conn = sqlite3.connect(db_file)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                source TEXT DEFAULT '',
                model TEXT DEFAULT '',
                created_at REAL,
                ended_at REAL,
                end_reason TEXT,
                parent_session_id TEXT,
                message_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT,
                content TEXT,
                timestamp REAL,
                active INTEGER NOT NULL DEFAULT 1
            );
        """)
        # Insert a session with 10 messages
        conn.execute(
            "INSERT INTO sessions (id, source, created_at, message_count) VALUES (?, 'test', ?, 10)",
            ("original-session", 1000.0),
        )
        import time as _time
        ts = 1000.0
        for i in range(10):
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, active) VALUES (?, ?, ?, ?, 1)",
                ("original-session", "user" if i % 2 == 0 else "assistant", f"message {i}", ts),
            )
            ts += 1.0
        conn.commit()
        conn.close()
        return db_file

    def test_rotation_preserves_original_messages(self, db_path):
        """After session rotation, original session messages must be intact."""
        import sqlite3
        import time
        import uuid
        from datetime import datetime

        conn = sqlite3.connect(db_path)

        old_sid = "original-session"
        old_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ? AND active = 1",
            (old_sid,),
        ).fetchone()[0]
        assert old_count == 10

        # Simulate compress_context session rotation
        new_sid = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # end_session
        conn.execute(
            "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ? AND ended_at IS NULL",
            (time.time(), "compression", old_sid),
        )
        # create_session with parent
        conn.execute(
            "INSERT INTO sessions (id, source, created_at, parent_session_id, message_count) VALUES (?, 'test', ?, ?, 0)",
            (new_sid, time.time(), old_sid),
        )
        # rewrite_transcript on NEW session (simulates replace_messages on new_sid)
        compressed = [
            ("user", "[summary 1]"),
            ("assistant", "[summary 2]"),
        ]
        ts = time.time()
        for role, content in compressed:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, active) VALUES (?, ?, ?, ?, 1)",
                (new_sid, role, content, ts),
            )
            ts += 1e-6
        conn.execute(
            "UPDATE sessions SET message_count = ? WHERE id = ?",
            (len(compressed), new_sid),
        )
        conn.commit()

        # Verify: old session messages still intact
        old_count_after = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ? AND active = 1",
            (old_sid,),
        ).fetchone()[0]
        assert old_count_after == 10, (
            f"Old session lost messages: was 10, now {old_count_after}. "
            "Session rotation should preserve original messages."
        )

        # Verify: new session has compressed messages
        new_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ? AND active = 1",
            (new_sid,),
        ).fetchone()[0]
        assert new_count == 2

        # Verify: parent link
        parent = conn.execute(
            "SELECT parent_session_id FROM sessions WHERE id = ?", (new_sid,)
        ).fetchone()[0]
        assert parent == old_sid

        conn.close()

    def test_no_rotation_with_missing_session_db_overwrites(self, db_path):
        """Without session_db (old buggy behavior), rewrite goes to the SAME
        session and destroys original messages. This test documents the bug."""
        import sqlite3
        import time

        conn = sqlite3.connect(db_path)

        old_sid = "original-session"
        old_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ? AND active = 1",
            (old_sid,),
        ).fetchone()[0]
        assert old_count == 10

        # BUG PATH: no session rotation (session_db was None)
        # session_id stays the same
        new_sid = old_sid  # unchanged

        # rewrite_transcript is called unconditionally on the original session
        # This is the replace_messages DELETE + INSERT:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (old_sid,))
        compressed = [
            ("user", "[summary 1]"),
            ("assistant", "[summary 2]"),
        ]
        ts = time.time()
        for role, content in compressed:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, active) VALUES (?, ?, ?, ?, 1)",
                (old_sid, role, content, ts),
            )
            ts += 1e-6
        conn.commit()

        # BUG: original 10 messages are gone, replaced by 2
        count_after = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ? AND active = 1",
            (old_sid,),
        ).fetchone()[0]
        assert count_after == 2, (
            f"Expected 2 (compressed) but got {count_after}. "
            "This demonstrates the data loss bug when session_db is not passed."
        )

        conn.close()
