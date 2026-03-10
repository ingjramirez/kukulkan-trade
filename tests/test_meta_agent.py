"""Tests for the meta-agent self-improvement runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.analysis.meta_agent import MetaAgentRunner


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.get_snapshots.return_value = []
    db.get_positions.return_value = []
    db.get_trades.return_value = []
    db.get_agent_decisions.return_value = []
    db.get_tenant.return_value = MagicMock(
        strategy_mode="aggressive",
        trailing_stop_multiplier=1.0,
        ticker_exclusions=None,
    )
    db.get_improvement_snapshots.return_value = []
    db.get_parameter_changelog.return_value = []
    return db


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a temporary repo directory."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".venv" / "bin").mkdir(parents=True)
    return repo


@pytest.fixture
def runner(mock_db, tmp_repo):
    return MetaAgentRunner(
        db=mock_db,
        tenant_id="default",
        repo_path=tmp_repo,
        max_turns=5,
        timeout_s=30,
        model="opus",
    )


class TestSessionManagement:
    def test_get_session_id_no_file(self, runner: MetaAgentRunner):
        assert runner._get_session_id() is None

    def test_save_and_get_session_id(self, runner: MetaAgentRunner):
        runner._save_session_id("sess-123")
        assert runner._get_session_id() == "sess-123"

    def test_get_session_id_corrupt_file(self, runner: MetaAgentRunner, tmp_repo: Path):
        (tmp_repo / ".meta-session-id").write_text("not json")
        assert runner._get_session_id() is None


class TestBuildCmd:
    def test_without_session(self, runner: MetaAgentRunner):
        cmd = runner._build_cmd("test prompt", None)
        assert "claude" in cmd
        assert "-p" in cmd
        assert "test prompt" in cmd
        assert "--resume" not in cmd
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "opus"

    def test_with_session(self, runner: MetaAgentRunner):
        cmd = runner._build_cmd("test prompt", "sess-456")
        assert "--resume" in cmd
        idx = cmd.index("--resume")
        assert cmd[idx + 1] == "sess-456"

    def test_max_turns(self, runner: MetaAgentRunner):
        cmd = runner._build_cmd("p", None)
        idx = cmd.index("--max-turns")
        assert cmd[idx + 1] == "5"


class TestParseOutput:
    def test_empty(self, runner: MetaAgentRunner):
        assert runner._parse_output("") == {}

    def test_valid_json(self, runner: MetaAgentRunner):
        data = {"result": "some text", "session_id": "abc"}
        result = runner._parse_output(json.dumps(data))
        assert result["session_id"] == "abc"
        assert result["result"] == "some text"

    def test_invalid_json(self, runner: MetaAgentRunner):
        result = runner._parse_output("not json at all")
        assert "result" in result


class TestExtractPrUrls:
    def test_single_url(self, runner: MetaAgentRunner):
        text = "Created PR: https://github.com/user/repo/pull/42"
        assert runner._extract_pr_urls(text) == ["https://github.com/user/repo/pull/42"]

    def test_multiple_urls(self, runner: MetaAgentRunner):
        text = (
            "PR1: https://github.com/user/repo/pull/1 "
            "PR2: https://github.com/user/repo/pull/2"
        )
        assert len(runner._extract_pr_urls(text)) == 2

    def test_no_urls(self, runner: MetaAgentRunner):
        assert runner._extract_pr_urls("no PRs here") == []


class TestExtractSummary:
    def test_json_summary(self, runner: MetaAgentRunner):
        text = json.dumps({"summary": "Changed signal weights"})
        assert runner._extract_summary(text) == "Changed signal weights"

    def test_plain_text(self, runner: MetaAgentRunner):
        assert runner._extract_summary("plain text") == "plain text"

    def test_empty(self, runner: MetaAgentRunner):
        assert runner._extract_summary("") == ""


class TestSyncRepo:
    @patch("src.analysis.meta_agent.subprocess.run")
    def test_success(self, mock_run, runner: MetaAgentRunner):
        mock_run.return_value = MagicMock(returncode=0)
        assert runner._sync_repo() is True
        assert mock_run.call_count == 4  # fetch, checkout, reset, clean

    @patch("src.analysis.meta_agent.subprocess.run")
    def test_failure(self, mock_run, runner: MetaAgentRunner):
        from subprocess import CalledProcessError

        mock_run.side_effect = CalledProcessError(1, "git")
        assert runner._sync_repo() is False


class TestCollectContext:
    async def test_produces_markdown(self, runner: MetaAgentRunner):
        ctx = await runner._collect_context()
        assert "# Meta-Agent Context" in ctx
        assert "## Current Positions" in ctx
        assert "## Key Source Files" in ctx

    async def test_includes_positions(self, mock_db, tmp_repo):
        pos = MagicMock(
            ticker="AAPL",
            shares=10,
            avg_price=150.0,
            current_price=160.0,
        )
        mock_db.get_positions.return_value = [pos]
        runner = MetaAgentRunner(db=mock_db, repo_path=tmp_repo)
        ctx = await runner._collect_context()
        assert "AAPL" in ctx
        assert "+6.7%" in ctx


class TestRun:
    @patch("src.analysis.meta_agent.asyncio.to_thread")
    @patch("src.analysis.meta_agent.subprocess.run")
    async def test_full_run_with_pr(self, mock_sync_run, mock_to_thread, runner: MetaAgentRunner):
        mock_sync_run.return_value = MagicMock(returncode=0)

        cli_output = json.dumps({
            "result": "Created PR: https://github.com/user/repo/pull/99\nChanged signal weights.",
            "session_id": "new-sess",
        })
        mock_to_thread.return_value = MagicMock(
            returncode=0,
            stdout=cli_output,
            stderr="",
        )

        result = await runner.run()
        assert result.status == "completed"
        assert result.session_id == "new-sess"
        assert "https://github.com/user/repo/pull/99" in result.prs_opened
        # Session should be persisted
        assert runner._get_session_id() == "new-sess"

    @patch("src.analysis.meta_agent.subprocess.run")
    async def test_sync_failure(self, mock_run, runner: MetaAgentRunner):
        from subprocess import CalledProcessError

        mock_run.side_effect = CalledProcessError(1, "git")
        result = await runner.run()
        assert result.status == "error"
        assert "sync" in result.error.lower()

    @patch("src.analysis.meta_agent.asyncio.to_thread")
    @patch("src.analysis.meta_agent.subprocess.run")
    async def test_no_changes(self, mock_sync_run, mock_to_thread, runner: MetaAgentRunner):
        mock_sync_run.return_value = MagicMock(returncode=0)

        cli_output = json.dumps({
            "result": "No clear improvement found this week.",
            "session_id": "sess-2",
        })
        mock_to_thread.return_value = MagicMock(
            returncode=0,
            stdout=cli_output,
            stderr="",
        )

        result = await runner.run()
        assert result.status == "no_changes"
        assert result.prs_opened == []

    @patch("src.analysis.meta_agent.asyncio.to_thread")
    @patch("src.analysis.meta_agent.subprocess.run")
    async def test_notify_on_pr(self, mock_sync_run, mock_to_thread, runner: MetaAgentRunner):
        mock_sync_run.return_value = MagicMock(returncode=0)

        cli_output = json.dumps({
            "result": "https://github.com/user/repo/pull/5",
            "session_id": "s1",
        })
        mock_to_thread.return_value = MagicMock(returncode=0, stdout=cli_output, stderr="")

        notifier = AsyncMock()
        await runner.run(notifier=notifier)
        notifier.send_message.assert_called_once()
        call_text = notifier.send_message.call_args[0][0]
        assert "pull/5" in call_text
