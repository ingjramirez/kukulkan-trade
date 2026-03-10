"""Meta-agent: self-improving code agent for Portfolio B.

Runs on a schedule via Claude Code CLI against a cloned repo.
Collects performance data, writes context, and lets the agent
propose code changes via GitHub PRs to maximize paper trading returns.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import structlog

from src.storage.database import Database

log = structlog.get_logger()

CONTEXT_FILENAME = "meta-agent-context.md"
SESSION_FILENAME = ".meta-session-id"


@dataclass
class MetaAgentResult:
    """Result of a meta-agent run."""

    status: str = "pending"  # completed, error, skipped, no_changes
    session_id: str | None = None
    prs_opened: list[str] = field(default_factory=list)
    summary: str = ""
    error: str | None = None


class MetaAgentRunner:
    """Self-improvement meta-agent that proposes code changes via PRs."""

    def __init__(
        self,
        db: Database,
        tenant_id: str = "default",
        repo_path: Path | None = None,
        max_turns: int = 25,
        timeout_s: int = 900,
        model: str = "opus",
    ) -> None:
        self._db = db
        self._tenant_id = tenant_id
        self._repo_path = repo_path or Path("/opt/kukulkan-improve")
        self._max_turns = max_turns
        self._timeout_s = timeout_s
        self._model = model
        self._session_file = self._repo_path / SESSION_FILENAME
        self._context_file = self._repo_path / CONTEXT_FILENAME

    async def run(self, notifier: object | None = None) -> MetaAgentResult:
        """Run the meta-agent: collect data, invoke Claude, report."""
        result = MetaAgentResult()

        try:
            # 1. Sync repo to latest main
            if not self._sync_repo():
                result.status = "error"
                result.error = "Failed to sync repo"
                return result

            # 2. Collect performance data and write context
            context_md = await self._collect_context()
            self._context_file.write_text(context_md)
            log.info("meta_agent_context_written", path=str(self._context_file), chars=len(context_md))

            # 3. Get persistent session ID
            session_id = self._get_session_id()

            # 4. Invoke Claude Code CLI
            prompt = self._build_prompt()
            cmd = self._build_cmd(prompt, session_id)

            log.info(
                "meta_agent_starting",
                tenant_id=self._tenant_id,
                session_id=session_id,
                repo=str(self._repo_path),
                model=self._model,
            )

            env = os.environ.copy()
            env["PATH"] = f"{self._repo_path / '.venv' / 'bin'}:{env.get('PATH', '')}"

            proc = await asyncio.to_thread(
                subprocess.run,
                cmd,
                cwd=self._repo_path,
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                env=env,
            )

            # 5. Parse output
            stdout = proc.stdout.strip()
            if proc.returncode != 0 and not stdout:
                log.error("meta_agent_cli_failed", returncode=proc.returncode, stderr=proc.stderr[:500])
                result.status = "error"
                result.error = proc.stderr[:500] if proc.stderr else f"Exit code {proc.returncode}"
                return result

            parsed = self._parse_output(stdout)

            # 6. Save session ID for next run
            new_session_id = parsed.get("session_id")
            if new_session_id:
                self._save_session_id(new_session_id)
                result.session_id = new_session_id

            # 7. Extract PR URLs and summary
            result_text = parsed.get("result", "")
            if isinstance(result_text, dict):
                result_text = json.dumps(result_text)

            result.prs_opened = self._extract_pr_urls(result_text)
            result.summary = self._extract_summary(result_text)
            result.status = "completed" if result.prs_opened else "no_changes"

            log.info(
                "meta_agent_complete",
                tenant_id=self._tenant_id,
                prs=len(result.prs_opened),
                status=result.status,
                summary=result.summary[:200],
            )

            # 8. Notify
            if notifier and hasattr(notifier, "send_message"):
                await self._notify(notifier, result)

        except subprocess.TimeoutExpired:
            result.status = "error"
            result.error = f"Session timed out ({self._timeout_s}s)"
            log.error("meta_agent_timeout", timeout=self._timeout_s)

        except Exception as e:
            result.status = "error"
            result.error = str(e)
            log.exception("meta_agent_failed", error=str(e))

        return result

    # ── Repo sync ──────────────────────────────────────────────────────────

    def _sync_repo(self) -> bool:
        """Reset clone to latest main."""
        try:
            for cmd in [
                ["git", "fetch", "origin"],
                ["git", "checkout", "main"],
                ["git", "reset", "--hard", "origin/main"],
                ["git", "clean", "-fd"],
            ]:
                subprocess.run(cmd, cwd=self._repo_path, capture_output=True, check=True, timeout=60)
            log.info("meta_agent_repo_synced")
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log.error("meta_agent_sync_failed", error=str(e))
            return False

    # ── Data collection ────────────────────────────────────────────────────

    async def _collect_context(self) -> str:
        """Build the context markdown with performance data."""
        today = date.today()
        month_ago = today - timedelta(days=30)
        week_ago = today - timedelta(days=7)
        tid = self._tenant_id
        lines: list[str] = []

        lines.append("# Meta-Agent Context — Performance Review")
        lines.append(f"**Date:** {today.isoformat()}")
        lines.append(f"**Tenant:** {tid}")
        lines.append(f"**Window:** {month_ago} to {today} (30 days)")
        lines.append("")

        # ── Equity curve ──
        lines.append("## Equity Curve (Portfolio B)")
        snapshots = await self._db.get_snapshots("B", tenant_id=tid)
        recent_snaps = [s for s in snapshots if s.date >= month_ago]
        if recent_snaps:
            values = [s.total_value for s in recent_snaps]
            peak = max(values)
            current = values[-1]
            start = values[0]
            total_return = ((current - start) / start) * 100 if start > 0 else 0
            max_dd = ((peak - min(values)) / peak) * 100 if peak > 0 else 0

            lines.append(f"- Start: ${start:,.0f} | Current: ${current:,.0f} | Return: {total_return:+.2f}%")
            lines.append(f"- Peak: ${peak:,.0f} | Max drawdown: {max_dd:.1f}%")
            lines.append(f"- Data points: {len(recent_snaps)}")
            lines.append("")
            lines.append("Daily values:")
            for s in recent_snaps[-14:]:  # Last 2 weeks
                dr = f" ({s.daily_return_pct:+.2f}%)" if s.daily_return_pct else ""
                lines.append(f"  {s.date}: ${s.total_value:,.0f}{dr}")
        else:
            lines.append("  No snapshots available.")
        lines.append("")

        # ── Current positions ──
        lines.append("## Current Positions")
        positions = await self._db.get_positions("B", tenant_id=tid)
        if positions:
            total_mv = 0.0
            for p in positions:
                cp = p.current_price or p.avg_price
                mv = p.shares * cp
                pnl_pct = ((cp - p.avg_price) / p.avg_price) * 100 if p.avg_price else 0
                total_mv += mv
                lines.append(
                    f"- {p.ticker}: {p.shares} @ ${p.avg_price:.2f}, now ${cp:.2f} ({pnl_pct:+.1f}%), ${mv:,.0f}"
                )
            lines.append(f"- **Total market value: ${total_mv:,.0f}**")
        else:
            lines.append("  No positions.")
        lines.append("")

        # ── Recent trades ──
        lines.append("## Recent Trades (30 days)")
        trades = await self._db.get_trades("B", tenant_id=tid)
        recent_trades = [t for t in trades if t.executed_at.date() >= month_ago]
        if recent_trades:
            buys = [t for t in recent_trades if t.side == "BUY"]
            sells = [t for t in recent_trades if t.side == "SELL"]
            lines.append(f"Total: {len(recent_trades)} ({len(buys)} buys, {len(sells)} sells)")
            lines.append("")
            for t in recent_trades[-20:]:
                lines.append(f"  {t.executed_at.date()} {t.side:4} {t.shares:>6.0f}x {t.ticker:<6} @ ${t.price:.2f}")
        else:
            lines.append("  No trades in the last 30 days.")
        lines.append("")

        # ── Trade outcomes / track record ──
        lines.append("## Trade Outcomes")
        try:
            from src.analysis.outcome_tracker import OutcomeTracker
            from src.analysis.track_record import TrackRecord

            tracker = OutcomeTracker(self._db)
            outcomes = await tracker.get_recent_outcomes(days=30, tenant_id=tid)
            if outcomes:
                record = TrackRecord()
                stats = record.compute(outcomes)
                if stats:
                    lines.append(f"- Win rate: {stats.win_rate_pct:.1f}%")
                    lines.append(f"- Avg P&L: {stats.avg_pnl_pct:+.2f}%")
                    if stats.avg_alpha_vs_spy is not None:
                        lines.append(f"- Alpha vs SPY: {stats.avg_alpha_vs_spy:+.2f}%")
                    if stats.best_sector:
                        lines.append(f"- Best sector: {stats.best_sector}")
                    if stats.worst_sector:
                        lines.append(f"- Worst sector: {stats.worst_sector}")

                    if stats.by_sector:
                        lines.append("\nBy sector:")
                        for s in stats.by_sector:
                            lines.append(
                                f"  {s.value}: WR={s.win_rate_pct:.0f}%, P&L={s.avg_pnl_pct:+.1f}%, n={s.total}"
                            )

                # Losers detail
                losers = [o for o in outcomes if o.pnl_pct < -0.5]
                if losers:
                    lines.append(f"\nLosing trades ({len(losers)}):")
                    for o in sorted(losers, key=lambda x: x.pnl_pct)[:10]:
                        lines.append(f"  {o.ticker}: {o.pnl_pct:+.1f}%, sector={o.sector}")
            else:
                lines.append("  No closed trade outcomes yet.")
        except Exception as e:
            lines.append(f"  (outcome tracking unavailable: {e})")
        lines.append("")

        # ── Agent decisions ──
        lines.append("## Recent Agent Decisions")
        decisions = await self._db.get_agent_decisions(limit=10, tenant_id=tid)
        week_decisions = [d for d in decisions if d.date >= week_ago]
        if week_decisions:
            for d in week_decisions:
                reasoning = (d.reasoning or "")[:150]
                regime = getattr(d, "regime", "") or ""
                lines.append(f"  {d.date}: regime={regime}, {reasoning}")
        else:
            lines.append("  No decisions this week.")
        lines.append("")

        # ── Current configuration ──
        lines.append("## Current Configuration")
        tenant = await self._db.get_tenant(tid)
        if tenant:
            lines.append(f"- Strategy mode: {tenant.strategy_mode}")
            trail_mult = getattr(tenant, "trailing_stop_multiplier", None) or 1.0
            lines.append(f"- Trailing stop multiplier: {trail_mult}")
            exclusions = "[]"
            if tenant.ticker_exclusions:
                exclusions = tenant.ticker_exclusions
            lines.append(f"- Ticker exclusions: {exclusions}")
        lines.append("")

        # ── Previous improvement attempts ──
        lines.append("## Previous Improvement Cycles")
        try:
            imp_snaps = await self._db.get_improvement_snapshots(tid, limit=5)
            if imp_snaps:
                for s in imp_snaps:
                    wr = f"WR={s.win_rate_pct:.0f}%" if s.win_rate_pct is not None else "WR=N/A"
                    lines.append(f"  {s.week_start} to {s.week_end}: {s.total_trades} trades, {wr}")
                    if s.applied_changes:
                        lines.append(f"    Changes: {s.applied_changes[:200]}")
            else:
                lines.append("  No previous improvement cycles.")
        except Exception:
            lines.append("  (improvement history unavailable)")
        lines.append("")

        # ── Parameter changelog ──
        lines.append("## Recent Parameter Changes")
        try:
            changelog = await self._db.get_parameter_changelog(tid, limit=10)
            if changelog:
                for e in changelog:
                    lines.append(f"  {e.parameter}: {e.old_value} -> {e.new_value} ({e.reason})")
            else:
                lines.append("  No parameter changes recorded.")
        except Exception:
            lines.append("  (changelog unavailable)")
        lines.append("")

        # ── Source file map ──
        lines.append("## Key Source Files (ranked by expected impact)")
        lines.append("- `src/analysis/signal_engine.py` — SIGNAL_WEIGHTS dict, indicator calculations, scoring")
        lines.append("- `data/agent-workspace/CLAUDE.md` — trading agent instructions, strategy rules")
        lines.append("- `src/analysis/risk_manager.py` — position sizing, risk checks, sector limits")
        lines.append("- `src/analysis/regime_detector.py` — regime classification thresholds (VIX, breadth)")
        lines.append("- `src/strategies/` — strategy logic, entry/exit criteria")
        lines.append("- `config/universe.py` — ticker universe (which tickers the agent can trade)")
        lines.append("- `src/analysis/meta_agent.py` — this meta-agent's own code (improve yourself)")
        lines.append("- `src/analysis/weekly_improvement.py` — parameter tuning analyzer")
        lines.append("")

        return "\n".join(lines)

    # ── Prompt ─────────────────────────────────────────────────────────────

    def _build_prompt(self) -> str:
        """Build the Claude CLI prompt."""
        today = date.today().isoformat()
        return (
            f"Self-improvement session — {today}.\n"
            "You are Kukulkan's meta-agent. Your SOLE mission: maximize Portfolio B paper trading returns.\n"
            "This is paper money — there is zero risk. Be bold. Experiment aggressively.\n"
            "\n"
            "Read meta-agent-context.md for this week's performance data.\n"
            "\n"
            "WORKFLOW:\n"
            "1. Read meta-agent-context.md for performance data and current config\n"
            "2. Analyze: what's working, what's failing, what to try next\n"
            "3. Read relevant source files to understand current behavior\n"
            f"4. git checkout -b meta-agent/{today}\n"
            "5. Make targeted code changes to improve trading performance\n"
            "6. Run tests: .venv/bin/python -m pytest tests/ -x -q\n"
            "7. Run lint: .venv/bin/ruff check src/ --fix && .venv/bin/ruff format src/\n"
            "8. git add + commit with descriptive message\n"
            "9. git push origin HEAD\n"
            "10. gh pr create --title '...' --body '...'\n"
            "\n"
            "FOCUS AREAS (ranked by expected impact):\n"
            "- Signal weights in src/analysis/signal_engine.py (7 floats controlling the entire ranking)\n"
            "- Trading agent instructions in data/agent-workspace/CLAUDE.md\n"
            "- Risk params in src/analysis/risk_manager.py\n"
            "- Regime thresholds in src/analysis/regime_detector.py\n"
            "- Strategy logic in src/strategies/\n"
            "- Ticker universe in config/universe.py\n"
            "- Your own runner: src/analysis/meta_agent.py\n"
            "\n"
            "DO NOT TOUCH: src/storage/, src/api/, deploy configs, .env, credentials, auth.\n"
            "\n"
            "RULES:\n"
            "- ONE PR per session — focused and reviewable\n"
            "- Tests MUST pass before opening the PR\n"
            "- Never push to main. Only create PRs from branches.\n"
            "- PR body must include: HYPOTHESIS (what you expect to improve and why),\n"
            "  CHANGES (what you modified), and ROLLBACK (how to revert if it hurts performance)\n"
            "- Think in EXPECTANCY: win_rate * avg_win - loss_rate * avg_loss\n"
            "- If no clear improvement exists, say so and skip the PR\n"
        )

    # ── CLI invocation ─────────────────────────────────────────────────────

    def _build_cmd(self, prompt: str, session_id: str | None) -> list[str]:
        """Build the claude CLI command."""
        cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--max-turns",
            str(self._max_turns),
            "--model",
            self._model,
            "--dangerously-skip-permissions",
        ]
        if session_id:
            cmd.extend(["--resume", session_id])
        return cmd

    # ── Session management ─────────────────────────────────────────────────

    def _get_session_id(self) -> str | None:
        """Read persistent session ID."""
        if not self._session_file.exists():
            return None
        try:
            data = json.loads(self._session_file.read_text())
            return data.get("session_id")
        except (json.JSONDecodeError, KeyError):
            return None

    def _save_session_id(self, session_id: str) -> None:
        """Persist session ID for --resume."""
        self._session_file.write_text(json.dumps({"session_id": session_id}))

    # ── Output parsing ─────────────────────────────────────────────────────

    def _parse_output(self, stdout: str) -> dict:
        """Parse Claude Code CLI JSON output."""
        if not stdout.strip():
            return {}
        try:
            data = json.loads(stdout)
            # Claude Code wraps in {"result": "...", "session_id": "..."}
            return data
        except json.JSONDecodeError:
            return {"result": stdout[:2000]}

    def _extract_pr_urls(self, text: str) -> list[str]:
        """Extract GitHub PR URLs from output text."""
        return re.findall(r"https://github\.com/[^\s\"']+/pull/\d+", text)

    def _extract_summary(self, text: str) -> str:
        """Extract a concise summary from the result text."""
        if not text:
            return ""
        # If it's JSON with a summary field, use that
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "summary" in obj:
                return obj["summary"][:500]
        except (json.JSONDecodeError, TypeError):
            pass
        # Otherwise truncate the raw text
        return text[:500]

    # ── Notification ───────────────────────────────────────────────────────

    async def _notify(self, notifier: object, result: MetaAgentResult) -> None:
        """Send Telegram notification about the meta-agent run."""
        lines = ["<b>Meta-Agent Self-Improvement</b>", ""]

        if result.prs_opened:
            lines.append(f"PRs opened: {len(result.prs_opened)}")
            for url in result.prs_opened:
                lines.append(f"  {url}")
        elif result.status == "no_changes":
            lines.append("No changes this session.")
        elif result.status == "error":
            lines.append(f"Error: {result.error}")

        if result.summary:
            # Escape HTML for Telegram
            safe_summary = result.summary.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            lines.append(f"\n{safe_summary[:300]}")

        try:
            await notifier.send_message("\n".join(lines), parse_mode="HTML")
        except Exception as e:
            log.warning("meta_agent_notify_failed", error=str(e))
