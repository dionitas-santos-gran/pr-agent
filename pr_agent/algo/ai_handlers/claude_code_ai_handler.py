import asyncio
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
from pr_agent.algo.run_details import record_ai_call
from pr_agent.log import get_logger

# Host-only settings: read from the environment so a reviewed repository's .pr_agent.toml cannot
# choose the executable that runs on this machine.
CLAUDE_CODE_BIN_ENV = "PR_AGENT_CLAUDE_CODE_BIN"
CLAUDE_CODE_TIMEOUT_ENV = "PR_AGENT_CLAUDE_CODE_TIMEOUT"
DEFAULT_TIMEOUT_SECONDS = 600

# These outrank the subscription login in Claude Code, so removing them keeps usage on the login.
_API_CREDENTIAL_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
_MODEL_ALIASES = frozenset({"fable", "haiku", "opus", "sonnet"})
_MODEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:\[\]-]{0,127}")
_MAX_ERROR_LENGTH = 500


class ClaudeCodeError(RuntimeError):
    """Raised when the Claude Code CLI does not return a usable completion."""


def _cli_model_name(model: str) -> str:
    """Return the name to pass to the CLI, rejecting anything Claude Code cannot serve."""
    name = str(model or "").strip().removeprefix("anthropic/")
    if not _MODEL_PATTERN.fullmatch(name) or not (name in _MODEL_ALIASES or name.startswith("claude-")):
        raise ClaudeCodeError(
            f"Claude Code cannot serve model {name[:64]!r}. Use a Claude model in config.model and "
            "remove non-Claude entries from config.fallback_models."
        )
    return name


def _timeout_seconds() -> float:
    raw_value = os.environ.get(CLAUDE_CODE_TIMEOUT_ENV, "").strip()
    if not raw_value:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = float(raw_value)
    except ValueError as e:
        raise ClaudeCodeError(f"{CLAUDE_CODE_TIMEOUT_ENV} must be a number of seconds") from e
    if timeout <= 0:
        raise ClaudeCodeError(f"{CLAUDE_CODE_TIMEOUT_ENV} must be greater than zero")
    return timeout


def _build_usage(usage: dict) -> dict:
    """Translate Claude Code usage to the prompt/completion fields used by run details."""
    cache_read = usage.get("cache_read_input_tokens") or 0
    cache_creation = usage.get("cache_creation_input_tokens") or 0
    prompt_tokens = (usage.get("input_tokens") or 0) + cache_read + cache_creation
    completion_tokens = usage.get("output_tokens") or 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_creation,
    }


def _parse_payload(stdout: bytes) -> dict | None:
    try:
        payload = json.loads(stdout)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


class ClaudeCodeAIHandler(BaseAiHandler):
    """Run completions through the official `claude -p` CLI with the local user's own login.

    Intended for one person running PR-Agent on their own machine. The handler never reads or
    forwards the OAuth credential; authentication stays inside the unmodified Claude Code binary.
    Reviewed diffs are untrusted input, so the CLI runs with no tools, MCP servers or customizations.
    """

    def __init__(self):
        super().__init__()

    @property
    def deployment_id(self):
        return None

    async def chat_completion(self, model: str, system: str, user: str, temperature: float = 0.2, img_path: str = None):
        if img_path:
            get_logger().warning("Image input is not supported by ClaudeCodeAIHandler; ignoring it")
        cli_model = _cli_model_name(model)
        binary_name = os.environ.get(CLAUDE_CODE_BIN_ENV, "").strip() or "claude"
        binary = shutil.which(binary_name)
        if not binary:
            raise ClaudeCodeError(f"Claude Code executable {binary_name!r} was not found")
        timeout = _timeout_seconds()

        environment = {key: value for key, value in os.environ.items() if key not in _API_CREDENTIAL_ENV_VARS}
        # An empty working directory keeps the checkout and its CLAUDE.md out of the request.
        with tempfile.TemporaryDirectory(prefix="pr-agent-claude-code-") as working_directory:
            system_prompt_path = Path(working_directory) / "system-prompt.txt"
            system_prompt_path.write_text(system or "No system prompt provided", encoding="utf-8")
            command = [
                binary,
                "-p",
                f"--model={cli_model}",
                "--output-format=json",
                f"--system-prompt-file={system_prompt_path}",
                "--tools=",
                "--setting-sources=",
                "--safe-mode",
                "--strict-mcp-config",
                "--disable-slash-commands",
                "--no-session-persistence",
                "--max-turns=1",
            ]
            stdout, stderr, return_code = await self._run(command, user, working_directory, environment, timeout)

        return self._parse_completion(stdout, stderr, return_code, model)

    @staticmethod
    async def _run(command, user, working_directory, environment, timeout) -> tuple[bytes, bytes, int]:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_directory,
            env=environment,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate((user or "").encode("utf-8")), timeout)
        except TimeoutError as e:
            raise ClaudeCodeError(f"Claude Code did not finish within {timeout:g} seconds") from e
        finally:
            # Also covers task cancellation, so no CLI process outlives the request.
            if process.returncode is None:
                process.kill()
                await process.wait()
        return stdout, stderr, process.returncode

    @staticmethod
    def _parse_completion(stdout: bytes, stderr: bytes, return_code: int, model: str) -> tuple[str, str]:
        payload = _parse_payload(stdout)
        response = payload.get("result") if payload else None
        if return_code != 0 or not payload or payload.get("is_error") or not isinstance(response, str) or not response:
            detail = response if isinstance(response, str) and response else stderr.decode("utf-8", "replace").strip()
            detail = detail[:_MAX_ERROR_LENGTH] or "no details were reported"
            raise ClaudeCodeError(f"Claude Code failed with exit code {return_code}: {detail}")

        usage = payload.get("usage")
        # total_cost_usd is an API-list estimate, not what a subscription is charged, so it is not recorded.
        record_ai_call(_build_usage(usage) if isinstance(usage, dict) else None, model=model)
        finish_reason = "length" if payload.get("stop_reason") == "max_tokens" else "stop"
        get_logger().debug("Claude Code completion finished", finish_reason=finish_reason, model=model)
        return response, finish_reason
