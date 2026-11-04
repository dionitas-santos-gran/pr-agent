import asyncio
import json
import os
import stat
import sys

import pytest

from pr_agent import cli
from pr_agent.algo.ai_handlers.claude_code_ai_handler import ClaudeCodeAIHandler, ClaudeCodeError
from pr_agent.algo.run_details import get_run_details, init_run_details

FAKE_CLI = """#!{python}
import json, os, sys, time
args = sys.argv[1:]
system_file = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--system-prompt-file="))
with open(os.environ["FAKE_CLI_RECORD"], "w", encoding="utf-8") as record:
    json.dump({{
        "args": args,
        "stdin": sys.stdin.read(),
        "system": open(system_file, encoding="utf-8").read(),
        "cwd": os.getcwd(),
        "cwd_entries": os.listdir("."),
        "api_key": os.environ.get("ANTHROPIC_API_KEY"),
        "auth_token": os.environ.get("ANTHROPIC_AUTH_TOKEN"),
        "oauth_token": os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"),
    }}, record)
time.sleep(float(os.environ.get("FAKE_CLI_SLEEP", "0")))
sys.stderr.write(os.environ.get("FAKE_CLI_STDERR", ""))
sys.stdout.write(os.environ["FAKE_CLI_STDOUT"])
sys.exit(int(os.environ.get("FAKE_CLI_EXIT", "0")))
"""

SUCCESS = {
    "type": "result",
    "is_error": False,
    "result": "the answer",
    "stop_reason": "end_turn",
    "total_cost_usd": 12.5,
    "usage": {
        "input_tokens": 10,
        "cache_read_input_tokens": 5,
        "cache_creation_input_tokens": 3,
        "output_tokens": 7,
    },
}


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    executable = tmp_path / "fake-claude"
    executable.write_text(FAKE_CLI.format(python=sys.executable), encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
    record_path = tmp_path / "record.json"
    monkeypatch.setenv("PR_AGENT_CLAUDE_CODE_BIN", str(executable))
    monkeypatch.setenv("FAKE_CLI_RECORD", str(record_path))
    monkeypatch.setenv("FAKE_CLI_STDOUT", json.dumps(SUCCESS))
    monkeypatch.delenv("PR_AGENT_CLAUDE_CODE_TIMEOUT", raising=False)
    return lambda: json.loads(record_path.read_text(encoding="utf-8"))


async def test_returns_completion_and_sends_prompts_outside_argv(fake_cli):
    response, finish_reason = await ClaudeCodeAIHandler().chat_completion(
        "claude-sonnet-5-5", "system text", "untrusted diff --tools=Bash"
    )

    record = fake_cli()
    assert (response, finish_reason) == ("the answer", "stop")
    assert record["system"] == "system text"
    assert record["stdin"] == "untrusted diff --tools=Bash"
    assert "--model=claude-sonnet-5-5" in record["args"]
    assert not any("untrusted" in arg or "system text" in arg for arg in record["args"])


async def test_runs_without_tools_or_customizations_in_an_empty_directory(fake_cli):
    await ClaudeCodeAIHandler().chat_completion("claude-sonnet-5-5", "system", "user")

    record = fake_cli()
    for flag in (
        "-p",
        "--tools=",
        "--setting-sources=",
        "--safe-mode",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--max-turns=1",
    ):
        assert flag in record["args"]
    assert record["cwd_entries"] == ["system-prompt.txt"]
    assert not os.path.exists(record["cwd"])


async def test_removes_api_credentials_but_keeps_the_subscription_token(fake_cli, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "auth-token")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-token")

    await ClaudeCodeAIHandler().chat_completion("claude-sonnet-5-5", "system", "user")

    record = fake_cli()
    assert record["api_key"] is None
    assert record["auth_token"] is None
    assert record["oauth_token"] == "oauth-token"


async def test_records_tokens_without_a_subscription_cost(fake_cli):
    init_run_details()

    await ClaudeCodeAIHandler().chat_completion("claude-sonnet-5-5", "system", "user")

    details = get_run_details()
    assert details.num_ai_calls == 1
    assert details.prompt_tokens == 18
    assert details.completion_tokens == 7
    assert details.cache_read_tokens == 5
    assert details.cache_creation_tokens == 3
    assert details.known_cost_call_count == 0


async def test_maps_max_tokens_to_length(fake_cli, monkeypatch):
    monkeypatch.setenv("FAKE_CLI_STDOUT", json.dumps({**SUCCESS, "stop_reason": "max_tokens"}))

    _, finish_reason = await ClaudeCodeAIHandler().chat_completion("claude-sonnet-5-5", "system", "user")

    assert finish_reason == "length"


@pytest.mark.parametrize(
    ("model", "expected"),
    [("anthropic/claude-opus-5-5", "--model=claude-opus-5-5"), ("sonnet", "--model=sonnet")],
)
async def test_accepts_prefixed_models_and_aliases(fake_cli, model, expected):
    await ClaudeCodeAIHandler().chat_completion(model, "system", "user")

    assert expected in fake_cli()["args"]


@pytest.mark.parametrize("model", ["gpt-6-luna", "--tools=Bash", "claude-x --safe-mode", "", None])
async def test_rejects_models_the_cli_cannot_serve(fake_cli, model):
    with pytest.raises(ClaudeCodeError, match="cannot serve"):
        await ClaudeCodeAIHandler().chat_completion(model, "system", "user")


@pytest.mark.parametrize(
    ("stdout", "exit_code", "stderr", "message"),
    [
        (json.dumps({**SUCCESS, "is_error": True, "result": "usage limit reached"}), "1", "", "usage limit reached"),
        (json.dumps({**SUCCESS, "is_error": True, "result": "not logged in"}), "0", "", "not logged in"),
        ("not json", "1", "invalid option", "invalid option"),
        (json.dumps({**SUCCESS, "result": ""}), "0", "", "no details"),
        ("[]", "0", "", "no details"),
    ],
)
async def test_raises_on_cli_failures(fake_cli, monkeypatch, stdout, exit_code, stderr, message):
    monkeypatch.setenv("FAKE_CLI_STDOUT", stdout)
    monkeypatch.setenv("FAKE_CLI_EXIT", exit_code)
    monkeypatch.setenv("FAKE_CLI_STDERR", stderr)

    with pytest.raises(ClaudeCodeError, match=message):
        await ClaudeCodeAIHandler().chat_completion("claude-sonnet-5-5", "system", "user")


async def test_missing_executable_has_a_clear_error(monkeypatch):
    monkeypatch.setenv("PR_AGENT_CLAUDE_CODE_BIN", "does-not-exist-claude")

    with pytest.raises(ClaudeCodeError, match="was not found"):
        await ClaudeCodeAIHandler().chat_completion("claude-sonnet-5-5", "system", "user")


async def test_times_out_and_stops_the_process(fake_cli, monkeypatch):
    monkeypatch.setenv("FAKE_CLI_SLEEP", "30")
    monkeypatch.setenv("PR_AGENT_CLAUDE_CODE_TIMEOUT", "0.5")

    with pytest.raises(ClaudeCodeError, match="did not finish"):
        await asyncio.wait_for(ClaudeCodeAIHandler().chat_completion("claude-sonnet-5-5", "system", "user"), 15)


@pytest.mark.parametrize("value", ["soon", "0", "-1"])
async def test_rejects_invalid_timeouts(fake_cli, monkeypatch, value):
    monkeypatch.setenv("PR_AGENT_CLAUDE_CODE_TIMEOUT", value)

    with pytest.raises(ClaudeCodeError, match="PR_AGENT_CLAUDE_CODE_TIMEOUT"):
        await ClaudeCodeAIHandler().chat_completion("claude-sonnet-5-5", "system", "user")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("claude_code", ClaudeCodeAIHandler), (" Claude_Code ", ClaudeCodeAIHandler)],
)
def test_cli_selects_the_handler_from_the_environment(monkeypatch, value, expected):
    monkeypatch.setenv("PR_AGENT_AI_HANDLER", value)

    assert cli._build_agent().ai_handler is expected


def test_cli_keeps_the_default_handler_otherwise(monkeypatch):
    monkeypatch.delenv("PR_AGENT_AI_HANDLER", raising=False)

    assert cli._build_agent().ai_handler is not ClaudeCodeAIHandler
