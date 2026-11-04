# Run PR-Agent locally with your Claude Code login

This fork adds an AI handler that calls the official `claude -p` command, so PR-Agent uses your own Claude Code login
instead of an API key. PR-Agent never reads or forwards the OAuth token.

It is meant for one person running the CLI on their own machine. For a shared bot, a webhook or team automation, use an
API key or a cloud provider instead. See
[Claude Code's authentication rules](https://code.claude.com/docs/en/legal-and-compliance).

## Setup

1. Install Claude Code and sign in with `claude`.
2. Install this fork with `uv sync`.
3. Set the variables below.

```bash
export PR_AGENT_AI_HANDLER=claude_code
export CONFIG__MODEL=claude-sonnet-5-5
export CONFIG__FALLBACK_MODELS='[]'
```

## Usage

Review a local diff without posting anything:

```bash
git diff main...HEAD | uv run pr-agent --stdin review
```

Review a pull request. This posts the result as a comment, using your GitHub token:

```bash
export GITHUB__USER_TOKEN=...
uv run pr-agent --pr_url https://github.com/owner/repo/pull/123 review
```

## Settings

| Variable | Purpose | Default |
| --- | --- | --- |
| `PR_AGENT_AI_HANDLER` | Set to `claude_code` to use this handler | LiteLLM |
| `PR_AGENT_CLAUDE_CODE_BIN` | Claude Code executable | `claude` |
| `PR_AGENT_CLAUDE_CODE_TIMEOUT` | Seconds allowed for each completion | `600` |

These are environment variables on purpose, so a reviewed repository's `.pr_agent.toml` cannot change them.

## Behavior

- `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` are removed from the CLI's environment, because they would otherwise
  take priority over your login.
- The diff is untrusted input, so the CLI runs in an empty directory with no tools, MCP servers, hooks, or `CLAUDE.md`.
- Usage counts against your Claude plan. Token totals are recorded, but no cost is, because the CLI's cost figure is an
  API-price estimate.
- Images and `temperature` are ignored.
- `pr-agent doctor` still checks LiteLLM, so it does not validate this setup.
