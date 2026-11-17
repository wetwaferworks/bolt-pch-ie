# PMCP client profiles

PMCP is the MCP interface Peachy serves. Every profile here starts the same small host program over stdio:

    peachy_host serve --lib peachy.dll --allowlist allowlist.json --audit-dir audit --client <name> \
                      --token-file tokens/<name>.token --mode record [planner and game options]

Paths are placeholders under `C:/PMCP/`; change them to where the files really live. The game command line
(`lithokraft_headless.exe --mcp-stdio`) is also a placeholder for however the headless game starts its MCP server.

## Setting up a client (done once, by the person at the console)

1. `peachy_host token-new --out C:/PMCP/tokens/<name>.token` makes a token file and prints only its SHA-256.
2. `peachy_host allow-add --allowlist C:/PMCP/allowlist.json --client <name> --token-file C:/PMCP/tokens/<name>.token`
3. `peachy_host grant --allowlist C:/PMCP/allowlist.json --client <name> observe propose proposal_status`
   (add `plan` and `audit_head` if wanted). Grants never come over MCP.
4. Copy the client's profile below into that client's MCP settings.

The token stays in its file. No profile holds a token, and none should: the host reads the file and the library
compares hashes. `allowlist.example.json` shows the format with zero hashes, which match no token.

Every profile starts in `record` mode: proposals are checked and logged, nothing runs. `advisory` reports what
Peachy would do; still nothing runs. Actuation is not available in this release.

## Profiles

| Client | File | How it connects | Notes |
|---|---|---|---|
| Grok (Grok Build CLI) | `clients/grok.json` | local stdio | `grok mcp add pmcp -- C:/PMCP/peachy_host.exe serve ...` with the same arguments as the file. |
| GrokBot | `clients/grokbot.json` | local stdio, through a harness that runs on the same machine | A cloud-side agent cannot start a local process; it needs a local stdio MCP harness. |
| Gemini (Gemini CLI) | `clients/gemini.json` | local stdio | Goes in `settings.json` under `mcpServers`. `trust` stays false so every tool call is confirmed. |
| Muse (Muse Code) | `clients/muse.json` | local stdio | Goes in the `mcp_servers` block of Muse Code settings. Check field names against Muse Code's own docs. |
| Claude Desktop | `clients/claude-desktop.json` | local stdio | `claude_desktop_config.json`. |
| Any other stdio MCP client | `clients/generic-stdio.json` | local stdio | Most clients take this `command` plus `args` form. |

Remote-only paths (the xAI API's remote MCP tools, grok.com connectors, the Meta Model API) need a public HTTP
endpoint. PMCP has no HTTP transport in this release, by design; adding one is a separate decision.
