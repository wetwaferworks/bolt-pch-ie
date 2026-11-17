# PMCP client profiles

PMCP is the MCP interface Peachy serves. Every profile here starts the same small host program over stdio:

    peachy_host serve --lib peachy.dll --allowlist allowlist.json --audit-dir audit --client <name> \
                      --token-file tokens/<name>.token --mode record [planner and game options]

Paths are placeholders under `C:/PMCP/`, except the allowlist, which lives in a protected folder under
`C:/ProgramData/PMCP/` (see [Protecting the files](#protecting-the-files)). Change them to where the files really live. The game command line
(`lithokraft_headless.exe --mcp-stdio`) is also a placeholder for however the headless game starts its MCP server.

## Setting up a client (done once, by the person at the console)

The outside models run under your account, and their own tools can run commands, so anything your account can
change, they can change too. That's why the grants live where only an administrator can change them, and every step
that changes them runs from an elevated prompt ("Run as administrator"; `sudo` on Linux). `serve` itself runs from
a normal prompt, and refuses to run elevated.

1. Elevated, once: `peachy_host protect --allowlist C:/ProgramData/PMCP/allowlist.json`
   This makes the folder if it's missing. The folder and the file become owned by Administrators, with full
   control for SYSTEM and Administrators, read-only for your account, and inheritance off. The folder must hold
   nothing but the allowlist.
2. Normal prompt: `peachy_host token-new --out C:/PMCP/tokens/<name>.token` makes a token file that only your
   account (plus SYSTEM and Administrators) can open, and prints only its SHA-256.
3. Elevated: `peachy_host allow-add --allowlist C:/ProgramData/PMCP/allowlist.json --client <name> --token-file C:/PMCP/tokens/<name>.token`
4. Elevated: `peachy_host grant --allowlist C:/ProgramData/PMCP/allowlist.json --client <name> observe propose proposal_status`
   (add `plan` and `audit_head` if wanted). Grants never come over MCP.
5. Normal prompt: `peachy_host check --allowlist C:/ProgramData/PMCP/allowlist.json --token-file C:/PMCP/tokens/<name>.token`
   should print `OK`. If it prints `REFUSED`, the reason is on the line above it.
6. Copy the client's profile below into that client's MCP settings.

The token stays in its file. No profile holds a token, and none should: the host reads the file and the library
compares hashes. `allowlist.example.json` shows the format with zero hashes, which match no token.

Every profile starts in `record` mode: proposals are checked and logged, nothing runs. `advisory` reports what
Peachy would do; still nothing runs. Actuation is not available in this release.

## Protecting the files

Before it starts, `serve` checks its files and refuses to start if any check fails:

- **Elevation.** `serve` never runs elevated or as root, since such a process could change its own grants.
- **The allowlist** must be an ordinary file with no link or junction anywhere in its path. Your account must not
  be able to write it, delete it, re-permission it or take ownership of it. It also must not be able to delete,
  rename or re-permission any folder above it, or delete what's inside them. Put it in a folder of its own under
  `C:/ProgramData` (on Linux, under `/etc`), not in your user profile, which your account can always change.
  `protect` sets the folder and the file up. On Linux it makes them owned by root, with modes 0755 and 0644.
- **A token file** must be an ordinary file that no account but yours, SYSTEM and Administrators can read. On Linux,
  it must be owned by you with no group or other bits (0600). `token-new` makes it that way.

`peachy_host check` runs the same checks from a normal prompt and says which one failed. After `allow-add`,
`grant` or `revoke`, the rewritten allowlist is locked down again.

What this covers, and what it doesn't:

- It stops a client from granting itself tools through Peachy. A client can't edit the allowlist, and it can't run
  `grant` or `allow-add` without an administrator saying yes.
- It doesn't keep clients apart from each other. The supported setup runs every client under one shared account, by
  choice: it keeps setup simple, and the grants, not the account, are what limit each client. The trade-off is that
  any client can read every client's token and connect as another client, though only with the grants you gave that
  client. Keep each client's grants as narrow as its job, and `actuate` stays refused for all of them. (Running each
  client under its own account, with only that account able to read its token, would close this gap. `serve`
  already checks that a token is private to the account running it, so that remains possible later without code
  changes.)
- A client with a shell can still run programs of its own, including a copy of Peachy built without these checks.
  Keep `peachy_host.exe` and `peachy.dll` where your account can't replace them (for example
  `C:/Program Files/PMCP`), so the server each profile starts is the real one. `actuate` stays refused everywhere.
  Before it exists, the game will have to accept actions only from Peachy.

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
