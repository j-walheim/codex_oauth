# Codex OAuth Proxy

`codex_oauth_proxy.py` exposes Hermes-managed OpenAI Codex OAuth credentials as a small local OpenAI-compatible HTTP proxy.

It is useful when a local tool expects an OpenAI-style `base_url` and `api_key`, but you want requests to go through Codex OAuth accounts stored in Hermes.

It also serves Grok Build: send `"model": "grok"` and the request goes to your
`grok` CLI subscription instead of Codex. See [Grok Build](#grok-build).

## Start The Proxy

```bash
cd /home/jonas/codex_oauth_proxy
python codex_oauth_proxy.py --port 8765
```

By default the proxy looks for the Hermes repo at `../hermes-agent`. If your
checkout is elsewhere, set:

```bash
export HERMES_AGENT_ROOT=/path/to/hermes-agent
```

The local base URL is:

```text
http://127.0.0.1:8765/v1
```

Supported endpoints:

```text
/v1/responses
/v1/responses/*
/v1/chat/completions
/v1/models
/accounts
/health
```

`/v1/responses/*` includes subpaths such as `/v1/responses/compact`.

The proxy does not require a real OpenAI API key from local clients. If a client library requires one, use any non-empty placeholder value.

## Import Existing Codex CLI Auth

Hermes can import Codex CLI credentials from:

```text
$CODEX_HOME/auth.json
~/.codex/auth.json
```

Run:

```bash
python codex_oauth_proxy.py --import-codex-auth --label codex-cli
```

By default this imports and exits. To import and then serve:

```bash
python codex_oauth_proxy.py --import-codex-auth --serve-after-import
```

Imported credentials are saved into Hermes' `openai-codex` credential pool.

## Sign In Directly With Codex OAuth

The proxy can also start the same OpenAI Codex device-code OAuth flow that
Hermes uses for `hermes auth add openai-codex`. This creates a fresh Hermes
credential instead of copying tokens from Codex CLI:

```bash
python codex_oauth_proxy.py --codex-oauth-auth --label codex-direct
```

To sign in and then start serving:

```bash
python codex_oauth_proxy.py \
  --codex-oauth-auth \
  --serve-after-auth \
  --no-auto-sync-codex-cli-auth
```

This follows Hermes' device-code approach:

1. Request a user code from `https://auth.openai.com/api/accounts/deviceauth/usercode`.
2. Ask you to open `https://auth.openai.com/codex/device` and enter the code.
3. Poll `deviceauth/token` until OpenAI returns an authorization code.
4. Exchange that code at the OpenAI OAuth token endpoint.
5. Store the new access and refresh tokens in Hermes' `openai-codex` pool.

## Remove Stale OAuth Configs

Remove stale Codex OAuth entries from the Hermes credential pool:

```bash
python codex_oauth_proxy.py --remove-stale-oauth-configs
```

Preview the cleanup first:

```bash
python codex_oauth_proxy.py --remove-stale-oauth-configs --dry-run
```

The cleanup only targets clearly stale OAuth entries: missing access tokens,
expired access tokens without refresh tokens, and credentials already marked
with hard auth failures such as `401`, `403`, `invalid_grant`,
`invalid_token`, `refresh_token_reused`, or `token_invalidated`.

## Detect And Delete Dead Login Credentials

`--remove-stale-oauth-configs` only catches credentials that are *locally*
obviously stale (missing/expired access token with no refresh token, or a
recorded hard auth failure). It cannot detect a token that OpenAI has
**invalidated server-side** — the JWT still looks valid and nothing is recorded
locally, so the cleanup reports "No stale Codex OAuth configs found" even though
requests fail.

To catch those, probe each credential live against the Codex backend:

```bash
python codex_oauth_proxy.py --check-credentials
```

This makes one lightweight authenticated request per credential and reports a
health category for each:

```text
ok           reachable (HTTP 200)
invalidated  rejected with 401/403 or token_invalidated / invalid_token /
             invalid_grant / refresh_token_reused (refresh does not recover it)
expired      access token expired with no refresh token
unknown      transient/network error or unexpected status — left alone
```

It refreshes refreshable tokens before probing, so healthy accounts that merely
need a refresh are not flagged.

To probe and then interactively delete the dead ones:

```bash
python codex_oauth_proxy.py --prune-credentials
```

This prints the health report and then asks `y/N` for each dead credential
before removing it from the Hermes pool. Only `invalidated` and `expired`
credentials are offered for deletion; `unknown` ones are never touched.

To delete every dead credential without prompting (e.g. in a script):

```bash
python codex_oauth_proxy.py --prune-credentials --yes
```

After deleting dead credentials, sign in again with `--codex-oauth-auth` (or
`--import-codex-auth`) to add fresh ones.

When serving, the proxy now auto-syncs from Codex CLI auth (`$CODEX_HOME/auth.json`
or `~/.codex/auth.json`) before credential selection. If Codex has a newer token,
the proxy updates the imported Hermes entry automatically.

Disable this behavior with:

```bash
export CODEX_PROXY_AUTO_SYNC_CLI_AUTH=0
```

Or per proxy process:

```bash
python codex_oauth_proxy.py --no-auto-sync-codex-cli-auth
```

## List Accounts

```bash
curl http://127.0.0.1:8765/accounts
```

Each account includes its Hermes credential `id`, `label`, ChatGPT `account_id`, plan type, source, and status.

## Select One Account

Select a default account when starting the proxy:

```bash
python codex_oauth_proxy.py \
  --port 8765 \
  --account credentials-new-46c68ec7
```

Or select per request:

```bash
curl http://127.0.0.1:8765/v1/responses \
  -H 'Content-Type: application/json' \
  -H 'X-Hermes-Codex-Account: credentials-new-46c68ec7' \
  -d '{"model":"gpt-5.4","input":"hello"}'
```

The account selector can be a credential label, credential id, or ChatGPT account id.

## Ranked Account Failover

Start with a default ranking:

```bash
python codex_oauth_proxy.py \
  --port 8765 \
  --account-order credentials-new-46c68ec7,device_code
```

Or set the ranking per request:

```bash
curl http://127.0.0.1:8765/v1/responses \
  -H 'Content-Type: application/json' \
  -H 'X-Hermes-Codex-Account-Order: credentials-new-46c68ec7,device_code' \
  -d '{"model":"gpt-5.4","input":"hello"}'
```

The proxy tries accounts in order. If Codex returns `401`, `402`, `403`, or `429`, the proxy marks that credential exhausted in the Hermes pool and retries the next account when possible.

If Codex includes quota reset metadata, the proxy stores that reset time so Hermes can avoid the exhausted credential until the cooldown expires.

## Track Codex Quota

`/accounts` includes each login's live primary and secondary quota windows,
their used percentages, and reset times. Fetch a snapshot from the command line:

```bash
python codex_oauth_proxy.py --check-quota
```

To stop routing requests to any account when either quota window reaches a
chosen usage percentage, set a limit in `.env`:

```text
CODEX_PROXY_MAX_QUOTA_USED_PERCENT=80
```

Or pass `--max-quota-used-percent 80` when starting the proxy. Accounts at or
above the limit are skipped, allowing failover to another account below the
limit. If quota status cannot be fetched while this guard is enabled, that
account is skipped until its status can be checked. The guard is disabled by
default; use `--no-quota-limit` to override a configured limit for one run.

## Responses API Example

```bash
curl http://127.0.0.1:8765/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-5.4","input":"Reply exactly: proxy-ok"}'
```

For non-streaming `/v1/responses` calls, the proxy streams upstream to Codex internally and returns the final JSON response to the client.

For streaming passthrough:

```bash
curl -N http://127.0.0.1:8765/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-5.4","input":"hello","stream":true}'
```

## Chat Completions Example

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-5.4",
    "messages": [
      {"role": "user", "content": "Reply exactly: chat-ok"}
    ]
  }'
```

The chat completions wrapper is non-streaming. Use `/v1/responses` for streaming.

## Grok Build

Pass `grok` as the model and the proxy routes the request to Grok Build's own
backend instead of Codex:

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"grok","messages":[{"role":"user","content":"Reply exactly: grok-ok"}]}'
```

Any model name starting with `grok` routes there. Bare `grok` and `grok-build`
resolve to `grok-4.5`; anything else is passed through as given.

Credentials come from the `grok` CLI's own login file (`~/.grok/auth.json`,
written by `grok login`) — not from the Hermes credential pool. Nothing needs to
be imported: the file is read fresh on each request, so a token the `grok` CLI
refreshed underneath the proxy is picked up automatically. When the token is
within `GROK_PROXY_REFRESH_SKEW_SECONDS` of expiry the proxy refreshes it itself
and writes the new pair back to `auth.json`, so a long-running proxy keeps
working without you touching the CLI. Write-back is necessary because xAI may
rotate the refresh token, which would otherwise leave the `grok` CLI holding a
dead one. Set `GROK_PROXY_AUTO_REFRESH=0` to never write to `auth.json`; the
proxy then works until the token expires and you re-run `grok login`.

The first write copies the original file to `auth.json.proxy-bak`. That is a
one-shot snapshot guarding the first write only — it is not refreshed
afterwards, so it goes stale and is not a general restore point. If Grok auth
ever breaks, `grok login` is the fix.

Both `/v1/responses` and `/v1/chat/completions` are supported, streaming and
non-streaming — Grok Build implements both natively, so bodies pass through
untouched apart from the model alias and the `model(effort)` suffix:

```bash
curl -N http://127.0.0.1:8765/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"grok(high)","input":"hello","stream":true}'
```

Grok advertises only `high`, `medium`, and `low` effort, so `minimal` clamps to
`low`, `xhigh`/`max` clamp to `high`, and `none`/`auto` are dropped.

Grok responses carry `X-Proxy-Backend: grok` and `X-Proxy-Account: <email>`.

Multiple Grok logins work the same way as the Codex ones — each is a separate
`GROK_HOME=<dir> grok login`:

```bash
python codex_oauth_proxy.py --port 8765 --grok-home ~/.grok --grok-home ~/.grok-acct-work
```

The first is the default; select another per request with `X-Grok-Home`. That
header takes any login dir, so `--grok-home` only sets which one is used by
default — it is not an allowlist. Unlike Codex there is no pool behind Grok, so
there is no automatic failover between Grok accounts — upstream errors go
straight back to the client.

To force a backend regardless of the model name, send `X-Proxy-Backend: grok`
or `X-Proxy-Backend: codex` (or `?backend=grok`). This is also how to list Grok
models, since a `GET` carries no model:

```bash
curl 'http://127.0.0.1:8765/v1/models?backend=grok'
```

Grok settings:

```text
GROK_PROXY_GROK_HOMES           comma-separated login dirs (default: $GROK_HOME or ~/.grok)
GROK_PROXY_DEFAULT_MODEL        what bare `grok` resolves to (default: grok-4.5)
GROK_PROXY_BASE_URL             upstream (default: https://cli-chat-proxy.grok.com/v1)
GROK_PROXY_CLIENT_VERSION       client version sent upstream (default: 0.2.112)
GROK_PROXY_AUTO_REFRESH=0       never refresh/write the grok token
GROK_PROXY_REFRESH_SKEW_SECONDS refresh this long before expiry (default: 300)
```

Grok Build gates on a client-version header; if the backend starts answering
`426 ... version (none) is outdated`, set `GROK_PROXY_CLIENT_VERSION` to your
installed version (`grok --version`).

## OpenAI SDK Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key="local-placeholder",
)

response = client.responses.create(
    model="gpt-5.4",
    input="Reply exactly: sdk-ok",
    extra_headers={
        "X-Hermes-Codex-Account-Order": "credentials-new-46c68ec7,device_code",
    },
)

print(response.output_text)
```

## Health Check

```bash
curl http://127.0.0.1:8765/health
```

With account selection:

```bash
curl http://127.0.0.1:8765/health \
  -H 'X-Hermes-Codex-Account: credentials-new-46c68ec7'
```

## Notes

- The proxy reuses Hermes Codex OAuth handling and credential pool state.
- Codex and Grok Build are separate backends. Grok requests never touch the
  Hermes pool, and the Codex normalization below does not apply to them; they
  work even when no Codex credentials are configured at all.
- Request fields unsupported by the Codex backend, such as `temperature`, `top_p`, and token limit fields, are stripped before forwarding.
- `system` and `developer` input messages are lifted into Responses `instructions`.
- The proxy backfills empty final output from streamed `response.output_item.done` events, matching Hermes' Codex stream recovery behavior.
