# Codex OAuth Proxy

`codex_oauth_proxy.py` exposes Hermes-managed OpenAI Codex OAuth credentials as a small local OpenAI-compatible HTTP proxy.

It is useful when a local tool expects an OpenAI-style `base_url` and `api_key`, but you want requests to go through Codex OAuth accounts stored in Hermes.

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
- Request fields unsupported by the Codex backend, such as `temperature`, `top_p`, and token limit fields, are stripped before forwarding.
- `system` and `developer` input messages are lifted into Responses `instructions`.
- The proxy backfills empty final output from streamed `response.output_item.done` events, matching Hermes' Codex stream recovery behavior.
