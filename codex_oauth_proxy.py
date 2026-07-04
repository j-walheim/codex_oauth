#!/usr/bin/env python3
"""Local OpenAI-compatible proxy backed by Hermes Codex OAuth."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

import httpx

DEBUG = os.getenv("CODEX_PROXY_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def debug(msg: str) -> None:
    if DEBUG:
        print(msg, file=sys.stderr, flush=True)


def _find_hermes_root() -> Path:
    explicit = os.getenv("HERMES_AGENT_ROOT", "").strip()
    candidates = [Path(explicit)] if explicit else []
    here = Path(__file__).resolve()
    candidates += [
        here.parent / "hermes-agent",
        here.parent.parent / "hermes-agent",
        Path.cwd(),
    ]
    for candidate in candidates:
        if candidate and (candidate / "hermes_cli" / "auth.py").is_file():
            return candidate.resolve()
    raise RuntimeError("Could not find Hermes repo. Set HERMES_AGENT_ROOT=/path/to/hermes-agent.")


ROOT = _find_hermes_root()
sys.path.insert(0, str(ROOT))

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
HOP = {"host", "content-length", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "trailers", "transfer-encoding", "upgrade", "authorization", "x-hermes-codex-account"}
CODEX_DROP_FIELDS = {"context_management", "max_completion_tokens", "max_output_tokens", "max_tokens", "metadata", "previous_response_id", "prompt_cache_retention", "safety_identifier", "stream_options", "temperature", "text", "top_k", "top_p", "truncation", "user"}
FAILOVER_STATUS = {401, 402, 403, 429}
REASONING_SUFFIXES = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "xhigh",
    "none": "none",
    "auto": "auto",
}

AUTO_SYNC_CLI_AUTH = os.getenv("CODEX_PROXY_AUTO_SYNC_CLI_AUTH", "1").strip().lower() not in {"0", "false", "no", "off"}


def _parse_codex_homes(raw: str | None) -> list[str]:
    """Split a comma/os.pathsep-separated list of Codex home dirs into expanded paths."""
    if not raw:
        return []
    parts = raw.replace(os.pathsep, ",").split(",")
    return [str(Path(p.strip()).expanduser()) for p in parts if p.strip()]


# Codex CLI auth dirs the proxy imports / auto-syncs from, one per account you
# want to flip between. Override via CODEX_PROXY_CODEX_HOMES (comma-separated) or
# one/more --codex-home flags. Each dir is a separate Codex login
# (`CODEX_HOME=<dir> codex login`).
_DEFAULT_CODEX_HOMES = "~/.codex,~/.codex-acct-jonas"
CODEX_HOME_DIRS: list[str] = _parse_codex_homes(
    os.getenv("CODEX_PROXY_CODEX_HOMES") or _DEFAULT_CODEX_HOMES
)

# Auto-sync each Codex login dir at most once per this interval, and stagger the
# dirs evenly across it so two accounts never refresh in the same window (e.g.
# 6h interval + 2 dirs => each dir every 6h, 3h apart). Override via
# CODEX_PROXY_SYNC_INTERVAL_SECONDS.
CODEX_SYNC_INTERVAL_SECONDS = int(os.getenv("CODEX_PROXY_SYNC_INTERVAL_SECONDS", str(6 * 3600)))
_LAST_SYNC: dict[str, float] = {}


@contextmanager
def _codex_home_env(codex_home: str | None):
    """Temporarily point hermes' Codex CLI reader at a specific login dir."""
    if not codex_home:
        yield
        return
    prev = os.environ.get("CODEX_HOME")
    os.environ["CODEX_HOME"] = str(Path(codex_home).expanduser())
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("CODEX_HOME", None)
        else:
            os.environ["CODEX_HOME"] = prev


def _b(obj: object) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode()


def _body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    raw = handler.rfile.read(int(handler.headers.get("Content-Length", "0") or "0")) or b"{}"
    data = json.loads(raw.decode())
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data


def _strip_v1(path: str) -> str:
    return path[3:] if path.startswith("/v1/") else path


def _route_rel(path: str) -> str:
    if path.startswith("/api/provider/"):
        parts = path.split("/")
        if len(parts) >= 4:
            rest = "/" + "/".join(parts[4:])
            return _strip_v1(rest)
    if path.startswith("/backend-api/codex/"):
        return path[len("/backend-api/codex"):]
    return _strip_v1(path)


def _is_responses_path(rel: str) -> bool:
    return rel == "/responses" or rel.startswith("/responses/")


def normalize_responses_subpath_request(rel: str, body: dict[str, Any]) -> dict[str, Any]:
    """Normalize Responses API request payloads for Codex backend compatibility.

    Root /responses gets full request normalization; /responses/* subpaths keep
    request shape mostly intact to avoid dropping endpoint-specific fields.
    """
    if rel == "/responses":
        return normalize_responses_request(body)

    req = dict(body)
    req.setdefault("model", "gpt-5.4")
    if req.get("instructions") is None:
        req["instructions"] = ""
    if "input" in req and not isinstance(req.get("input"), list):
        text = "" if req.get("input") is None else str(req.get("input"))
        req["input"] = [{"role": "user", "content": [{"type": "input_text", "text": text}]}]
    return codex_wire_request(normalize_codex_builtin_tools(req))


def _req_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {k: v for k, v in headers if k.lower() not in HOP}


def _resp_headers(headers: httpx.Headers) -> list[tuple[str, str]]:
    return [
        (k, v) for k, v in headers.items()
        if k.lower() not in HOP and k.lower() not in {"content-length", "content-encoding"}
    ]


def _jwt_account_id(token: str) -> str:
    return str(_jwt_identity(token).get("account_id") or "")


def _jwt_user_key(token: str) -> str:
    """Stable per-USER identity for a token. chatgpt_account_id is the
    workspace/org (shared by every member of a Team), so it must NOT be used to
    tell two logins apart — key on the user instead."""
    ident = _jwt_identity(token)
    return str(ident.get("user_id") or ident.get("subject") or "")


def _label_for_home(codex_home: str | None) -> str | None:
    """Derive a readable default label from a Codex login dir, e.g.
    ~/.codex-acct-jonas -> 'codex-acct-jonas'."""
    if not codex_home:
        return None
    base = Path(codex_home).expanduser().name.lstrip(".")
    return base or None


def _identity_summary(token: str) -> str:
    """Human-readable 'who is this' suffix decoded from the token, e.g.
    'Jane Doe <jane@acme.com>, plan=pro'. Empty string if nothing is known."""
    ident = _jwt_identity(token)
    name = ident.get("name") or ident.get("account_name") or ""
    email = ident.get("email", "")
    plan = ident.get("plan_type", "")
    who = name
    if email:
        who = f"{name} <{email}>" if name else email
    bits = [b for b in (who, f"plan={plan}" if plan else "") if b]
    return ", ".join(bits)


def _with_identity(message: str, token: str) -> str:
    summary = _identity_summary(token)
    return f"{message} — {summary}" if summary else message


def _jwt_identity(token: str) -> dict[str, str]:
    try:
        from hermes_cli.auth import _decode_jwt_claims
        claims = _decode_jwt_claims(token)
        auth = claims.get("https://api.openai.com/auth")
        out = {
            "email": str(claims.get("email") or claims.get("preferred_username") or ""),
            "name": str(claims.get("name") or ""),
            "subject": str(claims.get("sub") or ""),
        }
        if isinstance(auth, dict):
            out.update({
                "account_id": str(auth.get("chatgpt_account_id") or ""),
                "account_name": str(auth.get("chatgpt_account_name") or ""),
                "plan_type": str(auth.get("chatgpt_plan_type") or ""),
                "user_id": str(auth.get("chatgpt_user_id") or ""),
            })
        return {k: v for k, v in out.items() if v}
    except Exception:
        return {}


def _matches(entry: object, selector: str) -> bool:
    wanted = selector.strip().lower()
    token = str(getattr(entry, "access_token", "") or "")
    ident = _jwt_identity(token)
    # account_id (org) is intentionally last and ambiguous across Team members;
    # prefer id/label/user_id to target an individual login.
    vals = [
        getattr(entry, "id", ""),
        getattr(entry, "label", ""),
        ident.get("user_id", ""),
        ident.get("email", ""),
        ident.get("account_id", ""),
    ]
    return any(isinstance(v, str) and v.strip().lower() == wanted for v in vals)


@dataclass(frozen=True)
class Cred:
    token: str
    headers: dict[str, str]
    label: str
    cid: str
    base_url: str


def _selectors(raw: str | None) -> list[str]:
    return [p.strip() for p in str(raw or "").split(",") if p.strip()]


def _cred_from_entry(entry: object) -> Cred:
    from agent.auxiliary_client import _codex_cloudflare_headers
    from hermes_cli.auth import DEFAULT_CODEX_BASE_URL

    token = str(getattr(entry, "access_token", "") or "").strip()
    if not token:
        raise RuntimeError(f"Codex credential {getattr(entry, 'label', '') or getattr(entry, 'id', '')!r} has no access token")
    return Cred(
        token=token,
        headers=_codex_cloudflare_headers(token),
        label=getattr(entry, "label", "") or getattr(entry, "id", ""),
        cid=getattr(entry, "id", ""),
        base_url=(getattr(entry, "base_url", "") or DEFAULT_CODEX_BASE_URL).rstrip("/"),
    )


def resolve_cred(account: str | None = None) -> Cred:
    from agent.credential_pool import load_pool

    _auto_sync_codex_cli_auth()
    pool = load_pool("openai-codex")
    if account:
        available = pool._available_entries(clear_expired=True, refresh=True)
        entry = next((e for e in available if _matches(e, account)), None)
        if entry is None:
            if _auto_sync_codex_cli_auth():
                pool = load_pool("openai-codex")
                available = pool._available_entries(clear_expired=True, refresh=True)
                entry = next((e for e in available if _matches(e, account)), None)
            if entry is None:
                known = ", ".join(f"{e.label or e.id} ({e.id})" for e in pool.entries()) or "none"
                raise RuntimeError(f"No available Codex account matching {account!r}. Known: {known}")
        pool.acquire_lease(entry.id)
    else:
        entry = pool.select()
        if entry is None and _auto_sync_codex_cli_auth():
            pool = load_pool("openai-codex")
            entry = pool.select()
    if entry is None:
        raise RuntimeError("No Codex credentials. Run `hermes auth add openai-codex`.")
    return _cred_from_entry(entry)


def resolve_creds(account: str | None = None, order: str | None = None) -> list[Cred]:
    """Ordered list of creds to try for one request, primary first.

    The explicitly chosen account(s) lead; every other currently-available
    account is queued behind them so a 429/401/403 on the primary rolls over to
    the next login mid-request. Accounts already in exhaustion cooldown are
    excluded by `_available_entries`, so once a 429 marks one exhausted the next
    account is preferred until its credits reset.
    """
    from agent.credential_pool import load_pool

    _auto_sync_codex_cli_auth()
    selectors = _selectors(order)
    if account and account not in selectors:
        selectors.insert(0, account)

    def snapshot() -> list[Cred]:
        pool = load_pool("openai-codex")
        available = pool._available_entries(clear_expired=True, refresh=True)
        picked: list[Cred] = []
        seen: set[str] = set()

        def take(entry: object) -> None:
            pool.acquire_lease(entry.id)
            picked.append(_cred_from_entry(entry))
            seen.add(entry.id)

        for selector in selectors:
            entry = next((e for e in available if _matches(e, selector) and e.id not in seen), None)
            if entry is not None:
                take(entry)
        if not selectors:
            primary = pool.select()
            if primary is not None and primary.id not in seen:
                take(primary)
        for entry in available:
            if entry.id not in seen:
                take(entry)
        return picked

    picked = snapshot()
    if not picked and _auto_sync_codex_cli_auth(force=True):
        picked = snapshot()
    if picked:
        return picked

    pool = load_pool("openai-codex")
    known = ", ".join(f"{e.label or e.id} ({e.id})" for e in pool.entries()) or "none"
    if selectors:
        raise RuntimeError(f"No available Codex account matching order {selectors!r}. Known: {known}")
    raise RuntimeError("No Codex credentials. Run `hermes auth add openai-codex`.")


def error_context(message: str) -> dict[str, Any]:
    out: dict[str, Any] = {"message": message, "reason": "proxy_failover"}
    try:
        err = json.loads(message).get("error") or {}
        if isinstance(err, dict):
            out["message"] = str(err.get("message") or message)
            out["reason"] = str(err.get("code") or err.get("type") or out["reason"])
            if err.get("resets_at"):
                out["reset_at"] = err.get("resets_at")
            elif err.get("resets_in_seconds"):
                out["reset_at"] = time.time() + float(err["resets_in_seconds"])
    except Exception:
        pass
    return out


def mark_exhausted(c: Cred, status: int, message: str = "") -> None:
    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    entry = next((e for e in pool.entries() if e.id == c.cid), None)
    if entry is not None:
        pool._mark_exhausted(entry, status, error_context(message))


def import_codex_cli_auth(label: str | None = None, codex_home: str | None = None) -> str:
    from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, label_from_token, load_pool
    from hermes_cli.auth import DEFAULT_CODEX_BASE_URL, _import_codex_cli_tokens, _save_codex_tokens

    with _codex_home_env(codex_home):
        tokens = _import_codex_cli_tokens()
    if not tokens:
        where = codex_home or os.getenv("CODEX_HOME") or "~/.codex"
        raise RuntimeError(f"No usable Codex CLI auth found at {where}/auth.json")
    pool = load_pool("openai-codex")
    user_key = _jwt_user_key(tokens["access_token"])
    refresh_token = tokens.get("refresh_token")
    for entry in pool.entries():
        same_pair = entry.access_token == tokens["access_token"] or (
            bool(refresh_token) and entry.refresh_token == refresh_token
        )
        if not same_pair:
            continue
        if entry.last_status:
            updated = replace(
                entry,
                last_status=None,
                last_status_at=None,
                last_error_code=None,
                last_error_reason=None,
                last_error_message=None,
                last_error_reset_at=None,
            )
            pool._replace_entry(entry, updated)
            pool._persist()
            return _with_identity(f"Codex CLI auth refreshed for {updated.label!r} ({updated.id})", updated.access_token)
        return _with_identity(f"Codex CLI auth already present as {entry.label!r} ({entry.id})", entry.access_token)

    # Match the entry to update by USER (not the org account_id, which every
    # Team member shares) so importing a different user creates its own entry
    # instead of clobbering an existing one. Fall back to the source marker only
    # when the token carries no decodable user identity.
    if user_key:
        manual_import = next(
            (entry for entry in pool.entries()
             if _jwt_user_key(entry.access_token) == user_key),
            None,
        )
    else:
        manual_import = next(
            (entry for entry in pool.entries()
             if (entry.source or "") == "manual:codex_cli_import"),
            None,
        )
    if manual_import is not None:
        updated = replace(
            manual_import,
            access_token=tokens["access_token"],
            refresh_token=refresh_token or manual_import.refresh_token,
            auth_type=AUTH_TYPE_OAUTH,
            base_url=manual_import.base_url or DEFAULT_CODEX_BASE_URL,
            source="manual:codex_cli_import",
            last_status=None,
            last_status_at=None,
            last_error_code=None,
            last_error_reason=None,
            last_error_message=None,
            last_error_reset_at=None,
        )
        pool._replace_entry(manual_import, updated)
        pool._persist()
        _save_codex_tokens(tokens)
        return _with_identity(f"Updated Codex CLI auth for {updated.label!r} ({updated.id})", updated.access_token)

    name = (
        label
        or _label_for_home(codex_home)
        or label_from_token(tokens["access_token"], f"codex-cli-{len(pool.entries()) + 1}")
    )
    entry = PooledCredential(
        provider="openai-codex",
        id=uuid.uuid4().hex[:6],
        label=name,
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="manual:codex_cli_import",
        access_token=tokens["access_token"],
        refresh_token=tokens.get("refresh_token"),
        base_url=DEFAULT_CODEX_BASE_URL,
    )
    added = pool.add_entry(entry)
    _save_codex_tokens(tokens)
    return _with_identity(f"Imported Codex CLI auth as {added.label!r} ({added.id})", added.access_token)


def authenticate_codex_oauth(label: str | None = None) -> str:
    """Run a fresh OpenAI Codex device-code login and save it to Hermes."""
    from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, label_from_token, load_pool
    from hermes_cli import auth as auth_mod

    auth_mod.unsuppress_credential_source("openai-codex", "device_code")
    creds = auth_mod._codex_device_code_login()
    tokens = creds["tokens"]
    access_token = tokens["access_token"]
    refresh_token = tokens.get("refresh_token")
    pool = load_pool("openai-codex")
    user_key = _jwt_user_key(access_token)
    name = label or label_from_token(access_token, f"codex-oauth-{len(pool.entries()) + 1}")

    for entry in pool.entries():
        same_account = user_key and _jwt_user_key(entry.access_token) == user_key
        same_refresh = refresh_token and entry.refresh_token == refresh_token
        if not (same_account or same_refresh):
            continue
        updated = replace(
            entry,
            label=label or entry.label or name,
            auth_type=AUTH_TYPE_OAUTH,
            source="manual:device_code",
            access_token=access_token,
            refresh_token=refresh_token or entry.refresh_token,
            base_url=creds.get("base_url") or entry.base_url,
            last_refresh=creds.get("last_refresh"),
            last_status=None,
            last_status_at=None,
            last_error_code=None,
            last_error_reason=None,
            last_error_message=None,
            last_error_reset_at=None,
        )
        pool._replace_entry(entry, updated)
        pool._persist()
        return _with_identity(f"Updated Codex OAuth auth for {updated.label!r} ({updated.id})", updated.access_token)

    entry = PooledCredential(
        provider="openai-codex",
        id=uuid.uuid4().hex[:6],
        label=name,
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="manual:device_code",
        access_token=access_token,
        refresh_token=refresh_token,
        base_url=creds.get("base_url"),
        last_refresh=creds.get("last_refresh"),
    )
    added = pool.add_entry(entry)
    return _with_identity(f"Added Codex OAuth auth as {added.label!r} ({added.id})", added.access_token)


def _oauth_entry_stale_reason(entry: object) -> str | None:
    from agent.credential_pool import AUTH_TYPE_OAUTH
    from hermes_cli.auth import _codex_access_token_is_expiring

    if getattr(entry, "auth_type", None) != AUTH_TYPE_OAUTH:
        return None
    access_token = str(getattr(entry, "access_token", "") or "").strip()
    refresh_token = str(getattr(entry, "refresh_token", "") or "").strip()
    if not access_token:
        return "missing access token"
    if _codex_access_token_is_expiring(access_token, 0) and not refresh_token:
        return "expired access token with no refresh token"
    status_code = getattr(entry, "last_error_code", None)
    reason = str(getattr(entry, "last_error_reason", "") or "").strip()
    if status_code in {401, 403}:
        return f"auth failure status {status_code}"
    if reason in {"invalid_grant", "invalid_token", "refresh_token_reused", "token_invalidated"}:
        return reason
    return None


def remove_stale_oauth_configs(*, dry_run: bool = False) -> str:
    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    stale: list[tuple[object, str]] = []
    retained = []
    for entry in pool.entries():
        reason = _oauth_entry_stale_reason(entry)
        if reason:
            stale.append((entry, reason))
        else:
            retained.append(entry)

    if not stale:
        return "No stale Codex OAuth configs found."

    lines = [
        f"{'Would remove' if dry_run else 'Removed'} {len(stale)} stale Codex OAuth config"
        f"{'' if len(stale) == 1 else 's'}:"
    ]
    for entry, reason in stale:
        lines.append(f"- {getattr(entry, 'label', '') or getattr(entry, 'id', '')} ({getattr(entry, 'id', '')}): {reason}")

    if not dry_run:
        pool._entries = [
            replace(entry, priority=idx)
            for idx, entry in enumerate(retained)
        ]
        pool._persist()
    return "\n".join(lines)


CREDENTIAL_PROBE_PATH = "/models"
# Error codes (and their JWT-claim cousins) that mean the login is dead and a
# refresh will not recover it. token_invalidated is what Codex returns when a
# previously valid access token has been revoked server-side even though its JWT
# `exp` is still in the future.
DEAD_CREDENTIAL_CODES = {"token_invalidated", "invalid_token", "invalid_grant", "refresh_token_reused"}
HEALTH_OK = "ok"
HEALTH_INVALIDATED = "invalidated"
HEALTH_EXPIRED = "expired"
HEALTH_UNKNOWN = "unknown"
DEAD_HEALTH = {HEALTH_INVALIDATED, HEALTH_EXPIRED}


def _error_payload_code(response: httpx.Response) -> tuple[str, str]:
    """Return (code, message) from a Codex JSON error body, best-effort."""
    try:
        err = (response.json() or {}).get("error") or {}
        if isinstance(err, dict):
            return str(err.get("code") or ""), str(err.get("message") or "")
    except Exception:
        pass
    return "", response.text[:200].replace("\n", " ").strip()


def _probe_entry(pool: object, entry: object) -> dict[str, Any]:
    """Probe a single credential's live health against the Codex backend.

    Refreshes the access token first when it is expiring (and refreshable), then
    makes one lightweight authenticated request. Returns a dict with the health
    category plus the upstream status/code/detail for reporting.
    """
    from agent.credential_pool import AUTH_TYPE_OAUTH
    from agent.auxiliary_client import _codex_cloudflare_headers
    from hermes_cli.auth import DEFAULT_CODEX_BASE_URL, _codex_access_token_is_expiring

    if getattr(entry, "auth_type", None) == AUTH_TYPE_OAUTH:
        try:
            if pool._entry_needs_refresh(entry):
                refreshed = pool._refresh_entry(entry, force=True)
                if refreshed is not None:
                    entry = refreshed
        except Exception as exc:
            debug(f"probe refresh failed for {getattr(entry, 'id', '')}: {exc}")

    token = str(getattr(entry, "access_token", "") or "").strip()
    refresh_token = str(getattr(entry, "refresh_token", "") or "").strip()
    if not token:
        return {"health": HEALTH_INVALIDATED, "http_status": None, "code": "missing_token", "detail": "no access token"}
    if _codex_access_token_is_expiring(token, 0) and not refresh_token:
        return {"health": HEALTH_EXPIRED, "http_status": None, "code": "expired", "detail": "expired access token with no refresh token"}

    base = (str(getattr(entry, "base_url", "") or "") or DEFAULT_CODEX_BASE_URL).rstrip("/")
    headers = _codex_cloudflare_headers(token)
    headers["Authorization"] = f"Bearer {token}"
    try:
        with httpx.Client(timeout=httpx.Timeout(30, connect=15)) as client:
            r = client.get(f"{base}{CREDENTIAL_PROBE_PATH}", headers=headers, params={"client_version": "1.0.0"}, follow_redirects=False)
    except Exception as exc:
        return {"health": HEALTH_UNKNOWN, "http_status": None, "code": "network_error", "detail": str(exc)[:200]}

    if r.status_code == 200:
        return {"health": HEALTH_OK, "http_status": 200, "code": "", "detail": "reachable"}
    code, message = _error_payload_code(r)
    if r.status_code in {401, 403} or code in DEAD_CREDENTIAL_CODES:
        return {"health": HEALTH_INVALIDATED, "http_status": r.status_code, "code": code or f"http_{r.status_code}", "detail": message or "authentication rejected"}
    return {"health": HEALTH_UNKNOWN, "http_status": r.status_code, "code": code or f"http_{r.status_code}", "detail": message or f"unexpected status {r.status_code}"}


def check_credentials() -> list[dict[str, Any]]:
    """Probe every Codex credential and return per-account health records."""
    from agent.credential_pool import load_pool

    _auto_sync_codex_cli_auth()
    pool = load_pool("openai-codex")
    out: list[dict[str, Any]] = []
    for entry in pool.entries():
        ident = _jwt_identity(str(getattr(entry, "access_token", "") or ""))
        record = {
            "id": getattr(entry, "id", ""),
            "label": getattr(entry, "label", "") or getattr(entry, "id", ""),
            "email": ident.get("email", ""),
            "account_id": ident.get("account_id", ""),
            "name": ident.get("name", "") or ident.get("account_name", ""),
            "plan_type": ident.get("plan_type", ""),
        }
        record.update(_probe_entry(pool, entry))
        out.append(record)
    return out


def _format_health_record(rec: dict[str, Any]) -> str:
    head = f"{rec.get('label') or rec.get('id')} ({rec.get('id')})"
    bits = [f"health={rec.get('health')}"]
    if rec.get("name"):
        bits.append(f"name={rec['name']}")
    if rec.get("email"):
        bits.append(f"email={rec['email']}")
    if rec.get("plan_type"):
        bits.append(f"plan={rec['plan_type']}")
    if rec.get("http_status") is not None:
        bits.append(f"http={rec['http_status']}")
    if rec.get("code"):
        bits.append(f"code={rec['code']}")
    if rec.get("detail"):
        bits.append(f"detail={rec['detail']}")
    return f"- {head}: " + " | ".join(bits)


def report_credential_health(records: list[dict[str, Any]] | None = None) -> str:
    records = check_credentials() if records is None else records
    if not records:
        return "No Codex credentials. Run `hermes auth add openai-codex` or `--codex-oauth-auth`."
    healthy = [r for r in records if r["health"] == HEALTH_OK]
    dead = [r for r in records if r["health"] in DEAD_HEALTH]
    unknown = [r for r in records if r["health"] == HEALTH_UNKNOWN]
    lines = [f"Checked {len(records)} Codex credential{'' if len(records) == 1 else 's'}: "
             f"{len(healthy)} ok, {len(dead)} dead, {len(unknown)} unknown."]
    lines.extend(_format_health_record(r) for r in records)
    return "\n".join(lines)


def _delete_entries_by_id(pool: object, ids: set[str]) -> int:
    retained = [e for e in pool.entries() if getattr(e, "id", "") not in ids]
    removed = len(pool.entries()) - len(retained)
    if removed:
        pool._entries = [replace(entry, priority=idx) for idx, entry in enumerate(retained)]
        pool._persist()
    return removed


def _prompt_yes_no(question: str) -> bool:
    while True:
        print(f"{question} [y/N]: ", end="", file=sys.stderr, flush=True)
        answer = sys.stdin.readline().strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"", "n", "no"}:
            return False
        print("Please answer y or n.", file=sys.stderr)


def prune_credentials(*, assume_yes: bool = False) -> str:
    """Probe credentials and interactively delete the dead (invalidated/expired) ones."""
    from agent.credential_pool import load_pool

    records = check_credentials()
    print(report_credential_health(records), file=sys.stderr)
    dead = [r for r in records if r["health"] in DEAD_HEALTH]
    if not dead:
        return "No dead Codex credentials to delete."

    to_delete: set[str] = set()
    if assume_yes:
        to_delete = {r["id"] for r in dead}
    else:
        if not sys.stdin.isatty():
            return (
                f"Found {len(dead)} dead credential(s) but stdin is not interactive. "
                "Re-run with --yes to delete them non-interactively."
            )
        print("", file=sys.stderr)
        for r in dead:
            head = f"{r.get('label') or r.get('id')} ({r.get('id')})"
            detail = r.get("detail") or r.get("code") or r.get("health")
            if _prompt_yes_no(f"Delete dead credential {head} — {detail}?"):
                to_delete.add(r["id"])

    if not to_delete:
        return "No credentials deleted."

    pool = load_pool("openai-codex")
    removed = _delete_entries_by_id(pool, to_delete)
    deleted_labels = ", ".join(r["label"] for r in dead if r["id"] in to_delete) or "none"
    return f"Deleted {removed} Codex credential(s): {deleted_labels}"


def _sync_due(key: str, index: int, count: int, now: float) -> bool:
    """Throttle + stagger: each dir syncs at most once per interval, and dir
    `index` is offset by index*(interval/count) so dirs never sync together."""
    interval = max(0, CODEX_SYNC_INTERVAL_SECONDS)
    if not interval:
        return True
    last = _LAST_SYNC.get(key)
    if last is None:
        # Seed so dir 0 is due immediately and dir i waits i*(interval/count)
        # before its first sync — establishing the staggered cadence.
        last = _LAST_SYNC[key] = now - interval + (interval * index) / (count or 1)
    return (now - last) >= interval


def _auto_sync_codex_cli_auth(force: bool = False) -> bool:
    if not AUTO_SYNC_CLI_AUTH:
        return False
    homes = CODEX_HOME_DIRS or [None]
    now = time.time()
    synced = False
    for index, home in enumerate(homes):
        key = home or "default"
        if not force and not _sync_due(key, index, len(homes), now):
            continue
        try:
            message = import_codex_cli_auth(codex_home=home)
        except Exception as exc:
            debug(f"auto-sync codex auth skipped for {key}: {exc}")
            continue
        _LAST_SYNC[key] = now
        debug(f"auto-sync codex auth ({key}): {message}")
        synced = True
    return synced


def accounts() -> list[dict[str, object]]:
    from agent.credential_pool import STATUS_EXHAUSTED, _exhausted_until, load_pool
    pool = load_pool("openai-codex")
    out = []
    for e in pool.entries():
        status = e.last_status or "ok"
        ident = _jwt_identity(e.access_token)
        out.append({
            "id": e.id,
            "label": e.label,
            "account_id": ident.get("account_id", ""),
            "account_name": ident.get("account_name", ""),
            "email": ident.get("email", ""),
            "name": ident.get("name", ""),
            "plan_type": ident.get("plan_type", ""),
            "user_id": ident.get("user_id", ""),
            "subject": ident.get("subject", ""),
            "source": e.source,
            "status": status,
            "exhausted_until": _exhausted_until(e) if status == STATUS_EXHAUSTED else None,
        })
    return out


def select_account_interactively() -> str:
    available = [acct for acct in accounts() if acct.get("status") == "ok"]
    if not available:
        raise RuntimeError("No available Codex credentials. Run `hermes auth add openai-codex`.")
    if not sys.stdin.isatty():
        raise RuntimeError("--select-account requires an interactive terminal.")
    if len(available) == 1:
        acct = available[0]
        print(f"Selected only available account: {acct.get('label') or acct.get('id')}", file=sys.stderr)
        return str(acct["id"])

    print("Available Codex accounts:", file=sys.stderr)
    for idx, acct in enumerate(available, start=1):
        details = [acct.get("label") or acct.get("id"), f"id={acct.get('id')}"]
        if acct.get("name"):
            details.append(f"name={acct['name']}")
        if acct.get("email"):
            details.append(f"email={acct['email']}")
        if acct.get("account_name"):
            details.append(f"account={acct['account_name']}")
        elif acct.get("account_id"):
            details.append(f"account_id={acct['account_id']}")
        if acct.get("plan_type"):
            details.append(f"plan={acct['plan_type']}")
        print(f"  {idx}. " + " | ".join(str(part) for part in details if part), file=sys.stderr)

    while True:
        print(f"Select account [1-{len(available)}]: ", end="", file=sys.stderr, flush=True)
        choice = sys.stdin.readline().strip()
        if not choice:
            continue
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(available):
                return str(available[index - 1]["id"])
        match = next(
            (
                acct for acct in available
                if choice.lower() in {
                    str(acct.get("id") or "").lower(),
                    str(acct.get("label") or "").lower(),
                    str(acct.get("account_id") or "").lower(),
                    str(acct.get("email") or "").lower(),
                }
            ),
            None,
        )
        if match is not None:
            return str(match["id"])
        print("Invalid selection. Enter a number, id, label, email, or account_id.", file=sys.stderr)


def chat_to_responses(body: dict[str, Any]) -> dict[str, Any]:
    from agent.codex_responses_adapter import _chat_messages_to_responses_input

    instructions, messages = [], []
    for msg in body.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") in {"system", "developer"}:
            instructions.append(str(msg.get("content") or ""))
        elif isinstance(msg, dict):
            messages.append(msg)
    req = {
        "model": body.get("model") or "gpt-5.4",
        "store": False,
        "input": _chat_messages_to_responses_input(messages) or [
            {"role": "user", "content": [{"type": "input_text", "text": ""}]}
        ],
    }
    req["instructions"] = "\n\n".join(filter(None, instructions)) or "You are a helpful assistant."
    tools = []
    for t in body.get("tools") or []:
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        if fn.get("name"):
            tools.append({
                "type": "function",
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            })
    if tools:
        req["tools"] = tools
    return codex_wire_request(normalize_codex_builtin_tools(apply_codex_defaults(req)))


CODEX_BUILTIN_TOOL_ALIASES = {
    "web_search_preview": "web_search",
    "web_search_preview_2025_03_11": "web_search",
}


def normalize_codex_builtin_tool_type(tool_type: Any) -> Any:
    if not isinstance(tool_type, str):
        return tool_type
    return CODEX_BUILTIN_TOOL_ALIASES.get(tool_type, tool_type)


def normalize_codex_builtin_tools(req: dict[str, Any]) -> dict[str, Any]:
    out = dict(req)
    tools = out.get("tools")
    if isinstance(tools, list):
        normalized_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                tool = dict(tool)
                tool["type"] = normalize_codex_builtin_tool_type(tool.get("type"))
            normalized_tools.append(tool)
        out["tools"] = normalized_tools

    tool_choice = out.get("tool_choice")
    if isinstance(tool_choice, dict):
        choice = dict(tool_choice)
        choice["type"] = normalize_codex_builtin_tool_type(choice.get("type"))
        choice_tools = choice.get("tools")
        if isinstance(choice_tools, list):
            normalized_choice_tools = []
            for tool in choice_tools:
                if isinstance(tool, dict):
                    tool = dict(tool)
                    tool["type"] = normalize_codex_builtin_tool_type(tool.get("type"))
                normalized_choice_tools.append(tool)
            choice["tools"] = normalized_choice_tools
        out["tool_choice"] = choice
    return out


def normalize_responses_request(body: dict[str, Any]) -> dict[str, Any]:
    req = dict(body)
    if req.get("stream") is not True:
        req.pop("stream", None)
    req.setdefault("model", "gpt-5.4")
    req.setdefault("instructions", "You are a helpful assistant.")
    req["store"] = False
    if not isinstance(req.get("input"), list):
        text = "" if req.get("input") is None else str(req.get("input"))
        req["input"] = [{"role": "user", "content": [{"type": "input_text", "text": text}]}]
    return codex_wire_request(normalize_codex_builtin_tools(apply_codex_defaults(req)))


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts = [part if isinstance(part, str) else part.get("text") for part in content if isinstance(part, (str, dict))]
    return "\n".join(str(p) for p in parts if p)


def split_model_suffix(model: Any) -> tuple[Any, str | None]:
    if not isinstance(model, str):
        return model, None
    end = model.rstrip()
    if not end.endswith(")"):
        return model, None
    start = end.rfind("(")
    if start < 0:
        return model, None
    base = end[:start].strip()
    suffix = end[start + 1:-1].strip()
    if not base or not suffix:
        return model, None
    return base, suffix


def apply_model_suffix(req: dict[str, Any]) -> dict[str, Any]:
    out = dict(req)
    model, suffix = split_model_suffix(out.get("model"))
    if suffix is None:
        return out
    out["model"] = model

    effort = REASONING_SUFFIXES.get(suffix.lower())
    if effort:
        reasoning = out.get("reasoning")
        if not isinstance(reasoning, dict):
            reasoning = {}
        reasoning["effort"] = effort
        out["reasoning"] = reasoning
    return out


def apply_codex_defaults(req: dict[str, Any]) -> dict[str, Any]:
    req = apply_model_suffix(req)
    out = {k: v for k, v in req.items() if k not in CODEX_DROP_FIELDS}
    lifted, items = [], []
    for item in out.get("input") or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role in {"system", "developer"}:
            text = content_text(item.get("content"))
            if text:
                lifted.append(text)
            continue
        items.append(item)
    if lifted:
        existing = str(out.get("instructions") or "").strip()
        out["instructions"] = "\n\n".join([p for p in [existing, *lifted] if p])
        out["input"] = items or [{"role": "user", "content": ""}]
    reasoning = out.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("enabled") is False:
        out.pop("reasoning", None)
        out.setdefault("include", [])
    else:
        effort = str(reasoning.get("effort") if isinstance(reasoning, dict) and reasoning.get("effort") else "medium")
        out["reasoning"] = {"effort": "low" if effort == "minimal" else effort, "summary": "auto"}
        out.setdefault("include", ["reasoning.encrypted_content"])
    if out.get("tools") and "parallel_tool_calls" not in out:
        out["parallel_tool_calls"] = True
    return out


def codex_wire_request(req: dict[str, Any]) -> dict[str, Any]:
    out = dict(req)
    extra_body = out.pop("extra_body", None)
    out.pop("extra_headers", None)
    if isinstance(extra_body, dict):
        out.update(extra_body)
    items = []
    for item in out.get("input") or []:
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"} and not item.get("type"):
            item = {"type": "message", **item}
        items.append(item)
    out["input"] = items
    return out


def responses_to_chat(payload: dict[str, Any], model: str) -> dict[str, Any]:
    text, calls = [], []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                    text.append(str(part.get("text") or ""))
        elif item.get("type") == "function_call":
            calls.append({
                "id": item.get("call_id") or item.get("id") or f"call_{len(calls)}",
                "type": "function",
                "function": {"name": item.get("name") or "", "arguments": item.get("arguments") or "{}"},
            })
    message = {"role": "assistant", "content": None if calls else "".join(text)}
    if calls:
        message["tool_calls"] = calls
    result = {
        "id": payload.get("id") or f"chatcmpl_codex_{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model") or model,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
    }
    usage = payload.get("usage")
    if isinstance(usage, dict):
        inp, out = int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
        result["usage"] = {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": int(usage.get("total_tokens") or inp + out)}
    return result


def chat_stream_chunk(
    model: str,
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    chunk_id: str | None = None,
) -> dict[str, Any]:
    choice: dict[str, Any] = {"index": 0, "delta": {}, "finish_reason": finish_reason}
    if content is not None:
        choice["delta"] = {"content": content}
    return {
        "id": chunk_id or f"chatcmpl_codex_{int(time.time() * 1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
    }


def collect_responses_stream(client: httpx.Client, url: str, req: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    final, text, output_items = None, [], []
    body = dict(req)
    body["stream"] = True
    with client.stream("POST", url, json=body, headers=headers, follow_redirects=False) as r:
        if r.status_code >= 400:
            r.read()
            raise httpx.HTTPStatusError(r.text, request=r.request, response=r)
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            raw = line[6:].strip()
            if not raw or raw == "[DONE]":
                continue
            event = json.loads(raw)
            etype = event.get("type")
            if etype == "response.output_item.done" and isinstance(event.get("item"), dict):
                output_items.append(event["item"])
            elif etype == "response.output_text.delta" and isinstance(event.get("delta"), str):
                text.append(event["delta"])
            elif etype in {"response.completed", "response.incomplete", "response.failed"}:
                final = event.get("response")
            elif etype == "error":
                raise RuntimeError(event.get("message") or event.get("error") or "Codex stream error")
    if not isinstance(final, dict):
        final = {"output": []}
    if not final.get("output") and output_items:
        final["output"] = output_items
    elif not final.get("output") and text:
        final["output"] = [{"type": "message", "content": [{"type": "output_text", "text": "".join(text)}]}]
    return final


def stream_responses_as_chat(
    client: httpx.Client,
    url: str,
    req: dict[str, Any],
    headers: dict[str, str],
    write: Any,
    start: Any | None = None,
) -> None:
    body = dict(req)
    body["stream"] = True
    model = str(body.get("model") or "gpt-5.4")
    with client.stream("POST", url, json=body, headers=headers, follow_redirects=False) as r:
        if r.status_code >= 400:
            r.read()
            raise httpx.HTTPStatusError(r.text, request=r.request, response=r)
        if start is not None:
            start()
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            raw = line[6:].strip()
            if not raw or raw == "[DONE]":
                continue
            event = json.loads(raw)
            etype = event.get("type")
            if etype == "response.output_text.delta" and isinstance(event.get("delta"), str):
                chunk = chat_stream_chunk(model, content=event["delta"], chunk_id=event.get("response_id"))
                write(b"data: " + _b(chunk) + b"\n\n")
            elif etype in {"response.completed", "response.incomplete"}:
                response = event.get("response") if isinstance(event.get("response"), dict) else {}
                finish_reason = "stop" if etype == "response.completed" else "length"
                chunk = chat_stream_chunk(model, finish_reason=finish_reason, chunk_id=response.get("id"))
                write(b"data: " + _b(chunk) + b"\n\n")
                write(b"data: [DONE]\n\n")
                return
            elif etype == "response.failed":
                response = event.get("response") if isinstance(event.get("response"), dict) else {}
                error = response.get("error") if isinstance(response.get("error"), dict) else {}
                raise RuntimeError(error.get("message") or "Codex stream failed")
            elif etype == "error":
                raise RuntimeError(event.get("message") or event.get("error") or "Codex stream error")
        write(b"data: " + _b(chat_stream_chunk(model, finish_reason="stop")) + b"\n\n")
        write(b"data: [DONE]\n\n")


class Handler(BaseHTTPRequestHandler):
    server_version = "codex-oauth-proxy/0.2"

    def log_message(self, fmt: str, *args: object) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        sys.stderr.write("%s %s - %s\n" % (ts, self.address_string(), fmt % args))

    def json(self, status: int, obj: object, extra: dict[str, str] | None = None) -> None:
        body = _b(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def account(self) -> str | None:
        return self.headers.get("X-Hermes-Codex-Account") or getattr(self.server, "account", None)

    def account_order(self) -> str | None:
        return self.headers.get("X-Hermes-Codex-Account-Order") or getattr(self.server, "account_order", None)

    def do_GET(self) -> None:
        self.route()

    def do_POST(self) -> None:
        self.route()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1")
        self.send_header("Access-Control-Allow-Headers", "authorization,content-type,x-hermes-codex-account,x-hermes-codex-account-order")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def route(self) -> None:
        path = urlsplit(self.path).path
        if path == "/health":
            try:
                c = resolve_cred(self.account())
                self.json(200, {"status": "ok", "account": c.label})
            except Exception as e:
                self.json(401, {"status": "error", "error": str(e)})
            return
        if path == "/accounts":
            self.json(200, {"accounts": accounts()})
            return
        rel = _route_rel(path)
        if rel == "/chat/completions":
            return self.chat()
        if rel == "/models" or _is_responses_path(rel):
            return self.pass_through(rel)
        self.json(404, {"error": {"message": f"Unsupported path {path}", "code": "path_not_allowed"}})

    def upstream_headers(self, c: Cred, req: dict[str, Any] | None = None) -> dict[str, str]:
        h = _req_headers(self.headers.items())
        h.update(c.headers)
        h["Authorization"] = f"Bearer {c.token}"
        cache_key = str((req or {}).get("prompt_cache_key") or "").strip()
        if cache_key:
            h.setdefault("session_id", cache_key)
            h.setdefault("x-client-request-id", cache_key)
        return h

    def pass_through(self, rel: str) -> None:
        try:
            body = None
            req = None
            downstream_stream = False
            is_responses = _is_responses_path(rel)
            if self.command != "GET":
                if is_responses:
                    raw_req = _body(self)
                    debug(f"downstream {self.command} {self.path} raw={json.dumps(raw_req, separators=(',', ':'))}")
                    downstream_stream = raw_req.get("stream") is True
                    req = normalize_responses_subpath_request(rel, raw_req)
                    debug(f"upstream {rel} normalized={json.dumps(req, separators=(',', ':'))}")
                    body = _b(req)
                else:
                    body = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
            with httpx.Client(timeout=httpx.Timeout(300, connect=15)) as client:
                creds = resolve_creds(self.account(), self.account_order())
                for idx, c in enumerate(creds):
                    query = urlsplit(self.path).query or ("client_version=1.0.0" if rel == "/models" else "")
                    url = urlunsplit(("", "", f"{c.base_url}{rel}", query, ""))
                    try:
                        if rel == "/responses" and not downstream_stream and self.command != "GET":
                            payload = collect_responses_stream(client, url, req or {}, self.upstream_headers(c, req))
                            self.json(200, payload, {"X-Hermes-Codex-Account": c.label, "X-Hermes-Codex-Credential-ID": c.cid})
                            return
                        with client.stream(self.command, url, content=body, headers=self.upstream_headers(c, req), follow_redirects=False) as r:
                            if r.status_code in FAILOVER_STATUS and idx + 1 < len(creds):
                                mark_exhausted(c, r.status_code, r.text[:500])
                                continue
                            if r.status_code >= 400:
                                r.read()
                                debug(f"upstream {url} status={r.status_code} body={r.text}")
                            self.send_response(r.status_code)
                            for k, v in _resp_headers(r.headers):
                                self.send_header(k, v)
                            self.send_header("X-Hermes-Codex-Account", c.label)
                            self.send_header("X-Hermes-Codex-Credential-ID", c.cid)
                            self.end_headers()
                            for chunk in r.iter_bytes():
                                self.wfile.write(chunk)
                            return
                    except httpx.HTTPStatusError as e:
                        status = e.response.status_code if e.response is not None else 502
                        if status in FAILOVER_STATUS and idx + 1 < len(creds):
                            mark_exhausted(c, status, str(e)[:500])
                            continue
                        raise
        except BrokenPipeError:
            return
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else 502
            try:
                payload = e.response.json() if e.response is not None else {"error": {"message": str(e)}}
            except Exception:
                text = e.response.text if e.response is not None else str(e)
                payload = {"error": {"message": text, "code": "upstream_error"}}
            self.json(status, payload)
        except Exception as e:
            self.json(502, {"error": {"message": str(e), "code": "proxy_error"}})

    def chat(self) -> None:
        try:
            req = _body(self)
            responses_req = chat_to_responses(req)
            downstream_stream = req.get("stream") is True
            with httpx.Client(timeout=httpx.Timeout(300, connect=15)) as client:
                creds = resolve_creds(self.account(), self.account_order())
                for idx, c in enumerate(creds):
                    try:
                        if downstream_stream:
                            started = False

                            def start_stream() -> None:
                                nonlocal started
                                if started:
                                    return
                                started = True
                                self.send_response(200)
                                self.send_header("Content-Type", "text/event-stream")
                                self.send_header("Cache-Control", "no-cache")
                                self.send_header("Connection", "keep-alive")
                                self.send_header("X-Hermes-Codex-Account", c.label)
                                self.send_header("X-Hermes-Codex-Credential-ID", c.cid)
                                self.end_headers()

                            stream_responses_as_chat(
                                client,
                                f"{c.base_url}/responses",
                                responses_req,
                                self.upstream_headers(c, responses_req),
                                self.wfile.write,
                                start_stream,
                            )
                            return
                        payload = collect_responses_stream(client, f"{c.base_url}/responses", responses_req, self.upstream_headers(c, responses_req))
                        extra = {"X-Hermes-Codex-Account": c.label, "X-Hermes-Codex-Credential-ID": c.cid}
                        self.json(200, responses_to_chat(payload, str(responses_req["model"])), extra)
                        return
                    except httpx.HTTPStatusError as e:
                        status = e.response.status_code if e.response is not None else 502
                        if status in FAILOVER_STATUS and idx + 1 < len(creds):
                            mark_exhausted(c, status, str(e)[:500])
                            continue
                        raise
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else 502
            try:
                payload = e.response.json() if e.response is not None else {"error": {"message": str(e)}}
            except Exception:
                payload = {"error": {"message": str(e), "code": "upstream_error"}}
            self.json(status, payload)
        except Exception as e:
            self.json(502, {"error": {"message": str(e), "code": "proxy_error"}})


def main() -> int:
    global AUTO_SYNC_CLI_AUTH

    p = argparse.ArgumentParser(description="Local OpenAI-compatible proxy for Hermes Codex OAuth.")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    account_group = p.add_mutually_exclusive_group()
    account_group.add_argument("--account", help="Default Codex credential label/id/account_id")
    account_group.add_argument("--select-account", action="store_true", help="Interactively select the default Codex account")
    p.add_argument("--account-order", help="Comma-separated failover order by label/id/account_id")
    p.add_argument("--import-codex-auth", action="store_true", help="Copy existing Codex CLI auth into Hermes and exit")
    p.add_argument("--codex-oauth-auth", action="store_true", help="Run a fresh Codex OAuth device-code login into Hermes and exit")
    p.add_argument("--serve-after-import", action="store_true", help="Start the proxy after --import-codex-auth")
    p.add_argument("--serve-after-auth", action="store_true", help="Start the proxy after --codex-oauth-auth")
    p.add_argument("--import-only", action="store_true", help="Deprecated alias for --import-codex-auth")
    p.add_argument("--label", help="Label for imported Codex CLI auth")
    p.add_argument("--codex-home", action="append", metavar="DIR", help="Codex CLI login dir to import/auto-sync from (repeatable). Overrides CODEX_PROXY_CODEX_HOMES. Default: $CODEX_HOME or ~/.codex")
    p.add_argument("--no-auto-sync-codex-cli-auth", action="store_true", help="Do not auto-copy Codex CLI auth while serving")
    p.add_argument("--remove-stale-oauth-configs", action="store_true", help="Remove stale Codex OAuth credentials from the Hermes pool and exit")
    p.add_argument("--dry-run", action="store_true", help="Show what --remove-stale-oauth-configs would remove")
    p.add_argument("--check-credentials", action="store_true", help="Probe each Codex credential against the backend, report health (ok/invalidated/expired), and exit")
    p.add_argument("--prune-credentials", action="store_true", help="Probe credentials and interactively delete dead (invalidated/expired) ones, then exit")
    p.add_argument("--yes", "-y", action="store_true", help="With --prune-credentials, delete all dead credentials without prompting")
    args = p.parse_args()
    if args.no_auto_sync_codex_cli_auth:
        AUTO_SYNC_CLI_AUTH = False
    if args.codex_home:
        CODEX_HOME_DIRS[:] = [str(Path(d).expanduser()) for d in args.codex_home]
    if args.remove_stale_oauth_configs:
        print(remove_stale_oauth_configs(dry_run=args.dry_run), file=sys.stderr)
        return 0
    if args.check_credentials:
        print(report_credential_health(), file=sys.stderr)
        return 0
    if args.prune_credentials:
        print(prune_credentials(assume_yes=args.yes), file=sys.stderr)
        return 0
    if args.codex_oauth_auth:
        print(authenticate_codex_oauth(args.label), file=sys.stderr)
    if args.codex_oauth_auth and not args.serve_after_auth:
        return 0
    if args.import_codex_auth or args.import_only:
        homes = CODEX_HOME_DIRS or [None]
        for home in homes:
            # A single --label is ambiguous across multiple dirs, so only apply it
            # when importing from exactly one.
            label = args.label if len(homes) == 1 else None
            print(import_codex_cli_auth(label, codex_home=home), file=sys.stderr)
    if (args.import_codex_auth or args.import_only) and not args.serve_after_import:
        return 0
    if args.select_account:
        args.account = select_account_interactively()
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as e:
        print(f"Could not bind http://{args.host}:{args.port}: {e}", file=sys.stderr)
        print("If the proxy is already running, use that instance or choose --port <other-port>.", file=sys.stderr)
        return 1
    server.account = args.account
    server.account_order = args.account_order
    print(f"Listening on http://{args.host}:{args.port}/v1", file=sys.stderr)
    print("Endpoints: /v1/responses, /v1/responses/*, /v1/chat/completions, /v1/models, /accounts, /health", file=sys.stderr)
    if args.account:
        print(f"Default account: {args.account}", file=sys.stderr)
    if args.account_order:
        print(f"Account order: {args.account_order}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
