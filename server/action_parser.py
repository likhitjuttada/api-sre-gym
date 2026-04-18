from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

ActionKind = Literal["http", "found", "invalid"]

_HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")

_FENCE_RE = re.compile(r"```(?:\w+)?\n?(.*?)```", re.DOTALL)
_FOUND_RE = re.compile(r"FOUND\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)
_NATIVE_RE = re.compile(
    rf"^\s*({'|'.join(_HTTP_METHODS)})\s+(\S+)(?:\s+(.+))?\s*$",
    re.IGNORECASE,
)
_CURL_METHOD_RE = re.compile(r"-X\s+(\w+)", re.IGNORECASE)
_CURL_URL_RE = re.compile(r"(?:\"|')?(https?://[^\s\"']+)(?:\"|')?")
_CURL_HEADER_RE = re.compile(r"-H\s+(?:\"([^\"]+)\"|'([^']+)')")
_CURL_DATA_RE = re.compile(r"(?:--data(?:-raw)?|-d)\s+(?:\"([^\"]*)\"|'([^']*)')")


@dataclass
class ParsedAction:
    kind: ActionKind
    method: str | None = None
    path: str | None = None
    body: dict | None = None
    headers: dict[str, str] = field(default_factory=dict)
    raw_claim: str | None = None
    error: str | None = None

    @classmethod
    def invalid(cls, reason: str) -> "ParsedAction":
        return cls(kind="invalid", error=reason)

    @classmethod
    def found(cls, claim: str) -> "ParsedAction":
        return cls(kind="found", raw_claim=claim.strip())


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _parse_body(raw: str | None) -> dict | None:
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(value, dict):
        return value
    return {"_value": value}


def _parse_native(line: str) -> ParsedAction | None:
    m = _NATIVE_RE.match(line)
    if not m:
        return None
    method = m.group(1).upper()
    path = m.group(2)
    parsed_url = urlparse(path)
    if parsed_url.scheme:
        path = parsed_url.path + (f"?{parsed_url.query}" if parsed_url.query else "")
    if not path.startswith("/"):
        path = "/" + path
    body = _parse_body(m.group(3))
    return ParsedAction(kind="http", method=method, path=path, body=body)


def _parse_curl(text: str) -> ParsedAction | None:
    if "curl" not in text.lower():
        return None
    url_m = _CURL_URL_RE.search(text)
    if not url_m:
        return None
    parsed = urlparse(url_m.group(1))
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    method_m = _CURL_METHOD_RE.search(text)
    method = method_m.group(1).upper() if method_m else "GET"
    if method not in _HTTP_METHODS:
        return None
    headers: dict[str, str] = {}
    for dq, sq in _CURL_HEADER_RE.findall(text):
        h = dq or sq
        if ":" in h:
            k, _, v = h.partition(":")
            headers[k.strip()] = v.strip()
    body = None
    data_m = _CURL_DATA_RE.search(text)
    if data_m:
        raw = data_m.group(1) if data_m.group(1) is not None else data_m.group(2)
        body = _parse_body(raw)
        if method == "GET":
            method = "POST"
    return ParsedAction(kind="http", method=method, path=path, body=body, headers=headers)


def parse(text: str) -> ParsedAction:
    if not text or not text.strip():
        return ParsedAction.invalid("empty output")

    cleaned = _strip_fences(text)

    found_m = _FOUND_RE.search(cleaned)
    if found_m:
        claim = found_m.group(1).strip()
        if claim:
            return ParsedAction.found(claim)

    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        native = _parse_native(line)
        if native:
            return native

    curl = _parse_curl(cleaned)
    if curl:
        return curl

    return ParsedAction.invalid("could not parse HTTP request or FOUND claim")


def canonical_hash_key(action: ParsedAction) -> str:
    if action.kind != "http":
        return f"{action.kind}:{action.raw_claim or action.error or ''}"
    method = (action.method or "GET").upper()
    path = (action.path or "/").strip()
    body = json.dumps(action.body, sort_keys=True) if action.body else ""
    return f"{method} {path} {body}"
