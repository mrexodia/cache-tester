from __future__ import annotations

import argparse
import errno
import json
import mimetypes
import os
import re
import sys
import time
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests

from cache_tester.core import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_CONTEXT_WORDS,
    DEFAULT_TOOL_COUNT,
    RawClient,
    RequestResult,
    assess_mode,
    build_body,
    discover_model,
    endpoint_for,
    headers_for,
    make_large_context,
    normalize_base_url,
    run_session,
)

DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8765
DEFAULT_ENDPOINT = "http://127.0.0.1:1234"
DEFAULT_API_KEY = ""
API_TYPES = ("chat", "responses", "anthropic")
STREAM_MODES = (False, True)


WWW_DIR = Path(__file__).with_name("www")
INDEX_HTML_PATH = WWW_DIR / "index.html"


def list_models(base_url: str, api_key: str, timeout: float) -> list[str]:
    base = normalize_base_url(base_url)
    response = requests.get(
        f"{base}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=(timeout, timeout),
    )
    response.raise_for_status()
    data = response.json()
    models: list[str] = []
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        for item in data["data"]:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                models.append(item["id"])
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        for item in data["models"]:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                models.append(item["name"])
    return models


def safe_run_id(value: str | None) -> str:
    if not value:
        value = "web-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    return value[:120] or "web-run"


def bool_from_json(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on", "stream"}
    return bool(value)


def is_client_disconnect(exc: BaseException) -> bool:
    if isinstance(exc, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)):
        return True
    if isinstance(exc, OSError):
        return exc.errno in {errno.EPIPE, errno.ECONNABORTED, errno.ECONNRESET} or getattr(exc, "winerror", None) in {10053, 10054}
    return False


def config_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    api_key = payload.get("api_key") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or DEFAULT_API_KEY
    base_url = normalize_base_url(str(payload.get("base_url") or DEFAULT_ENDPOINT))
    model = str(payload.get("model") or "").strip()
    if not model:
        model = discover_model(base_url, api_key, DEFAULT_CONNECT_TIMEOUT_SECONDS) or "local-model"
    read_timeout = payload.get("read_timeout")
    if read_timeout in ("", None):
        read_timeout = None
    else:
        read_timeout = float(read_timeout)
    return {
        "run_id": safe_run_id(str(payload.get("run_id") or "")),
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "api": str(payload.get("api") or "chat"),
        "stream": bool_from_json(payload.get("stream")),
        "context_tokens": max(100, int(payload.get("context_tokens") or DEFAULT_CONTEXT_WORDS)),
        "tool_count": max(0, int(payload.get("tool_count") or DEFAULT_TOOL_COUNT)),
        "turns": max(2, int(payload.get("turns") or 2)),
        "temperature": float(payload.get("temperature") or 0.0),
        "connect_timeout": float(payload.get("connect_timeout") or DEFAULT_CONNECT_TIMEOUT_SECONDS),
        "read_timeout": read_timeout,
    }


def api_log_name(api: str) -> str:
    return "completions" if api == "chat" else api


def log_dir_for(root: Path, cfg: dict[str, Any], phase: str) -> Path:
    mode = "stream" if cfg["stream"] else "nonstream"
    return root / cfg["run_id"] / phase / f"{api_log_name(cfg['api'])}-{mode}"


def result_to_dict(result: RequestResult) -> dict[str, Any]:
    return asdict(result)


def write_config(log_dir: Path, cfg: dict[str, Any], phase: str) -> None:
    public_cfg = {k: v for k, v in cfg.items() if k != "api_key"}
    public_cfg["phase"] = phase
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(json.dumps(public_cfg, indent=2), encoding="utf-8")


def read_log_files(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    requested = payload.get("files")
    if not isinstance(requested, dict):
        raise ValueError("files must be an object")

    root_path = root.resolve()
    max_bytes = 5_000_000
    files: dict[str, str] = {}
    for label, raw_path in requested.items():
        if not isinstance(label, str) or not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = Path.cwd() / path
        resolved = path.resolve()
        try:
            resolved.relative_to(root_path)
        except ValueError as exc:
            raise PermissionError(f"refusing to read outside log directory: {label}") from exc
        if not resolved.exists() or not resolved.is_file():
            files[label] = "<missing>"
            continue
        data = resolved.read_bytes()
        suffix = ""
        if len(data) > max_bytes:
            data = data[:max_bytes]
            suffix = f"\n\n<truncated after {max_bytes} bytes>"
        files[label] = data.decode("utf-8", errors="replace") + suffix
    return {"ok": True, "files": files}


def make_smoke_tools(api: str, nonce: str) -> list[dict[str, Any]]:
    """One deliberately unique tool so smoke requests cannot share a tool prefix."""
    short = re.sub(r"[^a-zA-Z0-9_]", "_", nonce)[:24]
    name = f"x{short}"
    description = f"{nonce} unique smoke-test tool. Do not call this tool."
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "description": f"{nonce} unique smoke query."}
        },
    }
    if api == "chat":
        return [{"type": "function", "function": {"name": name, "description": description, "parameters": schema}}]
    if api == "responses":
        return [{"type": "function", "name": name, "description": description, "parameters": schema}]
    if api == "anthropic":
        return [{"name": name, "description": description, "input_schema": schema}]
    return []


def run_hidden_warmup(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Run one tiny visible warmup request before a measured phase.

    The first request to a local server often pays model/session warmup costs
    unrelated to endpoint support or prompt-cache behavior. This request uses a
    unique nonce so it does not share a prefix with smoke or cache-test prompts.
    """
    cfg = config_from_payload(payload)
    phase = safe_run_id(str(payload.get("warmup_phase") or "manual"))
    errors: list[str] = []
    for api in API_TYPES:
        warm_cfg = dict(cfg)
        warm_cfg["api"] = api
        warm_cfg["stream"] = False
        log_dir = log_dir_for(root, warm_cfg, f"warmup-{phase}")
        write_config(log_dir, warm_cfg, f"warmup-{phase}")
        client = RawClient(
            api_key=cfg["api_key"],
            log_dir=log_dir,
            connect_timeout=cfg["connect_timeout"],
            read_timeout=cfg["read_timeout"],
            log_secrets=False,
        )
        warmup_nonce = uuid.uuid4().hex
        body = build_body(
            api=api,
            model=cfg["model"],
            system_prompt=f"{warmup_nonce}\nVisible {phase} warmup. Reply exactly OK.",
            conversation=[("user", "Reply exactly OK.")],
            tools=[],
            stream=False,
            temperature=cfg["temperature"],
        )
        result = client.post_json(
            label=f"warmup-{api}",
            api=api,
            url=endpoint_for(api, cfg["base_url"]),
            headers=headers_for(api, cfg["api_key"]),
            body=body,
            stream=False,
        )
        summary = {"ok": result.ok, "phase": phase, "api": api, "log_dir": str(log_dir), "result": result_to_dict(result)}
        (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if result.ok:
            return summary
        errors.append(f"{api}: {result.error or result.status_code}")
    return {"ok": False, "error": "; ".join(errors)}


def run_smoke_one(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    cfg = config_from_payload(payload)
    if cfg["api"] not in API_TYPES:
        raise ValueError(f"unsupported api: {cfg['api']}")
    log_dir = log_dir_for(root, cfg, "smoke")
    write_config(log_dir, cfg, "smoke")
    client = RawClient(
        api_key=cfg["api_key"],
        log_dir=log_dir,
        connect_timeout=cfg["connect_timeout"],
        read_timeout=cfg["read_timeout"],
        log_secrets=False,
    )
    smoke_nonce = uuid.uuid4().hex
    system_prompt = (
        f"{smoke_nonce}\n"
        f"Smoke test metadata: run={cfg['run_id']} api={cfg['api']} "
        f"mode={'stream' if cfg['stream'] else 'nonstream'}.\n"
        "You are a smoke-test assistant. Do not call tools. "
        "If the user says ping, reply exactly pong."
    )
    body = build_body(
        api=cfg["api"],
        model=cfg["model"],
        system_prompt=system_prompt,
        conversation=[("user", "ping. Reply exactly pong.")],
        tools=make_smoke_tools(cfg["api"], smoke_nonce),
        stream=cfg["stream"],
        temperature=cfg["temperature"],
    )
    result = client.post_json(
        label=f"smoke-{cfg['api']}-{'stream' if cfg['stream'] else 'nonstream'}",
        api=cfg["api"],
        url=endpoint_for(cfg["api"], cfg["base_url"]),
        headers=headers_for(cfg["api"], cfg["api_key"]),
        body=body,
        stream=cfg["stream"],
    )
    summary = {"ok": result.ok, "log_dir": str(log_dir), "result": result_to_dict(result)}
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_full_one(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    cfg = config_from_payload(payload)
    if cfg["api"] not in API_TYPES:
        raise ValueError(f"unsupported api: {cfg['api']}")
    log_dir = log_dir_for(root, cfg, "full")
    write_config(log_dir, cfg, "full")
    client = RawClient(
        api_key=cfg["api_key"],
        log_dir=log_dir,
        connect_timeout=cfg["connect_timeout"],
        read_timeout=cfg["read_timeout"],
        log_secrets=False,
    )
    context = make_large_context(cfg["context_tokens"], seed=1)
    results = run_session(
        client=client,
        api=cfg["api"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        context=context,
        stream=cfg["stream"],
        turns=cfg["turns"],
        tool_count=cfg["tool_count"],
        temperature=cfg["temperature"],
    )
    assessment = assess_mode(results)
    ok = bool(results) and all(r.ok for r in results)
    summary = {
        "ok": ok,
        "log_dir": str(log_dir),
        "assessment": assessment,
        "results": [result_to_dict(r) for r in results],
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class CacheTesterHandler(BaseHTTPRequestHandler):
    server_version = "cache-tester/0.2"
    log_root: Path = Path(".cache-tester-logs")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_json({"ok": True, "time": time.time()})
            return
        if parsed.path == "/api/models":
            params = parse_qs(parsed.query)
            base_url = params.get("base_url", [DEFAULT_ENDPOINT])[0]
            api_key = params.get("api_key", [DEFAULT_API_KEY])[0] or DEFAULT_API_KEY
            try:
                models = list_models(base_url, api_key, DEFAULT_CONNECT_TIMEOUT_SECONDS)
                self.send_json({"ok": True, "base_url": normalize_base_url(base_url), "models": models})
            except Exception as exc:
                self.send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.OK)
            return
        if parsed.path.startswith("/api/"):
            self.send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        self.serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self.read_json_body()
            if parsed.path == "/api/warmup":
                self.send_json(run_hidden_warmup(self.log_root, payload))
                return
            if parsed.path == "/api/smoke-one":
                self.send_json(run_smoke_one(self.log_root, payload))
                return
            if parsed.path == "/api/full-one":
                self.send_json(run_full_one(self.log_root, payload))
                return
            if parsed.path == "/api/log-files":
                self.send_json(read_log_files(self.log_root, payload))
                return
            self.send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            if is_client_disconnect(exc):
                return
            self.send_json(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=6),
                },
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def serve_static(self, request_path: str) -> None:
        relative = unquote(request_path).lstrip("/")
        if not relative or relative.endswith("/"):
            relative = f"{relative}index.html"
        root = WWW_DIR.resolve()
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            self.send_bytes(b"Forbidden", "text/plain; charset=utf-8", HTTPStatus.FORBIDDEN)
            return
        if not path.exists() or not path.is_file():
            self.send_bytes(b"Not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path == INDEX_HTML_PATH.resolve():
            content_type = "text/html; charset=utf-8"
        elif content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self.send_bytes(path.read_bytes(), content_type)

    def send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        try:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except OSError as exc:
            if is_client_disconnect(exc):
                return
            raise

    def send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_bytes(raw, "application/json; charset=utf-8", status)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}", file=sys.stderr)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the cache tester web UI.")
    parser.add_argument("--host", default=DEFAULT_WEB_HOST, help=f"Bind host (default: {DEFAULT_WEB_HOST}).")
    parser.add_argument("--port", type=int, default=DEFAULT_WEB_PORT, help=f"Bind port (default: {DEFAULT_WEB_PORT}).")
    parser.add_argument("--log-dir", type=Path, default=Path(".cache-tester-logs"), help="Log directory root (default: .cache-tester-logs).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    CacheTesterHandler.log_root = args.log_dir
    server = ThreadingHTTPServer((args.host, args.port), CacheTesterHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Cache tester web UI: {url}")
    print(f"Logs: {args.log_dir}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
