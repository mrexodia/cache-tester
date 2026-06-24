from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests

DEFAULT_CONTEXT_WORDS = 56_000
DEFAULT_TOOL_COUNT = 8
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5
DEFAULT_READ_TIMEOUT_SECONDS: float | None = None

COMMON_WORDS = " ".join(
    (
        "the of and to in is that for with as on by from at this it be are was an or",
        "reference context material section note record system prompt model cache token",
        "agent session append history turn request response local server timing measure",
        "simple stable deterministic unrelated information paragraph document example",
        "alpha beta gamma delta river mountain forest market signal vector matrix",
        "analysis design implementation result report log raw json http stream chunk",
        "weather calendar file search edit write read bash function schema parameter",
    )
).split()


@dataclass
class RequestResult:
    label: str
    api: str
    stream: bool
    url: str
    status_code: int | None
    ok: bool
    total_ms: float
    ttft_ms: float | None = None
    text: str = ""
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    prompt_ms: float | None = None
    prompt_n: int | None = None
    response_json_path: str | None = None
    response_http_path: str | None = None
    request_json_path: str | None = None
    request_http_path: str | None = None


class RawClient:
    def __init__(
        self,
        *,
        api_key: str,
        log_dir: Path,
        connect_timeout: float,
        read_timeout: float | None,
        log_secrets: bool,
    ) -> None:
        self.api_key = api_key
        self.log_dir = log_dir
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.log_secrets = log_secrets
        self.counter = 0
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def post_json(
        self,
        *,
        label: str,
        api: str,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        stream: bool,
    ) -> RequestResult:
        self.counter += 1
        safe_label = label.replace(" ", "-").replace("/", "-")
        prefix = f"{self.counter:03d}-{safe_label}"
        body_bytes = json.dumps(
            body,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        sent_headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            **headers,
        }

        request_json_path = self.log_dir / f"{prefix}.request.json"
        request_http_path = self.log_dir / f"{prefix}.request.http"
        request_json_path.write_bytes(body_bytes)
        request_http_path.write_text(
            render_http_request("POST", url, sent_headers, body_bytes, self.log_secrets),
            encoding="utf-8",
        )

        start = time.perf_counter()
        status_code: int | None = None
        response_headers: dict[str, str] = {}
        response_body = b""
        parsed_json: Any = None
        parsed_events: list[Any] = []
        ttft_ms: float | None = None
        text = ""
        error: str | None = None

        try:
            response = requests.post(
                url,
                headers=sent_headers,
                data=body_bytes,
                timeout=(self.connect_timeout, self.read_timeout),
                stream=stream,
            )
            status_code = response.status_code
            response_headers = dict(response.headers)

            if stream:
                chunks, parsed_events, text, ttft_ms = read_streaming_response(
                    response, api, start
                )
                response_body = b"".join(chunks)
            else:
                response_body = response.content
                if response_body:
                    try:
                        parsed_json = response.json()
                        text = extract_text(api, parsed_json)
                    except Exception:
                        parsed_json = None

            if status_code < 200 or status_code >= 300:
                if parsed_json is None and response_body:
                    try:
                        parsed_json = json.loads(response_body.decode("utf-8"))
                    except Exception:
                        pass
                error = response_body.decode("utf-8", errors="replace")[:4000]
        except Exception as exc:  # requests exceptions and stream parsing failures
            error = f"{type(exc).__name__}: {exc}"

        total_ms = (time.perf_counter() - start) * 1000.0
        response_http_path = self.log_dir / f"{prefix}.response.http"
        response_http_path.write_bytes(
            render_http_response(status_code, response_headers, response_body)
        )

        response_json_path: Path | None = None
        metrics_source: Any = parsed_json
        if stream:
            events_path = self.log_dir / f"{prefix}.response-events.jsonl"
            with events_path.open("w", encoding="utf-8") as f:
                for event in parsed_events:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
            metrics_source = coalesce_stream_metrics(api, parsed_events)
            response_json_path = events_path
        elif parsed_json is not None:
            response_json_path = self.log_dir / f"{prefix}.response.json"
            response_json_path.write_text(
                json.dumps(parsed_json, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        result = RequestResult(
            label=label,
            api=api,
            stream=stream,
            url=url,
            status_code=status_code,
            ok=error is None and status_code is not None and 200 <= status_code < 300,
            total_ms=total_ms,
            ttft_ms=ttft_ms,
            text=text,
            error=error,
            request_json_path=str(request_json_path),
            request_http_path=str(request_http_path),
            response_json_path=str(response_json_path) if response_json_path else None,
            response_http_path=str(response_http_path),
        )
        apply_usage_and_timing(result, metrics_source)

        metrics_path = self.log_dir / f"{prefix}.metrics.json"
        metrics_path.write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return result


def normalize_base_url(raw: str) -> str:
    base = raw.rstrip("/")
    parsed = urlparse(base)
    if not parsed.scheme:
        base = "http://" + base
        parsed = urlparse(base)
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        base += "/v1"
    return base.rstrip("/")


def discover_model(base_url: str, api_key: str, timeout: float) -> str | None:
    try:
        response = requests.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=(timeout, timeout),
        )
        if response.status_code < 200 or response.status_code >= 300:
            return None
        data = response.json()
        models = data.get("data") if isinstance(data, dict) else None
        if isinstance(models, list):
            for model in models:
                if isinstance(model, dict) and isinstance(model.get("id"), str):
                    return model["id"]
        if isinstance(data, dict) and isinstance(data.get("models"), list):
            for model in data["models"]:
                if isinstance(model, dict) and isinstance(model.get("name"), str):
                    return model["name"]
    except Exception:
        return None
    return None


def make_large_context(word_count: int, seed: int) -> str:
    rng = random.Random(seed)
    lines: list[str] = [
        "BEGIN LARGE UNRELATED REFERENCE CONTEXT",
        "This deterministic block exists only to fill the prompt cache. It is unrelated to ping.",
    ]
    line_words: list[str] = []
    for i in range(word_count):
        if i % 97 == 0:
            line_words.append("section")
        line_words.append(rng.choice(COMMON_WORDS))
        if len(line_words) >= 18:
            lines.append(" ".join(line_words) + ".")
            line_words = []
    if line_words:
        lines.append(" ".join(line_words) + ".")
    lines.append("END LARGE UNRELATED REFERENCE CONTEXT")
    return "\n".join(lines)


def make_system_prompt(*, context: str, session_id: str, api: str, stream: bool) -> str:
    stream_name = "enabled" if stream else "disabled"
    return "\n".join(
        [
            "You are a cache benchmark assistant.",
            "The reference context below is intentionally unrelated to the task.",
            "Do not call tools. If asked for ACK, answer exactly ACK. If asked ping, answer exactly pong.",
            f"Benchmark session: {session_id}",
            f"API under test: {api}; streaming: {stream_name}",
            context,
        ]
    )


def tool_description(index: int) -> str:
    words = [COMMON_WORDS[(index + i) % len(COMMON_WORDS)] for i in range(120)]
    return (
        f"Synthetic benchmark tool {index}. This tool is irrelevant to the ping task. "
        + " ".join(words)
        + ". Use only when explicitly requested by name."
    )


def tool_schema(index: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "description": f"Lookup query for synthetic benchmark records in tool {index}.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Maximum number of records to return.",
            },
            "mode": {
                "type": "string",
                "enum": ["summary", "detail"],
                "description": "Response detail level.",
            },
        },
    }


def make_tools(api: str, count: int) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for i in range(1, count + 1):
        name = f"benchmark_lookup_{i:02d}"
        description = tool_description(i)
        schema = tool_schema(i)
        if api == "chat":
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": schema,
                    },
                }
            )
        elif api == "responses":
            tools.append(
                {
                    "type": "function",
                    "name": name,
                    "description": description,
                    "parameters": schema,
                }
            )
        elif api == "anthropic":
            tools.append(
                {
                    "name": name,
                    "description": description,
                    "input_schema": schema,
                }
            )
    return tools


def endpoint_for(api: str, base_url: str) -> str:
    if api == "chat":
        return f"{base_url}/chat/completions"
    if api == "responses":
        return f"{base_url}/responses"
    if api == "anthropic":
        return f"{base_url}/messages"
    raise ValueError(f"unknown api: {api}")


def headers_for(api: str, api_key: str) -> dict[str, str]:
    if api == "anthropic":
        return {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
    return {"Authorization": f"Bearer {api_key}"}


def build_body(
    *,
    api: str,
    model: str,
    system_prompt: str,
    conversation: list[tuple[str, str]],
    tools: list[dict[str, Any]],
    stream: bool,
    temperature: float,
) -> dict[str, Any]:
    if api == "chat":
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}]
            + [{"role": role, "content": content} for role, content in conversation],
            "temperature": temperature,
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        if tools:
            body["tools"] = tools
        return body

    if api == "responses":
        input_items: list[dict[str, Any]] = []
        for role, content in conversation:
            item_type = "output_text" if role == "assistant" else "input_text"
            input_items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": [{"type": item_type, "text": content}],
                }
            )
        body = {
            "model": model,
            "instructions": system_prompt,
            "input": input_items,
            "temperature": temperature,
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        if tools:
            body["tools"] = tools
        return body

    if api == "anthropic":
        body = {
            "model": model,
            "system": system_prompt,
            "messages": [{"role": role, "content": content} for role, content in conversation],
            "temperature": temperature,
            "max_tokens": 4096,
            "stream": stream,
        }
        if tools:
            body["tools"] = tools
        return body

    raise ValueError(f"unknown api: {api}")


def build_warmup_body(
    *,
    api: str,
    model: str,
    stream: bool,
    temperature: float,
) -> dict[str, Any]:
    return build_body(
        api=api,
        model=model,
        system_prompt="You are a warmup assistant. Answer with OK.",
        conversation=[("user", "Reply exactly OK.")],
        tools=[],
        stream=stream,
        temperature=temperature,
    )


def render_http_request(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
    log_secrets: bool,
) -> str:
    redacted = redact_headers(headers, log_secrets)
    lines = [f"{method} {url} HTTP/1.1"]
    for key, value in redacted.items():
        lines.append(f"{key}: {value}")
    lines.append(f"Content-Length: {len(body)}")
    lines.append("")
    lines.append(body.decode("utf-8", errors="replace"))
    return "\n".join(lines)


def render_http_response(
    status_code: int | None, headers: dict[str, str], body: bytes
) -> bytes:
    status = status_code if status_code is not None else "NO_RESPONSE"
    lines = [f"HTTP/1.1 {status}"]
    for key, value in headers.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    return ("\n".join(lines) + "\n").encode("utf-8") + pretty_json_bytes(body)


def pretty_json_bytes(body: bytes) -> bytes:
    if not body.strip():
        return body
    try:
        parsed = json.loads(body.decode("utf-8"))
    except Exception:
        return body
    return json.dumps(parsed, indent=2, ensure_ascii=False).encode("utf-8")


def redact_headers(headers: dict[str, str], log_secrets: bool) -> dict[str, str]:
    if log_secrets:
        return dict(headers)
    redacted = dict(headers)
    for key in list(redacted):
        if key.lower() in {"authorization", "x-api-key", "api-key"}:
            redacted[key] = "<redacted>"
    return redacted


def read_streaming_response(
    response: requests.Response,
    api: str,
    start: float,
) -> tuple[list[bytes], list[Any], str, float | None]:
    chunks: list[bytes] = []
    parsed_events: list[Any] = []
    text_parts: list[str] = []
    ttft_ms: float | None = None
    buffer = b""
    event_name: str | None = None

    for chunk in response.iter_content(chunk_size=1024):
        if not chunk:
            continue
        chunks.append(chunk)
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.rstrip(b"\r")
            if not line:
                event_name = None
                continue
            decoded = line.decode("utf-8", errors="replace")
            if decoded.startswith("event:"):
                event_name = decoded[len("event:") :].strip()
                continue
            if not decoded.startswith("data:"):
                continue
            data = decoded[len("data:") :].strip()
            if not data or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except Exception:
                parsed_events.append({"event": event_name, "raw": data})
                continue
            event_record = {"event": event_name, "data": obj}
            parsed_events.append(event_record)
            delta = extract_stream_delta(api, obj, event_name)
            if delta:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000.0
                text_parts.append(delta)

    if buffer.strip():
        # Some servers return a single JSON object even when stream=True on errors.
        try:
            obj = json.loads(buffer.decode("utf-8"))
            parsed_events.append({"event": None, "data": obj})
        except Exception:
            pass

    return chunks, parsed_events, "".join(text_parts), ttft_ms


def extract_stream_delta(api: str, obj: Any, event_name: str | None) -> str:
    if not isinstance(obj, dict):
        return ""

    if api == "chat":
        choices = obj.get("choices")
        if isinstance(choices, list) and choices:
            delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    return content
        return ""

    if api == "responses":
        delta = obj.get("delta")
        event_type = obj.get("type")
        if (
            event_name == "response.output_text.delta"
            or event_type == "response.output_text.delta"
        ) and isinstance(delta, str):
            return delta
        return ""

    if api == "anthropic":
        if obj.get("type") == "content_block_delta":
            delta_obj = obj.get("delta")
            if (
                isinstance(delta_obj, dict)
                and delta_obj.get("type") == "text_delta"
                and isinstance(delta_obj.get("text"), str)
            ):
                return delta_obj["text"]
        return ""

    return ""


def extract_text(api: str, obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""

    if api == "chat":
        choices = obj.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content:
                    return content
        return ""

    if api == "responses":
        if isinstance(obj.get("output_text"), str):
            return obj["output_text"]
        parts: list[str] = []
        output = obj.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for c in content:
                        if (
                            isinstance(c, dict)
                            and c.get("type") == "output_text"
                            and isinstance(c.get("text"), str)
                        ):
                            parts.append(c["text"])
        return "".join(parts)

    if api == "anthropic":
        parts = []
        content = obj.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        return "".join(parts)

    return ""


def coalesce_stream_metrics(api: str, events: list[Any]) -> Any:
    # Prefer final response-like objects; otherwise merge any usage/timing
    # fragments seen in the stream. Anthropic, for example, sends input usage in
    # message_start.message.usage and output usage in message_delta.usage.
    usage_obj: dict[str, Any] = {}
    timings_obj: dict[str, Any] = {}
    final_response: dict[str, Any] | None = None
    for event in events:
        data = event.get("data") if isinstance(event, dict) else None
        if not isinstance(data, dict):
            continue
        if api == "responses" and isinstance(data.get("response"), dict):
            response_obj = data["response"]
            if response_obj.get("status") == "completed" or isinstance(response_obj.get("usage"), dict):
                final_response = response_obj
            elif final_response is None:
                final_response = response_obj
        nested_message = data.get("message")
        if isinstance(nested_message, dict) and isinstance(nested_message.get("usage"), dict):
            usage_obj.update(nested_message["usage"])
        if isinstance(data.get("usage"), dict):
            usage_obj.update(data["usage"])
        if isinstance(data.get("timings"), dict):
            timings_obj.update(data["timings"])
    if final_response is not None and isinstance(final_response.get("usage"), dict):
        return final_response
    merged: dict[str, Any] = {}
    if final_response is not None:
        merged.update(final_response)
    if usage_obj:
        merged["usage"] = usage_obj
    if timings_obj:
        merged["timings"] = timings_obj
    return merged


def apply_usage_and_timing(result: RequestResult, obj: Any) -> None:
    if not isinstance(obj, dict):
        return
    usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else {}

    result.input_tokens = first_int(
        usage,
        "prompt_tokens",
        "input_tokens",
        "input_token_count",
    )
    result.output_tokens = first_int(
        usage,
        "completion_tokens",
        "output_tokens",
        "output_token_count",
    )
    result.total_tokens = first_int(usage, "total_tokens")

    prompt_details = usage.get("prompt_tokens_details")
    input_details = usage.get("input_tokens_details")
    cached = None
    if isinstance(prompt_details, dict):
        cached = first_int(prompt_details, "cached_tokens")
    if cached is None and isinstance(input_details, dict):
        cached = first_int(input_details, "cached_tokens")
    if cached is None:
        cached = first_int(usage, "cache_read_input_tokens", "cached_tokens")
    result.cached_tokens = cached
    result.cache_write_tokens = first_int(
        usage,
        "cache_creation_input_tokens",
        "cache_write_input_tokens",
        "cache_written_tokens",
    )

    timings = obj.get("timings") if isinstance(obj.get("timings"), dict) else {}
    prompt_ms = first_float(timings, "prompt_ms", "prompt_eval_ms")
    if prompt_ms is None:
        ns = first_float(obj, "prompt_eval_duration")
        if ns is not None:
            prompt_ms = ns / 1_000_000.0
    result.prompt_ms = prompt_ms
    result.prompt_n = first_int(timings, "prompt_n", "prompt_eval_count") or first_int(
        obj, "prompt_eval_count"
    )


def first_int(mapping: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return None


def first_float(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return float(value)
    return None


def run_session(
    *,
    client: RawClient,
    api: str,
    base_url: str,
    api_key: str,
    model: str,
    context: str,
    stream: bool,
    turns: int,
    tool_count: int,
    temperature: float,
) -> list[RequestResult]:
    endpoint = endpoint_for(api, base_url)
    headers = headers_for(api, api_key)
    session_id = str(uuid.uuid4())
    system_prompt = make_system_prompt(
        context=context,
        session_id=session_id,
        api=api,
        stream=stream,
    )
    tools = make_tools(api, tool_count)
    conversation: list[tuple[str, str]] = []
    results: list[RequestResult] = []
    mode = "stream" if stream else "nonstream"

    for turn in range(1, turns + 1):
        user_text = "Reply exactly ACK." if turn == 1 else "ping. Reply exactly pong."
        expected = "ACK" if turn == 1 else "pong"
        conversation.append(("user", user_text))
        body = build_body(
            api=api,
            model=model,
            system_prompt=system_prompt,
            conversation=conversation,
            tools=tools,
            stream=stream,
            temperature=temperature,
        )
        label = f"{mode}-turn{turn}"
        result = client.post_json(
            label=label,
            api=api,
            url=endpoint,
            headers=headers,
            body=body,
            stream=stream,
        )
        results.append(result)
        assistant_text = result.text if result.text else expected
        conversation.append(("assistant", assistant_text))
        if not result.ok:
            break

    return results


def run_warmup(
    *,
    client: RawClient,
    api: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
) -> RequestResult:
    return client.post_json(
        label="warmup",
        api=api,
        url=endpoint_for(api, base_url),
        headers=headers_for(api, api_key),
        body=build_warmup_body(
            api=api,
            model=model,
            stream=False,
            temperature=temperature,
        ),
        stream=False,
    )


def metric_for_comparison(result: RequestResult) -> tuple[str, float | None]:
    if result.prompt_ms is not None:
        return "prompt_ms", result.prompt_ms
    if result.stream and result.ttft_ms is not None:
        return "ttft_ms", result.ttft_ms
    return "total_ms", result.total_ms


def assess_mode(results: list[RequestResult]) -> str:
    turns = [r for r in results if "turn" in r.label and r.ok]
    if len(turns) < 2:
        return "not enough successful turns"
    first = turns[0]
    second = turns[1]
    metric_name, first_value = metric_for_comparison(first)
    _, second_value = metric_for_comparison(second)
    if first_value is None or second_value is None or first_value <= 0:
        return "not enough timing data"
    ratio = second_value / first_value
    if second.cached_tokens and second.cached_tokens > 0:
        return f"likely caching: turn2 reports {second.cached_tokens} cached tokens"
    if ratio < 0.35:
        return f"likely caching: turn2 {metric_name} is {ratio:.2f}x turn1"
    if ratio < 0.70:
        return f"possible caching: turn2 {metric_name} is {ratio:.2f}x turn1"
    return f"no clear cache speedup: turn2 {metric_name} is {ratio:.2f}x turn1"


def print_report(results: list[RequestResult], log_dir: Path) -> None:
    print("\nCache tester report")
    print(f"Logs: {log_dir}")
    print()
    headers = [
        "label",
        "status",
        "stream",
        "input",
        "cached",
        "prompt_ms",
        "ttft_ms",
        "total_ms",
        "text",
    ]
    rows = []
    for r in results:
        rows.append(
            [
                r.label,
                str(r.status_code) if r.status_code is not None else "ERR",
                "yes" if r.stream else "no",
                fmt_int(r.input_tokens),
                fmt_int(r.cached_tokens),
                fmt_ms(r.prompt_ms),
                fmt_ms(r.ttft_ms),
                fmt_ms(r.total_ms),
                compact_text(r.text or r.error or ""),
            ]
        )
    print_table(headers, rows)
    print()
    for stream in (False, True):
        mode_results = [r for r in results if r.label.startswith("stream" if stream else "nonstream")]
        if mode_results:
            print(f"{('streaming' if stream else 'non-streaming')}: {assess_mode(mode_results)}")


def write_summary(results: list[RequestResult], log_dir: Path, config: dict[str, Any]) -> None:
    summary = {
        "config": config,
        "results": [asdict(r) for r in results],
        "assessments": {
            "nonstream": assess_mode([r for r in results if r.label.startswith("nonstream")]),
            "stream": assess_mode([r for r in results if r.label.startswith("stream")]),
        },
    }
    (log_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = ["# Cache tester report", "", f"Logs: `{log_dir}`", "", "## Results", ""]
    headers = ["label", "status", "stream", "input", "cached", "prompt_ms", "ttft_ms", "total_ms", "text"]
    rows = []
    for r in results:
        rows.append(
            [
                r.label,
                str(r.status_code) if r.status_code is not None else "ERR",
                "yes" if r.stream else "no",
                fmt_int(r.input_tokens),
                fmt_int(r.cached_tokens),
                fmt_ms(r.prompt_ms),
                fmt_ms(r.ttft_ms),
                fmt_ms(r.total_ms),
                compact_text(r.text or r.error or ""),
            ]
        )
    lines.extend(markdown_table(headers, rows))
    lines.extend(
        [
            "",
            "## Assessment",
            "",
            f"- Non-streaming: {summary['assessments']['nonstream']}",
            f"- Streaming: {summary['assessments']['stream']}",
            "",
            "This is an append-only session test. Turn 2 resends the same large prefix plus a small new user message.",
        ]
    )
    (log_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.0f}"


def fmt_int(value: int | None) -> str:
    if value is None:
        return "-"
    return str(value)


def compact_text(text: str) -> str:
    text = " ".join(text.split())
    if len(text) > 40:
        return text[:37] + "..."
    return text


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join("{:<" + str(width) + "}" for width in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * width for width in widths]))
    for row in rows:
        print(fmt.format(*row))


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |")
    return lines


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a large append-only agent-style session to a local LLM endpoint and report cache behavior.",
    )
    parser.add_argument(
        "base_url",
        help="Server base URL. Example: http://localhost:1234 (the script appends /v1 when needed).",
    )
    parser.add_argument(
        "--api",
        choices=["chat", "responses", "anthropic"],
        default="chat",
        help="API dialect to test (default: chat).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id. Defaults to the first model from /v1/models, then local-model.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key. Defaults to OPENAI_API_KEY, ANTHROPIC_API_KEY, then sk-local.",
    )
    parser.add_argument(
        "--context-tokens",
        "--context-words",
        dest="context_words",
        type=int,
        default=DEFAULT_CONTEXT_WORDS,
        help=f"Approximate large-context size using common-word tokens (default: {DEFAULT_CONTEXT_WORDS}).",
    )
    parser.add_argument(
        "--tool-count",
        type=int,
        default=DEFAULT_TOOL_COUNT,
        help=f"Number of synthetic tools to include (default: {DEFAULT_TOOL_COUNT}).",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=2,
        help="Append-only turns per streaming mode; minimum 2 (default: 2).",
    )
    parser.add_argument(
        "--streaming",
        choices=["both", "on", "off"],
        default="both",
        help="Whether to test streaming, non-streaming, or both (default: both).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Deterministic context seed (default: 1).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(".cache-tester-logs"),
        help="Parent directory for logs (default: .cache-tester-logs).",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the small warmup request.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=DEFAULT_CONNECT_TIMEOUT_SECONDS,
        help=f"HTTP connect timeout in seconds (default: {DEFAULT_CONNECT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=DEFAULT_READ_TIMEOUT_SECONDS,
        help="HTTP read timeout in seconds (default: none).",
    )
    parser.add_argument(
        "--log-secrets",
        action="store_true",
        help="Do not redact API keys in .request.http logs.",
    )
    return parser.parse_args(list(argv))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    base_url = normalize_base_url(args.base_url)
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or "sk-local"
    model = args.model or discover_model(base_url, api_key, args.connect_timeout) or "local-model"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = args.log_dir / timestamp
    client = RawClient(
        api_key=api_key,
        log_dir=log_dir,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        log_secrets=args.log_secrets,
    )

    turns = max(2, args.turns)
    config = {
        "base_url": base_url,
        "api": args.api,
        "model": model,
        "context_words": args.context_words,
        "tool_count": args.tool_count,
        "turns": turns,
        "streaming": args.streaming,
        "temperature": args.temperature,
        "seed": args.seed,
    }
    (log_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Base URL: {base_url}")
    print(f"API: {args.api}")
    print(f"Model: {model}")
    print(f"Large context: ~{args.context_words} words; tools: {args.tool_count}; turns: {turns}")
    print(f"Logs: {log_dir}")

    results: list[RequestResult] = []
    if not args.no_warmup:
        print("Warmup...")
        warmup = run_warmup(
            client=client,
            api=args.api,
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=args.temperature,
        )
        results.append(warmup)
        if not warmup.ok:
            print_report(results, log_dir)
            write_summary(results, log_dir, config)
            print("\nWarmup failed. Check the logged response for details.")
            return 1

    context = make_large_context(args.context_words, args.seed)
    modes: list[bool]
    if args.streaming == "both":
        modes = [False, True]
    elif args.streaming == "on":
        modes = [True]
    else:
        modes = [False]

    for stream in modes:
        mode_name = "streaming" if stream else "non-streaming"
        print(f"Running {mode_name} append-only session...")
        session_results = run_session(
            client=client,
            api=args.api,
            base_url=base_url,
            api_key=api_key,
            model=model,
            context=context,
            stream=stream,
            turns=turns,
            tool_count=args.tool_count,
            temperature=args.temperature,
        )
        results.extend(session_results)

    print_report(results, log_dir)
    write_summary(results, log_dir, config)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
