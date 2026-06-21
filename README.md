# cache-tester

Simple web UI for checking whether a local LLM endpoint is suitable for realistic append-only agent sessions with prompt caching.

Start the UI:

```bash
uv run cache-tester
```

Then open:

```text
http://127.0.0.1:8765
```

The UI defaults to:

- endpoint: `http://127.0.0.1:1234`
- API key: `sk-local`
- model: auto-discovered from `/v1/models`
- context size: ~56k common-word tokens
- tools: 8 synthetic tool definitions

Flow:

1. Configure endpoint/API key/model.
2. A visible unique warmup runs before smoke tests so the first visible smoke result is not penalized by model/session startup.
3. Run smoke tests for:
   - Completions: `/v1/chat/completions`
   - Responses: `/v1/responses`
   - Anthropic Messages: `/v1/messages`
   - each with streaming off and on
4. A second visible unique warmup runs immediately before full cache tests.
5. Run full cache tests for smoke-passing combinations.

Streaming OpenAI requests include:

```json
"stream_options": { "include_usage": true }
```

Warmup, smoke, and cache-test tables show timing plus usage information. Raw request/response logs can be expanded in the browser by clicking completed request rows. The files are also stored under `.cache-tester-logs/<run-id>/`.

Server options:

```bash
uv run cache-tester --host 127.0.0.1 --port 8765 --log-dir .cache-tester-logs
```
