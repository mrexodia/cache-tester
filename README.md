# cache-tester

<img width="1274" height="750" alt="image" src="https://github.com/user-attachments/assets/f388a99f-9f39-4276-bf62-45dac571f664" /> <img width="1272" height="905" alt="image" src="https://github.com/user-attachments/assets/206f01d8-3a6e-44b7-aabf-f158b3645c49" />

Simple web UI for checking whether an LLM endpoint is suitable for realistic append-only agent sessions with prompt caching.

Start the UI:

```bash
uv run cache-tester
```

Then open:

```text
http://127.0.0.1:8765
```

The UI defaults to:

- endpoint: `http://127.0.0.1:1234` (LM Studio)
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
