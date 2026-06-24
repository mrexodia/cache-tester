from __future__ import annotations

import argparse
import json
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
from urllib.parse import parse_qs, urlparse

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
DEFAULT_API_KEY = "sk-local"
API_TYPES = ("chat", "responses", "anthropic")
STREAM_MODES = (False, True)


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LLM Cache Tester</title>
  <style>
    :root {
      color-scheme: dark light;
      --bg: #0b1020;
      --panel: #121a2e;
      --panel2: #18223a;
      --text: #e9eefc;
      --muted: #9aa8c7;
      --border: #2b3858;
      --ok: #3ddc97;
      --bad: #ff6b6b;
      --warn: #ffd166;
      --blue: #7aa2ff;
      --shadow: rgba(0,0,0,0.25);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, #182749, var(--bg) 45%);
      color: var(--text);
    }
    header {
      padding: 28px 32px 16px;
      max-width: 1320px;
      margin: 0 auto;
    }
    h1 { margin: 0 0 6px; font-size: 28px; }
    h2 { margin: 0 0 14px; font-size: 18px; }
    p { color: var(--muted); margin: 0 0 12px; }
    main {
      max-width: 1320px;
      margin: 0 auto;
      padding: 0 32px 40px;
      display: grid;
      gap: 18px;
    }
    .panel {
      background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.01)), var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 8px 32px var(--shadow);
      padding: 18px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 12px;
      align-items: end;
    }
    .field { display: grid; gap: 6px; }
    .col-2 { grid-column: span 2; }
    .col-3 { grid-column: span 3; }
    .col-4 { grid-column: span 4; }
    .col-5 { grid-column: span 5; }
    .col-6 { grid-column: span 6; }
    .col-12 { grid-column: span 12; }
    label { color: var(--muted); font-size: 12px; font-weight: 650; letter-spacing: .02em; text-transform: uppercase; }
    input, select {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 10px;
      color: var(--text);
      background: #0d1426;
      padding: 10px 11px;
      outline: none;
    }
    input:focus, select:focus { border-color: var(--blue); box-shadow: 0 0 0 3px rgba(122,162,255,.14); }
    .buttons { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    button {
      border: 1px solid var(--border);
      background: var(--panel2);
      color: var(--text);
      padding: 10px 13px;
      border-radius: 10px;
      cursor: pointer;
      font-weight: 700;
    }
    button.primary { background: #3157d5; border-color: #4f72ea; }
    button.good { background: #157f55; border-color: #2fb77e; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .card { background: rgba(255,255,255,.035); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }
    .card .value { font-size: 24px; font-weight: 800; margin-top: 4px; }
    .muted { color: var(--muted); }
    .ok { color: var(--ok); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .status-line { color: var(--muted); min-height: 22px; }
    table { width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 12px; }
    th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
    th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .02em; background: rgba(255,255,255,.035); }
    tr:last-child td { border-bottom: 0; }
    .pill { display: inline-flex; align-items: center; border-radius: 999px; padding: 2px 8px; font-size: 12px; font-weight: 750; background: rgba(255,255,255,.08); }
    .pill.ok { background: rgba(61,220,151,.12); color: var(--ok); }
    .pill.bad { background: rgba(255,107,107,.12); color: var(--bad); }
    .pill.warn { background: rgba(255,209,102,.12); color: var(--warn); }
    .small { font-size: 12px; }
    .scroll { overflow-x: auto; }
    details { color: var(--muted); }
    summary { cursor: pointer; }
    tr.request-row { cursor: pointer; }
    tr.request-row:hover td { background: rgba(122,162,255,.08); }
    tr.log-row td { background: rgba(0,0,0,.18); }
    .log-panel { display: grid; gap: 12px; }
    .log-grid { display: grid; grid-template-columns: 1fr; gap: 12px; }
    .log-block { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; background: #060a14; }
    .log-title { padding: 8px 10px; background: rgba(255,255,255,.06); color: var(--muted); font-weight: 800; font-size: 12px; text-transform: uppercase; letter-spacing: .02em; }
    .log-block pre { margin: 0; padding: 10px; max-height: 360px; overflow: auto; white-space: pre-wrap; word-break: break-word; color: #dbe6ff; }
    @media (max-width: 900px) {
      header, main { padding-left: 16px; padding-right: 16px; }
      .grid { grid-template-columns: 1fr; }
      .col-2, .col-3, .col-4, .col-5, .col-6, .col-12 { grid-column: span 1; }
      .cards { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
<header>
  <h1>LLM Cache Tester</h1>
  <p>Smoke-test Completions, OpenAI Responses, and Anthropic Messages endpoints, then run append-only cache checks with streaming and non-streaming requests.</p>
</header>
<main>
  <section class="panel">
    <h2>1. Configure endpoint</h2>
    <div class="grid">
      <div class="field col-4">
        <label for="endpoint">Endpoint</label>
        <input id="endpoint" value="http://127.0.0.1:1234" autocomplete="off" />
      </div>
      <div class="field col-3">
        <label for="apiKey">API key</label>
        <input id="apiKey" value="sk-local" autocomplete="off" />
      </div>
      <div class="field col-3">
        <label for="model">Model</label>
        <input id="model" placeholder="auto-discover first model" autocomplete="off" />
      </div>
      <div class="field col-2">
        <label for="contextTokens">Context tokens</label>
        <input id="contextTokens" type="number" min="100" step="1000" value="56000" />
      </div>
      <div class="field col-2">
        <label for="toolCount">Tools</label>
        <input id="toolCount" type="number" min="0" max="50" value="8" />
      </div>
      <div class="field col-2">
        <label for="turns">Turns</label>
        <input id="turns" type="number" min="2" max="8" value="2" />
      </div>
      <div class="field col-2">
        <label for="readTimeout">Read timeout sec</label>
        <input id="readTimeout" type="number" min="1" step="1" placeholder="none" />
      </div>
      <div class="field col-4">
        <label>Actions</label>
        <div class="buttons">
          <button id="discoverBtn">Discover models</button>
          <button id="smokeBtn">Run smoke</button>
          <button id="fullBtn">Run full cache test</button>
          <button id="allBtn" class="primary">Run all</button>
        </div>
      </div>
      <div class="col-12 status-line" id="statusLine"></div>
    </div>
  </section>

  <section class="cards">
    <div class="card"><div class="muted small">Run id</div><div class="value mono" id="runIdCard">-</div></div>
    <div class="card"><div class="muted small">Smoke</div><div class="value" id="smokeCard">0 / 6</div></div>
    <div class="card"><div class="muted small">Full cache tests</div><div class="value" id="fullCard">0</div></div>
    <div class="card"><div class="muted small">Tests passed</div><div class="value small" id="logCard">0 / 0</div></div>
  </section>

  <section class="panel">
    <h2>2. Warmups</h2>
    <p>Warmups are visible but not counted. They use unique prompts so they warm the model/session without sharing a prefix with smoke or cache-test prompts.</p>
    <div class="scroll"><table id="warmupTable"></table></div>
  </section>

  <section class="panel">
    <h2>3. Smoke tests</h2>
    <p>Each API dialect is tested both without streaming and with streaming, using a tiny prompt and one unique synthetic tool definition. Click any completed request to expand raw request/response logs.</p>
    <div class="scroll"><table id="smokeTable"></table></div>
  </section>

  <section class="panel">
    <h2>4. Full append-only cache tests</h2>
    <p>Full tests run only for smoke-passing combinations. A visible unique warmup runs immediately before cache tests. Turn 2 resends the same large prefix plus one new user message, matching realistic append-only agent sessions. Click any completed request to expand raw logs.</p>
    <div class="scroll"><table id="fullTable"></table></div>
  </section>
</main>
<script>
const API_LABELS = {
  chat: 'Completions',
  responses: 'Responses',
  anthropic: 'Anthropic Messages'
};
const COMBOS = [
  {api:'chat', stream:false}, {api:'chat', stream:true},
  {api:'responses', stream:false}, {api:'responses', stream:true},
  {api:'anthropic', stream:false}, {api:'anthropic', stream:true},
];
let runId = newRunId();
let warmupResults = [];
let smokeResults = new Map();
let fullResults = new Map();
let expandedLogs = new Set();
let logCache = new Map();
let busy = false;

const el = id => document.getElementById(id);
function comboKey(c) { return `${c.api}:${c.stream ? 'stream' : 'nonstream'}`; }
function streamLabel(v) { return v ? 'stream' : 'non-stream'; }
function newRunId() {
  return 'web-' + new Date().toISOString().replace(/[:.]/g, '-') + '-' + Math.random().toString(16).slice(2, 8);
}
function setStatus(text, cls='') {
  el('statusLine').className = 'col-12 status-line ' + cls;
  el('statusLine').textContent = text;
}
function setBusy(value) {
  busy = value;
  for (const id of ['discoverBtn','smokeBtn','fullBtn','allBtn']) el(id).disabled = value;
}
function cfg() {
  const readTimeoutRaw = el('readTimeout').value.trim();
  return {
    run_id: runId,
    base_url: el('endpoint').value.trim(),
    api_key: el('apiKey').value,
    model: el('model').value.trim(),
    context_tokens: parseInt(el('contextTokens').value || '56000', 10),
    tool_count: parseInt(el('toolCount').value || '8', 10),
    turns: parseInt(el('turns').value || '2', 10),
    temperature: 0,
    read_timeout: readTimeoutRaw ? parseFloat(readTimeoutRaw) : null
  };
}
function saveCfg() {
  for (const id of ['endpoint','apiKey','model','contextTokens','toolCount','turns','readTimeout']) {
    localStorage.setItem('cacheTester.' + id, el(id).value);
  }
}
function loadCfg() {
  for (const id of ['endpoint','apiKey','model','contextTokens','toolCount','turns','readTimeout']) {
    const v = localStorage.getItem('cacheTester.' + id);
    if (v !== null) el(id).value = v;
  }
}
async function postJson(path, body) {
  const r = await fetch(path, {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const text = await r.text();
  let data;
  try { data = JSON.parse(text); } catch { data = {ok:false, error:text}; }
  if (!r.ok) data.ok = false;
  return data;
}
async function runVisibleWarmup(phase) {
  const label = phase === 'cache' ? 'cache-test warmup' : 'smoke warmup';
  const index = warmupResults.length;
  warmupResults.push({phase, label, running: true});
  renderAll();
  setStatus(`Running ${label}...`);
  const data = await postJson('/api/warmup', {...cfg(), warmup_phase: phase});
  warmupResults[index] = {...data, phase, label};
  renderAll();
  if (!data.ok) {
    setStatus(`${label} did not succeed; continuing with visible tests. ` + (data.error || ''), 'warn');
  }
  return data;
}
async function discoverModels(silent=false) {
  saveCfg();
  const c = cfg();
  if (!silent) setStatus('Discovering models...');
  const q = new URLSearchParams({base_url: c.base_url, api_key: c.api_key});
  const r = await fetch('/api/models?' + q.toString());
  const data = await r.json();
  if (data.ok && data.models && data.models.length) {
    el('model').value = data.models[0];
    setStatus(`Discovered ${data.models.length} model(s). Selected ${el('model').value}.`, 'ok');
    saveCfg();
    return true;
  }
  setStatus('Model discovery failed: ' + (data.error || 'no models'), 'warn');
  return false;
}
async function ensureModel() {
  if (el('model').value.trim()) return true;
  return await discoverModels(true);
}
function resetRun() {
  runId = newRunId();
  warmupResults = [];
  smokeResults = new Map();
  fullResults = new Map();
  expandedLogs = new Set();
  logCache = new Map();
  renderAll();
}
function pill(text, kind) { return `<span class="pill ${kind || ''}">${escapeHtml(text)}</span>`; }
function escapeHtml(s) { return String(s ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }
function ms(v) { return v == null ? '-' : Math.round(v); }
function tok(v) { return v == null ? '-' : v; }
function compact(s, n=80) { s = String(s || '').replace(/\s+/g, ' ').trim(); return s.length > n ? s.slice(0, n - 1) + '…' : s; }
function resultFiles(result) {
  if (!result) return {};
  return {
    'request.http': result.request_http_path,
    'response.http': result.response_http_path,
  };
}
async function toggleLog(id, result) {
  if (!result) return;
  if (expandedLogs.has(id)) {
    expandedLogs.delete(id);
    renderAll();
    return;
  }
  expandedLogs.add(id);
  if (!logCache.has(id)) {
    logCache.set(id, {loading: true});
    renderAll();
    const data = await postJson('/api/log-files', {files: resultFiles(result)});
    logCache.set(id, data);
  }
  renderAll();
}
function toggleWarmupLog(index) {
  const r = warmupResults[index];
  if (r && !r.running && r.result) toggleLog('warmup:' + index, r.result);
}
function toggleSmokeLog(k) {
  const r = smokeResults.get(k);
  if (r && r !== 'running' && r.result) toggleLog('smoke:' + k, r.result);
}
function toggleFullLog(k, index) {
  const data = fullResults.get(k);
  const result = data && data !== 'running' && Array.isArray(data.results) ? data.results[index] : null;
  if (result) toggleLog(`full:${k}:${index}`, result);
}
function renderLogRow(colspan, id) {
  if (!expandedLogs.has(id)) return '';
  const data = logCache.get(id);
  if (!data || data.loading) {
    return `<tr class="log-row"><td colspan="${colspan}"><div class="muted">Loading raw logs...</div></td></tr>`;
  }
  if (!data.ok) {
    return `<tr class="log-row"><td colspan="${colspan}"><div class="bad">${escapeHtml(data.error || 'Failed to load logs')}</div></td></tr>`;
  }
  const order = ['request.http', 'response.http'];
  const blocks = order.filter(name => data.files && data.files[name] != null).map(name => {
    return `<div class="log-block"><div class="log-title">${escapeHtml(name)}</div><pre>${escapeHtml(data.files[name])}</pre></div>`;
  }).join('');
  return `<tr class="log-row"><td colspan="${colspan}"><div class="log-panel"><div class="muted small">Raw request/response logs. Authorization headers are redacted.</div><div class="log-grid">${blocks || '<div class="muted">No log files available.</div>'}</div></div></td></tr>`;
}
function renderWarmupTable() {
  if (!warmupResults.length) {
    el('warmupTable').innerHTML = `<thead><tr><th>Phase</th><th>API used</th><th>Status</th><th>Total ms</th><th>Input</th><th>Output</th><th>Total tokens</th><th>Cached</th><th>Output / error</th></tr></thead><tbody><tr><td colspan="9" class="muted">No warmups yet. Warmups run automatically before smoke tests and before full cache tests.</td></tr></tbody>`;
    return;
  }
  const rows = warmupResults.map((r, index) => {
    const id = 'warmup:' + index;
    let api = '-', status = pill('running', 'warn'), latency = '-', input = '-', output = '-', total = '-', cached = '-', text = '-', rowClass = '';
    if (!r.running) {
      const rr = r.result || {};
      api = r.api ? escapeHtml(API_LABELS[r.api] || r.api) : '-';
      status = r.ok ? pill('ok ' + (rr.status_code || ''), 'ok') : pill('failed ' + (rr.status_code || ''), 'bad');
      latency = ms(rr.total_ms);
      input = tok(rr.input_tokens);
      output = tok(rr.output_tokens);
      total = tok(rr.total_tokens);
      cached = tok(rr.cached_tokens);
      text = escapeHtml(compact(rr.text || rr.error || r.error || '(empty)'));
      rowClass = r.result ? 'request-row' : '';
    }
    const main = `<tr class="${rowClass}" onclick="toggleWarmupLog(${index})"><td>${escapeHtml(r.label || r.phase || 'warmup')}</td><td>${api}</td><td>${status}</td><td>${latency}</td><td>${input}</td><td>${output}</td><td>${total}</td><td>${cached}</td><td>${text}</td></tr>`;
    return main + renderLogRow(9, id);
  }).join('');
  el('warmupTable').innerHTML = `<thead><tr><th>Phase</th><th>API used</th><th>Status</th><th>Total ms</th><th>Input</th><th>Output</th><th>Total tokens</th><th>Cached</th><th>Output / error</th></tr></thead><tbody>${rows}</tbody>`;
}
function renderSmokeTable() {
  const rows = COMBOS.map(c => {
    const k = comboKey(c), r = smokeResults.get(k);
    const id = 'smoke:' + k;
    let status = pill('pending', '');
    let latency = '-', ttft = '-', input = '-', output = '-', total = '-', cached = '-', text = '-', rowClass = '';
    if (r === 'running') status = pill('running', 'warn');
    else if (r) {
      const rr = r.result || {};
      status = r.ok ? pill('ok ' + (rr.status_code || ''), 'ok') : pill('failed ' + (rr.status_code || ''), 'bad');
      latency = ms(rr.total_ms);
      ttft = ms(rr.ttft_ms);
      input = tok(rr.input_tokens);
      output = tok(rr.output_tokens);
      total = tok(rr.total_tokens);
      cached = tok(rr.cached_tokens);
      text = escapeHtml(compact(rr.text || rr.error || r.error || '(empty)'));
      rowClass = 'request-row';
    }
    const main = `<tr class="${rowClass}" onclick="toggleSmokeLog('${k}')"><td>${escapeHtml(API_LABELS[c.api])}</td><td>${streamLabel(c.stream)}</td><td>${status}</td><td>${latency}</td><td>${ttft}</td><td>${input}</td><td>${output}</td><td>${total}</td><td>${cached}</td><td>${text}</td></tr>`;
    return main + renderLogRow(10, id);
  }).join('');
  el('smokeTable').innerHTML = `<thead><tr><th>API</th><th>Mode</th><th>Status</th><th>Total ms</th><th>TTFT ms</th><th>Input</th><th>Output</th><th>Total tokens</th><th>Cached</th><th>Output / reasoning / error</th></tr></thead><tbody>${rows}</tbody>`;
}
function resultTurn(data, turn) {
  if (!data || !Array.isArray(data.results)) return null;
  return data.results.find(r => String(r.label || '').endsWith('turn' + turn));
}
function renderFullTable() {
  const combos = COMBOS.filter(c => fullResults.has(comboKey(c)) || smokeResults.has(comboKey(c)));
  const use = combos.length ? combos : COMBOS;
  const emptyCells = '<td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>';
  const rows = use.map(c => {
    const k = comboKey(c), data = fullResults.get(k);
    if (data === 'running') {
      return `<tr><td>${escapeHtml(API_LABELS[c.api])}</td><td>${streamLabel(c.stream)}</td><td>-</td><td>${pill('running', 'warn')}</td>${emptyCells}</tr>`;
    }
    if (!data) {
      return `<tr><td>${escapeHtml(API_LABELS[c.api])}</td><td>${streamLabel(c.stream)}</td><td>-</td><td>${pill('pending', '')}</td>${emptyCells}</tr>`;
    }
    if (!Array.isArray(data.results) || !data.results.length) {
      return `<tr><td>${escapeHtml(API_LABELS[c.api])}</td><td>${streamLabel(c.stream)}</td><td>-</td><td>${pill('failed', 'bad')}</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>${escapeHtml(data.error || 'no results')}</td></tr>`;
    }
    return data.results.map((r, index) => {
      const id = `full:${k}:${index}`;
      const status = r.ok ? pill('ok ' + (r.status_code || ''), 'ok') : pill('failed ' + (r.status_code || ''), 'bad');
      const note = index === data.results.length - 1 ? (data.assessment || '') : compact(r.text || r.error || '');
      const main = `<tr class="request-row" onclick="toggleFullLog('${k}', ${index})"><td>${escapeHtml(API_LABELS[c.api])}</td><td>${streamLabel(c.stream)}</td><td>${escapeHtml(r.label || ('request ' + (index + 1)))}</td><td>${status}</td><td>${ms(r.total_ms)}</td><td>${ms(r.ttft_ms)}</td><td>${tok(r.input_tokens)}</td><td>${tok(r.output_tokens)}</td><td>${tok(r.total_tokens)}</td><td>${tok(r.cached_tokens)}</td><td>${escapeHtml(note || '-')}</td></tr>`;
      return main + renderLogRow(11, id);
    }).join('');
  }).join('');
  el('fullTable').innerHTML = `<thead><tr><th>API</th><th>Mode</th><th>Request</th><th>Status</th><th>Total ms</th><th>TTFT ms</th><th>Input</th><th>Output</th><th>Total tokens</th><th>Cached</th><th>Assessment / output</th></tr></thead><tbody>${rows}</tbody>`;
}
function renderCards() {
  el('runIdCard').textContent = runId.replace(/^web-/, '').slice(0, 19);
  const smokePassed = [...smokeResults.values()].filter(v => v && v !== 'running' && v.ok).length;
  const smokeCompleted = [...smokeResults.values()].filter(v => v && v !== 'running').length;
  el('smokeCard').textContent = `${smokePassed} / ${COMBOS.length}`;
  const fullCompleted = [...fullResults.values()].filter(v => v && v !== 'running').length;
  const fullPassed = [...fullResults.values()].filter(v => v && v !== 'running' && v.ok).length;
  el('fullCard').textContent = String(fullCompleted);
  el('logCard').textContent = `${smokePassed + fullPassed} / ${smokeCompleted + fullCompleted}`;
}
function renderAll() { renderWarmupTable(); renderSmokeTable(); renderFullTable(); renderCards(); }
async function runSmoke(reset=true) {
  if (busy) return;
  saveCfg();
  if (reset) resetRun();
  setBusy(true);
  try {
    await ensureModel();
    await runVisibleWarmup('smoke');
    for (const combo of COMBOS) {
      const k = comboKey(combo);
      smokeResults.set(k, 'running'); renderAll();
      setStatus(`Smoke: ${API_LABELS[combo.api]} ${streamLabel(combo.stream)}...`);
      const data = await postJson('/api/smoke-one', {...cfg(), ...combo});
      smokeResults.set(k, data); renderAll();
    }
    const pass = [...smokeResults.values()].filter(v => v && v !== 'running' && v.ok).length;
    setStatus(`Smoke complete: ${pass}/${COMBOS.length} combinations passed.`, pass ? 'ok' : 'bad');
  } finally { setBusy(false); }
}
function passingCombosOrAll() {
  const passing = COMBOS.filter(c => {
    const r = smokeResults.get(comboKey(c));
    return r && r !== 'running' && r.ok;
  });
  return passing.length ? passing : COMBOS;
}
async function runFull() {
  if (busy) return;
  saveCfg();
  setBusy(true);
  try {
    await ensureModel();
    await runVisibleWarmup('cache');
    const combos = passingCombosOrAll();
    for (const combo of combos) {
      const k = comboKey(combo);
      fullResults.set(k, 'running'); renderAll();
      setStatus(`Full cache test: ${API_LABELS[combo.api]} ${streamLabel(combo.stream)}...`);
      const data = await postJson('/api/full-one', {...cfg(), ...combo});
      fullResults.set(k, data); renderAll();
    }
    setStatus(`Full cache tests complete (${combos.length} combination${combos.length === 1 ? '' : 's'}).`, 'ok');
  } finally { setBusy(false); }
}
async function runAll() {
  if (busy) return;
  resetRun();
  setBusy(true);
  try {
    saveCfg();
    await ensureModel();
    await runVisibleWarmup('smoke');
    for (const combo of COMBOS) {
      const k = comboKey(combo);
      smokeResults.set(k, 'running'); renderAll();
      setStatus(`Smoke: ${API_LABELS[combo.api]} ${streamLabel(combo.stream)}...`);
      const data = await postJson('/api/smoke-one', {...cfg(), ...combo});
      smokeResults.set(k, data); renderAll();
    }
    const combos = passingCombosOrAll();
    setStatus(`Smoke complete. Running cache-test warmup before ${combos.length} full cache test combination(s)...`, 'ok');
    await runVisibleWarmup('cache');
    for (const combo of combos) {
      const k = comboKey(combo);
      fullResults.set(k, 'running'); renderAll();
      setStatus(`Full cache test: ${API_LABELS[combo.api]} ${streamLabel(combo.stream)}...`);
      const data = await postJson('/api/full-one', {...cfg(), ...combo});
      fullResults.set(k, data); renderAll();
    }
    setStatus('All tests complete.', 'ok');
  } finally { setBusy(false); }
}

el('discoverBtn').onclick = () => discoverModels(false);
el('smokeBtn').onclick = () => runSmoke(true);
el('fullBtn').onclick = () => runFull();
el('allBtn').onclick = () => runAll();
for (const id of ['endpoint','apiKey','model','contextTokens','toolCount','turns','readTimeout']) {
  el(id).addEventListener('change', saveCfg);
}
loadCfg();
renderAll();
</script>
</body>
</html>
"""


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


def log_dir_for(root: Path, cfg: dict[str, Any], phase: str) -> Path:
    mode = "stream" if cfg["stream"] else "nonstream"
    return root / cfg["run_id"] / phase / f"{cfg['api']}-{mode}"


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
        if parsed.path in {"/", "/index.html"}:
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
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
        self.send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)

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

    def send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

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
