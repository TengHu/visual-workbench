"""OpenRouter chat completions: any model OpenRouter lists, always streamed and rebuilt from the chunks.

The same behaviour as ../../custom-harness/src/models/openrouter.ts: the assistant message goes back unchanged on
later turns, its `reasoning_details` included, which Gemini and Claude need to continue a tool-calling conversation.

Messages are kept in a neutral shape and turned into the wire format at send time:
    {"role": "system", "content": str}
    {"role": "user", "content": [ {"type": "text", "text": str} | {"type": "image", "file": path} ... ]}
    {"role": "assistant", "raw": {...}}            exactly what the API returned, to send back
    {"role": "tool", "call_id": str, "content": str}
"""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
ATTEMPTS = 4
IDLE_SECONDS = 300.0  # a long think can send nothing for minutes
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


class ModelError(RuntimeError):
    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class Reply:
    text: str
    reasoning: str
    tool_calls: list[dict]  # {"id", "name", "args": dict | None, "error": str | None}
    assistant: dict  # the neutral assistant message to append to the conversation
    finish: str | None
    provider: str | None
    usage: dict = field(default_factory=dict)  # input, output, reasoning, cached, usd
    seconds: float = 0.0


def data_url(file: str) -> str:
    mime = MIME.get(Path(file).suffix.lower())
    if not mime:
        raise ValueError(f"{file}: not a PNG, JPEG or WebP")
    return f"data:{mime};base64," + base64.b64encode(Path(file).read_bytes()).decode()


def load_env(start: Path) -> dict:
    """KEY=value pairs of the nearest .env at or above `start`, without touching this process's environment."""
    for d in [start, *start.parents]:
        f = d / ".env"
        if f.exists():
            out = {}
            for line in f.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip().removeprefix("export ").strip()] = v.strip().strip('"').strip("'")
            return out
    return {}


class OpenRouter:
    def __init__(self, model: str, api_key: str, base: str | None = None, providers: list[str] | None = None, detail: str | None = None):
        import uuid
        self.session = uuid.uuid4().hex  # keeps one run on one provider, so its prompt cache is reused
        self.id = model
        self.key = api_key
        base = (base or "https://openrouter.ai/api/v1").rstrip("/")
        self.base = base[:-3] if base.endswith("/v1") else base
        self.providers = providers
        self.detail = detail

    # -- wire format -------------------------------------------------------------------------------------------

    def _wire(self, m: dict, with_images: bool):
        if m["role"] == "assistant":
            return m["raw"]
        if m["role"] == "tool":
            return {"role": "tool", "tool_call_id": m["call_id"], "content": m["content"]}
        if isinstance(m["content"], str):
            return {"role": m["role"], "content": m["content"]}
        parts = []
        for p in m["content"]:
            if p["type"] == "text":
                parts.append({"type": "text", "text": p["text"]})
            else:
                image = {"url": data_url(p["file"]) if with_images else f"file:{p['file']}"}
                if self.detail:
                    image["detail"] = self.detail
                parts.append({"type": "image_url", "image_url": image})
        return {"role": "user", "content": parts}

    def body(self, messages: list[dict], tools: list[dict], effort: str, max_tokens: int, with_images: bool = True) -> dict:
        b = {
            "model": self.id,
            "messages": [self._wire(m, with_images) for m in messages],
            "reasoning": {"effort": effort},
            "max_tokens": max_tokens,
            "stream": True,
            "session_id": self.session,
        }
        if self.id.startswith("anthropic/"):
            b["cache_control"] = {"type": "ephemeral"}  # Claude caches only when asked; Gemini does it by itself
        if tools:
            b["tools"] = [{"type": "function", "function": t} for t in tools]
        if self.providers:
            b["provider"] = {"order": self.providers, "allow_fallbacks": False}
        return b

    # -- sending -----------------------------------------------------------------------------------------------

    def send(self, messages, tools, effort, max_tokens, on_progress=None) -> Reply:
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._once(messages, tools, effort, max_tokens, on_progress)
            except ModelError as e:
                if attempt >= ATTEMPTS or not e.retryable:
                    raise
                wait = 2.0 * 2 ** (attempt - 1)
                print(f"    attempt {attempt} failed ({str(e)[:200]}); retrying in {wait:.0f}s")
                time.sleep(wait)

    def _once(self, messages, tools, effort, max_tokens, on_progress) -> Reply:
        t0 = time.time()
        text, reasoning = [], []
        details: list[dict] = []
        calls: dict[int, dict] = {}
        finish = provider = None
        usage: dict = {}
        last_tick = time.time()
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json", "X-Title": "visual-workbench"}
        timeout = httpx.Timeout(connect=30.0, read=IDLE_SECONDS, write=300.0, pool=30.0)
        try:
            with httpx.Client(timeout=timeout) as client, client.stream("POST", f"{self.base}/v1/chat/completions", headers=headers, content=json.dumps(self.body(messages, tools, effort, max_tokens))) as res:
                if res.status_code != 200:
                    detail = res.read().decode("utf-8", "replace")[:600]
                    raise ModelError(f"HTTP {res.status_code}: {detail}", retryable=res.status_code in RETRYABLE)
                for line in res.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    chunk = json.loads(data)
                    if chunk.get("error"):  # provider errors arrive inside the stream, after a 200
                        raise ModelError(f"stream error: {json.dumps(chunk['error'])[:500]}")
                    provider = provider or chunk.get("provider")
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choice = (chunk.get("choices") or [None])[0]
                    if not choice:
                        continue
                    finish = choice.get("native_finish_reason") or choice.get("finish_reason") or finish
                    d = choice.get("delta") or {}
                    if d.get("content"):
                        text.append(d["content"])
                    if d.get("reasoning"):
                        reasoning.append(d["reasoning"])
                    for item in d.get("reasoning_details") or []:
                        _add_detail(details, item)
                    for tc in d.get("tool_calls") or []:
                        call = calls.setdefault(tc.get("index", len(calls)), {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        if tc.get("id"):
                            call["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            call["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            call["function"]["arguments"] += fn["arguments"]
                    if on_progress and time.time() - last_tick > 30:
                        last_tick = time.time()
                        on_progress(time.time() - t0, sum(map(len, reasoning)), sum(map(len, text)), sum(len(c["function"]["arguments"]) for c in calls.values()))
        except httpx.TimeoutException as e:
            raise ModelError(f"no data for {IDLE_SECONDS:.0f} seconds ({type(e).__name__})") from e
        except httpx.HTTPError as e:
            raise ModelError(f"connection failed: {type(e).__name__}: {e}") from e

        made = [calls[i] for i in sorted(calls)]
        tool_calls, sendable = [], []
        for c in made:
            raw = c["function"]["arguments"]
            try:
                args = json.loads(raw) if raw else {}
                if not isinstance(args, dict):
                    raise ValueError("the arguments are not a JSON object")
                tool_calls.append({"id": c["id"], "name": c["function"]["name"], "args": args, "error": None})
                sendable.append(c)
            except ValueError as e:
                tool_calls.append({"id": c["id"], "name": c["function"]["name"], "args": None, "error": f"the arguments were not valid JSON ({e}); the reply may have been cut off"})
                # An argument string that does not parse cannot be sent back as it is: keep the call, emptied.
                sendable.append({**c, "function": {**c["function"], "arguments": "{}"}})
        content = "".join(text)
        raw = {"role": "assistant", "content": content or None}
        if sendable:
            raw["tool_calls"] = sendable
        if details:
            raw["reasoning_details"] = details
        ptd = usage.get("prompt_tokens_details") or {}
        ctd = usage.get("completion_tokens_details") or {}
        return Reply(
            text=content,
            reasoning="".join(reasoning),
            tool_calls=tool_calls,
            assistant={"role": "assistant", "raw": raw},
            finish=finish,
            provider=provider,
            usage={"input": usage.get("prompt_tokens"), "output": usage.get("completion_tokens"), "reasoning": ctd.get("reasoning_tokens"), "cached": ptd.get("cached_tokens"), "usd": usage.get("cost")},
            seconds=time.time() - t0,
        )


def _add_detail(details: list[dict], item: dict) -> None:
    """Text-like fields of a reasoning item continue across chunks; the rest (ids, formats) are set once."""
    prev = None
    if item.get("index") is not None:
        prev = next((x for x in details if x.get("index") == item["index"] and x.get("type") == item.get("type")), None)
    if prev is None:
        details.append(dict(item))
        return
    for k, v in item.items():
        if k in ("text", "summary", "data") and isinstance(v, str):
            prev[k] = (prev.get(k) or "") + v
        elif v is not None:
            prev[k] = v


def create(name: str, app_dir: Path, providers: list[str] | None = None, detail: str | None = None) -> OpenRouter:
    """`argon` is TARGET_MODEL from the repo's .env; anything else is an OpenRouter model id."""
    env = {**load_env(app_dir), **{k: v for k, v in os.environ.items() if k in ("GATEWAY_API_KEY", "OPENROUTER_API_KEY", "GATEWAY_ENDPOINT", "TARGET_MODEL")}}
    model = name
    if name == "argon":
        model = env.get("TARGET_MODEL") or "google/gemini-4-argon"
    key = env.get("GATEWAY_API_KEY") or env.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("No OpenRouter key: set GATEWAY_API_KEY (or OPENROUTER_API_KEY) in the repo's .env")
    return OpenRouter(model, key, env.get("GATEWAY_ENDPOINT"), providers, detail)
