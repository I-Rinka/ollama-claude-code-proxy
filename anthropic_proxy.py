"""
Anthropic Messages API -> OpenAI Chat Completions API Proxy  v3
Full tool-use support, real token tracking, enhanced system prompt.
Run:  bash ~/start_proxy.sh
"""

import json, uuid, traceback, uvicorn, httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI()

# ====== Config ======
OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL   = "qwen3:30b-a3b"
HOST, PORT      = "0.0.0.0", 8056
# ====================

ENHANCED_SYSTEM = (
    "You are a powerful AI coding assistant operating inside Claude Code, "
    "an interactive CLI tool for software development.\n"
    "You have full access to the local filesystem, shell, git, and IDE integrations.\n\n"

    "## Tool-Use Rules\n"
    "- ALWAYS use the provided tools rather than asking the user to run commands manually.\n"
    "- Use Bash for: shell commands, git, build tools, package managers, running tests.\n"
    "- Use Read for reading files (never use cat/head/tail to read files).\n"
    "- Use Edit for surgical edits to existing files (old_string/new_string replacement).\n"
    "- Use Write only for creating brand-new files or complete rewrites.\n"
    "- Use NotebookEdit for Jupyter notebook cell edits.\n"
    "- When the task needs information you don't already have, use tools to look it up "
    "(grep/find via Bash, Read files, etc.) before answering.\n"
    "- When given a coding task, write or modify code with tools, then verify with tests "
    "or build commands.\n"
    "- For multi-step tasks, plan your approach, then execute step by step using tools.\n"
    "- You can call multiple independent tools in parallel for efficiency.\n\n"

    "## Style\n"
    "- Be direct and concise. Lead with actions, not explanations.\n"
    "- Do not narrate obvious steps.\n"
    "- After making changes, briefly summarise what you did and any next steps.\n"
    "- Prefer editing existing files over creating new ones.\n"
    "- Never create .md or README files unless explicitly asked.\n\n"

    "## Safety\n"
    "- Avoid introducing security vulnerabilities (injection, XSS, etc.).\n"
    "- Ask before destructive operations (force push, rm -rf, dropping tables).\n"
    "- Never commit unless the user explicitly asks.\n"
)


# ---------------------------------------------------------------------------
# Anthropic <-> OpenAI format conversion
# ---------------------------------------------------------------------------

def _convert_tools(tools):
    """Anthropic tool defs -> OpenAI function defs."""
    if not tools:
        return None
    out = []
    for t in tools:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return out


def _extract_text(content) -> str:
    """Pull plain text out of an Anthropic content field (str | list[blocks])."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content else ""
    parts = []
    for b in content:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict):
            t = b.get("type", "")
            if t == "text":
                parts.append(b.get("text", ""))
            elif t == "thinking":
                pass
    return "\n".join(parts)


def _convert_message(msg):
    """Convert one Anthropic message -> one or more OpenAI messages."""
    role    = msg.get("role", "user")
    content = msg.get("content", "")

    # --- assistant: may contain text + tool_use blocks ---
    if role == "assistant":
        text, tool_calls = [], []
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict):
                    if b.get("type") == "text":
                        text.append(b.get("text", ""))
                    elif b.get("type") == "tool_use":
                        tool_calls.append({
                            "id": b.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                            "type": "function",
                            "function": {
                                "name": b.get("name", ""),
                                "arguments": json.dumps(b.get("input", {}), ensure_ascii=False),
                            },
                        })
        elif isinstance(content, str):
            text.append(content)

        result = {"role": "assistant", "content": "\n".join(text) or ""}
        if tool_calls:
            result["tool_calls"] = tool_calls
        return [result]

    # --- user: may contain text + tool_result blocks ---
    if isinstance(content, list):
        text_parts, tool_results = [], []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    text_parts.append(b.get("text", ""))
                elif b.get("type") == "tool_result":
                    rc = b.get("content", "")
                    if isinstance(rc, list):
                        rc = "\n".join(
                            s.get("text", "") if isinstance(s, dict) else str(s) for s in rc
                        )
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": b.get("tool_use_id", ""),
                        "content": str(rc),
                    })
            elif isinstance(b, str):
                text_parts.append(b)

        msgs = []
        if text_parts:
            msgs.append({"role": "user", "content": "\n".join(text_parts)})
        msgs.extend(tool_results)
        return msgs if msgs else [{"role": "user", "content": ""}]

    return [{"role": role, "content": str(content)}]


def anthropic_to_openai(body):
    """Full Anthropic Messages API body -> OpenAI Chat Completions body."""
    messages = []

    # system prompt
    raw_sys = body.get("system", "")
    sys_text = _extract_text(raw_sys).strip() if raw_sys else ""
    full_sys = (sys_text + "\n\n" + ENHANCED_SYSTEM).strip() if sys_text else ENHANCED_SYSTEM
    full_sys += "\n\n/no_think"          # suppress Qwen3 thinking
    messages.append({"role": "system", "content": full_sys})

    # messages
    for m in body.get("messages", []):
        messages.extend(_convert_message(m))

    # tools
    tools = _convert_tools(body.get("tools"))

    # build body
    ob = {
        "model":      body.get("model", DEFAULT_MODEL),
        "messages":   messages,
        "max_tokens": max(body.get("max_tokens", 4096), 8192),
        "stream":     body.get("stream", False),
    }
    if tools:
        ob["tools"] = tools
    if "temperature" in body:
        ob["temperature"] = body["temperature"]
    if "top_p" in body:
        ob["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        ob["stop"] = body["stop_sequences"]
    return ob


# ---------------------------------------------------------------------------
# Non-streaming response conversion
# ---------------------------------------------------------------------------

def openai_to_anthropic(resp, model):
    choice  = resp["choices"][0]
    message = choice["message"]
    usage   = resp.get("usage", {})

    content_blocks = []
    text = message.get("content", "") or ""
    if not text and message.get("reasoning"):
        text = message["reasoning"]
    if text:
        content_blocks.append({"type": "text", "text": text})

    for tc in (message.get("tool_calls") or []):
        try:
            inp = json.loads(tc["function"].get("arguments", "{}"))
        except json.JSONDecodeError:
            inp = {}
        content_blocks.append({
            "type":  "tool_use",
            "id":    tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
            "name":  tc["function"]["name"],
            "input": inp,
        })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    fin = choice.get("finish_reason", "stop") or "stop"
    stop = "end_turn" if fin in ("stop", "length") else ("tool_use" if fin == "tool_calls" else fin)

    return {
        "id":            f"msg_{uuid.uuid4().hex[:24]}",
        "type":          "message",
        "role":          "assistant",
        "content":       content_blocks,
        "model":         model,
        "stop_reason":   stop,
        "stop_sequence": None,
        "usage": {
            "input_tokens":              usage.get("prompt_tokens", 0),
            "output_tokens":             usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens":     0,
        },
    }


# ---------------------------------------------------------------------------
# Streaming chunk -> Anthropic SSE events
# ---------------------------------------------------------------------------

def _sse(event_type, data_dict):
    """Build one SSE frame: 'event: X\ndata: {...}\n\n'."""
    return f"event: {event_type}\ndata: {json.dumps(data_dict, ensure_ascii=False)}\n\n"


def _sse_chunk(chunk, model, msg_id, S):
    """
    Convert one OpenAI streaming chunk into zero or more Anthropic SSE event strings.
    S = mutable state dict that persists across the whole stream.
    """
    events = []
    choices = chunk.get("choices") or []
    if not choices:
        return events

    c      = choices[0]
    delta  = c.get("delta") or {}
    finish = c.get("finish_reason")

    # --- first chunk: message_start ---
    if not S["started"]:
        S["started"] = True
        events.append(_sse("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "content": [], "model": model,
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0,
                          "cache_creation_input_tokens": 0,
                          "cache_read_input_tokens": 0},
            },
        }))

    # --- text delta ---
    txt = delta.get("content") or ""
    if txt:
        if S["block_type"] is None:
            S["block_type"] = "text"
            events.append(_sse("content_block_start", {
                "type": "content_block_start", "index": S["block_idx"],
                "content_block": {"type": "text", "text": ""},
            }))
        events.append(_sse("content_block_delta", {
            "type": "content_block_delta", "index": S["block_idx"],
            "delta": {"type": "text_delta", "text": txt},
        }))

    # --- tool call deltas ---
    tcs = delta.get("tool_calls")
    if tcs:
        for tc in tcs:
            idx = tc.get("index", 0)

            # first time we see this tool index -> start a tool_use block
            if idx not in S["tools"]:
                # close previous block
                if S["block_type"] is not None:
                    events.append(_sse("content_block_stop", {
                        "type": "content_block_stop", "index": S["block_idx"],
                    }))
                    S["block_idx"] += 1

                tid   = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                tname = (tc.get("function") or {}).get("name", "")
                S["tools"][idx]  = {"id": tid, "name": tname}
                S["block_type"]  = "tool"

                events.append(_sse("content_block_start", {
                    "type": "content_block_start", "index": S["block_idx"],
                    "content_block": {"type": "tool_use", "id": tid,
                                      "name": tname, "input": {}},
                }))

            # stream argument fragment
            arg = (tc.get("function") or {}).get("arguments", "")
            if arg:
                events.append(_sse("content_block_delta", {
                    "type": "content_block_delta", "index": S["block_idx"],
                    "delta": {"type": "input_json_delta", "partial_json": arg},
                }))

    # --- finish ---
    if finish:
        if S["block_type"] is not None:
            events.append(_sse("content_block_stop", {
                "type": "content_block_stop", "index": S["block_idx"],
            }))

        if finish == "tool_calls":
            stop = "tool_use"
        elif finish == "length":
            stop = "max_tokens"
        else:
            stop = "end_turn"

        events.append(_sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop, "stop_sequence": None},
            "usage": {"output_tokens": S["out_tok"]},
        }))
        events.append(_sse("message_stop", {"type": "message_stop"}))

    # accumulate token counts from the final chunk
    u = chunk.get("usage")
    if u:
        S["in_tok"]  = u.get("prompt_tokens",    S["in_tok"])
        S["out_tok"] = u.get("completion_tokens", S["out_tok"])

    return events


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

def _err(code, msg, t="api_error"):
    return JSONResponse(code, {"type": "error", "error": {"type": t, "message": msg}})


@app.post("/v1/messages")
async def messages(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        return _err(400, f"Invalid JSON: {e}", "invalid_request_error")

    model  = body.get("model", DEFAULT_MODEL)
    stream = body.get("stream", False)

    try:
        ob = anthropic_to_openai(body)
    except Exception as e:
        traceback.print_exc()
        return _err(400, f"Conversion error: {e}", "invalid_request_error")

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    # ---- streaming ----
    if stream:
        async def gen():
            S = dict(started=False, block_type=None, block_idx=0,
                     tools={}, in_tok=0, out_tok=0)
            try:
                async with httpx.AsyncClient(timeout=600.0) as cli:
                    async with cli.stream("POST",
                            f"{OLLAMA_BASE_URL}/chat/completions", json=ob) as r:
                        async for line in r.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            ds = line[6:]
                            if ds.strip() == "[DONE]":
                                return
                            try:
                                for ev in _sse_chunk(json.loads(ds), model, msg_id, S):
                                    yield ev
                            except json.JSONDecodeError:
                                pass
            except Exception:
                traceback.print_exc()
                err_msg = traceback.format_exc().splitlines()[-1]
                yield _sse("error", {
                    "type": "error",
                    "error": {"type": "api_error", "message": err_msg},
                })

        return StreamingResponse(gen(), media_type="text/event-stream")

    # ---- non-streaming ----
    try:
        async with httpx.AsyncClient(timeout=600.0) as cli:
            r = await cli.post(f"{OLLAMA_BASE_URL}/chat/completions", json=ob)
            return JSONResponse(openai_to_anthropic(r.json(), model))
    except Exception as e:
        traceback.print_exc()
        return _err(502, f"Backend error: {e}")


@app.get("/v1/models")
async def list_models():
    try:
        async with httpx.AsyncClient(timeout=5.0) as cli:
            r  = await cli.get(f"{OLLAMA_BASE_URL}/models")
            ms = r.json().get("data", [])
    except Exception:
        ms = [{"id": DEFAULT_MODEL}]
    return JSONResponse({
        "data": [{"id": m.get("id", DEFAULT_MODEL),
                  "display_name": m.get("id", DEFAULT_MODEL),
                  "created_at": "2026-01-01T00:00:00Z"} for m in ms],
        "type": "list",
    })


@app.get("/")
async def root():
    return {"status": "running", "version": 3, "backend": OLLAMA_BASE_URL}


if __name__ == "__main__":
    print(f"Anthropic -> OpenAI Proxy v3  on {HOST}:{PORT}")
    print(f"Backend: {OLLAMA_BASE_URL}")
    uvicorn.run(app, host=HOST, port=PORT)
