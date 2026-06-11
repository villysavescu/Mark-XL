"""
Repro + regression test for the Groq HTTP-400 bug on post-tool-call requests.

Scenario reproduced: the user says something that triggers `save_memory`, then
asks a follow-up question.  After the tool round, main.py keeps the assistant
tool-call message (Ollama shape: arguments as a dict, no "type", id maybe empty)
plus a role:"tool" reply in `self._conversation`.  On the NEXT user turn that
history is replayed to Groq, which rejects it with HTTP 400 + empty body.

This test:
  1. Builds that exact history.
  2. Asserts `_normalize_history_openai` produces strict OpenAI format.
  3. If Groq is reachable (api_keys.json + network), fires the raw history
     (expecting 400) and the normalized history (expecting 200) to prove the fix.

Run:  python test_groq_tool_history.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
from core.llm_client import (
    _normalize_history_openai,
    get_llm_settings,
    get_llm_provider,
    _openai_headers,
    _load_config,
)


def _build_ollama_shaped_history() -> list:
    """The history main.py accumulates after a silent save_memory round."""
    return [
        {"role": "system", "content": "You are JARVIS. Be concise."},
        {"role": "user", "content": "My name is Fatih and I live in Ankara."},
        # Assistant tool-call in OLLAMA shape: arguments is a DICT, no "type",
        # and (as Ollama produces) no id.
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "save_memory",
                        "arguments": {"category": "identity", "key": "name", "value": "Fatih"},
                    }
                }
            ],
        },
        # Tool reply with NO tool_call_id (Ollama doesn't use them).
        {"role": "tool", "content": "Done."},
        {"role": "assistant", "content": "Got it, Fatih."},
        # The follow-up question that triggers the failing request.
        {"role": "user", "content": "What's the capital of France?"},
    ]


def test_normalization():
    raw  = _build_ollama_shaped_history()
    norm = _normalize_history_openai(raw)

    # Same number of messages, order preserved.
    assert len(norm) == len(raw), "message count changed"

    asst = next(m for m in norm if m["role"] == "assistant" and m.get("tool_calls"))
    tc   = asst["tool_calls"][0]

    assert tc.get("type") == "function", "missing type:function"
    assert tc.get("id"), "missing tool_call id"
    assert isinstance(tc["function"]["arguments"], str), "arguments must be a JSON string"
    # arguments must be valid JSON encoding the original dict.
    assert json.loads(tc["function"]["arguments"]) == {
        "category": "identity", "key": "name", "value": "Fatih",
    }, "arguments JSON content mismatch"

    tool_msg = next(m for m in norm if m["role"] == "tool")
    assert tool_msg.get("tool_call_id") == tc["id"], "tool_call_id must match assistant tool call id"

    # Non-tool messages must be untouched.
    assert norm[0] == raw[0], "system message changed"
    assert norm[-1] == raw[-1], "trailing user message changed"

    print("[PASS] normalization produces strict OpenAI tool-call format")
    print(json.dumps(norm, indent=2, ensure_ascii=False))


def _groq_reachable() -> bool:
    if get_llm_provider() != "openai":
        return False
    if not _load_config().get("openai_api_key"):
        return False
    url, _ = get_llm_settings()
    try:
        return requests.get(f"{url}/v1/models", headers=_openai_headers(), timeout=8).status_code == 200
    except Exception:
        return False


def _post(messages: list) -> requests.Response:
    url, model = get_llm_settings()
    payload = {"model": model, "messages": messages, "stream": False, "max_tokens": 64}
    return requests.post(
        f"{url}/v1/chat/completions",
        json=payload,
        headers=_openai_headers(),
        timeout=60,
    )


def test_live_groq():
    if not _groq_reachable():
        print("[SKIP] Groq not reachable (no key / offline) — skipping live test")
        return

    raw  = _build_ollama_shaped_history()
    norm = _normalize_history_openai(raw)

    print("\n[LIVE] Sending RAW (Ollama-shaped) history -- expecting HTTP 400...")
    r_raw = _post(raw)
    print(f"[LIVE] raw  -> HTTP {r_raw.status_code}  body[:200]={r_raw.text[:200]!r}")

    print("\n[LIVE] Sending NORMALIZED history -- expecting HTTP 200...")
    r_norm = _post(norm)
    print(f"[LIVE] norm -> HTTP {r_norm.status_code}")

    if r_norm.status_code == 200:
        content = r_norm.json()["choices"][0]["message"].get("content", "")
        print(f"[LIVE] answer: {content.strip()[:200]}")

    assert r_norm.status_code == 200, f"normalized history still failed: {r_norm.status_code} {r_norm.text[:300]}"
    if r_raw.status_code == 400:
        print("[PASS] reproduced the 400 on raw history AND fixed it via normalization")
    else:
        print(f"[WARN] raw history returned {r_raw.status_code} (not the expected 400); "
              "fix still verified by the 200 on normalized history")


if __name__ == "__main__":
    test_normalization()
    test_live_groq()
    print("\nAll checks complete.")
