"""
Self-learning reflection pass for MARK XL.

Takes the current conversation, asks the LLM to extract ONLY durable signal —
new facts about the user, corrections/feedback the user gave, and observed
preferences — as strict JSON keyed by the long_term.json categories. The result
is merged into long_term.json WITHOUT overwriting any entry that already exists.

Runs automatically at shutdown and on demand via the reflect_on_conversation
tool. Conversations shorter than MIN_MESSAGES are skipped (too little signal).
"""
import json
import re
from datetime import datetime

from memory.memory_manager import load_memory, save_memory

# Below this many messages there is not enough signal to reflect on.
MIN_MESSAGES = 4

# Only these categories are accepted from the model's output. relationships /
# wishes are intentionally excluded — the prompt asks for facts/feedback/prefs.
_VALID_CATEGORIES = ("identity", "preferences", "projects", "notes")

_SYSTEM = (
    "Ești un analist care extrage informații durabile despre un utilizator dintr-o "
    "conversație cu asistentul lui. Returnezi STRICT un singur obiect JSON valid, "
    "fără text în plus, fără explicații, fără ghilimele de cod."
)


def _format_transcript(conversation: list) -> str:
    lines: list[str] = []
    for msg in conversation:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue  # skip system / tool messages — not user signal
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        speaker = "Utilizator" if role == "user" else "Asistent"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def _build_prompt(transcript: str) -> str:
    return (
        "Analizează conversația de mai jos și extrage DOAR:\n"
        "  1. Fapte noi despre utilizator (identitate, situație, context real).\n"
        "  2. Corecturi sau feedback pe care utilizatorul le-a dat asistentului.\n"
        "  3. Preferințe observate (cum vrea să i se vorbească / lucreze).\n\n"
        "NU inventa nimic. Dacă nu reiese clar din conversație, omite. "
        "Ignoră small talk și informațiile triviale.\n\n"
        "Returnează STRICT un obiect JSON cu aceste categorii (omite-le pe cele goale):\n"
        "{\n"
        '  "identity":    { "<cheie_snake_case>": "<valoare scurtă>" },\n'
        '  "preferences": { "<cheie_snake_case>": "<valoare scurtă>" },\n'
        '  "projects":    { "<cheie_snake_case>": "<valoare scurtă>" },\n'
        '  "notes":       { "<cheie_snake_case>": "<valoare scurtă>" }\n'
        "}\n\n"
        "Cheile în snake_case, valorile scurte și concrete, în limba română.\n\n"
        f"=== CONVERSAȚIE ===\n{transcript[:6000]}\n=== FINAL ===\n\n"
        "JSON:"
    )


def _parse_json(raw: str) -> dict:
    if not raw:
        return {}
    text = raw.strip()
    # Strip ```json / ``` fences if the model added them.
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text).strip()
    # Fall back to the first {...last } span if there's stray prose around it.
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[Reflect] ⚠️ JSON parse failed: {e}")
        return {}


def _merge_new_only(data: dict) -> list[str]:
    """
    Insert extracted entries into long_term.json, but ONLY for keys that don't
    already exist in that category — existing entries are never overwritten.
    Returns the list of "category/key" paths actually added.
    """
    memory = load_memory()
    today  = datetime.now().strftime("%Y-%m-%d")
    added: list[str] = []

    for cat in _VALID_CATEGORIES:
        items = data.get(cat)
        if not isinstance(items, dict):
            continue
        bucket = memory.setdefault(cat, {})
        for key, val in items.items():
            key = str(key).strip()
            if not key:
                continue
            value = val.get("value") if isinstance(val, dict) else val
            if not isinstance(value, str) or not value.strip():
                continue
            if key in bucket:
                continue  # never overwrite an existing entry
            bucket[key] = {"value": value.strip(), "updated": today}
            added.append(f"{cat}/{key}")

    if added:
        save_memory(memory)
        print(f"[Reflect] 💾 Learned {len(added)} new: {added}")
    return added


def reflect_on_conversation(conversation: list, player=None) -> str:
    """
    Analyse `conversation`, extract durable signal, merge new entries into
    long_term.json. Returns a short human-facing summary (spoken to the user
    when triggered via the tool). Best-effort: never raises.
    """
    msgs = [
        m for m in (conversation or [])
        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
    ]
    if len(msgs) < MIN_MESSAGES:
        print(f"[Reflect] Skipped — only {len(msgs)} message(s) (<{MIN_MESSAGES}).")
        return "Conversația e prea scurtă pentru reflecție."

    transcript = _format_transcript(conversation)
    if not transcript.strip():
        return "Nu am ce analiza din conversație."

    if player:
        try:
            player.write_log("SYS: Reflecting on conversation…")
        except Exception:
            pass

    try:
        from core.llm_client import call_llm_text
        raw = call_llm_text(_build_prompt(transcript), system=_SYSTEM, timeout=180)
    except Exception as e:
        print(f"[Reflect] ⚠️ LLM call failed: {e}")
        return "Nu am reușit să reflectez asupra conversației."

    data = _parse_json(raw)
    if not data:
        return "Nu am extras nimic nou din conversație."

    added = _merge_new_only(data)
    if not added:
        return "Am analizat conversația, dar nu era nimic nou de reținut."
    return f"Am reținut {len(added)} lucruri noi din conversație."
