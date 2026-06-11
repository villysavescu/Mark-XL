"""
Crash-safe conversation logging for MARK XL.

Every session gets ONE timestamped JSON file under memory/conversations/.
The full conversation is rewritten after every exchange (atomic temp-then-
replace), so a crash mid-session never corrupts or loses the log: the worst
case is losing the single in-progress turn.

The logged file is also the input for the reflection pass (see reflection.py),
but reflection normally runs on the live self._conversation list directly.
"""
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from threading import Lock


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR          = get_base_dir()
CONVERSATIONS_DIR = BASE_DIR / "memory" / "conversations"


class ConversationLogger:
    """One instance per session → one JSON file, rewritten on each save()."""

    def __init__(self) -> None:
        self._lock     = Lock()
        self._started  = datetime.now()
        stamp          = self._started.strftime("%Y%m%d_%H%M%S")
        self._path     = CONVERSATIONS_DIR / f"conversation_{stamp}.json"

    @property
    def path(self) -> Path:
        return self._path

    def save(self, conversation: list) -> None:
        """
        Persist the full conversation so far.  Safe to call after every turn.

        Writes to a temp file in the same directory, then os.replace() →
        atomic on Windows and POSIX, so the JSON on disk is always complete.
        """
        if not isinstance(conversation, list):
            return
        payload = {
            "started":  self._started.isoformat(timespec="seconds"),
            "updated":  datetime.now().isoformat(timespec="seconds"),
            "messages": conversation,
        }
        with self._lock:
            try:
                CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=str(CONVERSATIONS_DIR), suffix=".tmp")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2, ensure_ascii=False)
                    os.replace(tmp, self._path)
                except Exception:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                    raise
            except Exception as e:
                print(f"[ConvLog] ⚠️ Save error: {e}")
