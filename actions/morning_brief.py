"""morning_brief.py — Morning voice briefing: weather + news + reminders,
composed by the LLM in the voice of a warm-but-direct British butler speaking Romanian."""
import json
import sys
from pathlib import Path

import requests

from memory.memory_manager import load_memory


def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR        = _get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"

CITY = "Piatra Neamț,RO"


def _load_api_keys() -> dict:
    try:
        return json.loads(API_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_weather() -> str:
    api_key = _load_api_keys().get("openweathermap_api_key", "").strip()
    if not api_key:
        return "[vreme indisponibilă — adaugă openweathermap_api_key în config/api_keys.json]"

    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": CITY, "appid": api_key, "units": "metric", "lang": "ro"},
            timeout=10,
        )
        resp.raise_for_status()
        data  = resp.json()
        desc  = data["weather"][0]["description"]
        temp  = round(data["main"]["temp"])
        feels = round(data["main"]["feels_like"])
        return f"În Piatra Neamț sunt {temp}°C, {desc}, se simte ca {feels}°C."
    except Exception as e:
        print(f"[MorningBrief] Weather error: {e}")
        return "[vreme indisponibilă — eroare la interogarea OpenWeatherMap]"


def _get_news(max_items: int = 5) -> list[str]:
    newsapi_key = _load_api_keys().get("newsapi_key", "").strip()

    if newsapi_key:
        try:
            resp = requests.get(
                "https://newsapi.org/v2/top-headlines",
                params={"country": "ro", "pageSize": max_items, "apiKey": newsapi_key},
                timeout=10,
            )
            resp.raise_for_status()
            articles  = resp.json().get("articles", [])
            headlines = [a.get("title", "").strip() for a in articles if a.get("title")]
            if headlines:
                return headlines[:max_items]
        except Exception as e:
            print(f"[MorningBrief] NewsAPI error: {e}")

    # Fallback (or default path when no NewsAPI key is configured): reuse the
    # DuckDuckGo helper already used by the web_search tool.
    try:
        from actions.web_search import _ddg_search
        results = _ddg_search("știri România astăzi", max_results=max_items)
        return [r["title"] for r in results if r.get("title")][:max_items]
    except Exception as e:
        print(f"[MorningBrief] DDG news error: {e}")
        return []


def _get_reminders() -> list[str]:
    memory = load_memory()
    items  = []
    for section in ("notes", "wishes"):
        for entry in memory.get(section, {}).values():
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                items.append(str(val))
    return items


def _compose_briefing(weather: str, news: list[str], reminders: list[str]) -> str:
    news_block      = "\n".join(f"- {h}" for h in news) if news else "- (nicio știre disponibilă)"
    reminders_block = "\n".join(f"- {r}" for r in reminders) if reminders else "- (niciun memento activ)"

    system = (
        "Ești JARVIS, un majordom britanic impecabil care vorbește română, în slujba lui Vili Savescu. "
        "Compui briefingul lui de dimineață: cald dar direct, natural, ca un majordom de încredere care "
        "vorbește cu stăpânul casei — nu ca un robot care citește o listă. Adresează-te lui direct. "
        "Maxim 150-200 de cuvinte. Răspunde STRICT cu textul briefingului, fără titluri, fără explicații."
    )
    prompt = (
        f"Vremea de azi: {weather}\n\n"
        f"Știri de azi:\n{news_block}\n\n"
        f"Notițe și planuri active ale lui Vili:\n{reminders_block}\n\n"
        "Compune briefingul de dimineață pentru Vili."
    )

    try:
        from core.llm_client import call_llm_text
        text = call_llm_text(prompt, system=system)
        if text:
            return text
    except Exception as e:
        print(f"[MorningBrief] LLM compose error: {e}")

    fallback = [f"Bună dimineața, domnule Savescu. {weather}"]
    if news:
        fallback.append("Pe scurt din presă: " + "; ".join(news[:3]) + ".")
    if reminders:
        fallback.append("Și nu uitați: " + reminders[0] + ".")
    return " ".join(fallback)


def morning_brief(
    parameters: dict = None,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    weather   = _get_weather()
    news      = _get_news()
    reminders = _get_reminders()

    briefing = _compose_briefing(weather, news, reminders)

    if player:
        try:
            player.write_log(f"[MorningBrief] {briefing[:80]}…")
        except Exception:
            pass

    return briefing
