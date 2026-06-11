# generate_script.py
# Generează un script nou de social media imitând stilul scripturilor de top
# ale userului (content/scripturi_top.txt) și îl salvează în content/generated/.
import sys
from datetime import datetime
from pathlib import Path


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR      = get_base_dir()
CONTENT_DIR   = BASE_DIR / "content"
EXAMPLES_PATH = CONTENT_DIR / "scripturi_top.txt"
GENERATED_DIR = CONTENT_DIR / "generated"

# Formaturi acceptate → instrucțiuni specifice de structură/CTA.
_FORMATS = {
    "reel": (
        "Format REEL (Instagram/TikTok, 30-60 secunde, vorbit la cameră). "
        "Hook-ul trebuie să oprească scroll-ul în primele 3 secunde. "
        "Ține propozițiile scurte, ritmate, ușor de spus cu voce tare. "
        "CTA-ul OBLIGATORIU include exact: \"Scrie-mi DM cu COACH\"."
    ),
    "youtube": (
        "Format YOUTUBE (video lung, 5-12 minute). "
        "Hook-ul promite clar transformarea pe care o primește privitorul. "
        "Dezvoltă problema și soluția în profunzime, cu exemple din experiență. "
        "CTA-ul invită la abonare și la pasul următor de lucru cu tine."
    ),
    "carousel": (
        "Format CAROUSEL (Instagram, 6-10 slide-uri). "
        "Slide 1 = hook-ul. Fiecare slide următor = o idee scurtă, punchy. "
        "Ultimul slide = CTA clar. Scrie scriptul slide cu slide (Slide 1:, Slide 2: ...)."
    ),
}

_SYSTEM = (
    "Ești un copywriter de elită pentru content de social media în limba română. "
    "Scrii scripturi virale pentru un coach care vorbește din experiență reală, nu din teorie. "
    "Tonul este DIRECT, AUTORITAR, fără cuvinte de umplutură — ca un mentor care îți spune adevărul în față. "
    "Imiți cu fidelitate stilul, vocabularul și ritmul exemplelor primite. "
    "Returnezi DOAR scriptul final, fără explicații, fără introduceri de tipul 'Iată scriptul'."
)


def _load_examples() -> str:
    try:
        text = EXAMPLES_PATH.read_text(encoding="utf-8").strip()
        return text
    except Exception:
        return ""


def _log(message: str, player=None) -> None:
    print(f"[GenScript] {message}")
    if player:
        try:
            player.write_log(f"JARVIS: {message}")
        except Exception:
            pass


def generate_script(
    parameters:     dict,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    params = parameters or {}
    tema   = (params.get("tema") or "").strip()
    fmt    = (params.get("format") or "reel").strip().lower()

    if not tema:
        msg = "Sir, îmi trebuie tema scriptului ca să îl pot genera."
        _log(msg, player)
        return msg

    if fmt not in _FORMATS:
        fmt = "reel"

    examples = _load_examples()
    if not examples:
        msg = (
            f"Sir, nu găsesc exemple în {EXAMPLES_PATH}. "
            "Adaugă acolo scripturile tale de top ca să pot imita stilul."
        )
        _log(msg, player)
        return msg

    _log(f"Generez script {fmt} despre: {tema}", player)

    format_rules = _FORMATS[fmt]

    prompt = (
        "Acestea sunt scripturile mele de TOP, cu performanțele lor. "
        "Studiază-le stilul, vocabularul, ritmul și structura:\n\n"
        f"=== SCRIPTURI DE TOP ===\n{examples[:6000]}\n=== FINAL EXEMPLE ===\n\n"
        f"Acum scrie-mi UN script NOU despre tema: \"{tema}\".\n\n"
        "Reguli OBLIGATORII:\n"
        "1. Imită stilul, vocabularul și ritmul din exemplele de mai sus — să sune ca mine.\n"
        "2. Respectă structura: HOOK PUTERNIC → PROBLEMĂ → SOLUȚIE → CTA.\n"
        "3. Limba română, ton direct și autoritar de coach care vorbește din experiență, nu teorie.\n"
        f"4. {format_rules}\n\n"
        "Returnează DOAR scriptul final."
    )

    try:
        from core.llm_client import call_llm_text
        script = call_llm_text(prompt, system=_SYSTEM, timeout=180).strip()
    except Exception as e:
        msg = f"Sir, nu am reușit să generez scriptul: {e}"
        _log(msg, player)
        return msg

    if not script:
        msg = "Sir, modelul a returnat un script gol. Încearcă din nou."
        _log(msg, player)
        return msg

    # ── Salvare cu timestamp în nume ──────────────────────────────────────
    saved_path: Path | None = None
    try:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        stamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug      = "".join(c if c.isalnum() else "_" for c in tema.lower())[:40].strip("_")
        filename  = f"script_{fmt}_{slug}_{stamp}.txt"
        saved_path = GENERATED_DIR / filename
        header = f"# Temă: {tema}\n# Format: {fmt}\n# Generat: {stamp}\n\n"
        saved_path.write_text(header + script, encoding="utf-8")
        _log(f"Script salvat în {saved_path}", player)
    except Exception as e:
        _log(f"Scriptul a fost generat dar nu s-a putut salva: {e}", player)

    if saved_path:
        return f"{script}\n\n(Salvat în content/generated/{saved_path.name})"
    return script
