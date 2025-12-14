#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZHAB.py  (ZaonHe Anki Builder)

CSV -> (romanise via rime-yahwe_zaonhe dict) -> (Shanghainese TTS via HF Space) -> Anki .apkg

Notes:
- Correctly calls the HF Space using fn_index (Blocks apps often have no named api_name)
  * Auto-selects the endpoint that RETURNS Audio (in your dump: fn_index=1)
- Handles Gradio FileSerializable audio returns (local path string OR dict with base64)
- Romanisation field (multiple readings joined by "/")
- Mandarin field preserved (does NOT overwrite it)
- Full CJK block coverage per your ranges (incl. 𠲎 / 𡍲 etc.)
- Fix “double-layer PDF text” duplication: 我我也也… -> 我也…
- per-row retries with exponential backoff + jitter
- persistent resume cache (out/state.json)
- failed queue (out/failed.csv) with multi-pass retry
- Blue ANSI success lines only
- Progress bar (tqdm) + per-item logging (tqdm.write)

CSV required column: shanghainese
Recommended columns: mandarin, meaning
Optional: id, ipa_input, speed

Install:
  pip install pandas requests gradio-client genanki tqdm
  # Optional for MP3 conversion:
  # install ffmpeg system-wide

Run:
  python ZHAB.py
"""

import re
import json
import time
import hashlib
import random
import subprocess
import base64
from urllib.parse import quote
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Tuple, Optional

import pandas as pd
import requests
from gradio_client import Client
import genanki
from tqdm import tqdm


# =========================
# CONFIG
# =========================
SPACE_ID = "https://cjangcjengh-shanghainese-tts.hf.space"

RIME_DICT_URL = "https://raw.githubusercontent.com/wugniu/rime-yahwe_zaonhe/master/yahwe_zaonhe.dict.yaml"

MAX_ATTEMPTS_PER_ROW = 4
BASE_BACKOFF_SECONDS = 2.0
JITTER_SECONDS = 0.6
FAILED_QUEUE_PASSES = 6
FAILED_PASS_PAUSE_SECONDS = 20.0
RETRY_FAILED_EVERY_N_SUCCESSES = 50  # 0 disables mid-run retry
MIN_AUDIO_BYTES = 4096   # 4 KB - check to see if audio isn't empty.

OUT_DIR = Path("out")
AUDIO_DIR = OUT_DIR / "audio"
STATE_PATH = OUT_DIR / "state.json"
FAILED_PATH = OUT_DIR / "failed.csv"
RIME_CACHE_PATH = OUT_DIR / "yahwe_zaonhe.dict.yaml"

MODEL_ID = 190101001
DECK_ID = 190101002

BRACKET_UNKNOWN = True
KEEP_WAV = True
PRINT_ROMANISATION = False

# Force a specific fn_index (set to 1 to skip autodetect; deprecated):
FORCE_FN_INDEX: Optional[int] = None


# =========================
# ANSI
# =========================

# I chose blue as I am colour blind!
ANSI_BLUE = "\033[34m"
ANSI_RESET = "\033[0m"


def blue(s: str) -> str:
    return f"{ANSI_BLUE}{s}{ANSI_RESET}"


# =========================
# CJK regex (Ranges A-J)
# =========================
_CJK_RE = re.compile(
    r"["
    r"\u3400-\u4DBF"          # Extension A
    r"\u4E00-\u9FFF"          # Unified Ideographs
    r"\U00020000-\U0002A6DF"  # Extension B
    r"\U0002A700-\U0002B73F"  # Extension C
    r"\U0002B740-\U0002B81D"  # Extension D
    r"\U0002B820-\U0002CEAD"  # Extension E
    r"\U0002CEB0-\U0002EBE0"  # Extension F
    r"\U00031350-\U000323AF"  # Extension H
    r"\U0002EBF0-\U0002EE5D"  # Extension I
    r"\U000323B0-\U00033479"  # Extension J
    r"\U0002F800-\U0002FA1F"  # Compatibility Supplement
    r"]"
)


# =========================
# UTILITIES
# =========================
_PAGE_FRAC_RE = re.compile(r"\b\d+\s*/\s*\d+\b")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sanitize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    s = _PAGE_FRAC_RE.sub("", s).strip()
    return s


def undouble_text(s: str) -> str:
    """
    Fix pairwise duplicate glyph extraction:
      我我也也常常常常 -> 我也常常
    Also handles whole-string duplication:
      ABCABC -> ABC
    Removes internal whitespace.
    """
    s = sanitize_text(s)
    compact = re.sub(r"\s+", "", s)
    n = len(compact)
    if n >= 2 and n % 2 == 0:
        half = n // 2
        if compact[0::2] == compact[1::2]:
            return compact[0::2]
        if compact[:half] == compact[half:]:
            return compact[:half]
    return compact


def row_key(text: str, ipa_input: bool, speed: float) -> str:
    raw = f"{text}|ipa={ipa_input}|speed={speed}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def safe_audio_basename(key: str) -> str:
    return f"shh_{key[:16]}"


def load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"done": {}, "meta": {"space": SPACE_ID, "created": time.time()}}


def save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def append_failed_row(failed_row: Dict[str, Any]) -> None:
    df = pd.DataFrame([failed_row])
    if FAILED_PATH.exists():
        df.to_csv(FAILED_PATH, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(FAILED_PATH, mode="w", header=True, index=False, encoding="utf-8-sig")


def load_failed_df() -> pd.DataFrame:
    if FAILED_PATH.exists():
        return pd.read_csv(FAILED_PATH).fillna("")
    return pd.DataFrame(
        columns=["key", "id", "shanghainese", "mandarin", "meaning", "ipa_input", "speed", "error", "attempts"]
    )


def audio_nonempty(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size >= MIN_AUDIO_BYTES
    except Exception:
        return False

def rewrite_failed_df(df: pd.DataFrame) -> None:
    if len(df) == 0:
        FAILED_PATH.write_text(
            "key,id,shanghainese,mandarin,meaning,ipa_input,speed,error,attempts\n",
            encoding="utf-8-sig",
        )
        return
    df.to_csv(FAILED_PATH, index=False, encoding="utf-8-sig")


def sleep_backoff(attempt_index: int) -> None:
    delay = BASE_BACKOFF_SECONDS * (2 ** (attempt_index - 1))
    delay += random.uniform(0, JITTER_SECONDS)
    time.sleep(delay)


def log_success(pbar: tqdm, idx_ok: int, text: str, roman: str, mandarin: str, audio_filename: str) -> None:
    if PRINT_ROMANISATION:
        msg = (
            f"{blue(f'[OK {idx_ok:>4}]')} {text}\n"
            f"          → {roman}\n"
            f"          → zh: {mandarin}\n"
            f"          → audio: {audio_filename}"
        )
    else:
        msg = f"{blue(f'[OK {idx_ok:>4}]')} {text}  (audio: {audio_filename})"
    pbar.write(msg)


# =========================
# RIME ROMANISATION
# =========================
PUNCT = set("，。！？；：、（）「」『』《》〈〉…—～·" ",.!?;:()[]\"'")
OPENERS = set("（「『《〈(")

_ROM_FIX_SPACES = [
    (re.compile(r"\s+([，。！？；：、])"), r"\1"),  # remove space before CJK punct
    (re.compile(r"([，。！？；：、])\s*"), r"\1 "),  # single space after punct
    (re.compile(r"\s+([,.!?;:])"), r"\1"),        # remove space before ASCII punct
    (re.compile(r"([,.!?;:])\s*"), r"\1 "),       # single space after ASCII punct
]

def postprocess_romanisation(s: str) -> str:
    s = s.strip()
    # remove weird stray leading apostrophes before letters
    s = re.sub(r"(?<![A-Za-z0-9])'+(?=[A-Za-z])", "", s)
    # collapse spaces
    s = re.sub(r"\s+", " ", s)
    for pat, rep in _ROM_FIX_SPACES:
        s = pat.sub(rep, s)
    return s.strip()


def load_rime_dict(url: str = RIME_DICT_URL, cache_path: Optional[Path] = None) -> Tuple[Dict[str, set], int]:
    if cache_path and cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
    else:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        text = r.text
        if cache_path:
            ensure_dir(cache_path.parent)
            cache_path.write_text(text, encoding="utf-8")

    romap = defaultdict(set)
    maxlen = 1
    in_body = False

    for line in text.splitlines():
        line = line.rstrip("\n")
        if not in_body:
            if line.strip() == "...":
                in_body = True
            continue
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        word = parts[0].strip()
        roman = parts[1].strip()
        if not word or not roman:
            continue
        romap[word].add(roman)
        maxlen = max(maxlen, len(word))

    return dict(romap), maxlen


def _join_readings(readings: set[str]) -> str:
    return "/".join(sorted(readings))


def romanise_sentence(s: str, romap: Dict[str, set], maxlen: int) -> str:
    s = sanitize_text(s)
    if not s:
        return ""

    tokens: list[str] = []
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]

        if ch in PUNCT:
            tokens.append(ch)
            i += 1
            continue

        if ch.isspace():
            i += 1
            continue

        if not _CJK_RE.match(ch):
            tokens.append(ch)
            i += 1
            continue

        matched = None
        upper = min(maxlen, n - i)
        for L in range(upper, 0, -1):
            seg = s[i : i + L]
            if seg in romap:
                matched = seg
                break

        if matched is None:
            tokens.append(f"[{ch}]" if BRACKET_UNKNOWN else ch)
            i += 1
        else:
            tokens.append(_join_readings(romap[matched]))
            i += len(matched)

    out = []
    for t in tokens:
        if not out:
            out.append(t)
            continue
        if t in PUNCT:
            out.append(t)
        else:
            out.append(t if out[-1] in OPENERS else " " + t)

    return "".join(out).strip()


# =========================
# HF SPACE ENDPOINT (fn_index)
# =========================
def detect_fn_index(client: Client) -> int:
    """
    Choose the endpoint that RETURNS Audio from unnamed_endpoints.
    In your dump: key '1' returns component 'Audio' => fn_index 1.
    """
    if FORCE_FN_INDEX is not None:
        return int(FORCE_FN_INDEX)

    api = client.view_api(all_endpoints=True, return_format="dict")
    unnamed = api.get("unnamed_endpoints") or {}

    if not isinstance(unnamed, dict) or not unnamed:
        return 0

    # Prefer endpoint that returns Audio
    for k, ep in unnamed.items():
        returns = ep.get("returns", [])
        if any((r.get("component") == "Audio") for r in returns):
            return int(k) if str(k).isdigit() else 0

    # Fallback: most parameters
    k_best = max(unnamed.keys(), key=lambda kk: len(unnamed[kk].get("parameters", [])))
    return int(k_best) if str(k_best).isdigit() else 0


# =========================
# AUDIO RETURN HANDLING
# =========================
def wav_to_mp3(wav_path: Path, mp3_path: Path, pbar: Optional[tqdm] = None) -> Optional[Path]:
    """
    Convert using ffmpeg if available.
    HARD GUARD: Only attempt conversion for real local files.
    Also logs exactly what ffmpeg is being asked to open.
    """
    def _log(msg: str):
        if pbar is not None:
            pbar.write(msg)
        else:
            print(msg)

    try:
        # Guard 1: must be a Path and must exist
        if not isinstance(wav_path, Path):
            _log(f"[mp3-skip] wav_path not a Path: {type(wav_path)} {wav_path}")
            return None
        if not wav_path.exists():
            _log(f"[mp3-skip] input does not exist: {wav_path}")
            return None

        # Guard 2: block anything that looks like a URL (just in case)
        s = str(wav_path)
        if s.lower().startswith(("ws://", "wss://", "http://", "https://")):
            _log(f"[mp3-skip] refusing URL input to ffmpeg: {s}")
            return None

        # LOG what we're converting
        _log(f"[mp3] ffmpeg input: {wav_path}")

        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_path), str(mp3_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,   # capture errors to inspect if needed
            text=True,
        )
        return mp3_path

    except subprocess.CalledProcessError as e:
        # Show the actual ffmpeg error line
        err = (e.stderr or "").strip()
        _log(f"[mp3-fail] ffmpeg error: {err[:400]}")
        return None
    except Exception as e:
        _log(f"[mp3-fail] unexpected: {e}")
        return None

def save_audio_result(result: Any, out_path: Path) -> Path:
    """
    Save audio returned by Gradio.

    Handles:
      1) dict with base64 in ["data"]
      2) dict with {"is_file": True, "name": "/tmp/..."} => download via /file=
      3) local filesystem path (string)

    Rejects ws:// URLs.
    """

    # Case 1: base64 audio dict
    if isinstance(result, dict) and "data" in result and result.get("data"):
        b64 = result["data"]
        if isinstance(b64, str) and "base64," in b64:
            b64 = b64.split("base64,", 1)[1]
        audio_bytes = base64.b64decode(b64)
        out_path.write_bytes(audio_bytes)
        return out_path

    # Case 2: file reference dict (common with /run/predict)
    if isinstance(result, dict) and result.get("is_file") and result.get("name"):
        server_path = str(result["name"])

        # If it somehow is already a local file, just copy it
        p = Path(server_path)
        if p.exists():
            out_path.write_bytes(p.read_bytes())
            return out_path

        # Otherwise download from the Gradio app
        # Gradio exposes files via:  <base_url>/file=<server_path>
        file_url = SPACE_ID.rstrip("/") + "/file=" + quote(server_path, safe="")
        r = requests.get(file_url, timeout=300)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return out_path

    # Case 3: string → must be a LOCAL FILE PATH
    if isinstance(result, str):
        lowered = result.lower()
        if lowered.startswith(("ws://", "wss://")):
            raise RuntimeError(f"Refusing websocket URL for audio: {result}")
        if lowered.startswith(("http://", "https://")):
            # If you ever get https URLs, we can add safe downloading here too.
            raise RuntimeError(f"Got http(s) URL for audio; not handled yet: {result}")

        p = Path(result)
        if not p.exists():
            raise RuntimeError(f"Gradio returned a string but it is not a local file: {result}")

        out_path.write_bytes(p.read_bytes())
        return out_path

    raise RuntimeError(f"Unexpected audio return type: {type(result)} -> {result}")

def gradio_http_predict(base_url: str, fn_index: int, data: list, timeout: int = 300) -> Any:
    """
    Call a Gradio app over HTTP, bypassing websocket queueing.

    Tries /run/predict first (common), then /api/predict.
    Returns the first element of response["data"] if present.
    """
    base = base_url.rstrip("/")
    endpoints = ["/run/predict", "/api/predict", "/api/predict/"]

    payload = {"data": data, "fn_index": fn_index}
    last_err = None

    for ep in endpoints:
        try:
            r = requests.post(base + ep, json=payload, timeout=timeout)
            r.raise_for_status()
            j = r.json()

            # Gradio commonly returns {"data":[...], "is_generating":..., "duration":...}
            if isinstance(j, dict) and "data" in j:
                d = j["data"]
                if isinstance(d, list) and len(d) > 0:
                    return d[0]
                return d

            # Some variants wrap differently
            if isinstance(j, dict) and "output" in j:
                return j["output"]

            return j
        except Exception as e:
            last_err = e

    raise RuntimeError(f"HTTP Gradio call failed on all endpoints: {last_err}")

def tts_generate_with_retries(
    client: Client,
    fn_index: int,
    text: str,
    ipa_input: bool,
    speed: float,
    out_base: Path,
    pbar: Optional[tqdm] = None,
) -> Tuple[Path, str]:
    """
    Returns (audio_path, audio_filename) on success.
    Re-uses cached audio if present.
    """
    def _log(msg: str):
        if pbar is not None:
            pbar.write(msg)
        else:
            print(msg)

    wav_path = out_base.with_suffix(".wav")
    mp3_path = out_base.with_suffix(".mp3")

    if mp3_path.exists():
        return mp3_path, mp3_path.name
    if wav_path.exists():
        return wav_path, wav_path.name

    last_err = None
    for attempt in range(1, MAX_ATTEMPTS_PER_ROW + 1):
        try:
            # Call the Audio endpoint (your API dump shows fn_index=1 returns Audio)
            result = gradio_http_predict(SPACE_ID, fn_index, [text, ipa_input, speed], timeout=300)

            # If we got here, predict() returned something (debug it)
            _log(f"[DEBUG] predict() return type={type(result)} value={str(result)[:200]}")

            saved = save_audio_result(result, wav_path)

            # Check if the audio is empty.
            if not audio_nonempty(Path(saved)):
                raise RuntimeError(f"Empty or too-small audio file: {saved}")

            mp3 = wav_to_mp3(Path(saved), mp3_path, pbar=pbar)
            if mp3 and mp3.exists():
                # 🔒 NEW: sanity-check MP3
                if not audio_nonempty(mp3):
                    raise RuntimeError(f"Empty MP3 after conversion: {mp3}")

                if not KEEP_WAV:
                    try:
                        wav_path.unlink(missing_ok=True)
                    except Exception:
                        pass

                return mp3, mp3.name

                # Fallback: WAV only (already checked above)
                return Path(saved), Path(saved).name
                _log(f"[DEBUG] saved audio to: {saved} exists={Path(saved).exists()}")

        except Exception as e:
            last_err = e
            _log(f"[DEBUG] attempt {attempt} failed: {repr(e)}")
            if attempt < MAX_ATTEMPTS_PER_ROW:
                sleep_backoff(attempt)

    raise RuntimeError(f"TTS failed after {MAX_ATTEMPTS_PER_ROW} attempts: {last_err}")

# =========================
# BUILD ANKI DECK
# =========================
def build_anki(deck_name: str, notes: list, media_files: list, out_apkg: Path) -> None:
    model = genanki.Model(
    MODEL_ID,
    "ShanghaineseTTSModel",
    fields=[
        {"name": "Shanghainese"},
        {"name": "Romanisation"},
        {"name": "Mandarin"},
        {"name": "Meaning"},
        {"name": "Audio"},
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": (
                "<div style='font-size:28px'>{{Shanghainese}}{{Audio}}</div>"
                "<div style='margin-top:8px; font-size:20px'>{{Romanisation}}</div>"
            ),
            "afmt": (
                "{{FrontSide}}<hr>"
                "<div style='font-size:22px'>{{Mandarin}}</div>"
                "<div style='margin-top:6px;'>{{Meaning}}</div>"
                "<br>{{Audio}}"
            ),
        }
    ],
    css="""
/* ZHAB deck styling */
.card {
  font-size: 20px;
  line-height: 1.35;
  text-align: left;
  font-family:
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "Noto Sans CJK JP",
    "Noto Sans SC",
    "PingFang SC",
    "Hiragino Sans GB",
    "Microsoft YaHei",
    "SimSun",
    "Arial Unicode MS",
    sans-serif;
}
hr { opacity: 0.25; }
"""
)


    deck = genanki.Deck(DECK_ID, deck_name)

    for (text, roman, mandarin, meaning, audio_filename) in notes:
        audio_tag = f"[sound:{audio_filename}]"
        deck.add_note(genanki.Note(model=model, fields=[text, roman, mandarin, meaning, audio_tag]))

    pkg = genanki.Package(deck)
    pkg.media_files = media_files
    pkg.write_to_file(str(out_apkg))


# =========================
# PIPELINE
# =========================
def process_rows(
    df: pd.DataFrame,
    client: Client,
    fn_index: int,
    romap: Dict[str, set],
    maxlen: int,
    state: Dict[str, Any],
    pbar: tqdm,
) -> Tuple[int, int]:
    successes = 0
    failures = 0

    for _, row in df.iterrows():
        pbar.update(1)

        raw_text = sanitize_text(row.get("shanghainese", ""))
        if not raw_text:
            continue

        text = undouble_text(raw_text)
        mandarin = undouble_text(sanitize_text(row.get("mandarin", "")))
        meaning = sanitize_text(row.get("meaning", ""))

        ipa_input = str(row.get("ipa_input", "TRUE")).strip().upper() in ("TRUE", "1", "YES", "Y")
        try:
            speed = float(row.get("speed", 1.0) or 1.0)
        except ValueError:
            speed = 1.0

        key = row_key(text, ipa_input, speed)
        if key in state["done"]:
            continue

        roman = romanise_sentence(text, romap, maxlen)
        roman = postprocess_romanisation(roman)
        base = AUDIO_DIR / safe_audio_basename(key)

        try:
            audio_path, audio_filename = tts_generate_with_retries(
                client=client,
                fn_index=fn_index,
                text=text,
                ipa_input=ipa_input,
                speed=speed,
                out_base=base,
                pbar=pbar,  # <-- so debug shows up
            )

            state["done"][key] = {
                "text": text,
                "romanisation": roman,
                "mandarin": mandarin,
                "meaning": meaning,
                "ipa_input": ipa_input,
                "speed": speed,
                "audio_path": str(audio_path),
                "audio_filename": audio_filename,
                "updated": time.time(),
            }
            save_state(state)
            successes += 1

            log_success(pbar, successes, text, roman, mandarin, audio_filename)

        except Exception as e:
            failures += 1
            append_failed_row(
                {
                    "key": key,
                    "id": row.get("id", ""),
                    "shanghainese": text,
                    "mandarin": mandarin,
                    "meaning": meaning,
                    "ipa_input": ipa_input,
                    "speed": speed,
                    "error": str(e),
                    "attempts": MAX_ATTEMPTS_PER_ROW,
                }
            )

        if RETRY_FAILED_EVERY_N_SUCCESSES and successes > 0 and successes % RETRY_FAILED_EVERY_N_SUCCESSES == 0:
            retry_failed_queue(client, fn_index, romap, maxlen, state, pbar)

    return successes, failures


def retry_failed_queue(
    client: Client,
    fn_index: int,
    romap: Dict[str, set],
    maxlen: int,
    state: Dict[str, Any],
    pbar: tqdm,
) -> None:
    failed_df = load_failed_df()
    if len(failed_df) == 0:
        return

    remaining = []
    for _, r in failed_df.iterrows():
        key = str(r.get("key", "")).strip()
        if not key:
            continue
        if key in state["done"]:
            continue

        text = undouble_text(sanitize_text(r.get("shanghainese", "")))
        if not text:
            continue

        mandarin = undouble_text(sanitize_text(r.get("mandarin", "")))
        meaning = sanitize_text(r.get("meaning", ""))

        ipa_input = str(r.get("ipa_input", "TRUE")).strip().upper() in ("TRUE", "1", "YES", "Y")
        try:
            speed = float(r.get("speed", 1.0) or 1.0)
        except ValueError:
            speed = 1.0

        roman = romanise_sentence(text, romap, maxlen)
        base = AUDIO_DIR / safe_audio_basename(key)

        try:
            audio_path, audio_filename = tts_generate_with_retries(
                client=client,
                fn_index=fn_index,
                text=text,
                ipa_input=ipa_input,
                speed=speed,
                out_base=base,
            )

            state["done"][key] = {
                "text": text,
                "romanisation": roman,
                "mandarin": mandarin,
                "meaning": meaning,
                "ipa_input": ipa_input,
                "speed": speed,
                "audio_path": str(audio_path),
                "audio_filename": audio_filename,
                "updated": time.time(),
            }
            save_state(state)

            log_success(pbar, len(state["done"]), text, roman, mandarin, audio_filename)

        except Exception as e:
            remaining.append(
                {
                    "key": key,
                    "id": r.get("id", ""),
                    "shanghainese": text,
                    "mandarin": mandarin,
                    "meaning": meaning,
                    "ipa_input": ipa_input,
                    "speed": speed,
                    "error": str(e),
                    "attempts": int(r.get("attempts", 0)) + MAX_ATTEMPTS_PER_ROW,
                }
            )

    rewrite_failed_df(pd.DataFrame(remaining))


def main(csv_path: str = "sentences.csv", deck_name: str = "Shanghainese TTS (HF)"):
    ensure_dir(OUT_DIR)
    ensure_dir(AUDIO_DIR)

    df = pd.read_csv(csv_path, encoding="utf-8-sig").fillna("")
    if "shanghainese" not in df.columns:
        raise SystemExit("CSV must contain a 'shanghainese' column.")

    romap, maxlen = load_rime_dict(cache_path=RIME_CACHE_PATH)
    state = load_state()

    ensure_dir(OUT_DIR / "gradio_tmp")

    client = Client(
    SPACE_ID,
    verbose=False,
    download_files=str(OUT_DIR / "gradio_tmp"),
    httpx_kwargs={"timeout": 300, "follow_redirects": True},
)

    fn_index = detect_fn_index(client)
    print(f"Using HF Space fn_index={fn_index} for audio endpoint.")

    with tqdm(total=len(df), desc="Generating", unit="row", dynamic_ncols=True) as pbar:
        new_ok, new_bad = process_rows(df, client, fn_index, romap, maxlen, state, pbar)

        for p in range(1, FAILED_QUEUE_PASSES + 1):
            if len(load_failed_df()) == 0:
                break
            pbar.write(f"Retry pass {p}/{FAILED_QUEUE_PASSES} over failed queue…")
            retry_failed_queue(client, fn_index, romap, maxlen, state, pbar)
            if len(load_failed_df()) > 0 and p < FAILED_QUEUE_PASSES:
                time.sleep(FAILED_PASS_PAUSE_SECONDS)

    notes = []
    media_files = []
    for _, rec in state["done"].items():
        text = rec["text"]
        roman = rec.get("romanisation", "")
        mandarin = rec.get("mandarin", "")
        meaning = rec.get("meaning", "")
        audio_path = Path(rec["audio_path"])
        audio_filename = rec["audio_filename"]

        notes.append((text, roman, mandarin, meaning, audio_filename))
        if audio_path.exists():
            media_files.append(str(audio_path))

    out_apkg = OUT_DIR / f"{deck_name}.apkg"
    build_anki(deck_name, notes, media_files, out_apkg)

    print(f"\nSpace: {SPACE_ID}")
    print(f"fn_index used: {fn_index}")
    print(f"New successes this run: {new_ok}")
    print(f"New failures this run: {new_bad}")
    print(f"Total cards in package: {len(notes)}")
    print(f"Wrote: {out_apkg}")
    print(f"Remaining failed rows: {len(load_failed_df())}")
    print(f"Romanisation dict cached at: {RIME_CACHE_PATH}")


if __name__ == "__main__":
    main()
