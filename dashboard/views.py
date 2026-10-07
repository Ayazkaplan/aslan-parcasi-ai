import json
import logging
import os
import base64
import datetime as dt
import io
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import requests
from collections import namedtuple
from html import unescape
from urllib.parse import parse_qs, quote, unquote, urlparse
from zoneinfo import ZoneInfo
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST
from openai import OpenAI
from .forms import CustomUserCreationForm, EmailOrUsernameAuthenticationForm
from .models import AppClock, ChatHistory, UserProfile

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


MAX_CHAT_FILE_BYTES = 50 * 1024 * 1024
MAX_CHAT_IMAGE_BYTES = 3 * 1024 * 1024
MAX_IMAGE_EDIT_BYTES = 20 * 1024 * 1024
# Büyük dosya dökümleri yanıtı dakikalarca geciktirir; bu yüzden bilinçli olarak küçük.
MAX_EXTRACTED_TEXT = 24_000
IMAGE_REQUEST_MIN_INTERVAL = 8
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_FAST_MODEL = "gemini-2.5-flash-lite"
GEMINI_IMAGE_MODEL = "gemini-3-pro-image"
GEMINI_IMAGE_FALLBACK_MODELS = (
  "gemini-3.1-flash-image",
  "gemini-2.5-flash-image",
)
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
REASONING_MARKER = "\x00R\x00"

Endpoint = namedtuple("Endpoint", "provider key client models fallback")

# Free-tier providers, in priority order. Every endpoint receives the same
# personality prompt, temperature and max_tokens, so the voice never changes
# when the rotation moves to another provider.
#
# "chat": False keeps a provider out of the plain-text rotation (its quota is
# reserved for something else, e.g. Google for image generation and speech to
# text). "vision" lists the models that actually accept an image part; a
# provider without it is skipped when the user sends a photo.
PROVIDER_SPECS = (
    {
        "name": "groq",
        "env": "GROQ_API_KEY",
        "base": "https://api.groq.com/openai/v1",
        "normal": ("qwen/qwen3.8-27b", "openai/gpt-oss-120b"),
        "fast": ("qwen/qwen3.8-27b", "openai/gpt-oss-20b"),
        "vision": ("qwen/qwen3.8-27b",),
    },
    {
        "name": "mistral",
        "env": "MISTRAL_API_KEY",
        "base": "https://api.mistral.ai/v1",
        "normal": ("ministral-14b-latest", "ministral-8b-latest", "mistral-small-latest"),
        "fast": ("ministral-8b-latest", "ministral-3b-latest"),
        "vision": ("ministral-14b-latest", "ministral-8b-latest"),
    },
    {
        "name": "openrouter",
        "env": "OPENROUTER_API_KEY",
        "base": "https://openrouter.ai/api/v1",
        "normal": (
            "qwen/qwen3.8-27b:free",
            "nvidia/nemotron-3.5-lightning:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
        ),
        "fast": ("qwen/qwen3.8-27b:free", "nvidia/nemotron-3.5-lightning:free"),
        "vision": ("qwen/qwen3.8-27b:free", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"),
    },
    {
        "name": "nvidia",
        "env": "NVIDIA_API_KEY",
        "base": "https://integrate.api.nvidia.com/v1",
        "chat": False,
        "normal": ("meta/llama-3.2-11b-vision-instruct",),
        "fast": ("meta/llama-3.2-11b-vision-instruct",),
        "vision": ("meta/llama-3.2-11b-vision-instruct", "meta/llama-3.2-90b-vision-instruct"),
    },
    {
        # Google kotası görsel üretimi, ses çevirisi ve görsel anlama için saklı;
        # düz metin sohbeti diğer sağlayıcılardan döner.
        "name": "gemini",
        "env": "GEMINI_API_KEY",
        "base": GEMINI_BASE_URL,
        "chat": False,
        "normal": (GEMINI_MODEL,),
        "fast": (GEMINI_FAST_MODEL,),
        "vision": (GEMINI_MODEL, GEMINI_FAST_MODEL),
    },
)
_image_request_lock = threading.Lock()
_last_image_request_at = {}
# Görsel üretimi Google kotasını kullanır; tek bir kullanıcının herkesin hakkını
# bitirmemesi için günlük bir üst sınır var.
IMAGE_DAILY_LIMIT = 20
_image_daily_usage = {}

logger = logging.getLogger(__name__)

MODEL_BACKOFFS = (1, 6, 15)
QUOTA_STATE = {
    "last_error": "",
    "last_status": None,
    "last_at": "",
    "last_kind": "",
}

DAILY_QUOTA_MESSAGE = (
    "Aslan Parçası'nın günlük ücretsiz kullanım hakkı şu an doldu; bu yüzden yanıt "
    "üretemiyorum. Hak, Türkiye saatiyle sabah 10:00 civarında kendiliğinden yenilenir; "
    "o saatten sonra aynı soruyu sorabilirsin. Bu arada saat, hava durumu ve internet "
    "araması sorularına topladığım gerçek verilerle cevap verebiliyorum."
)
DAILY_QUOTA_FALLBACK_NOTE = (
    "(Not: günlük ücretsiz kullanım hakkı şu an dolu olduğu için yanıtı model yazamadı; "
    "yukarıdaki gerçek verileri doğrudan sundum. Hak Türkiye saatiyle ~10:00'da yenilenir.)"
)
BUSY_FALLBACK_NOTE = (
    "(Not: yanıt modeli şu an aşırı yoğun olduğu için topladığım canlı verileri doğrudan sundum.)"
)
EMPTY_ANSWER_MESSAGE = (
    "Aslan Parçası bu isteğe boş yanıt döndürdü; yedek beyinleri de denedim ama metin "
    "gelmedi. Mesajını bir kez daha gönderir misin? Uzun dosya veya görsel eklediysen "
    "kısaltıp denemek genelde çözer."
)
IMAGE_UNAVAILABLE_MESSAGE = (
  "Görsel oluşturma servislerim şu an bu isteği tamamlayamadı. "
    "Birkaç dakika sonra tekrar dene, büyük ihtimalle düzelir. Bu arada sohbet, hava "
    "durumu ve internet araştırması özelliklerim çalışmaya devam ediyor."
)

HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-dev")
_TURKISH_PROMPT_MAP = {
    "kedi": "cat",
    "köpek": "dog",
    "daire": "circle",
    "çember": "circle",
    "kare": "square",
    "dikdörtgen": "rectangle",
    "üçgen": "triangle",
    "kırmızı": "red",
    "kirmizi": "red",
    "mavi": "blue",
    "yeşil": "green",
    "yesil": "green",
    "sarı": "yellow",
    "sari": "yellow",
    "siyah": "black",
    "beyaz": "white",
    "uçak": "airplane",
    "araba": "car",
    "şehir": "city",
    "sehir": "city",
    "kadın": "woman",
    "erkek": "man",
    "çocuk": "child",
    "insan": "person",
    "orman": "forest",
    "deniz": "sea",
    "güneş": "sun",
    "gunes": "sun",
    "kahve": "coffee",
    "yemek": "food",
    "köy": "village",
    "ev": "house",
    "oda": "room",
    "masa": "table",
    "mousepad": "mouse pad",
    "mouse pad": "mouse pad",
    "mouse mat": "mouse pad",
    "fare altlığı": "mouse pad",
    "fare altligi": "mouse pad",
    "kız": "girl",
    "erkek çocuk": "boy",
    "otobüs": "bus",
    "tren": "train",
    "gökyüzü": "sky",
    "gokyuzu": "sky",
    "fotoğraf": "photo",
    "fotograf": "photo",
    "gerçekçi": "photorealistic",
    "fotogerçekçi": "photorealistic",
    "gercekci": "photorealistic",
    "açık": "open",
    "kapalı": "closed",
    "güzel": "beautiful",
    "guzel": "beautiful",
    "kötü": "bad",
    "kotu": "bad",
}

GEOMETRIC_SHAPE_RE = re.compile(
    r"\b(circle|square|rectangle|triangle|ellipse|oval|line|dot|shape|"
    r"daire|çember|kare|dikdörtgen|ucgen|üçgen|oval|çizgi|nokta|şekil|sekil)\b",
    re.I,
)
OTHER_SUBJECT_RE = re.compile(
    r"\b(person|woman|man|girl|boy|child|people|cat|dog|bird|horse|car|house|"
    r"tree|flower|face|human|kadın|erkek|çocuk|insan|kedi|köpek|kuş|at|araba|"
    r"ev|ağaç|çiçek|yüz)\b",
    re.I,
)
SIMPLE_GEOMETRIC_SHAPES = {
    "circle": "circle",
    "daire": "circle",
    "çember": "circle",
    "square": "square",
    "kare": "square",
    "rectangle": "rectangle",
    "dikdörtgen": "rectangle",
    "triangle": "triangle",
    "üçgen": "triangle",
    "oval": "oval",
    "ellipse": "oval",
}
IMAGE_COLOR_HEX = {
    "red": "#ff0000",
    "kırmızı": "#ff0000",
    "blue": "#0000ff",
    "mavi": "#0000ff",
    "green": "#008000",
    "yeşil": "#008000",
    "yellow": "#ffff00",
    "sarı": "#ffff00",
    "black": "#000000",
    "siyah": "#000000",
    "white": "#ffffff",
    "beyaz": "#ffffff",
    "orange": "#ff8c00",
    "turuncu": "#ff8c00",
    "purple": "#800080",
    "mor": "#800080",
    "pink": "#ff69b4",
    "pembe": "#ff69b4",
}
IMAGE_COLOR_RE = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, IMAGE_COLOR_HEX), key=len, reverse=True)) + r")\b",
    re.I,
)
IMAGE_BACKGROUND_PATTERNS = (
    re.compile(
        r"(?:arka\s+plan(?:ı|i|u|ü)?|zemin)(?:\s+rengi)?\s*"
        r"(?:(?:olan|rengi)\s*)?(?P<color>" +
        "|".join(sorted(map(re.escape, IMAGE_COLOR_HEX), key=len, reverse=True)) +
        r")",
        re.I,
    ),
    re.compile(
        r"(?:arka\s+plan(?:ı|i|u|ü)?|zemin)(?:\s+rengi)?\s*"
        r"(?P<color>" +
        "|".join(sorted(map(re.escape, IMAGE_COLOR_HEX), key=len, reverse=True)) +
        r")\s*(?:olan|renkli)?",
        re.I,
    ),
    re.compile(
        r"(?P<color>" +
        "|".join(sorted(map(re.escape, IMAGE_COLOR_HEX), key=len, reverse=True)) +
        r")\s+(?:renkli\s+)?(?:arka\s+plan(?:ı|i|u|ü)?|zemin)",
        re.I,
    ),
    re.compile(
        r"background(?:\s+color)?\s*(?:is|:|=)?\s*(?P<color>" +
        "|".join(sorted(map(re.escape, IMAGE_COLOR_HEX), key=len, reverse=True)) +
        r")",
        re.I,
    ),
    re.compile(
        r"(?P<color>" +
        "|".join(sorted(map(re.escape, IMAGE_COLOR_HEX), key=len, reverse=True)) +
        r")\s+background",
        re.I,
    ),
)


def is_simple_geometric_prompt(prompt):
  """Keep a minimal shape request literal instead of adding photographic subjects."""
  return bool(
      GEOMETRIC_SHAPE_RE.search(prompt or "")
      and not OTHER_SUBJECT_RE.search(prompt or "")
      and not re.search(
          r"\b(photo(?:graph)?|photorealistic|realistic|watercolor|oil painting|"
          r"anime|manga|cartoon|logo|poster|fotoğraf|fotogerçekçi|suluboya|"
          r"yağlı boya|karikatür|afiş)\b",
          prompt or "",
          re.I,
      )
  )


def generate_simple_geometric_image(prompt):
  """Render a single, plain geometric shape exactly instead of sampling an image model."""
  text = (prompt or "").strip()
  shape_matches = list(re.finditer(
      r"\b(circle|square|rectangle|triangle|ellipse|oval|daire|çember|kare|"
      r"dikdörtgen|üçgen)\b",
      text,
      re.I,
  ))
  if (
      not is_simple_geometric_prompt(prompt)
      or len(shape_matches) != 1
      or re.search(r"\b(two|three|four|iki|üç|dört|birkaç|multiple|several)\b", text, re.I)
  ):
    return None

  shape = SIMPLE_GEOMETRIC_SHAPES.get(shape_matches[0].group(0).casefold())
  colors = list(IMAGE_COLOR_RE.finditer(text))
  background_match = next(
      (match for pattern in IMAGE_BACKGROUND_PATTERNS
       if (match := pattern.search(text))),
      None,
  )
  background_color_name = (
      background_match.group("color") if background_match else None
  )
  if background_color_name:
    colors = [
        match for match in colors
        if match.group(0).casefold() != background_color_name.casefold()
        or not (
            background_match.start() <= match.start() < background_match.end()
        )
    ]
  shape_color = min(colors, key=lambda match: abs(match.start() - shape_matches[0].start())) if colors else None
  fill_color = (
      IMAGE_COLOR_HEX[shape_color.group(0).casefold()]
      if shape_color
      else "#000000"
  )
  background_color = "#ffffff"
  if background_match:
    background_color = IMAGE_COLOR_HEX[background_color_name.casefold()]

  elements = []
  if not re.search(r"\btransparent\b|şeffaf", text, re.I):
    elements.append(f'<rect width="4096" height="4096" fill="{background_color}"/>')
  if shape == "circle":
    elements.append(f'<circle cx="2048" cy="2048" r="1200" fill="{fill_color}"/>')
  elif shape == "square":
    elements.append(f'<rect x="848" y="848" width="2400" height="2400" fill="{fill_color}"/>')
  elif shape == "rectangle":
    elements.append(f'<rect x="528" y="1248" width="3040" height="1600" fill="{fill_color}"/>')
  elif shape == "triangle":
    elements.append(f'<path d="M2048 680 3520 3280H576Z" fill="{fill_color}"/>')
  elif shape == "oval":
    elements.append(f'<ellipse cx="2048" cy="2048" rx="1440" ry="960" fill="{fill_color}"/>')
  else:
    return None
  svg = (
      '<svg xmlns="http://www.w3.org/2000/svg" width="4096" height="4096" '
      'viewBox="0 0 4096 4096">' + "".join(elements) + "</svg>"
  )
  encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
  return f"data:image/svg+xml;base64,{encoded}"


def translate_prompt_to_english(prompt):
  """Translate the most common Turkish prompts to English, then enrich them."""
  text = (prompt or "").strip()
  if not text:
    return ""
  lowered = text.lower().strip()
  if not re.search(r"[çğıöşüÇĞİÖŞÜ]/|[a-zA-Z]", text):
    return text
  if not re.search(r"[çğıöşüÇĞİÖŞÜ]", text):
    return text

  translated = text
  for source, target in sorted(_TURKISH_PROMPT_MAP.items(), key=lambda item: len(item[0]), reverse=True):
    translated = re.sub(rf"\b{re.escape(source)}\b", target, translated, flags=re.IGNORECASE)
    translated = translated.replace(source.title(), target.title())

  translated = re.sub(r"\s+", " ", translated).strip()
  if translated.lower() == lowered:
    translated = f"{text}"
  return translated


def enhance_image_prompt(prompt):
  """Turn a normal user prompt into a stable, polished image request."""
  raw = (prompt or "").strip()
  if not raw:
    return ""
  english = translate_prompt_to_english(raw)
  english = re.sub(r"\s+", " ", english).strip()
  topic = english.strip("., ")
  if not topic:
    return ""
  if re.search(
      r"\b(anime|manga|cartoon|illustration|drawing|sketch|logo|icon|sticker|pixel art|watercolor|oil painting)\b|"
      r"(çizim|karikatür|illüstrasyon|suluboya|yağlı boya|çıkartma)",
      raw,
      re.IGNORECASE,
  ):
    return topic
  if is_simple_geometric_prompt(raw):
    return (
        "A clean, simple, flat geometric illustration. Show only the exact shape, "
        "color, and count requested in this brief. Center it on a plain uncluttered "
        "background. No people, faces, animals, objects, text, or extra shapes. "
        f"Exact user brief: {topic}"
    )

  if re.search(r"\b(mouse\s*pad|mouse\s*mat|mousepad|fare\s+altl[ıi]ğ[ıi])\b", raw, re.I):
    return (
        "Ultra-photorealistic native 4K UHD studio product photo. Main subject: exactly "
        "one empty, flat rectangular mousepad (the desk pad only, not a computer mouse). "
        "Show its complete horizontal shape, straight parallel edges, softly rounded "
        "corners, realistic proportions, fine woven-cloth surface, precise stitched "
        "perimeter, and thin rubber base visible only along the edge. The pad lies flat "
        "and level, centered on a bright light-gray tabletop; use clean, even softbox "
        "lighting, balanced exposure, crisp fabric detail, and a natural contact shadow. "
        "Keep the pad geometrically straight and undistorted. Exclude the computer mouse "
        "device, hands, keyboard, other electronics, text, logos, packaging, extra objects, "
        "and watermarks. Do not make the image dark or underexposed. Match the requested "
        "color and design exactly. "
        f"Exact user brief: {topic}"
    )

  if re.search(
      r"\b(brand(?:ed)?|bottle|can|soda|cola|şişe\w*|kutu\w*|ambalaj\w*|"
      r"markalı|markali|packaging|label)\b",
      raw,
      re.IGNORECASE,
  ):
    topic = f"{topic}, clearly visible branded product packaging, readable label, actual bottle/can design, not a generic cup, centered in frame"

  if re.search(
      r"\b(brand(?:ed)?|bottle|can|soda|cola|şişe\w*|kutu\w*|ambalaj\w*|"
      r"markalı|markali|packaging|label)\b",
      raw,
      re.IGNORECASE,
  ):
    topic = f"{topic}, clearly visible branded product packaging, readable label, actual bottle/can design, not a generic cup, centered in frame"

  descriptors = [
      "ultra-photorealistic 4K UHD photograph",
      "crisp lifelike detail",
      "sharp focus",
      "physically accurate shape and proportions",
      "natural studio lighting",
      "realistic texture and contact shadows",
      "professional product-photography composition",
      "no warping, distortion, or unrelated objects",
  ]
  # The user may already have a rich prompt; avoid repeating the same style tokens.
  detail_prefix = ""
  if not any(token.lower() in topic.lower() for token in ["photorealistic", "realistic", "cinematic", "8k", "detailed"]):
    detail_prefix = ", ".join(descriptors)
  return f"{topic}, {detail_prefix}".strip(", ")


def build_flux_image_params(prompt, seed=None, width=1024, height=1024):
  """Use a safe Flux-ready payload: explicit prompt, fixed resolution and encoding."""
  refined_prompt = enhance_image_prompt(prompt)
  resolved_seed = seed if seed is not None else random.randint(1, 999_999_999)
  return {
      "prompt": refined_prompt,
      "width": int(width),
      "height": int(height),
      "seed": int(resolved_seed),
      "model": "flux",
      "nologo": "true",
      "safe": "true",
      "enhance": "false",
      "aspect_ratio": "1:1",
  }


def compute_deep_think_deadline(seconds):
  """Return the strict wall-clock deadline for the deep-think session."""
  total = max(30, min(1800, int(seconds or 300)))
  return time.monotonic() + total


class ImageUnavailableError(RuntimeError):
    """Every image provider failed; the message is safe to show the user."""


class QuotaBreakerError(RuntimeError):
    """Raised fast while the quota circuit breaker is open."""


class DailyQuotaError(RuntimeError):
    """The daily allowance of one API key is gone; try another key."""


class KeyRateLimitedError(RuntimeError):
    """This key hit a per-minute limit; another key can serve the request."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


class KeyRejectedError(RuntimeError):
    """The key itself is invalid or not allowed to use this feature."""


class ModelUnavailableError(RuntimeError):
    """No configured model answered for this key."""


class UpstreamBusyError(RuntimeError):
    """5xx / overloaded / timeout: worth trying the next key."""


class UpstreamHTTPError(RuntimeError):
    """A raw HTTP failure from the model service, keeping the status code."""

    def __init__(self, status_code, body):
        super().__init__(f"Error code: {status_code} - {body}")
        self.status_code = status_code


def is_daily_quota_error(error):
  """True when the provider reports the daily allowance is exhausted."""
  text = str(error).lower()
  return "exceeded your current quota" in text or "plan and billing" in text


def record_quota_error(error, kind):
  QUOTA_STATE["last_error"] = str(error)[:300]
  QUOTA_STATE["last_status"] = getattr(error, "status_code", None)
  QUOTA_STATE["last_at"] = timezone.now().isoformat()
  QUOTA_STATE["last_kind"] = kind
  logger.warning("Aslan model quota error (%s): %s", kind, str(error)[:300])


KEY_STATE = {}
_KEY_LOCK = threading.Lock()
_CLIENT_CACHE = {}
_rr_counter = 0


def _key_state(key):
  state = KEY_STATE.get(key)
  if state is None:
    state = {
        "provider": "",
        "daily_until": 0.0,
        "rate_until": 0.0,
        "invalid": False,
        "calls": 0,
        "last_error": "",
        "last_at": "",
    }
    KEY_STATE[key] = state
  return state


def bind_key_provider(key, provider):
  with _KEY_LOCK:
    _key_state(key)["provider"] = provider


def key_is_available(key):
  state = _key_state(key)
  if state["invalid"]:
    return False
  now = time.monotonic()
  return now >= state["daily_until"] and now >= state["rate_until"]


def seconds_until_daily_reset():
  """The provider refills the free daily allowance at midnight US Pacific."""
  try:
    pacific = ZoneInfo("America/Los_Angeles")
    now_local = dt.datetime.now(dt.timezone.utc).astimezone(pacific)
    reset_local = (now_local + dt.timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return max(60.0, min((reset_local - now_local).total_seconds(), 24 * 3600))
  except Exception:
    return 6 * 3600


def _mark_key(key, error, **updates):
  with _KEY_LOCK:
    state = _key_state(key)
    state.update(updates)
    state["last_error"] = str(error)[:200]
    state["last_at"] = timezone.now().isoformat()
    record_quota_error(error, updates.get("kind", "rate"))


def mark_key_daily(key, error):
  _mark_key(
      key,
      error,
      kind="daily",
      daily_until=time.monotonic() + seconds_until_daily_reset(),
      rate_until=0.0,
  )


def mark_key_rate(key, error, retry_after=None):
  cooldown = max(10.0, min(120.0, float(retry_after or 45)))
  _mark_key(key, error, kind="rate", rate_until=time.monotonic() + cooldown)


def mark_key_invalid(key, error):
  _mark_key(key, error, kind="invalid", invalid=True)


def mark_key_ok(key):
  with _KEY_LOCK:
    state = _key_state(key)
    state["calls"] += 1
    state["rate_until"] = 0.0
    state["daily_until"] = 0.0
    state["invalid"] = False


def ordered_clients(pairs=None):
  """Round-robin the healthy keys first, exhausted ones only as a last resort."""
  global _rr_counter
  items = list(pairs if pairs is not None else get_gemini_clients())
  if not items:
    return []
  with _KEY_LOCK:
    offset = _rr_counter % len(items)
    _rr_counter += 1
  rotated = items[offset:] + items[:offset]
  healthy = [item for item in rotated if key_is_available(item[0])]
  return healthy + [item for item in rotated if not key_is_available(item[0])]


def ordered_endpoints(endpoints):
  """Round-robin healthy provider endpoints first, blocked ones last resort."""
  global _rr_counter
  items = list(endpoints)
  if not items:
    return []
  with _KEY_LOCK:
    offset = _rr_counter % len(items)
    _rr_counter += 1
  rotated = items[offset:] + items[:offset]
  healthy = [item for item in rotated if key_is_available(item.key)]
  return healthy + [item for item in rotated if not key_is_available(item.key)]


def ordered_keys(keys=None):
  pairs = [(key, key) for key in (keys if keys is not None else get_api_keys())]
  return [key for key, _ in ordered_clients(pairs)]


def _env_keys(env):
  raw = (os.environ.get(env) or "").strip()
  return [key.strip() for key in raw.replace(";", ",").split(",") if key.strip()]


def get_api_keys():
  return _env_keys("GEMINI_API_KEY")


def _client_for(base_url, key):
  cache_key = (base_url, key)
  client = _CLIENT_CACHE.get(cache_key)
  if client is None:
    client = OpenAI(base_url=base_url, api_key=key)
    _CLIENT_CACHE[cache_key] = client
  return client


def get_gemini_clients():
  """One cached client per configured key, in the configured order."""
  return [(key, _client_for(GEMINI_BASE_URL, key)) for key in get_api_keys()]


def _build_endpoints(tier="normal", need_vision=False, chat_only=True):
  endpoints = []
  for spec in PROVIDER_SPECS:
    if need_vision:
      models = tuple(spec.get("vision") or ())
      fallback = None
    else:
      if chat_only and not spec.get("chat", True):
        continue
      models = spec["normal"] if tier == "normal" else spec["fast"]
      fallback = spec["fast"][0] if tier == "normal" else None
    if not models:
      continue
    for key in _env_keys(spec["env"]):
      endpoints.append(Endpoint(spec["name"], key, _client_for(spec["base"], key), models, fallback))
  return endpoints


def get_chat_endpoints(tier="normal", need_vision=False):
  """Every configured provider key as an Endpoint, in provider priority order.

  Plain-text chat skips the providers whose quota is reserved for image
  generation and speech, but when nothing else is configured they are used
  anyway instead of leaving the app without a brain.
  """
  endpoints = _build_endpoints(tier, need_vision)
  if not endpoints and not need_vision:
    endpoints = _build_endpoints(tier, False, chat_only=False)
  if not endpoints:
    client = get_gemini_client()
    if client is not None:
      endpoints = [
          Endpoint("gemini", "fallback-key", client, (GEMINI_MODEL,), None),
      ]
  return endpoints


def get_gemini_client():
  clients = get_gemini_clients()
  return clients[0][1] if clients else None


def get_user_profile(user):
  profile, _ = UserProfile.objects.get_or_create(user=user)
  return profile


def get_chat_history(user):
  history, _ = ChatHistory.objects.get_or_create(user=user)
  return history


def sync_app_clock():
  """Persist today's Istanbul date whenever the app is used."""
  current_date = timezone.localdate()
  clock, _ = AppClock.objects.get_or_create(
      singleton=True,
      defaults={"current_date": current_date},
  )
  if clock.current_date != current_date:
    clock.current_date = current_date
    clock.save(update_fields=["current_date", "updated_at"])
  return clock


def _clip_extracted(text):
  """Keep the prompt small and tell the model the dump was shortened."""
  content = text or ""
  clipped = content[:MAX_EXTRACTED_TEXT]
  if len(content) > MAX_EXTRACTED_TEXT:
    clipped += (
        f"\n\n[... dosyanın geri kalanı kısaltıldı, ilk {MAX_EXTRACTED_TEXT} karakter işlendi. "
        "Kullanıcıya dosyanın tamamını değil bu bölümü görebildiğini söyle. ...]"
    )
  return clipped


def extract_uploaded_file_text(file_item):
  """Extract useful text from common uploaded formats before sending to AI."""
  if not isinstance(file_item, dict):
    return ""
  filename = os.path.basename(str(file_item.get("name") or "dosya"))
  name = filename.lower()
  content_type = str(file_item.get("type") or "").lower()
  extension = os.path.splitext(name)[1]
  encoded = file_item.get("base64") or ""
  text_content = file_item.get("text")
  if text_content is not None and str(text_content).strip():
    return _clip_extracted(str(text_content))
  if not encoded:
    if content_type.startswith("text/") or extension in {
        ".txt", ".md", ".json", ".csv", ".py", ".js", ".ts", ".html",
        ".css", ".java", ".c", ".cpp", ".sql", ".xml", ".yaml", ".yml",
        ".log",
    }:
      return ""
    raise ValueError(f"{filename} dosyasının içeriği alınamadı.")
  try:
    raw = base64.b64decode(encoded, validate=True)
  except (ValueError, TypeError) as error:
    raise ValueError(f"{filename} dosyası geçerli bir yükleme değil.") from error

  if len(raw) > MAX_CHAT_FILE_BYTES:
    raise ValueError("Dosya boyutu 50 MB sınırını aşamaz.")
  try:
    declared_size = int(file_item.get("size") or 0)
  except (TypeError, ValueError):
    declared_size = 0
  if declared_size > MAX_CHAT_FILE_BYTES:
    raise ValueError("Dosya boyutu 50 MB sınırını aşamaz.")

  try:
    if content_type.startswith("text/") or extension in {
        ".txt", ".md", ".json", ".csv", ".py", ".js", ".ts", ".html",
        ".css", ".java", ".c", ".cpp", ".sql", ".xml", ".yaml", ".yml",
        ".log",
    }:
      return _clip_extracted(raw.decode("utf-8", errors="replace"))
    if (extension == ".pdf" or content_type == "application/pdf") and PdfReader:
      pages = PdfReader(io.BytesIO(raw)).pages
      extracted = "\n\n".join((page.extract_text() or "") for page in pages)
      if not extracted.strip():
        raise ValueError(
            f"{filename} içinde seçilebilir metin yok. Taranmış PDF'ler şu anda okunamıyor."
        )
      return _clip_extracted(extracted)
    if (extension == ".docx" or content_type.endswith("wordprocessingml.document")) and Document:
      document = Document(io.BytesIO(raw))
      extracted = "\n".join(paragraph.text for paragraph in document.paragraphs)
      if not extracted.strip():
        raise ValueError(f"{filename} içinde okunabilir metin bulunamadı.")
      return _clip_extracted(extracted)
  except ValueError:
    raise
  except Exception as error:
    raise ValueError(f"{filename} dosyası okunamadı: {error}") from error

  raise ValueError(
      f"{filename} dosya türü desteklenmiyor. Metin, PDF ve DOCX dosyası yükleyin."
  )


def extract_image_url(message):
  """Accept the different image shapes returned by OpenRouter models."""
  def get_value(value, key):
    if isinstance(value, dict):
      return value.get(key)
    return getattr(value, key, None)

  images = get_value(message, "images") if message else None
  if images:
    for image in images:
      image_url = get_value(image, "image_url")
      if isinstance(image_url, dict):
        image_url = image_url.get("url")
      elif image_url:
        image_url = getattr(image_url, "url", image_url)
      if image_url:
        return image_url

  content = get_value(message, "content") if message else None
  if isinstance(content, list):
    for part in content:
      if get_value(part, "type") in ("image_url", "image"):
        image_url = get_value(part, "image_url") or get_value(part, "url")
        if isinstance(image_url, dict):
          image_url = image_url.get("url")
        if image_url:
          return image_url
  if isinstance(content, str):
    match = re.search(r"!\[[^\]]*\]\(([^)]+)\)", content)
    if match:
      return match.group(1)
    match = re.search(r"(https?://\S+|data:image/[^\s]+)", content)
    if match:
      return match.group(1).rstrip(").,")
  return None


# Function definitions for AI tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time in Turkey (Europe/Istanbul timezone)",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (e.g., Istanbul, Ankara, Izmir)"
                    },
                    "country": {
                        "type": "string",
                        "description": "Country code (default: TR)",
                        "default": "TR"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def execute_function(function_name, arguments):
    """Execute a function call and return the result"""
    try:
        if function_name == "get_current_time":
            return get_current_time()
        elif function_name == "get_weather":
            return get_weather(arguments.get("city"), arguments.get("country", "TR"))
        elif function_name == "web_search":
            return web_search(arguments.get("query"), arguments.get("num_results", 5))
        else:
            return {"error": f"Unknown function: {function_name}"}
    except Exception as e:
        return {"error": f"Function execution failed: {str(e)}"}


def get_current_time():
    """Get current time in Turkey"""
    sync_app_clock()
    tz = timezone.get_current_timezone()
    now = timezone.now().astimezone(tz)
    return {
        "time": now.strftime("%H:%M"),
        "date": now.strftime("%d.%m.%Y"),
        "day": now.strftime("%A"),
        "timezone": "Europe/Istanbul (UTC+3)"
    }


TURKISH_ASCII_MAP = str.maketrans("çğıöşüÇĞİÖŞÜâîûÂÎÛ", "cgiosuCGIOSUaiuAIU")


def _ascii_fold(text):
    return (text or "").translate(TURKISH_ASCII_MAP)


def _city_candidates(city):
    """Geocoders index districts on their own, so 'balıkesir erdek' -> 'erdek' -> 'balıkesir'."""
    cleaned = re.sub(r"\s+", " ", (city or "").strip(" .,;:!?'’\"")).strip()
    cleaned = re.sub(
        r"^(?:şehir|sehir|city|ben|benim|için|icin)\s+", "", cleaned, flags=re.I
    ).strip()
    cleaned = _strip_locative(cleaned)
    candidates = []
    seen = set()

    def push(value):
        value = (value or "").strip()
        if len(value) < 2:
            return
        marker = _ascii_fold(value).lower()
        if marker in seen:
            return
        seen.add(marker)
        candidates.append(value)

    push(cleaned)
    words = cleaned.split()
    if len(words) > 1:
        for word in reversed(words):
            push(word)
    push(_ascii_fold(cleaned))
    for word in reversed(words):
        push(_ascii_fold(word))
    return candidates


def _rank_geocode_matches(matches, wanted, country):
    """Exact name match wins, then the requested country, then the biggest place."""
    wanted_key = _ascii_fold(wanted).lower()

    def score(result):
        name_key = _ascii_fold(result.get("name") or "").lower()
        return (
            0 if name_key == wanted_key else 1,
            0 if (result.get("country_code") or "").upper() == country else 1,
            -(result.get("population") or 0),
        )

    return sorted(matches, key=score)


def _geocode_city(city, country="TR"):
    """Resolve a Turkish city/district name (any case, with or without suffix)."""
    last_error = None
    for candidate in _city_candidates(city):
        try:
            response = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": candidate, "count": 10, "language": "tr", "format": "json"},
                timeout=10,
            )
            response.raise_for_status()
            matches = response.json().get("results") or []
        except requests.RequestException as error:
            last_error = error
            continue
        if matches:
            return _rank_geocode_matches(matches, candidate, country)[0]
    if last_error is not None:
        raise last_error
    return None


def _fresh_wttr_observation(location, timezone_name, now=None):
    try:
        latitude = float(location["latitude"])
        longitude = float(location["longitude"])
        response = requests.get(
            f"https://wttr.in/{latitude},{longitude}?format=j1",
            headers={**SEARCH_HEADERS, "Accept": "application/json"},
            timeout=5,
        )
        response.raise_for_status()
        current = (response.json().get("current_condition") or [])[0]
        temperature = float(current["temp_C"])
        time_text = str(current.get("observation_time") or "").strip()
        observation_clock = None
        for time_format in ("%I:%M %p", "%H:%M"):
            try:
                observation_clock = dt.datetime.strptime(time_text, time_format).time()
                break
            except ValueError:
                continue
        if observation_clock is None:
            return None

        zone = ZoneInfo(timezone_name or "UTC")
        local_now = now.astimezone(zone) if now is not None else dt.datetime.now(zone)
        observed_at = dt.datetime.combine(local_now.date(), observation_clock, tzinfo=zone)
        if observed_at - local_now > dt.timedelta(minutes=15):
            observed_at -= dt.timedelta(days=1)
        age = local_now - observed_at
        if age < dt.timedelta(0) or age > dt.timedelta(minutes=90):
            return None
        feels_like = current.get("FeelsLikeC")
        return {
            "temperature": temperature,
            "feels_like": float(feels_like) if feels_like not in (None, "") else None,
            "observed_at": observed_at.isoformat(),
            "source": "wttr.in",
            "age_minutes": age.total_seconds() / 60,
        }
    except (requests.RequestException, KeyError, TypeError, ValueError, IndexError):
        return None


def format_weather_answer(weather_data):
    temperature = weather_data.get("temperature")
    city = weather_data.get("city") or "Seçilen şehir"
    if weather_data.get("temperature_conflict"):
        observed = weather_data.get("observed_temperature")
        forecast = weather_data.get("forecast_temperature")
        observed_time = weather_data.get("observed_at") or "güncel gözlem"
        return (
            f"🌤 {city} için kaynaklar uyuşmuyor: {weather_data.get('source')} gözlemi "
            f"{observed:g}°C ({observed_time}), Open-Meteo tahmini {forecast:g}°C. "
            "Bu fark nedeniyle tek bir kesin sıcaklık vermiyorum."
        )

    temperature_text = f"{float(temperature):g}"
    feels_like = weather_data.get("feels_like")
    answer = f"🌤 {city} için güncel sıcaklık {temperature_text}°C"
    if feels_like is not None:
        answer += f", hissedilen {float(feels_like):g}°C"
    description = weather_data.get("description")
    if description:
        answer += f", {description.lower()}"
    answer += "."
    if weather_data.get("source"):
        answer += f" Kaynak: {weather_data['source']}"
    if weather_data.get("observed_at"):
      timestamp = weather_data["observed_at"]
      timezone_name = weather_data.get("timezone")
      if timezone_name and not re.search(r"[+-]\d{2}:?\d{2}$", str(timestamp)):
        timestamp = f"{timestamp} {timezone_name}"
      answer += f" ({timestamp})"
    return answer


def get_weather(city, country="TR"):
    """Get current weather from Open-Meteo without a paid API key."""
    city = (city or "").strip()
    country = (country or "TR").strip().upper()
    if not city:
        return {"error": "Hava durumunu bulmak için bir şehir adı gerekli."}

    try:
        location = _geocode_city(city, country)
        if not location:
            return {
                "not_found": True,
                "error": f"'{city}' adında bir yer bulunamadı.",
            }
        forecast_params = {
          "latitude": location["latitude"],
          "longitude": location["longitude"],
          "current": (
            "temperature_2m,relative_humidity_2m,apparent_temperature,"
            "weather_code,wind_speed_10m"
          ),
          "timezone": "auto",
        }
        with ThreadPoolExecutor(max_workers=2) as executor:
          forecast_future = executor.submit(
            requests.get,
            "https://api.open-meteo.com/v1/forecast",
            params=forecast_params,
            timeout=10,
          )
          observation_future = executor.submit(
            _fresh_wttr_observation,
            location,
            location.get("timezone") or "Europe/Istanbul",
          )
          weather_response = forecast_future.result()
          observation = observation_future.result()
        weather_response.raise_for_status()
        weather_payload = weather_response.json()
        current = weather_payload.get("current") or {}
        weather_descriptions = {
            0: "Açık",
            1: "Çoğunlukla açık",
            2: "Parçalı bulutlu",
            3: "Kapalı",
            45: "Sisli",
            48: "Kırağılı sis",
            51: "Hafif çisenti",
            53: "Çisenti",
            55: "Yoğun çisenti",
            61: "Hafif yağmur",
            63: "Yağmur",
            65: "Kuvvetli yağmur",
            71: "Hafif kar",
            73: "Kar",
            75: "Kuvvetli kar",
            80: "Sağanak",
            81: "Kuvvetli sağanak",
            82: "Şiddetli sağanak",
            95: "Gök gürültülü fırtına",
            96: "Dolu ihtimalli fırtına",
            99: "Kuvvetli dolulu fırtına",
        }
        weather_code = current.get("weather_code")
        place_name = location.get("name") or city
        region = location.get("admin1") or ""
        display_city = (
            f"{place_name}, {region}"
            if region and _ascii_fold(region).lower() != _ascii_fold(place_name).lower()
            else place_name
        )
        forecast_temperature = current.get("temperature_2m")
        temperature = forecast_temperature
        temperature_source = "Open-Meteo tahmini"
        temperature_conflict = False
        if observation is not None:
          temperature = observation["temperature"]
          temperature_source = observation["source"]
          temperature_conflict = (
            forecast_temperature is not None
            and abs(float(forecast_temperature) - temperature) >= 3
          )
        return {
            "city": display_city,
            "country": location.get("country", country),
          "temperature": temperature,
          "feels_like": (
            observation.get("feels_like")
            if observation and observation.get("feels_like") is not None
            else current.get("apparent_temperature")
          ),
            "humidity": current.get("relative_humidity_2m"),
            "description": weather_descriptions.get(weather_code, "Güncel hava durumu"),
            "wind_speed": current.get("wind_speed_10m"),
          "observed_at": (
            observation.get("observed_at") if observation else current.get("time")
          ),
          "source": temperature_source,
          "forecast_temperature": forecast_temperature,
          "observed_temperature": observation.get("temperature") if observation else None,
          "temperature_conflict": temperature_conflict,
          "forecast_source": "Open-Meteo",
          "timezone": weather_payload.get("timezone"),
        }
    except requests.Timeout:
        return {"error": "Weather service timeout"}
    except requests.RequestException as e:
        return {"error": f"Weather service error: {str(e)}"}
    except Exception as e:
        return {"error": f"Weather error: {str(e)}"}


def _strip_tags(fragment):
    text = re.sub(r"<[^>]+>", " ", fragment or "")
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _decode_bing_redirect(url):
    """Bing hides the real target in /ck/a?u=a1<base64url>; the user needs the target."""
    encoded = (parse_qs(urlparse(url).query).get("u") or [""])[0]
    if not encoded.startswith("a1"):
        return ""
    payload = encoded[2:]
    payload += "=" * (-len(payload) % 4)
    try:
        return base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8", "replace")
    except Exception:
        return ""


def _clean_result_url(href):
    """Search engines wrap every hit in a redirect; show the real target instead."""
    url = unescape((href or "").strip())
    if not url:
        return ""
    if url.startswith("//"):
        url = "https:" + url
    if "duckduckgo.com/l/" in url:
        target = (parse_qs(urlparse(url).query).get("uddg") or [""])[0]
        if target:
            return unquote(target)
    if "bing.com/ck/" in url:
        decoded = _decode_bing_redirect(url)
        return decoded if decoded.startswith("http") else ""
    if "bing.com/news/apiclick" in url:
        target = (parse_qs(urlparse(url).query).get("url") or [""])[0]
        if target.startswith("http"):
            return unquote(target)
    return url


def _parse_ddg_results(html_text, limit):
    """Parse DuckDuckGo html/lite result pages with plain regular expressions."""
    results = []
    blocks = re.findall(
        r'<a[^>]+class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        html_text,
        re.S,
    )
    if not blocks:
        blocks = re.findall(
            r'<a[^>]+href="([^"]*)"[^>]*class="result__a"[^>]*>(.*?)</a>',
            html_text,
            re.S,
        )
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html_text, re.S)
    for index, (href, title_html) in enumerate(blocks[:limit]):
        title = _strip_tags(title_html)
        url = _clean_result_url(href)
        if not title and not url:
            continue
        snippet_html = snippets[index] if index < len(snippets) else ""
        results.append({"title": title, "url": url, "snippet": _strip_tags(snippet_html)})
    if not results:
        # lite.duckduckgo.com serves a plain table of follow links.
        for href, title_html in re.findall(
            r'<a[^>]+rel="nofollow"[^>]+href="([^"]*)"[^>]*>(.*?)</a>', html_text, re.S
        ):
            title = _strip_tags(title_html)
            url = _clean_result_url(href)
            if title and url:
                results.append({"title": title, "url": url, "snippet": ""})
            if len(results) >= limit:
                break
    return results


def _parse_bing_results(html_text, limit):
    """Backup engine when DuckDuckGo blocks the server."""
    results = []
    for block in re.findall(r'<li class="b_algo".*?</li>', html_text, re.S):
        match = re.search(r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not match:
            continue
        snippet_match = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
        url = _clean_result_url(match.group(1))
        title = _strip_tags(match.group(2))
        if not title or not url:
            continue
        results.append({
            "title": title,
            "url": url,
            "snippet": _strip_tags(snippet_match.group(1)) if snippet_match else "",
        })
        if len(results) >= limit:
            break
    return results


def _parse_google_news_rss(xml_text, limit):
    """Keep article links so search findings can be opened and verified."""
    results = []
    for item in re.findall(r"<item>(.*?)</item>", xml_text or "", re.S):
        title_match = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", item, re.S)
        link_match = re.search(r"<link>\s*(\S+)\s*</link>", item, re.S)
        if not title_match or not link_match:
            continue
        title = _strip_tags(title_match.group(1))
        if not title:
            continue
        source_match = re.search(r"<source[^>]*>(.*?)</source>", item, re.S)
        source_url_match = (
            re.search(r"<source[^>]*\burl=[\"']([^\"']+)[\"']", item, re.S)
            if source_match
            else None
        )
        date_match = re.search(r"<pubDate>(.*?)</pubDate>", item, re.S)
        published = _rss_date(_strip_tags(date_match.group(1)) if date_match else "")
        meta = " · ".join(part for part in (
            published,
            _strip_tags(source_match.group(1)) if source_match else "",
        ) if part)
        entry = {
            "title": title,
            "url": _clean_result_url(link_match.group(1)),
            "snippet": meta,
        }
        if source_url_match:
            entry["publisher_url"] = source_url_match.group(1)
        if published:
            entry["published"] = published
        results.append(entry)
        if len(results) >= limit:
            break
    return results


RSS_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
RSS_DATE_RE = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3})[A-Za-z]*\s+(\d{4})")


def _rss_date(raw):
    """Feed dates as ISO, so the newest match sorts first without a locale."""
    match = RSS_DATE_RE.search(raw or "")
    if not match:
        return ""
    day, month, year = match.group(1), match.group(2).lower(), match.group(3)
    if month not in RSS_MONTHS:
        return ""
    return f"{year}-{RSS_MONTHS[month]:02d}-{int(day):02d}"


WIKI_SEARCH_PARAMS = {
    "action": "query",
    "list": "search",
    "srlimit": 6,
    "srprop": "snippet",
    "format": "json",
    "utf8": 1,
}


def _parse_wikipedia_results(json_text, limit, lang):
    """Keyless encyclopedia search: reachable from the IPs DuckDuckGo blocks."""
    try:
        payload = json.loads(json_text or "{}")
    except (TypeError, ValueError):
        return []
    results = []
    for entry in ((payload.get("query") or {}).get("search")) or []:
        title = str(entry.get("title") or "").strip()
        snippet = _strip_tags(str(entry.get("snippet") or ""))
        if not title or not snippet:
            continue
        results.append({
            "title": f"{title} ({'Vikipedi' if lang == 'tr' else 'Wikipedia'})",
            "url": f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
            "snippet": snippet,
        })
        if len(results) >= limit:
            break
    return results


def _wikipedia_parser(lang):
    def parse(json_text, limit):
        return _parse_wikipedia_results(json_text, limit, lang)

    return parse


ARTICLE_BLOCK_RE = re.compile(
    r"<(script|style|noscript|svg|header|footer|nav|aside|form|iframe|figure)"
    r"[^>]*>.*?</\1>",
    re.S | re.I,
)
ARTICLE_SENTENCE_RE = re.compile(r"(?<=[.!?…])(?<!\d\.)\s+")
ARTICLE_NOISE_SENTENCE_RE = re.compile(
    r"(uygulamayı\s+aç|uygulamayi\s+ac|web'de\s+devam|kaynak\s+ekle|abone\s+ol|"
    r"bilgi\s+rehberi|reklam|çerez|cerez|gizlilik|kvkk|bizi\s+takip|"
    r"haber\s+kaynağınız|youtube|instagram|twitter|facebook|telegram|"
    r"son\s+dakika\s+haberleri|tüm\s+hakları\s+saklıdır)",
    re.I,
)
TURKISH_MARK_RE = re.compile(r"[ıİğüşöç]")


def _decode_html_body(response):
    """Some Turkish news sites serve cp1254 without saying so; mojibake loses names."""
    raw = response.content or b""
    fallback = ""
    for encoding in ("utf-8", response.encoding, "cp1254"):
        if not encoding:
            continue
        try:
            candidate = raw.decode(encoding, "strict")
        except (LookupError, UnicodeDecodeError):
            continue
        if not fallback:
            fallback = candidate
        if TURKISH_MARK_RE.search(candidate):
            return candidate
    return fallback


def _article_excerpt(url, tokens, timeout=6, limit=900):
    """Real article text: news feeds carry only a date, so scorers went missing."""
    target = (url or "").strip()
    if not target.startswith("http"):
        return ""
    try:
        response = requests.get(
            target, headers=SEARCH_HEADERS, timeout=timeout, allow_redirects=True
        )
    except Exception:
        return ""
    if response.status_code >= 400:
        return ""
    text = _strip_tags(ARTICLE_BLOCK_RE.sub(" ", _decode_html_body(response)))
    if len(text) < 60:
        return ""
    picked = []
    total = 0
    for sentence in ARTICLE_SENTENCE_RE.split(text):
        sentence = sentence.strip()
        if len(sentence) < 40 or ARTICLE_NOISE_SENTENCE_RE.search(sentence):
            continue
        folded = _ascii_fold(sentence).lower()
        if tokens and not any(token[:5] in folded for token in tokens):
            continue
        picked.append(sentence)
        total += len(sentence)
        if total >= limit:
            break
    return " ".join(picked)[:limit]


def _enrich_results(results, tokens, count, deadline=None):
    """Fill thin snippets with the article's own words, best matches first."""
    if count <= 0:
        return results
    done = 0
    for item in results:
        if done >= count:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        budget = 6.0 if deadline is None else min(6.0, max(1.0, deadline - time.monotonic()))
        excerpt = _article_excerpt(str(item.get("url") or ""), tokens, timeout=budget)
        if excerpt:
            item["excerpt"] = excerpt
            done += 1
    return results


SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.7,en;q=0.6",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


NEWS_HINT_RE = re.compile(
    r"(son\s*dakika|haber|maç|mac\b|skor|puan\s*durumu|dolar|euro|altın|altin|fiyat|zam\b|"
    r"seçim|secim|deprem|kaza|yangın|yangin|istifa|transfer|borsa|enflasyon|faiz|"
    r"cumhurbaşkan|cumhurbaskan|bakan|belediye|saldırı|saldiri|operasyon|fragman|"
    r"bilet|yeni\s+bölüm|yeni\s+bolum|son\s+bölüm|son\s+bolum|\b20\d\d\b)",
    re.I,
)
SEARCH_STOPWORDS = {
    "ve", "ile", "için", "icin", "bir", "bu", "şu", "onu", "değil", "degil", "nasıl",
    "nasil", "nedir", "hakkında", "hakkinda", "son", "güncel", "guncel", "araştır",
    "arastir", "mısın", "misin", "bana", "söyle", "soyle", "kaç", "kac", "derece",
    "the", "and", "for", "with", "what", "how",
}
# Kullanıcı dizi/film/haber sormadıysa bu başlıklar bulgu diye gösterilmez.
CHATTER_TITLE_RE = re.compile(
    r"(fragman|vizyon|dizi|film|sinema|oyuncu|çizgi roman|cizgi roman|magazin|şarkı|sarki|"
    r"klip|albüm|album|derbi|transfer|sezon finali|yarışma|yarisma|evlilik|boşanma|bosanma|"
    r"sevgili|son\s*dakika)",
    re.I,
)
ENTERTAINMENT_ASK_RE = re.compile(
    r"(dizi|film|sinema|fragman|oyuncu|şarkı|sarki|klip|albüm|album|belgesel|kitap|roman|"
    r"oyun\b|marvel|dc\b|netflix)",
    re.I,
)
# "evreni" aranınca motor yalnızca magazin buluyor; yalın hâl ("evren") ansiklopedik sonuç verir.
TURKISH_CASE_ENDING_RE = re.compile(
    r"(?:ler|lar)?(?:in|ın|un|ün|de|da|te|ta|den|dan|ten|tan|i|ı|u|ü|e|a)$"
)


def _strip_case(word):
    """Drop a Turkish case ending: 'evreni' only finds entertainment news."""
    match = TURKISH_CASE_ENDING_RE.search(word)
    if not match:
        return word
    stem = word[: match.start()]
    return stem if len(stem) >= 4 else word


def _query_variants(query):
    """The question as asked, then the same keywords in their plain form."""
    cleaned = (query or "").strip()
    variants = [cleaned]
    stripped = " ".join(_strip_case(word) for word in cleaned.split()).strip()
    if stripped and stripped != cleaned:
        variants.append(stripped)
    return variants


def _query_tokens(text):
    words = re.findall(r"[a-z0-9]{3,}", _ascii_fold(text or "").lower())
    return [word for word in dict.fromkeys(words) if word not in SEARCH_STOPWORDS]


def _relevance(result, tokens):
    """How many query stems appear in the result text (Turkish suffixes ignored)."""
    haystack = _ascii_fold(f"{result.get('title', '')} {result.get('snippet', '')}").lower()
    if not haystack.strip():
        return 0
    return sum(1 for token in tokens if token[:5] in haystack)


def _is_relevant(result, tokens, allow_chatter, strict=False):
    """On-topic and, unless the user asked for it, not entertainment noise."""
    if tokens:
        haystack = _ascii_fold(
            f"{result.get('title', '')} {result.get('snippet', '')}"
        ).lower()
        minimum_matches = 1 if len(tokens) <= 2 else (len(tokens) + 1) // 2
        if strict:
            # Ansiklopedi her konuda bir madde bulur; nadir terim yoksa konu başkadır.
            rarest = max(tokens, key=len)
            if rarest[:6] not in haystack:
                return False
        if sum(1 for token in tokens if token[:5] in haystack) < minimum_matches:
            return False
    if allow_chatter:
        return True
    return not CHATTER_TITLE_RE.search(str(result.get("title") or ""))


SEARCH_ENGINE_COOLDOWN = 120.0
SEARCH_ENGINE_STATE = {}
SEARCH_ENGINE_LOCK = threading.Lock()
REFERENCE_ENGINES = frozenset({"vikipedi-tr", "vikipedi-en"})
ENGINE_QUERY_PARAM = {"vikipedi-tr": "srsearch", "vikipedi-en": "srsearch"}


def _engine_ready(engine):
    """A blocked engine is skipped for a while instead of eating the whole budget."""
    with SEARCH_ENGINE_LOCK:
        return time.monotonic() >= SEARCH_ENGINE_STATE.get(engine, 0.0)


def _engine_failed(engine):
    with SEARCH_ENGINE_LOCK:
        SEARCH_ENGINE_STATE[engine] = time.monotonic() + SEARCH_ENGINE_COOLDOWN


def _search_attempts(query, duckduckgo_only=False):
    """Keyless feeds first: the HTML scrapers are the ones cloud IPs get blocked on."""
    general = (
        ("ddg-html", "post", "https://html.duckduckgo.com/html/", {"kl": "tr-tr"}, _parse_ddg_results),
        ("ddg-lite", "post", "https://lite.duckduckgo.com/lite/", {"kl": "tr-tr"}, _parse_ddg_results),
        ("ddg-get", "get", "https://html.duckduckgo.com/html/", {}, _parse_ddg_results),
        ("bing", "get", "https://www.bing.com/search", {"setmkt": "tr-TR", "setlang": "tr"}, _parse_bing_results),
    )
    news = (
        ("haber-rss", "get", "https://news.google.com/rss/search",
         {"hl": "tr", "gl": "TR", "ceid": "TR:tr"}, _parse_google_news_rss),
        ("bing-haber-rss", "get", "https://www.bing.com/news/search",
         {"format": "rss", "setmkt": "tr-TR"}, _parse_google_news_rss),
    )
    reference = (
        ("vikipedi-tr", "get", "https://tr.wikipedia.org/w/api.php",
         dict(WIKI_SEARCH_PARAMS), _wikipedia_parser("tr")),
        ("vikipedi-en", "get", "https://en.wikipedia.org/w/api.php",
         dict(WIKI_SEARCH_PARAMS), _wikipedia_parser("en")),
    )
    if duckduckgo_only:
        return tuple(attempt for attempt in general if attempt[0].startswith("ddg-"))
    if NEWS_HINT_RE.search(query or ""):
        return news + reference + general
    return reference + news + general


def web_search(query, num_results=5, *, duckduckgo_only=False, deadline=None,
               recency_days=None, enrich=0, only_engines=None):
    """Current web results with no API key, filtered for relevance to the query."""
    limit = max(1, min(int(num_results), 10))
    clean_query = (query or "").strip()
    if not clean_query:
        return {"error": "Web search failed: empty query", "results": [], "query": query}
    allow_chatter = bool(
        NEWS_HINT_RE.search(clean_query) or ENTERTAINMENT_ASK_RE.search(clean_query)
    )
    last_error = None
    last_tokens = _query_tokens(clean_query)
    partial = None
    for variant in _query_variants(clean_query):
        if deadline is not None and time.monotonic() >= deadline:
            break
        tokens = last_tokens = _query_tokens(variant)
        for engine, method, url, extra, parser in _search_attempts(variant, duckduckgo_only):
            if deadline is not None and time.monotonic() >= deadline:
                break
            if only_engines and engine not in only_engines:
                continue
            if not _engine_ready(engine):
                last_error = f"{engine}: az önce başarısız oldu, bekleniyor"
                continue
            strict = engine in REFERENCE_ENGINES
            try:
                timeout = 4 if duckduckgo_only else 8
                if deadline is not None:
                    timeout = min(timeout, max(0.5, deadline - time.monotonic()))
                engine_query = variant
                if recency_days and engine == "haber-rss":
                    engine_query = f"{variant} when:{int(recency_days)}d"
                params = {ENGINE_QUERY_PARAM.get(engine, "q"): engine_query, **extra}
                if method == "post":
                    response = requests.post(
                        url, data=params, headers=SEARCH_HEADERS, timeout=timeout
                    )
                else:
                    response = requests.get(
                        url, params=params, headers=SEARCH_HEADERS, timeout=timeout
                    )
                if response.status_code == 202 or "anomalyDetectionBlock" in response.text:
                    _engine_failed(engine)
                    if duckduckgo_only:
                        return {
                            "error": "DuckDuckGo requested a security check for this server.",
                            "results": [],
                            "query": variant,
                            "engine": engine,
                            "blocked": True,
                        }
                    last_error = f"{engine}: güvenlik doğrulaması istedi"
                    continue
                if response.status_code >= 400:
                    _engine_failed(engine)
                    last_error = f"{engine} HTTP {response.status_code}"
                    continue
                results = parser(response.text, limit * 2)
                if not results:
                    last_error = f"{engine}: no results parsed"
                    continue
                relevant = [
                    item for item in results
                    if _is_relevant(item, tokens, allow_chatter, strict)
                ]
                relevant.sort(key=lambda item: -_relevance(item, tokens))
                if recency_days:
                    # "Maçı kaç kaç bitti?" son maçı sorar; tarih sırası şart.
                    relevant.sort(key=lambda item: str(item.get("published") or ""), reverse=True)
                # Tek sonuç tesadüf olabilir; yalın hâl varyantı genelde çok daha isabetli.
                if len(relevant) >= 2 or (strict and relevant):
                    chosen = relevant[:limit]
                    _enrich_results(chosen, tokens, enrich, deadline)
                    return {"query": variant, "results": chosen, "engine": engine}
                if relevant and partial is None:
                    partial = relevant[:limit]
                last_error = f"{engine}: sonuç soruyla ilgili değil"
            except Exception as error:
                _engine_failed(engine)
                last_error = f"{engine}: {error}"
    if partial:
        _enrich_results(partial, last_tokens, enrich, deadline)
        return {"query": clean_query, "results": partial, "engine": "weak-match"}
    return {"error": f"Web search failed: {last_error}", "results": [], "query": clean_query}


def web_search_multi(query, num_results=5, *, deadline=None, recency_days=None, enrich=2):
    """Combine independent news and web indexes under one bounded time budget."""
    news_query = bool(NEWS_HINT_RE.search(query or "") or _sports_club_names(query))
    engines = (
        ("haber-rss", "bing-haber-rss", "ddg-html", "bing")
        if news_query
        else ("ddg-html", "bing", "vikipedi-tr", "vikipedi-en")
    )
    search_deadline = deadline or time.monotonic() + 18
    results = []
    seen = set()
    found_engines = []
    tokens = _query_tokens(query)
    for engine in engines:
        if time.monotonic() >= search_deadline:
            break
        context = web_search(
            query,
            num_results,
            deadline=search_deadline,
            recency_days=recency_days,
            only_engines=(engine,),
        )
        if context.get("results"):
            found_engines.append(context.get("engine") or engine)
        for item in context.get("results") or []:
            url = str(item.get("url") or "").strip()
            parsed_url = urlparse(url)
            key = (
                f"{(parsed_url.hostname or '').lower()}{parsed_url.path.rstrip('/')}"
                if parsed_url.hostname
                else str(item.get("title") or "").strip().casefold()
            )
            if key and key not in seen:
                seen.add(key)
                results.append({**item, "source_engine": context.get("engine") or engine})
    results.sort(key=lambda item: _relevance(item, tokens), reverse=True)
    if recency_days:
        results.sort(key=lambda item: str(item.get("published") or ""), reverse=True)
    if results:
        _enrich_results(results, tokens, enrich, search_deadline)
    return {
        "query": query,
        "results": results[:max(1, min(int(num_results) * 2, 10))],
        "engine": ", ".join(dict.fromkeys(found_engines)),
        "engines": list(dict.fromkeys(found_engines)),
    }


SEARCH_FILLER_RE = re.compile(
    r"\b(bana|bir|acaba|lütfen|söyle|söyler\s+misin|bilir\s+misin|yapar\s+mısın|"
    r"araştır|araştırır\s+mısın|bul|bulabilir\s+misin|öğrenmek\s+istiyorum|"
    r"hakkında\s+bilgi\s+ver|nedir|ne\s+demek|mi|mı|mu|mü|ya|şey)\b",
    re.I,
)


def build_search_query(text):
    """Turn a chatty question into keywords a search engine actually matches."""
    source = (text or "").strip()
    cleaned = re.sub(r"[?!.,;:'\"()\[\]]", " ", source.lower())
    cleaned = SEARCH_FILLER_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return source
    if re.search(r"\b(maç|mac|skor|karşılaşma|karsilasma)", cleaned) and "sonuç" not in cleaned and "sonuc" not in cleaned:
        cleaned += " maç sonucu skor"
    return cleaned


def search_topic(text):
    """Nominative keywords, so appended facets stay grammatical ('evren' + 'tarihçesi')."""
    return " ".join(_strip_case(word) for word in build_search_query(text).split()).strip()


DETAILED_RESEARCH_RE = re.compile(
    r"\b(araştır\w*|arastir\w*|incele\w*|derinlemesine|detaylı\s+araştır\w*|"
    r"kapsamlı\s+araştır\w*|comprehensive\s+research)\b",
    re.I,
)


def is_detailed_research_request(text):
  return bool(DETAILED_RESEARCH_RE.search(text or ""))


def build_detailed_research_queries(text):
  """Search the requested subject from complementary angles, keeping the original first."""
  topic = search_topic(text)[:240]
  if not topic:
    return (build_search_query(text)[:500],)
  year = timezone.localdate().year
  return tuple(dict.fromkeys((
      build_search_query(text)[:500],
      f"{topic} {year} güncel gelişmeler",
      f"{topic} ayrıntılı analiz ve etkileri",
      f"{topic} temel veriler ve uzman değerlendirmesi",
  )))


BRAND_LEAK_RE = re.compile(r"\b(google|gemini)\b", re.I)


def is_brand_safe(text):
    """A source line the user is allowed to see (never names another AI brand)."""
    return not BRAND_LEAK_RE.search(text or "")


def format_source_lines(payload, limit, with_snippets=False):
    """Search results as display lines: another AI brand never reaches the screen.

    A result whose title is clean stays useful even when its link is not, so the
    link is dropped instead of the whole finding.
    """
    lines = []
    for item in (payload.get("results") or []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "")
        snippet = str(item.get("snippet") or "")
        url = str(item.get("url") or "")
        if not title or not is_brand_safe(f"{title} {snippet}"):
            continue
        line = title
        if url and is_brand_safe(url):
            line += f" — {url}"
        if with_snippets and snippet and is_brand_safe(snippet):
            line += f"\n   {snippet}"
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


FOUNDER_NAME_RE = re.compile(r"\bayaz\s+kaplan\b", re.I)
FOUNDER_ASK_RE = re.compile(
    r"(kurucu|kuran|sahibi|seni kim|kim yap|kim gelişt|kim gelistir|kimin eseri|yaratıcın|yaraticin)",
    re.I,
)
# Akış parça parça geldiği için adın yarısı bir parçada, yarısı diğerinde kalabilir;
# bu yüzden olası her yarım eşleşme tamamlanana kadar buffer'da tutulur.
FOUNDER_NAME_PARTIALS = sorted(
    ("ayaz kaplan"[:length] for length in range(1, len("ayaz kaplan"))),
    key=len,
    reverse=True,
)


class NameScrubber:
  """Keeps the founder's name out of samples; the stream is scanned across chunks."""

  def __init__(self, replacement, enabled=True):
    self.replacement = replacement or "Kullanıcı"
    self.enabled = enabled
    self.buffer = ""

  def feed(self, piece):
    if not piece:
      return ""
    if not self.enabled:
      return piece
    self.buffer += piece
    emit = self.buffer
    lowered = emit.lower()
    for partial in FOUNDER_NAME_PARTIALS:
      if lowered.endswith(partial):
        emit = emit[: len(emit) - len(partial)]
        break
    self.buffer = self.buffer[len(emit):]
    if not emit:
      return ""
    return FOUNDER_NAME_RE.sub(self.replacement, emit)

  def flush(self):
    if not self.enabled or not self.buffer:
      return ""
    emit, self.buffer = self.buffer, ""
    return FOUNDER_NAME_RE.sub(self.replacement, emit)


def should_fetch_live_context(text):
  """Use live research for explicit updates and substantive factual questions."""
  normalized = (text or "").lower()
  if re.fullmatch(
      r"\s*(?:merhaba|selam|sa(?:ğ|g)ol|teşekkür(?:ler)?|nasılsın|naber|"
      r"hi|hello|thanks|thank you)\s*[!.?]*\s*",
      normalized,
      re.I,
  ):
    return False
  hints = (
      "internetten", "güncel", "bugün", "şu an", "şuan", "son dakika",
      "haber", "maç", "mac ", "skor", "sonuç", "spor", "fiyat", "kur", "döviz",
      "kim kazandı", "ne zaman", "araştır", "web'de", "webte", "bitti",
      "kazandı", "puan durumu", "transfer", "deprem", "seçim", "altın",
      "dolar", "euro", "bitcoin", "hisse", "sürüm", "versiyon", "yeni çıkan",
  )
  if any(hint in normalized for hint in hints):
    return True
  factual_question = re.search(
      r"\b(ne|nedir|kim|kimdir|hangi|kaç|kac|nerede|neresi|nasıl|nasil|neden|"
      r"niçin|nicin|how|what|who|which|where|when|why)\b",
      normalized,
      re.I,
  )
  meaningful_words = re.findall(r"[a-z0-9çğıöşü]{3,}", normalized)
  return bool(factual_question and len(meaningful_words) >= 3)


SPORTS_FACTS_RE = re.compile(
  r"\b(maç|mac|futbol|gol\w*|skor|puan|derbi|lig|transfer|kadrosu|golcü\w*|golcu\w*)\b|"
    r"(fenerbahçe|fenerbahce|eyüpspor|eyupspor|galatasaray|beşiktaş|besiktas|"
    r"trabzonspor|başakşehir|basaksehir|süper lig|super lig)",
    re.I,
)
SPORTS_FOLLOWUP_RE = re.compile(
    r"\b(o|bu|şu|su|bunun|bunu|onun|onda|o\s+nasıl|bu\s+nasıl|neden|"
    r"nasıl\s+oldu|ne\s+yapmış|ne\s+yapmis|kim\s+attı|kim\s+atti|"
    r"detay|ayrıntı|ayrinti)\b",
    re.I,
)
SPORTS_SCORER_RE = re.compile(
    r"\b(gol\w*|golcü\w*|golcu\w*|kim\s+attı|kim\s+atti|"
    r"asist\w*|dakika\w*|oyuncu\w*)\b",
    re.I,
)
SPORTS_SCORE_ASK_RE = re.compile(
    r"\b(kaç\s*kaç|skor|sonuç|sonuc|bitti|kazandı|kazandi|ne\s+oldu)\b",
    re.I,
)
SPORTS_SCORE_RE = re.compile(r"\b\d{1,2}\s*[-–:]\s*\d{1,2}\b")
SPORTS_RESULT_CONTEXT_RE = re.compile(
    r"\b(maç|mac|skor|sonuç|sonuc|bitti|kazandı|kazandi|yendi|yenildi|"
    r"mağlup|maglup|berabere|score|result|final|won|beat|draw)\b",
    re.I,
)
SPORTS_CONFIRMED_SCORE_CONTEXT_RE = re.compile(
    r"\b(skor|sonuç|sonuc|bitti|kazandı|kazandi|yendi|yenildi|mağlup|"
    r"maglup|berabere|score|result|final|won|beat|draw|sona\s+erdi)\b",
    re.I,
)
SPORTS_LIVE_SCORE_CONTEXT_RE = re.compile(
    r"(canlı\s+skor|canli\s+skor|live\s+score|ikinci\s+y[ıi]r[ıi]|"
    r"maç\s+devam\s+ediyor|mac\s+devam\s+ediyor|in\s+progress)",
    re.I,
)
SPORTS_DATE_CONTEXT_RE = re.compile(
    r"\b(tarih\w*|takvim\w*|fikstür\w*|fikstur\w*|date|calendar|scheduled|schedule)\b",
    re.I,
)
SPORTS_BASKETBALL_CONTEXT_RE = re.compile(
    r"\b(basketbol|basketball|euroleague|nba|voleybol|volleyball|"
    r"hentbol|handball|set(?:i|te|ler)?)\b",
    re.I,
)
SPORTS_DATE_RE = re.compile(
    r"\b\d{1,2}\s+(?:ocak|şubat|subat|mart|nisan|mayıs|mayis|haziran|"
    r"temmuz|ağustos|agustos|eylül|eylul|ekim|kasım|kasim|aralık|aralik)\s+20\d{2}\b",
    re.I,
)
SPORTS_QUERY_NOISE = SEARCH_STOPWORDS | {
    "mac", "maci", "macta", "sonuc", "sonucu", "skor", "kac", "bitti",
    "gol", "goller", "golcu", "golculer", "kim", "atti", "ne", "oldu",
  }


def is_sports_question(text):
    return bool(SPORTS_FACTS_RE.search(text or ""))


def _previous_sports_user_question(history):
  for message in reversed(history or []):
    if not isinstance(message, dict) or message.get("sender") != "user":
      continue
    previous_question = str(message.get("text") or "").strip()
    return previous_question if is_sports_question(previous_question) else ""
  return ""


def is_sports_followup(text, history=None):
  """A short scorer/detail question inherits only the immediately prior user match."""
  return bool(
    SPORTS_FOLLOWUP_RE.search(text or "")
    and _previous_sports_user_question(history)
  )


def _compact_sports_match_terms(text):
  words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9-]+", text or "")
  return " ".join(
      word for word in words
      if len(_ascii_fold(word)) >= 3
      and _ascii_fold(word).lower() not in SPORTS_QUERY_NOISE
  )


def _previous_assistant_match_hints(history):
  for message in reversed(history or []):
    if not isinstance(message, dict):
      continue
    if message.get("sender") == "user":
      if is_sports_question(message.get("text")):
        return []
      continue
    if message.get("sender") != "ai":
      continue
    answer = str(message.get("text") or "")
    hints = []
    score = SPORTS_SCORE_RE.search(answer)
    date = SPORTS_DATE_RE.search(answer)
    if score:
      hints.append(score.group(0).replace("–", "-").replace(":", "-"))
    if date:
      hints.append(date.group(0))
    return hints
  return []


SPORTS_CLUB_RE = re.compile(
    r"\b(fenerbahçe|fenerbahce|galatasaray|beşiktaş|besiktas|trabzonspor|"
    r"başakşehir|basaksehir|eyüpspor|eyupspor|samsunspor|rizespor|antalyaspor|"
    r"konyaspor|kayserispor|gaziantep\s*fk|sivasspor|alanyaspor|kasımpaşa|kasimpasa|"
    r"gençlerbirliği|genclerbirligi|göztepe|goztepe|karagümrük|karagumruk|kocaelispor|"
    r"bursaspor|eskişehirspor|eskisehirspor|ankaragücü|ankaragucu|turan\s*tovuz)\b",
    re.I,
)


def _sports_club_names(text):
  """Club names in the order the user wrote them, deduplicated."""
  clubs = []
  seen = set()
  for match in SPORTS_CLUB_RE.finditer(text or ""):
    club = match.group(0).strip()
    marker = _ascii_fold(club).casefold()
    if marker not in seen:
      seen.add(marker)
      clubs.append(club)
  return clubs


def build_live_search_queries(text, history=None):
  """Return the contextual sports query and a focused detail query when needed."""
  current = (text or "").strip()
  previous_question = _previous_sports_user_question(history)
  if previous_question and SPORTS_FOLLOWUP_RE.search(current):
    if SPORTS_SCORER_RE.search(current):
      match_terms = _compact_sports_match_terms(previous_question)
      queries = []
      if match_terms:
        queries.extend((
          " ".join(part for part in (match_terms, "golleri kim attı") if part),
          " ".join(part for part in (
            match_terms,
            "gol atan oyuncular",
          ) if part),
        ))
      queries.append(build_search_query(f"{previous_question} {current}"))
    else:
      queries = [build_search_query(f"{previous_question} {current}")]
    return tuple(dict.fromkeys(queries))
  queries = [build_search_query(current)]
  clubs = _sports_club_names(current)
  if clubs and not SPORTS_SCORE_RE.search(current):
    # Rakip yazılmadıysa motor eski maçları getirir; "son maç" açıkça sorulur.
    queries.append(f"{' '.join(clubs)} son maç sonucu kaç kaç bitti golleri kim attı")
  return tuple(dict.fromkeys(queries))


def build_live_search_query(text, history=None):
  """Compatibility helper returning the primary contextual live-search query."""
  return build_live_search_queries(text, history)[0]


def _sports_result_rank(item, clubs, tokens):
  """Newest match of the asked club first, unrelated fixtures dropped."""
  text = _ascii_fold(
      f"{item.get('title', '')} {item.get('snippet', '')} {item.get('excerpt', '')}"
  ).lower()
  club_hits = sum(1 for club in clubs if _ascii_fold(club).lower() in text)
  return (club_hits, _relevance(item, tokens), str(item.get("published") or ""))


def search_live_sports(queries, num_results, *, followup=False, deadline=None, enrich=2):
  """Newest fixture first, with the article's own words so scorers are present."""
  deadline = deadline or time.monotonic() + 25
  combined = []
  seen = set()
  engine = ""
  used_query = ""
  recency_plan = (None,) if followup else (7, 30, None)
  for recency in recency_plan:
    for query in queries:
      if deadline is not None and time.monotonic() >= deadline:
        break
      found = web_search_multi(
          query,
          num_results,
          recency_days=recency,
          deadline=deadline,
          enrich=0,
      )
      for item in found.get("results") or []:
        key = str(item.get("url") or item.get("title") or "").strip().casefold()
        if key and key not in seen:
          seen.add(key)
          combined.append(item)
      if combined and not engine:
        engine = found.get("engine") or ""
        used_query = query
    if len(combined) >= 2:
      break
  clubs = [_ascii_fold(club).lower() for club in _sports_club_names(" ".join(queries))]
  def matches_requested_clubs(item):
    text = _ascii_fold(
        f"{item.get('title', '')} {item.get('snippet', '')} {item.get('excerpt', '')}"
    ).lower()
    # A named fixture must match both teams; a single-team hit can be another match.
    return (
        all(club in text for club in clubs)
        if len(clubs) > 1
        else any(club in text for club in clubs)
    )

  if clubs:
    combined = [item for item in combined if matches_requested_clubs(item)]
  if not combined:
    return {"results": [], "engine": engine, "query": " | ".join(queries)}
  tokens = _query_tokens(queries[0])
  if len(clubs) == 1 and not followup:
    combined.sort(
        key=lambda item: (
            str(item.get("published") or ""),
            *_sports_result_rank(item, clubs, tokens)[:2],
        ),
        reverse=True,
    )
  else:
    combined.sort(key=lambda item: _sports_result_rank(item, clubs, tokens), reverse=True)
  _enrich_results(combined, tokens, enrich, deadline)
  if len(clubs) == 1 and not followup:
    combined.sort(
        key=lambda item: (
            str(item.get("published") or ""),
            *_sports_result_rank(item, clubs, tokens)[:2],
        ),
        reverse=True,
    )
  else:
    combined.sort(key=lambda item: _sports_result_rank(item, clubs, tokens), reverse=True)
  return {"results": combined, "engine": engine,
          "query": used_query or queries[0]}


def should_retry_live_search_with_general_results(question, live_context):
    """Retry weak or blocked search results before answering a live-data question."""
    if not isinstance(live_context, dict):
        return False
    if live_context.get("blocked"):
        return True
    if live_context.get("engine") == "weak-match":
        return True
    results = live_context.get("results") or []
    return not results


def summarize_sourced_match_score(results, *, latest_fixture=False):
    """Report a sourced score, ignoring dates and unanchored score-like numbers."""
    results = [result for result in results or [] if isinstance(result, dict)]
    dated_results = [
        result for result in results
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(result.get("published") or ""))
    ]
    if latest_fixture and dated_results:
        latest_date = max(dt.date.fromisoformat(result["published"]) for result in dated_results)
        oldest_relevant_date = latest_date - dt.timedelta(days=1)
        results = [
            result for result in dated_results
            if dt.date.fromisoformat(result["published"]) >= oldest_relevant_date
        ]
    score_sources = {}
    for result in results or []:
        if not isinstance(result, dict):
            continue
        title = str(result.get("title") or "").strip()
        snippet = str(result.get("snippet") or "").strip()
        url = str(result.get("url") or "").strip()
        if not is_brand_safe(f"{title} {snippet} {url}"):
            continue
        excerpt = str(result.get("excerpt") or "").strip()
        host = (
            urlparse(str(result.get("publisher_url") or "")).hostname
            or urlparse(url).hostname
            or ""
        ).lower()
        if not host:
            continue
        for source_text in (title, snippet, excerpt):
            cleaned_text = re.sub(
                r"\b(?:20\d{2}[-./]\d{1,2}[-./]\d{1,2}|"
                r"\d{1,2}[-./]\d{1,2}[-./](?:20)?\d{2})\b",
                " ",
                source_text,
            )
            for match in SPORTS_SCORE_RE.finditer(cleaned_text):
                left, right = (
                    int(value)
                    for value in re.split(r"\s*[-–:]\s*", match.group(0))
                )
                context = cleaned_text[
                    max(0, match.start() - 70):min(len(cleaned_text), match.end() + 70)
                ]
                if SPORTS_LIVE_SCORE_CONTEXT_RE.search(context):
                    continue
                looks_like_calendar_date = (
                    (1 <= left <= 12 and 13 <= right <= 31)
                    or (13 <= left <= 31 and 1 <= right <= 12)
                )
                if looks_like_calendar_date:
                    continue
                if (
                    left <= 12
                    and right <= 12
                    and SPORTS_DATE_CONTEXT_RE.search(context)
                    and not SPORTS_CONFIRMED_SCORE_CONTEXT_RE.search(context)
                ):
                    continue
                if (
                    max(left, right) > 20
                    and not SPORTS_BASKETBALL_CONTEXT_RE.search(context)
                ):
                    continue
                if (
                    not SPORTS_RESULT_CONTEXT_RE.search(context)
                    and len(_sports_club_names(context)) < 2
                ):
                    continue
                score = f"{left}-{right}"
                score_sources.setdefault(score, {})[host] = {
                    "title": title,
                    "url": url,
                }

    if not score_sources:
        return (
            "Arama sonuçlarında maç skoru açıkça yer almıyor. Yanlış skor uydurmamak "
            "için kesin bir sonuç vermiyorum."
        )
    corroborated = {
        score: sources for score, sources in score_sources.items()
        if len(sources) >= 2
    }
    if len(corroborated) == 1:
        score, _sources = next(iter(corroborated.items()))
        return f"İki bağımsız kaynak maç skorunu {score} olarak doğruluyor."
    if len(corroborated) > 1 or len(score_sources) > 1:
        return (
            "Canlı kaynaklarda bu soruya uyan birden fazla maç veya çelişkili skor var. "
            "Kesin sonuç uydurmamak için skor vermiyorum; rakip takımı ve futbol/basketbol "
            "branşını belirtirsen yalnızca o maçı doğrulayabilirim."
        )
    score, sources = next(iter(score_sources.items()))
    return (
        f"Bir canlı arama sonucu skoru {score} olarak bildiriyor; bunu bağımsız bir "
        "kaynakla doğrulayamadığım için kesin bilgi olarak sunmuyorum."
    )


def summarize_sourced_match_details(results):
    """Summarize article evidence, preferring article text over search snippets."""
    lines = []
    seen = set()
    scorer_evidence = re.compile(
        r"\b(gol\w*|golcü\w*|golcu\w*|attı|atti|kaydetti|penaltı|penalti|"
        r"dakika|skorer|asist\w*|goals?|scored|scorer)\b|fileleri\s+buldu",
        re.I,
    )
    for result in results or []:
        if not isinstance(result, dict):
            continue
        title = str(result.get("title") or "").strip()
        snippet = str(result.get("snippet") or "").strip()
        excerpt = str(result.get("excerpt") or "").strip()
        detail = excerpt or snippet
        if not title and not detail:
            continue
        if not is_brand_safe(f"{title} {snippet} {excerpt}"):
            continue
        for sentence in ARTICLE_SENTENCE_RE.split(detail):
            sentence = re.sub(r"https?://\S+", "", sentence).strip(" \t-:;()")
            key = re.sub(r"\s+", " ", sentence).casefold()
            if (
                sentence
                and key not in seen
                and scorer_evidence.search(sentence)
                and not ARTICLE_NOISE_SENTENCE_RE.search(sentence)
            ):
                seen.add(key)
                lines.append(sentence)
                if len(lines) == 3:
                    break
        if len(lines) == 3:
            break
    if not lines:
        return "Arama sonuçlarında doğrulanabilir maç ayrıntısı bulamadım; oyuncu veya gol dakikası uydurmayacağım."
    return (
        "Maç raporlarında gol ve oyuncu ayrıntıları şöyle geçiyor:\n" + "\n".join(lines)
    )


TIME_INTENT_RE = re.compile(
    r"(saat\s+kaç|saat\s+kaçtır|şu\s+an\s+saat|saati\s+söyle|"
    r"tarih\s+(ne|kaç)|"
    r"bugün\s+(ayın\s+kaçı|ne\s+günü|hangi\s+gün)|günlerden\s+(ne|hangi)|"
    r"hangi\s+gündeyiz|bugün\s+günlerden)",
    re.I,
)
TIME_LOCATION_PATTERNS = (
    r"(?P<place>[\wÇĞİÖŞÜçğıöşü .'-]+?)['’]?(?:de|da|te|ta|nde|nda|nte|nta)"
    r"\s+(?:şu\s+an\s+)?saat\s+kaç",
    r"(?P<place>[\wÇĞİÖŞÜçğıöşü .'-]+?)\s+(?:şu\s+an\s+)?saat\s+kaç",
    r"saat\s+kaç(?:tır)?\s+(?P<place>[\wÇĞİÖŞÜçğıöşü .'-]+)",
)
TIME_LOCATION_ALIASES = {
    "cin": ("Şanghay", "Asia/Shanghai"),
    "china": ("Shanghai", "Asia/Shanghai"),
    "japonya": ("Tokyo", "Asia/Tokyo"),
    "japan": ("Tokyo", "Asia/Tokyo"),
}
WEEKDAY_NAMES_TR = (
    "Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar",
)


def detect_time_location(text):
  for pattern in TIME_LOCATION_PATTERNS:
    match = re.search(pattern, text or "", re.I)
    if not match:
      continue
    place = _strip_locative(match.group("place").strip(" ?!.,'’\""))
    if place and place.casefold() not in {"şu an", "şimdi", "bugün"}:
      return place
  return None


def get_location_time(place, now=None):
  """Resolve a city to its IANA timezone and return its current local time."""
  cleaned = _strip_locative((place or "").strip(" ?!.,'’\""))
  alias = TIME_LOCATION_ALIASES.get(_ascii_fold(cleaned).casefold())
  if alias:
    city_name, timezone_name = alias
  else:
    try:
      location = _geocode_city(cleaned, country="")
    except (requests.RequestException, ValueError, TypeError) as error:
      logger.warning("Local-time geocoding failed for %s: %s", cleaned, error)
      return {"error": "Şehrin saat dilimine şu anda ulaşılamadı."}
    if not location:
      return {"not_found": True, "error": f"'{cleaned}' için saat dilimi bulunamadı."}
    city_name = location.get("name") or cleaned
    timezone_name = location.get("timezone")
  if not timezone_name:
    return {"error": f"'{cleaned}' için saat dilimi bulunamadı."}
  try:
    zone = ZoneInfo(timezone_name)
  except (KeyError, ValueError):
    return {"error": f"'{cleaned}' için geçerli saat dilimi bulunamadı."}
  local_now = (now or timezone.now()).astimezone(zone)
  return {
      "city": city_name,
      "time": local_now.strftime("%H:%M"),
      "date": local_now.strftime("%d.%m.%Y"),
      "day": WEEKDAY_NAMES_TR[local_now.weekday()],
      "timezone": timezone_name,
      "utc_offset": local_now.strftime("%z"),
  }


def format_location_time_answer(time_data):
  offset = time_data.get("utc_offset") or ""
  offset = f" (UTC{offset[:3]}:{offset[3:]})" if len(offset) == 5 else ""
  return (
      f"{time_data['city']} için saat {time_data['time']}, "
      f"{time_data['date']} {time_data['day']}{offset}."
  )
WEATHER_INTENT_RE = re.compile(
    r"(hava\s+durumu|hava\s+nasıl|havası\s+nasıl|kaç\s+derece|sıcaklık\s+kaç|hava\s+kaç\s+derece)",
    re.I,
)
CITY_PATTERNS = (
    r"([A-Za-zÇĞİÖŞÜçğıöşü\-\.]+)['’]?(?:de|da|te|ta|nde|nda|nte|nta)\s+(?:için\s+)?hava",
    r"([A-Za-zÇĞİÖŞÜçğıöşü\-\.]+)\s+(?:için\s+)?hava\s+durumu",
    r"hava(?:\s+\w+){0,4}\s+([A-Za-zÇĞİÖŞÜçğıöşü\-\.]+)['’]?\s+için\b",
    r"([A-Za-zÇĞİÖŞÜçğıöşü\-\.]+)\s+hava\s+nasıl",
    r"hava\s+durumu\s+([A-Za-zÇĞİÖŞÜçğıöşü\-\.]+)",
    r"hava\s+durumu\s+nedir\s+([A-Za-zÇĞİÖŞÜçğıöşü\-\.]+)",
)
WEATHER_STOPWORDS = {
    "nasıl", "nedir", "kaç", "derece", "için", "ve", "bu", "şu", "an", "şuan",
    "bugün", "yarın", "şimdi", "orada", "burada", "tr", "türkiye",
}


def detect_weather_city(text):
    normalized = (text or "").strip()
    for pattern in CITY_PATTERNS:
        match = re.search(pattern, normalized, re.I)
        if not match:
            continue
        city = match.group(1).strip(" ?!.,'’\"")
        city = re.sub(r"\s+(için\s+)?hava\s+durumu.*$", "", city, flags=re.I).strip()
        if city and city.lower() not in WEATHER_STOPWORDS and len(city) >= 2:
            return city
    return None


WEATHER_ASK_CITY = (
    "Hangi şehir için hava durumu öğrenmek istiyorsun? 🌤 Şehri yazman yeterli, "
    "anlık sıcaklığı derece derece hemen getireyim."
)
WEATHER_ASK_MARKERS = ("hangi şehir için hava durumu", "şehri yazman yeterli")
CITY_REPLY_STOPWORDS = WEATHER_STOPWORDS | {
    "evet", "hayır", "hayir", "tamam", "ok", "teşekkürler", "tesekkurler",
    "sağol", "sagol", "merhaba", "selam", "hava", "durumu",
}


def _strip_locative(city):
    return re.sub(
        r"(?:['’]?(?:de|da|te|ta|nde|nda|nte|nta))$",
        "",
        (city or "").strip(),
        flags=re.I,
    ).strip()


def pending_weather_city(history, text):
    """Read the city out of a bare reply to the 'which city?' weather question."""
    reply = re.sub(r"[?!.,;:'\"]+", " ", (text or "")).strip()
    if not reply or len(reply) > 40 or len(reply.split()) > 4:
        return None
    if WEATHER_INTENT_RE.search(reply) or TIME_INTENT_RE.search(reply):
        return None
    last_answer = ""
    for item in reversed(history or []):
        if isinstance(item, dict) and item.get("sender") != "user":
            last_answer = str(item.get("text") or "").lower()
            break
    if not any(marker in last_answer for marker in WEATHER_ASK_MARKERS):
        return None
    city = _strip_locative(re.sub(r"^(şehir|sehir|city|ben|benim|için|icin)\s+", "", reply, flags=re.I).strip())
    if len(city) < 2 or city.lower() in CITY_REPLY_STOPWORDS:
        return None
    return city


def profile_payload(user):
  profile = get_user_profile(user)
  return {
      "username": user.username,
      "avatar": profile.avatar,
      "theme": profile.theme,
      "pattern": profile.pattern,
  }


def friendly_api_error(error):
  """User-facing error text. Never mentions Gemini or Google."""
  error_text = str(error).lower()
  if "max_tokens" in error_text or ("maximum" in error_text and "token" in error_text):
    return "Aslan Parçası için istek çok uzun. Daha kısa bir mesaj veya daha küçük bir dosya deneyin."
  if "401" in error_text or "403" in error_text or "unauthorized" in error_text or "api key" in error_text:
    return "Aslan Parçası'nın beyin bağlantısı şu an doğrulanamadı. Lütfen biraz sonra tekrar deneyin."
  if "404" in error_text or "not found" in error_text:
    return "Aslan Parçası'nın istediğin yeteneği şu an erişilemiyor. Lütfen tekrar deneyin."
  if is_daily_quota_error(error):
    return DAILY_QUOTA_MESSAGE
  if (
      "429" in error_text
      or "rate limit" in error_text
      or "resource_exhausted" in error_text
      or "quota" in error_text
  ):
    return "Aslan Parçası şu anda çok yoğun istek alıyor. Birkaç saniye sonra tekrar deneyin."
  if "503" in error_text or "unavailable" in error_text or "overloaded" in error_text:
    return "Aslan Parçası'nın sunucuları şu an yoğun. Birkaç saniye sonra tekrar deneyin."
  return "Aslan Parçası yanıtı alınamadı. Lütfen biraz sonra tekrar deneyin."


def sanitize_model_output(text):
  """Strip hidden reasoning traces and keep only the user's answer."""
  raw = (text or "").strip()
  if not raw:
    return ""
  cleaned = raw

  lower = cleaned.lower()
  answer_markers = [
      "final answer:", "answer:", "son cevap:", "cevap:", "özet:", "sonuç:", "result:",
      "live answer:", "kısa cevap:", "yanıt:", "net cevap:",
  ]
  for marker in answer_markers:
    idx = lower.rfind(marker)
    if idx >= 0:
      remainder = cleaned[idx + len(marker):].strip("\n \t-:;")
      if remainder:
        return re.sub(r"\n\s*\n+", "\n\n", remainder).strip()

  if re.search(
      r"(?im)^\s*(?:here's\s+a\s+thinking\s+process|thinking\s+process|"
      r"analysis|reasoning|chain\s+of\s+thought|internal\s+reasoning|"
      r"1\s*[.)]\s*(?:\*\*)?analyze\s+user\s+input)\s*[:\-]",
      cleaned,
  ):
    return ""

  cleaned = re.sub(r"(?is)^\s*(?:here's\s+a\s+thinking\s+process|thinking\s+process|analyze\s+user\s+input|check\s+knowledge|step\s*\d+\s*:|1\.\s*\*\*analyze\s+user\s+input\*\*)\b.*?(?:\n|$)", "", cleaned)
  cleaned = re.sub(r"(?is)\n\s*(?:here's\s+a\s+thinking\s+process|thinking\s+process|analyze\s+user\s+input|check\s+knowledge|step\s*\d+\s*:).*", "", cleaned)
  cleaned = re.sub(r"(?is)^\s*(?:[-*•]\s*)?(?:analysis|reasoning|thinking|plan|research)\s*[:\-].*?(?:\n|$)", "", cleaned)
  cleaned = re.sub(r"(?is)\n\s*(?:[-*•]\s*)?(?:analysis|reasoning|thinking|plan|research)\s*[:\-].*", "", cleaned)
  cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
  return cleaned


class ModelOutputStreamSanitizer:
  ANSWER_MARKERS = (
      "final answer:", "answer:", "son cevap:", "cevap:", "özet:", "sonuç:", "result:",
      "live answer:", "kısa cevap:", "yanıt:", "net cevap:",
  )
  REASONING_HEADER = re.compile(
      r"^\s*(?:here's\s+a\s+thinking\s+process|thinking\s+process|analysis|"
      r"reasoning|chain\s+of\s+thought|internal\s+reasoning|plan|research)\s*[:\-]",
      re.I,
  )

  def __init__(self):
    self.pending = ""
    self.suppressing_reasoning = False

  def feed(self, text, final=False):
    self.pending += text or ""
    output = []
    while "\n" in self.pending:
      line, self.pending = self.pending.split("\n", 1)
      cleaned = self._clean_line(line)
      if cleaned:
        output.append(cleaned + "\n")
      elif not self.suppressing_reasoning:
        output.append("\n")
    if final and self.pending:
      cleaned = self._clean_line(self.pending)
      if cleaned:
        output.append(cleaned)
      self.pending = ""
    return "".join(output)

  def _clean_line(self, line):
    lowered = line.lower()
    answer_matches = [
        (lowered.rfind(marker), marker)
        for marker in self.ANSWER_MARKERS
        if marker in lowered
    ]
    answer_position, answer_marker = max(answer_matches, default=(-1, ""))

    if self.suppressing_reasoning:
      if answer_position < 0:
        return ""
      line = line[answer_position + len(answer_marker):].strip(" \t-:;")
      self.suppressing_reasoning = False
    else:
      reasoning_match = self.REASONING_HEADER.search(line)
      if reasoning_match:
        if answer_position <= reasoning_match.start():
          self.suppressing_reasoning = True
          return ""
        line = line[answer_position + len(answer_marker):].strip(" \t-:;")

    return sanitize_model_output(line)


LIVE_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(\s*https?://[^)\s]+\s*\)", re.I)
LIVE_URL_RE = re.compile(r"https?://\S+", re.I)


def strip_live_answer_links(text):
  text = LIVE_MARKDOWN_LINK_RE.sub(r"\1", text or "")
  text = LIVE_URL_RE.sub("", text)
  text = re.sub(r"[ \t]{2,}", " ", text)
  text = re.sub(r"\(\s*\)|\[\s*\]", "", text)
  return re.sub(r"[ \t]+([,.;!?])", r"\1", text).strip()


class LiveAnswerLinkSanitizer:
  """Buffer complete lines so URLs split across model chunks are removed."""

  def __init__(self):
    self.pending = ""

  def feed(self, text, final=False):
    self.pending += text or ""
    output = []
    while "\n" in self.pending:
      line, self.pending = self.pending.split("\n", 1)
      output.append(strip_live_answer_links(line) + "\n")
    if final and self.pending:
      output.append(strip_live_answer_links(self.pending))
      self.pending = ""
    return "".join(output)


def image_usage_today(user_id):
  """How many images this user generated today (resets at midnight, in memory)."""
  today = timezone.localdate().isoformat()
  with _image_request_lock:
    for stale in [key for key in _image_daily_usage if key[1] != today]:
      _image_daily_usage.pop(stale, None)
    return _image_daily_usage.get((user_id, today), 0)


def record_image_usage(user_id):
  today = timezone.localdate().isoformat()
  with _image_request_lock:
    key = (user_id, today)
    _image_daily_usage[key] = _image_daily_usage.get(key, 0) + 1


IMAGE_PROMPT_FILLER_WORDS = frozenset({
    "bir", "ve", "ile", "icin", "lütfen", "lutfen", "görsel", "gorsel",
    "resim", "oluştur", "olustur", "üret", "uret", "çiz", "ciz", "yap",
    "create", "generate", "draw", "image", "picture", "please", "make",
    "daha", "güzel", "guzel", "iyi", "şey", "sey", "rastgele",
    "kırmızı", "kirmizi", "mavi", "yeşil", "yesil", "siyah", "beyaz",
    "sarı", "sari", "mor", "pembe", "turuncu", "renkli",
    "red", "blue", "green", "black", "white", "yellow", "purple", "pink",
    "orange", "colorful", "realistic", "photorealistic", "ultra", "detailed",
})
IMAGE_PROMPT_WORD_RE = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+|\d+")


def validate_image_prompt(prompt):
  """Reject empty, command-only, and obvious keyboard-mash prompts with guidance."""
  text = (prompt or "").strip()
  if not text:
    return "Ne oluşturulacağını anlayamadım. Görselde olmasını istediğin konu veya nesneyi yaz."
  if len(text) > 4000:
    return "Görsel açıklaması çok uzun. Lütfen isteğini en fazla 4000 karakterle anlat."

  words = IMAGE_PROMPT_WORD_RE.findall(text)
  normalized_words = [_ascii_fold(word).lower() for word in words]
  meaningful = [
      word for word in normalized_words
      if len(word) >= 3
      and word not in IMAGE_PROMPT_FILLER_WORDS
      and not word.isdigit()
  ]
  if not meaningful:
    return (
        "Ne oluşturulacağını anlayamadım. Bir konu veya nesne belirt; örneğin "
        "'kırmızı bir daire' ya da 'masada duran gerçekçi bir mousepad'."
    )
  if any(
      len(word) >= 6 and not re.search(r"[aeıioöuü]", word)
      for word in meaningful
  ) or re.search(r"(asdf|qwer|zxcv|hjkl)", " ".join(normalized_words)):
    return (
        "İsteğin anlaşılır görünmüyor. Görselde görmek istediğin nesneyi, sahneyi "
        "ve varsa renk veya düzen ayrıntılarını açıkça yaz."
    )
  return None


def parse_image_data_url(image_url):
  """Validate a generated data URL before passing its image bytes to a model."""
  if not isinstance(image_url, str):
    raise ValueError("Düzenlenecek görsel verisi bulunamadı.")
  match = re.fullmatch(
      r"data:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/]*={0,2})",
      image_url.strip(),
      re.I,
  )
  if not match:
    raise ValueError("Düzenlenecek görsel JPEG, PNG veya WebP biçiminde olmalı.")
  encoded = match.group(2)
  try:
    image_bytes = base64.b64decode(encoded, validate=True)
  except (ValueError, base64.binascii.Error) as error:
    raise ValueError("Düzenlenecek görsel verisi okunamadı.") from error
  if not image_bytes:
    raise ValueError("Düzenlenecek görsel boş.")
  if len(image_bytes) > MAX_IMAGE_EDIT_BYTES:
    raise ValueError("Düzenlenecek görsel 20 MB sınırını aşıyor.")
  return {"mime_type": match.group(1).lower(), "data": encoded}


def generate_image_with_gemini(prompt, image_model, api_key, source_image=None):
  """Generate a new image or edit the supplied image through generateContent."""
  image_size = "1K" if image_model == "gemini-2.5-flash-image" else "4K"
  if source_image:
    image_prompt = (
        "Edit the supplied image itself. Apply only the changes explicitly requested "
        "below; preserve every unmentioned subject, object, identity, color, layout, "
        "camera angle, and background. Do not replace the scene or add anything "
        "unrequested. Follow the user's exact wording and all listed constraints. "
        "Exact user edit instructions:\n" + prompt
    )
    parts = [
        {
            "inlineData": {
                "mimeType": source_image["mime_type"],
                "data": source_image["data"],
            }
        },
        {"text": image_prompt},
    ]
  else:
    refined_prompt = enhance_image_prompt(prompt)
    if is_simple_geometric_prompt(prompt):
      image_prompt = refined_prompt
    else:
      image_prompt = (
          "Create exactly one exceptionally detailed 4K UHD image that follows the user's brief. "
        "Unless the brief explicitly requests an illustration, cartoon, logo, or another "
        "non-photographic style, render it as an ultra-photorealistic photograph captured "
        "with a professional full-frame camera. Use physically plausible light, natural "
        "skin and material textures, accurate anatomy and perspective, realistic depth of "
        "field, crisp focus on the subject, nuanced shadows, and restrained true-to-life "
        "color grading. Preserve the requested subject, count, action, and composition; "
          "render all colors, backgrounds, shapes, and products exactly as requested; do not "
          "invent unrelated objects. Do not add text, signatures, logos, borders, or provider "
          "branding in the image itself unless explicitly requested. Follow any "
          "style explicitly requested by the user instead of forcing photorealism. Exact user "
          "brief: "
          + prompt
          + "\nPhotographic translation and detail cues: "
          + refined_prompt
      )
    parts = [{"text": image_prompt}]
  response = requests.post(
      f"https://generativelanguage.googleapis.com/v1beta/models/{image_model}:generateContent",
      headers={
          "Content-Type": "application/json",
      },
      params={"key": api_key},
      json={
          "contents": [{"parts": parts}],
          "generationConfig": {
            "responseModalities": ["IMAGE"],
              "imageConfig": {"imageSize": image_size},
          },
      },
      timeout=180,
  )
  if response.status_code >= 400:
    raise UpstreamHTTPError(response.status_code, response.text[:400])
  payload = response.json()
  for candidate in payload.get("candidates") or []:
    for part in (candidate.get("content") or {}).get("parts") or []:
      inline_data = part.get("inlineData") or part.get("inline_data")
      if inline_data and inline_data.get("data"):
        media_type = inline_data.get("mimeType") or inline_data.get("mime_type") or "image/png"
        return f"data:{media_type};base64,{inline_data['data']}"
  raise ValueError("Görsel servisi yanıtında görsel verisi bulunamadı.")


def generate_edited_image_with_retry(prompt, source_image, api_keys=None):
  """Edit an existing image; never fall back to a text-to-image-only provider."""
  configured_keys = list(api_keys if api_keys is not None else get_api_keys())
  keys = ordered_keys(configured_keys)
  if not keys:
    raise ImageUnavailableError(
        "Görsel düzenleme için sunucuda Gemini görsel anahtarı tanımlı değil."
    )

  models = tuple(dict.fromkeys((
      GEMINI_IMAGE_MODEL,
      *GEMINI_IMAGE_FALLBACK_MODELS,
  )))
  last_error = None
  for key in keys:
    if not key_is_available(key):
      continue
    for model in models:
      try:
        result = generate_image_with_gemini(
            prompt, model, key, source_image=source_image
        )
        mark_key_ok(key)
        return result
      except Exception as error:
        last_error = error
        status = getattr(error, "status_code", None)
        if is_daily_quota_error(error):
          mark_key_daily(key, error)
          break
        if status == 429:
          mark_key_rate(key, error, _retry_after(error))
          break
        if status in (401, 403):
          mark_key_invalid(key, error)
          break

  if last_error:
    logger.warning("Image editing provider failed: %s", last_error)
  raise ImageUnavailableError(
      "Görsel şu anda düzenlenemedi. Görsel sağlayıcısına erişim veya kota ayarlarını "
      "kontrol edip tekrar dene."
  ) from last_error


def generate_image_with_flux_hf(prompt, seed=None):
  """Preferred Flux.1 path over Pollinations when a Hugging Face token is configured."""
  token = os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
  if not token:
    raise ValueError("HF_API_TOKEN not configured")
  payload = build_flux_image_params(prompt, seed=seed, width=1024, height=1024)
  response = requests.post(
      f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}",
      headers={
          "Authorization": f"Bearer {token}",
          "Content-Type": "application/json",
      },
      json={
          "inputs": payload["prompt"],
          "parameters": {
              "guidance_scale": 4.5,
              "num_inference_steps": 25,
              "seed": payload["seed"],
              "width": payload["width"],
              "height": payload["height"],
          },
      },
      timeout=180,
  )
  if response.status_code >= 400:
    raise UpstreamHTTPError(response.status_code, response.text[:400])
  content_type = (response.headers.get("Content-Type") or "image/png").split(";")[0].strip().lower()
  if response.content and content_type.startswith("image/"):
    encoded = base64.b64encode(response.content).decode("ascii")
    return f"data:{content_type};base64,{encoded}"
  if response.headers.get("content-type", "").lower().startswith("application/json"):
    payload = response.json()
    if isinstance(payload, dict):
      image = payload.get("image") or payload.get("images")
      if isinstance(image, str):
        return f"data:image/png;base64,{image}"
      if isinstance(image, list) and image:
        first = image[0]
        if isinstance(first, str):
          return f"data:image/png;base64,{first}"
  raise ValueError("Flux.1 görsel üretimi boş veya beklenen formatta dönmedi.")


def generate_image_with_pollinations(prompt, seed=None):
  """Stable Pollinations fallback with explicit prompt refinement and URL-safe encoding."""
  params = build_flux_image_params(prompt, seed=seed, width=1024, height=1024)
  response = requests.get(
      f"https://image.pollinations.ai/prompt/{quote(params['prompt'], safe='')}",
      params={
          "width": params["width"],
          "height": params["height"],
          "nologo": "true",
          "safe": params["safe"],
          "model": params["model"],
          "seed": params["seed"],
          "enhance": "false",
      },
      headers={"User-Agent": "Mozilla/5.0 (compatible; AslanParcasi/1.0)"},
      timeout=150,
  )
  if response.status_code >= 400:
    raise UpstreamHTTPError(response.status_code, response.text[:200])
  content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
  if not response.content or not content_type.startswith("image/"):
    raise ValueError(f"Yedek görsel servisi görsel döndürmedi ({content_type or 'boş yanıt'}).")
  encoded = base64.b64encode(response.content).decode("ascii")
  return f"data:{content_type};base64,{encoded}"


def generate_image_with_retry(prompt, image_model=None, api_keys=None):
  """Try premium Gemini image models first, then configured Flux without public branding."""
  configured_keys = list(api_keys if api_keys is not None else get_api_keys())
  keys = ordered_keys(api_keys)
  flux_configured = bool(
      os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
  )
  if not configured_keys and not flux_configured:
    try:
      return generate_image_with_pollinations(prompt)
    except Exception as fallback_error:
      logger.warning("Keyless image fallback failed: %s", fallback_error)
      raise ImageUnavailableError(
          "Sunucuda görsel API anahtarı tanımlı değil ve yedek görsel servisi de yanıt vermedi."
      ) from fallback_error
  models = []
  for candidate in (image_model, GEMINI_IMAGE_MODEL, *GEMINI_IMAGE_FALLBACK_MODELS):
    if candidate and candidate not in models:
      models.append(candidate)

  last_error = None
  for key in keys:
    if not key_is_available(key):
      continue
    for model in models:
      try:
        result = generate_image_with_gemini(prompt, model, key)
        mark_key_ok(key)
        return result
      except Exception as error:
        last_error = error
        status = getattr(error, "status_code", None)
        if is_daily_quota_error(error):
          mark_key_daily(key, error)
          break
        if status == 429:
          mark_key_rate(key, error, _retry_after(error))
          break
        if status in (401, 403):
          mark_key_invalid(key, error)
          break

  if last_error is not None:
    logger.warning("Primary image provider failed, falling back: %s", last_error)
  flux_error = None
  if os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN"):
    try:
      return generate_image_with_flux_hf(prompt)
    except Exception as image_error:
      flux_error = image_error
      logger.warning("Flux image generation failed after Gemini fallbacks: %s", image_error)
  fallback_error = None
  try:
    return generate_image_with_pollinations(prompt)
  except Exception as image_error:
    fallback_error = image_error
    logger.warning("Keyless image fallback failed after primary providers: %s", image_error)
  if last_error is not None:
    status = getattr(last_error, "status_code", None)
    error_text = str(last_error).lower()
    if is_daily_quota_error(last_error) or status == 429 or "rate limit" in error_text:
      raise ImageUnavailableError(
          "Görsel üretim kotası dolmuş veya hız sınırına ulaşılmış. Bir süre sonra tekrar dene."
      ) from last_error
    if status in (401, 403):
      raise ImageUnavailableError(
          "Görsel üretim anahtarı doğrulanamadı veya bu modele erişemiyor. Sunucu yapılandırması kontrol edilmeli."
      ) from last_error
    if status == 404:
      raise ImageUnavailableError(
          "Görsel modelleri bu sunucu anahtarı için etkin değil. Sunucu model erişimi kontrol edilmeli."
      ) from last_error
    if status == 400:
      raise ImageUnavailableError(
          "Görsel isteği model tarafından kabul edilmedi. İstek boyutu veya görsel ayarları kontrol edilmeli."
      ) from last_error
  if flux_error is not None:
    status = getattr(flux_error, "status_code", None)
    if status in (401, 403):
      raise ImageUnavailableError(
          "Görsel yedeği doğrulanamadı. Sunucudaki görsel erişim anahtarı kontrol edilmeli."
      ) from flux_error
  if fallback_error is not None and not keys:
    raise ImageUnavailableError(
        "Görsel üretim anahtarları kota sınırında ve yedek görsel servisi yanıt vermedi."
    ) from fallback_error
  raise ImageUnavailableError(IMAGE_UNAVAILABLE_MESSAGE)


def transcribe_audio(encoded, mime_type, api_keys=None):
  """Server-side speech-to-text so voice notes work on every browser."""
  keys = ordered_keys(api_keys)
  last_error = None
  attempted = False
  for key in keys:
    if not key_is_available(key):
      continue
    for model in (GEMINI_MODEL, GEMINI_FAST_MODEL):
      attempted = True
      try:
        transcript = _transcribe_with_key(encoded, mime_type, key, model)
        mark_key_ok(key)
        return transcript
      except Exception as error:
        last_error = error
        status = getattr(error, "status_code", None)
        if is_daily_quota_error(error):
          mark_key_daily(key, error)
          break
        if status == 429:
          mark_key_rate(key, error, _retry_after(error))
          break
        if status in (401, 403):
          mark_key_invalid(key, error)
          break
  if not attempted:
    raise QuotaBreakerError(QUOTA_STATE["last_error"] or DAILY_QUOTA_MESSAGE)
  raise last_error


def _transcribe_with_key(encoded, mime_type, api_key, model):
  response = requests.post(
      f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
      headers={"Content-Type": "application/json"},
      params={"key": api_key},
      json={
          "contents": [{
              "parts": [
                  {"inline_data": {"mime_type": mime_type, "data": encoded}},
                  {
                      "text": (
                          "Bu ses kaydındaki konuşmayı olduğu gibi, eksiksiz ve düz metin "
                          "olarak yaz. Konuşma yoksa yalnızca NO_SPEECH yaz. Başka açıklama ekleme."
                      ),
                  },
              ],
          }],
      },
      timeout=120,
  )
  if response.status_code >= 400:
    raise UpstreamHTTPError(response.status_code, response.text[:400])
  payload = response.json()
  parts_text = []
  for candidate in payload.get("candidates") or []:
    for part in (candidate.get("content") or {}).get("parts") or []:
      if part.get("text"):
        parts_text.append(part["text"])
  transcript = " ".join(parts_text).strip()
  if not transcript or transcript.upper() == "NO_SPEECH":
    return ""
  return transcript


def _endpoint_list(client, model):
  """Normalise whatever was passed in to a list of Endpoint tuples."""
  if client is None:
    return []
  items = client if isinstance(client, (list, tuple)) else [client]
  endpoints = []
  for index, item in enumerate(items):
    if isinstance(item, Endpoint):
      if item.client is not None:
        endpoints.append(item)
    elif isinstance(item, (list, tuple)) and len(item) >= 2 and item[1] is not None:
      base_model = model or GEMINI_MODEL
      candidates = (base_model,) if base_model == GEMINI_MODEL else (base_model, GEMINI_MODEL)
      fallback = GEMINI_FAST_MODEL if base_model == GEMINI_MODEL else None
      endpoints.append(Endpoint("gemini", str(item[0]), item[1], candidates, fallback))
    elif item is not None:
      base_model = model or GEMINI_MODEL
      candidates = (base_model,) if base_model == GEMINI_MODEL else (base_model, GEMINI_MODEL)
      fallback = GEMINI_FAST_MODEL if base_model == GEMINI_MODEL else None
      endpoints.append(Endpoint("gemini", f"client-{index}", item, candidates, fallback))
  return endpoints


def safe_model_call(
    client,
    messages,
    model,
    temperature=0.7,
    max_tokens=4096,
    stream=False,
    deep_think=False,
    tools=None,
    timeout=None,
    deadline=None,
    skip_providers=(),
    trace=None,
):
    """Call the model, rotating across providers and keys so one limit never
    reaches the user.

    The personality is identical no matter which provider answers: every
    endpoint gets the same messages, temperature and max_tokens.
    `skip_providers` lets the caller retry on a different brain after a
    provider answered with an empty stream; `trace` receives the provider
    that finally served the request.
    """
    endpoints = _endpoint_list(client, model)
    if not endpoints:
        raise RuntimeError("Aslan Parçası için model istemcisi yapılandırılmamış.")
    skipped = {name for name in (skip_providers or ()) if name}
    if skipped:
        preferred = [endpoint for endpoint in endpoints if endpoint.provider not in skipped]
        if preferred:
            endpoints = preferred
    rotate = len(endpoints) > 1
    if rotate:
        endpoints = ordered_endpoints(endpoints)

    last_error = None
    attempted = False
    for endpoint in endpoints:
        if rotate and not key_is_available(endpoint.key):
            continue
        if deadline is not None and deadline - time.monotonic() <= 3:
            break
        attempted = True
        bind_key_provider(endpoint.key, endpoint.provider)
        try:
            result = _call_single_client(
                endpoint.client,
                messages,
                endpoint.models,
                endpoint.fallback,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                tools=tools,
                timeout=timeout,
                deadline=deadline,
                rotate=rotate,
            )
            mark_key_ok(endpoint.key)
            if trace is not None:
                trace["provider"] = endpoint.provider
            return result
        except DailyQuotaError as error:
            mark_key_daily(endpoint.key, error)
            last_error = error
        except KeyRateLimitedError as error:
            mark_key_rate(endpoint.key, error, error.retry_after)
            last_error = error
        except KeyRejectedError as error:
            mark_key_invalid(endpoint.key, error)
            last_error = error
        except (ModelUnavailableError, UpstreamBusyError) as error:
            last_error = error
        except Exception as error:
            # An unexpected upstream failure must never reach the user while
            # another provider can still answer.
            record_quota_error(error, "error")
            last_error = error
    if not attempted:
        raise QuotaBreakerError(QUOTA_STATE["last_error"] or DAILY_QUOTA_MESSAGE)
    raise last_error


def _call_single_client(
    client,
    messages,
    models,
    fallback_model=None,
    temperature=0.7,
    max_tokens=4096,
    stream=False,
    tools=None,
    timeout=None,
    deadline=None,
    rotate=False,
):
    candidates = [m for m in (models or ()) if m] or [GEMINI_MODEL]
    kwargs = {
        "model": GEMINI_MODEL if candidates[0] == "ignored-model-name" else candidates[0],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    last_error = None
    model_index = 0
    downgraded = False
    for attempt in range(3):
        call_timeout = timeout
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 3:
                break
            call_timeout = max(5, min(call_timeout or 60, int(remaining) - 2))
        call_kwargs = dict(kwargs)
        if call_timeout is not None:
            call_kwargs["timeout"] = int(call_timeout)
        try:
            result = client.chat.completions.create(**call_kwargs)
        except Exception as error:
            last_error = error
            error_text = str(error).lower()
            status = getattr(error, "status_code", None)
            retry_after = _retry_after(error)
            if is_daily_quota_error(error):
                raise DailyQuotaError(str(error)[:300]) from error
            if status in (401, 403) or "invalid api key" in error_text or "permission" in error_text:
                raise KeyRejectedError(str(error)[:300]) from error
            is_quota = (
                status == 429
                or "429" in error_text
                or "rate limit" in error_text
                or "resource_exhausted" in error_text
                or "quota" in error_text
            )
            if is_quota:
                # With several endpoints we keep the smart model and rotate
                # instead of downgrading, so the personality never changes.
                if not downgraded and not tools and not rotate and fallback_model and kwargs["model"] != fallback_model:
                    downgraded = True
                    kwargs["model"] = fallback_model
                    record_quota_error(error, "rate")
                    continue
                raise KeyRateLimitedError(str(error)[:300], retry_after) from error
            if status == 404 or "404" in error_text or "not found" in error_text:
                if model_index + 1 < len(candidates):
                    model_index += 1
                    kwargs["model"] = candidates[model_index]
                    continue
                raise ModelUnavailableError(str(error)[:300]) from error
            busy = (
                status in (500, 502, 503, 504)
                or "503" in error_text
                or "unavailable" in error_text
                or "overloaded" in error_text
                or "timeout" in error_text
                or "timed out" in error_text
            )
            if not busy:
                raise
            if rotate or attempt == 2:
                raise UpstreamBusyError(str(error)[:300]) from error
            if kwargs["model"] == "ignored-model-name":
              kwargs["model"] = GEMINI_MODEL
            sleep_for = max(float(MODEL_BACKOFFS[attempt]), retry_after or 0)
            if deadline is not None and time.monotonic() + sleep_for > deadline - 2:
                raise UpstreamBusyError(str(error)[:300]) from error
            time.sleep(min(sleep_for, 15))
            continue
        else:
            if not stream and not hasattr(result, "choices"):
                # Compatibility wrapper/tests may return a plain completion object
                # such as a string or a custom iterator-like payload; that still
                # counts as a valid reply unless the provider explicitly says it
                # is empty.
                return result
            if not stream and not (getattr(result, "choices", None) or []):
                # Bazı sağlayıcılar 200 dönüp gövdeyi boş bırakıyor; bunu yoğunluk
                # sayıp rotasyonu bir sonraki beyne taşıyoruz.
                last_error = UpstreamBusyError("Model boş yanıt döndürdü (choices yok).")
                if model_index + 1 < len(candidates):
                    model_index += 1
                    kwargs["model"] = candidates[model_index]
                continue
            return result
    raise UpstreamBusyError(str(last_error)[:300])


def _retry_after(error):
  headers = getattr(getattr(error, "response", None), "headers", None)
  if headers is None:
    return None
  try:
    value = float(headers.get("retry-after"))
  except (TypeError, ValueError):
    return None
  return value if 0 < value <= 120 else None


# Uzun düşünme bütçesi boşa harcanmasın diye sunucunun kendi başına denediği açılar.
RESEARCH_FACETS = (
    "nasıl çalışır",
    "örnekleri",
    "tarihçesi",
    "güncel gelişmeler",
    "istatistikler",
    "avantajları dezavantajları",
    "uzman yorumları",
    "bilimsel açıklama",
    "kullanım alanları",
    "riskleri ve eleştiriler",
)


def depth_instruction(seconds):
  evidence_guidance = (
    "Önce güçlü ve güncel kanıtları değerlendir; kaynaklar uyuşmuyorsa bunu belirt. "
    "Doğrulanmış bilgiyi, çıkarımı ve bilinmeyeni ayır. Gizli düşünce zincirini yazma; "
    "sonuçları ve kullanıcıya yararlı kısa gerekçeleri sun. "
  )
  if seconds <= 60:
    return (
        evidence_guidance
        + "Kısa düşünme bütçesi kullanıldı: net, öz ama gerekçeli bir yanıt ver; "
        "en fazla 3 madde."
    )
  if seconds <= 300:
    return (
        evidence_guidance
        + "Orta düzey düşünme bütçesi kullanıldı: başlıklarla yapılandırılmış, örnekli "
        "ve karşılaştırmalı bir yanıt ver."
    )
  if seconds <= 900:
    return (
        evidence_guidance
        + "Derin düşünme bütçesi kullanıldı: bölümler halinde ayrıntılı analiz yap, "
        "karşıt görüşleri ve riskleri değerlendir, sonunda net bir sonuç bölümü ver."
    )
  return (
      evidence_guidance
      + "Uzman düzey düşünme bütçesi kullanıldı: kapsamlı bir rapor yaz: yönetici özeti, "
      "yöntem, ayrıntılı bölümler, karşıt görüşler, riskler, kaynak değerlendirmesi ve "
      "sonuç önerileri. İlgili ve kanıtlanabilir ayrıntıları işle."
  )


def _last_user_text(messages):
  """The user's question as plain text, whatever shape the content has."""
  for item in reversed(messages or []):
    if not isinstance(item, dict) or item.get("role") != "user":
      continue
    content = item.get("content")
    if isinstance(content, str):
      return content
    if isinstance(content, list):
      return " ".join(
          str(part.get("text", "")) for part in content if isinstance(part, dict)
      ).strip()
  return ""


def deep_think_call(client, messages, model, seconds, tools=None, temperature=0.3, max_tokens=1024):
  """Compatibility wrapper used by the test suite and the UI when a deep-think response is streamed."""
  research = list(messages)
  initial = safe_model_call(
      client,
      research,
      model,
      temperature=temperature,
      max_tokens=max_tokens,
      stream=False,
      tools=tools,
      timeout=30,
  )
  choice = getattr(initial, "choices", None)
  if choice:
    assistant_message = choice[0].message
    tool_calls = getattr(assistant_message, "tool_calls", None) or []
    if tool_calls:
      research.append({
          "role": "assistant",
          "content": getattr(assistant_message, "content", None),
          "tool_calls": [
              {
                  "id": call.id,
                  "type": "function",
                  "function": {
                      "name": call.function.name,
                      "arguments": call.function.arguments,
                  },
              }
              for call in tool_calls
          ],
      })
      for call in tool_calls:
        try:
          arguments = json.loads(call.function.arguments or "{}")
        except (TypeError, json.JSONDecodeError):
          arguments = {}
        result = execute_function(call.function.name, arguments)
        research.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result, ensure_ascii=False),
        })
      final = safe_model_call(
          client,
          research,
          model,
          temperature=temperature,
          max_tokens=max_tokens,
          stream=True,
          timeout=30,
      )
      for chunk in final:
        choices = getattr(chunk, "choices", None)
        if not choices:
          continue
        delta = getattr(choices[0], "delta", None)
        piece = getattr(delta, "content", None) if delta else None
        if piece:
          yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])
      return

  if hasattr(initial, "choices"):
    for chunk in initial:
      choices = getattr(chunk, "choices", None)
      if not choices:
        continue
      delta = getattr(choices[0], "delta", None)
      piece = getattr(delta, "content", None) if delta else None
      if piece:
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])
    return

  if isinstance(initial, (list, tuple)):
    for item in initial:
      if hasattr(item, "choices"):
        yield item
        continue
      if isinstance(item, str):
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=item))])
    return

  if initial is not None:
    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=str(initial)))])


def deep_think_events(client, messages, seconds, temperature, max_tokens):
  """Research for the full selected budget, then stream the final answer.

  Yields ("reason", line) events for the thinking box and ("answer", text)
  chunks for the reply. The loop keeps working until the deadline minus a
  reserve for the final answer, so 30 s and 30 min budgets differ in depth.
  Real web research is guaranteed: the first search runs server-side before
  the model is asked for anything, so a model that never calls a tool still
  answers from live sources.
  """
  total_budget = max(30, min(1800, int(seconds)))
  deadline = compute_deep_think_deadline(total_budget)
  reserve = max(10, min(60, int(total_budget * 0.1)))
  question = _last_user_text(messages)
  research = [*messages]
  research[0] = {
      **messages[0],
      "content": (
          str(messages[0].get("content", ""))
          + " Araştırma modundasın: web_search ve get_weather araçlarını kullanarak "
          "konuyu derinlemesine incele. Her turda yeni bir açı dene, kaynakları "
          "karşılaştır. Gizli düşünce zincirini kullanıcıya yazma; bulguları final "
          "yanıtta özetle."
      ),
  }

  rounds = 0
  max_rounds = max(3, min(24, int(seconds) // 20))
  yield ("reason", f"Derin düşünme başladı · bütçe {int(seconds)} sn")

  findings = []
  sports_topic = is_sports_question(question)
  recency_days = 7 if sports_topic else None

  def add_findings(result):
    for item in (result.get("results") or []):
      if not isinstance(item, dict) or not (item.get("title") or item.get("url")):
        continue
      line = f"- {item.get('title', '')}"
      if item.get("url"):
        line += f" ({item['url']})"
      detail = item.get("excerpt") or item.get("snippet")
      if detail:
        line += f": {detail}"
      if line not in findings:
        findings.append(line)

  seen_lines = set()

  def report_sources(result, per_engine_limit=5):
    """Track distinct sources without exposing links in the user-facing research UI."""
    fresh = 0
    for line in format_source_lines(result, per_engine_limit * 2):
      if line in seen_lines:
        continue
      seen_lines.add(line)
      fresh += 1
      if fresh <= per_engine_limit:
        yield ("reason", "SRC:🌐 Kaynak: farklı bir yayın kontrol ediliyor")

  def search_with_fallback(query, num_results, enrich=2):
    result = web_search(
      query, num_results, duckduckgo_only=True, deadline=deadline - reserve
    )
    if result.get("results") and result.get("engine") != "weak-match":
      return result
    yield ("reason", "İlk arama engellendi veya yeterli kaynak vermedi; genel arama ve sade sorgu deneniyor.")
    best_fallback = None
    variants = dict.fromkeys((query, search_topic(query)))
    for variant in variants:
      if not variant:
        continue
      fallback = web_search(
        variant, num_results, duckduckgo_only=False, deadline=deadline - reserve,
        recency_days=recency_days, enrich=enrich,
      )
      if fallback.get("results") and fallback.get("engine") != "weak-match":
        return fallback
      if fallback.get("results") and best_fallback is None:
        best_fallback = fallback
    return best_fallback or result

  def facet_search(index):
    """Server-driven research step so the whole budget is used, never idled."""
    facet = RESEARCH_FACETS[index % len(RESEARCH_FACETS)]
    facet_query = f"{search_topic(question)} {facet}".strip()
    yield ("reason", f"SRC:🔎 Ek araştırma ({index + 1}. tur): {facet_query[:90]}")
    result = yield from search_with_fallback(facet_query, 5)
    left = max(0, int(deadline - time.monotonic()))
    if result.get("blocked") or not result.get("results"):
      yield ("reason", f"Bu açıdan kaynak çıkmadı; {left} sn kaldı, farklı bir açı deneniyor.")
      return
    if result.get("engine") == "weak-match":
      yield ("reason", f"Bu açıdan zayıf kaynak çıktı; {left} sn kaldı, farklı bir açı deneniyor.")
      add_findings(result)
      return
    add_findings(result)
    new_sources = 0
    for event in report_sources(result):
      new_sources += 1
      yield event
    if not new_sources:
      yield ("reason", f"Bu açının kaynakları zaten doğrulanmıştı; {left} sn kaldı, yeni açı deneniyor.")
    else:
      yield ("reason", f"{new_sources} yeni kaynak doğrulandı, bulgular karşılaştırılıyor... ({left} sn kaldı)")
    research.append({
        "role": "system",
        "content": (
            f"'{facet}' açısından toplanan ek web bulguları:\n"
            + "\n".join(
                f"- {item.get('title')}: {item.get('excerpt') or item.get('snippet') or ''}"
                for item in (result.get("results") or [])[:5]
            )
        ),
    })

  if question:
    yield ("reason", f"SRC:🔎 Canlı arama: {build_search_query(question)[:90]}")
    seeded = yield from search_with_fallback(build_search_query(question), 6, enrich=3)
    if not seeded.get("results"):
      # Kaynak bulunamamak araştırmanın sonu değil: bütçe boyunca farklı açı denenir.
      yield ("reason", "İlk arama kaynak vermedi; bütçe boyunca farklı motor ve açılar denenecek.")
    elif seeded.get("engine") == "weak-match":
      yield ("reason", "İlk arama konuyla ilgili güçlü kaynak vermedi; ek turlarda yeniden denenecek.")
      add_findings(seeded)
    else:
      add_findings(seeded)
      for event in report_sources(seeded, 6):
        yield event
  if findings:
    research.append({
        "role": "system",
        "content": (
            "Derin düşünme için sunucunun topladığı gerçek web bulguları (bunları "
            "doğrula, genişlet ve yanıtta kaynak olarak kullan):\n" + "\n".join(findings[:12])
        ),
    })
    yield ("reason", f"{len(findings)} kaynak toplandı, karşılaştırılıyor...")

  tools = TOOLS
  facet_index = 0
  while rounds < max_rounds:
    remaining = deadline - time.monotonic()
    if remaining <= reserve:
      break
    try:
      completion = safe_model_call(
          client,
          research,
          GEMINI_MODEL,
          temperature=0.3,
          max_tokens=1024,
          stream=False,
          tools=tools,
          timeout=max(5, min(45, int(remaining - reserve))),
          deadline=deadline - reserve,
      )
    except (
        QuotaBreakerError,
        DailyQuotaError,
        KeyRateLimitedError,
        KeyRejectedError,
        ModelUnavailableError,
    ):
      yield ("reason", "Model kotası şu an dolu; araştırma kısaltılıyor, mevcut bulgularla yanıt yazılacak.")
      break
    except Exception as error:
      error_text = str(error).lower()
      tool_rejected = (
          tools
          and ("tool" in error_text or "function" in error_text)
          and ("support" in error_text or "invalid" in error_text or "not allowed" in error_text)
      )
      if tool_rejected:
        tools = None
        yield ("reason", "Bu beyin araç çağrısını desteklemiyor; toplanan kaynaklarla derin analiz yapılıyor.")
        rounds += 1
        continue
      yield ("reason", f"Araştırma adımı aksadı, yeniden deneniyor ({type(error).__name__})")
      time.sleep(min(5, max(1, deadline - time.monotonic() - reserve)))
      rounds += 1
      continue

    choice = completion.choices[0] if completion.choices else None
    assistant_message = choice.message if choice else None
    tool_calls = getattr(assistant_message, "tool_calls", None) or []
    if not tool_calls:
      gap_note = (
          f"Kalan süre {max(0, int(deadline - time.monotonic()))} sn. Eksik kalan "
          "noktaları belirle; web_search aracıyla YENİ bir açı ara, bulduklarını "
          "karşılaştır ve bulgularını maddele."
      )
      research.append({"role": "user", "content": gap_note})
      yield ("reason", "Model araç çağırmadı; sunucu yeni bir açıdan araştırıyor...")
      yield from facet_search(facet_index)
      facet_index += 1
      rounds += 1
      continue

    research.append({
        "role": "assistant",
        "content": getattr(assistant_message, "content", None),
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ],
    })
    for call in tool_calls:
      if time.monotonic() >= deadline - reserve:
        break
      try:
        arguments = json.loads(call.function.arguments or "{}")
      except (TypeError, json.JSONDecodeError):
        arguments = {}
      short_args = json.dumps(arguments, ensure_ascii=False)[:120]
      yield ("reason", f"SRC:🔎 {call.function.name} çağrıldı: {short_args}")
      if call.function.name == "web_search":
        result = yield from search_with_fallback(
            arguments.get("query"), arguments.get("num_results", 5)
        )
      else:
        result = execute_function(call.function.name, arguments)
      research.append({
          "role": "tool",
          "tool_call_id": call.id,
          "content": json.dumps(result, ensure_ascii=False),
      })
      if call.function.name == "web_search":
        if result.get("blocked") or not result.get("results"):
          yield ("reason", "Modelin aradığı sorgu kaynak vermedi; sunucu farklı bir açı deniyor.")
        elif result.get("engine") != "weak-match":
          add_findings(result)
          for event in report_sources(result):
            yield event
        else:
          yield ("reason", "Modelin aradığı sorgu konuyla ilgili güçlü kaynak vermedi; atlandı.")
      elif call.function.name == "get_weather":
        if result.get("temperature") is not None:
          findings.append(
              f"- {result.get('city')} hava durumu: {result.get('temperature')}°C, "
              f"{result.get('description')} (kaynak: Open-Meteo)"
          )
          yield ("reason", f"SRC:🌤 Hava verisi: {result.get('city')} {result.get('temperature')}°C")
      elif call.function.name == "get_current_time":
        findings.append(f"- Sunucu saati: {result.get('time')} {result.get('date')} {result.get('day')}")
        yield ("reason", f"SRC:🕒 Saat verisi: {result.get('time')} {result.get('date')}")
    rounds += 1

  # Bütçe bitene kadar boş durma: her turda yeni bir açıdan gerçekten araştır.
  while time.monotonic() < deadline - reserve and question:
    yield from facet_search(facet_index)
    facet_index += 1
    remaining = max(0.0, deadline - time.monotonic())
    pause = min(8, max(1, remaining))
    if pause > 0:
      yield ("reason", f"Toplam {len(findings)} bulgu birikti; {int(remaining)} sn kaldı, yeni tur hazırlanıyor...")
      time.sleep(min(pause, 5))

  if not findings:
    # Kaynak yoksa susma: model kendi bilgisiyle yanıtlar, uydurması yasaklanır.
    research.append({
        "role": "system",
        "content": (
            "Bu turda doğrulanabilir web kaynağı toplanamadı. Yanıtı kendi bilgine "
            "dayandır, güncel olabilecek veri (skor, fiyat, tarih, sürüm) UYDURMA ve "
            "yanıtın başında tek cümleyle canlı kaynağa ulaşılamadığını belirt."
        ),
    })
    yield ("reason", "Canlı kaynak bulunamadı; yanıt genel bilgiyle, dürüst bir uyarıyla yazılıyor.")

  target_words = min(1800, 250 + int(seconds * 0.6))
  final_messages = [
      *research,
      {
          "role": "system",
          "content": (
              "Yanıtını araştırma bulgularına dayandır ve kaynakları (başlık + bağlantı) "
              "yanıtın sonundaki 'Kaynaklar' bölümünde listele. Araç hata verdiyse veri "
              "uydurma. Kaynakların tarihini ve iddialarını karşılaştır; çelişkileri belirt. "
              "Doğrulanmış bilgileri, çıkarımları ve bilinmeyenleri birbirinden ayır. İç "
              "düşünce zincirini veya gizli analiz notlarını gösterme; kısa gerekçeler ve "
              "kanıtları sun. Kaynaklarda başka yapay zeka markalarının adı geçerse bunları yanıtta "
              "ANMA; sen Aslan Parçası'sın. Yarım paragraf bırakma: giriş, başlıklı bölümler, "
              "maddeler, karşıt görüşler, riskler ve sonuç bölümü olan UZUN bir rapor yaz. "
              f"Hedef uzunluk en az {target_words} kelime. "
              + depth_instruction(int(seconds))
          ),
      },
  ]
  yield ("reason", "Yanıt yazılıyor...")
  answer_parts = []
  has_answer = False
  answer_released = False
  silent_providers = set()
  for _attempt in range(2):
    if time.monotonic() >= deadline:
      break
    trace = {}
    try:
      final_completion = safe_model_call(
          client,
          final_messages,
          GEMINI_MODEL,
          temperature=temperature,
          max_tokens=max_tokens,
          stream=True,
          timeout=max(15, min(120, int(seconds * 0.1) + 15)),
          skip_providers=silent_providers,
          trace=trace,
      )
    except Exception as error:
      yield ("reason", f"Yanıt yazımı aksadı ({type(error).__name__}); toplanan bulgular aktarılıyor.")
      break
    for chunk in final_completion:
      choices = getattr(chunk, "choices", None)
      if not choices:
        continue
      delta = getattr(choices[0], "delta", None)
      piece = getattr(delta, "content", None) if delta else None
      if piece:
        has_answer = True
        answer_parts.append(piece)
        if time.monotonic() >= deadline:
          if not answer_released:
            for buffered_piece in answer_parts[:-1]:
              yield ("answer", buffered_piece)
            answer_released = True
          yield ("answer", piece)
    if has_answer:
      if not answer_released:
        while time.monotonic() < deadline:
          time.sleep(min(0.25, deadline - time.monotonic()))
        for piece in answer_parts:
          yield ("answer", piece)
        answer_released = True
      break
    if trace.get("provider"):
      silent_providers.add(trace["provider"])

  if not has_answer:
    if findings:
      visible_findings = [line for line in findings if is_brand_safe(line)]
      answer_parts = [
          "Araştırmayı tamamladım ama yanıt modeli bu kez boş döndü; topladığım gerçek "
          "kaynakları olduğu gibi aktarıyorum:\n\n" + "\n".join(visible_findings[:15])
      ]
    else:
      answer_parts = [
          "DuckDuckGo bu süre içinde konuya uygun doğrulanabilir kaynak bulamadı; "
          "kaynaksız bir araştırma raporu yazmayacağım."
      ]
    while time.monotonic() < deadline:
      time.sleep(min(0.25, deadline - time.monotonic()))
    for piece in answer_parts:
      yield ("answer", piece)


@login_required(login_url="login")
def index(request):
  if not request.user.email:
    return redirect("update_email")
  sync_app_clock()
  profile = get_user_profile(request.user)
  return render(
      request,
      "dashboard/index.html",
      {"user_settings": json.dumps(profile_payload(request.user))},
  )


@login_required(login_url="login")
@require_POST
def update_profile_view(request):
  try:
    data = json.loads(request.body or "{}")
  except json.JSONDecodeError:
    return JsonResponse({"error": "Geçersiz JSON."}, status=400)

  profile = get_user_profile(request.user)
  avatar = data.get("avatar")
  theme = (data.get("theme") or profile.theme).strip()
  pattern = (data.get("pattern") or profile.pattern).strip()

  allowed_themes = {"theme-cyber", "theme-matrix", "theme-sunset", "theme-deepspace", "theme-minimal"}
  allowed_patterns = {"pattern-grid", "pattern-dots", "pattern-lines", "pattern-gradient", "pattern-plain"}
  if theme not in allowed_themes or pattern not in allowed_patterns:
    return JsonResponse({"error": "Geçersiz tema veya arka plan deseni."}, status=400)
  if avatar is not None:
    if avatar and (not isinstance(avatar, str) or not avatar.startswith(("data:image/", "http://", "https://"))):
      return JsonResponse({"error": "Geçersiz profil fotoğrafı."}, status=400)
    if isinstance(avatar, str) and len(avatar) > 5_000_000:
      return JsonResponse({"error": "Profil fotoğrafı çok büyük. Daha küçük bir görsel seçin."}, status=400)
    profile.avatar = avatar
  profile.theme = theme
  profile.pattern = pattern
  profile.save()
  return JsonResponse({"status": "success", "settings": profile_payload(request.user)})


@login_required(login_url="login")
def update_email_view(request):
  if request.user.email and request.method != "POST":
    return redirect("index")

  error = None
  if request.method == "POST":
    email = (request.POST.get("email") or "").strip()
    if not email:
      error = "E-posta adresi gerekli."
    elif User.objects.filter(email__iexact=email).exclude(pk=request.user.pk).exists():
      error = "Bu e-posta adresi zaten kullanılıyor."
    else:
      request.user.email = email
      request.user.save(update_fields=["email"])
      return redirect("index")
  return render(request, "dashboard/update_email.html", {"error": error})


@login_required(login_url="login")
@require_POST
def update_username_view(request):
  try:
    data = json.loads(request.body or "{}")
  except json.JSONDecodeError:
    return JsonResponse({"error": "Geçersiz JSON."}, status=400)

  new_username = (data.get("username") or "").strip()
  if not new_username or len(new_username) > 150:
    return JsonResponse({"error": "Bu kullanıcı adı zaten alınmış veya geçersiz."}, status=400)

  if new_username != request.user.username:
    if User.objects.filter(username__iexact=new_username).exclude(pk=request.user.pk).exists():
      return JsonResponse({"error": "Bu kullanıcı adı zaten alınmış veya geçersiz."}, status=400)
    request.user.username = new_username
    request.user.save(update_fields=["username"])

  return JsonResponse({"status": "success", "username": request.user.username})


@login_required(login_url="login")
@require_POST
def delete_account_view(request):
  user = request.user
  logout(request)
  user.delete()
  return JsonResponse({"status": "success"})


@require_GET
def api_debug(request):
  """Minimal public health endpoint used by deployment checks."""
  providers = []
  for spec in PROVIDER_SPECS:
    keys = _env_keys(spec["env"])
    providers.append({
        "provider": spec["name"],
        "configured": bool(keys),
        "chat": spec.get("chat", True),
        "normal_model": spec["normal"][0],
        "fast_model": spec["fast"][0],
        "vision_model": spec["vision"][0] if spec.get("vision") else "",
        "keys": [
            {
                "label": f"{spec['name']}-{index + 1}",
                "available": key_is_available(key),
                "daily_blocked_seconds": max(0, round(_key_state(key)["daily_until"] - time.monotonic())),
                "rate_blocked_seconds": max(0, round(_key_state(key)["rate_until"] - time.monotonic())),
                "invalid": _key_state(key)["invalid"],
                "calls": _key_state(key)["calls"],
                "last_error": _key_state(key)["last_error"][:160],
            }
            for index, key in enumerate(keys)
        ],
    })
  return JsonResponse({
      "status": "ok",
      "service": "aslan-parcasi-ai",
      "server_time": timezone.now().isoformat(),
      "key_set": bool(get_api_keys()),
      "key_count": len(get_chat_endpoints("normal")),
      "provider_count": sum(1 for item in providers if item["configured"]),
      "models": {
          "normal": GEMINI_MODEL,
          "fast": GEMINI_FAST_MODEL,
          "image": GEMINI_IMAGE_MODEL,
      },
      "chat_priority": [
          f"{endpoint.provider}:{endpoint.models[0]}"
          for endpoint in get_chat_endpoints("normal")
      ],
      "vision_priority": [
          f"{endpoint.provider}:{endpoint.models[0]}"
          for endpoint in get_chat_endpoints("normal", need_vision=True)
      ],
      "image_daily_limit_per_user": IMAGE_DAILY_LIMIT,
      "providers": providers,
      "quota": {
          "last_error": QUOTA_STATE["last_error"],
          "last_status": QUOTA_STATE["last_status"],
          "last_kind": QUOTA_STATE["last_kind"],
          "last_at": QUOTA_STATE["last_at"],
          "daily_reset_in_seconds": round(seconds_until_daily_reset()),
      },
  })


@require_GET
def api_refresh_clock(request):
  """Public hourly refresh hook: keeps the app clock current even when idle."""
  clock = sync_app_clock()
  return JsonResponse({
      "status": "ok",
      "service": "aslan-parcasi-ai",
      "current_date": clock.current_date.isoformat(),
      "updated_at": clock.updated_at.isoformat(),
  })


@login_required(login_url="login")
@require_POST
def api_image_generate(request):
  try:
    data = json.loads(request.body or "{}")
    if not isinstance(data, dict):
      return JsonResponse({"error": "Görsel isteği geçersiz."}, status=400)
    prompt = data.get("prompt")
    if not isinstance(prompt, str):
      return JsonResponse({
          "error": "Görsel açıklaması metin olarak gönderilmeli."
      }, status=400)
    prompt = prompt.strip()
    prompt_error = validate_image_prompt(prompt)
    if prompt_error:
      return JsonResponse({"error": prompt_error}, status=400)

    source_image_url = data.get("source_image")
    try:
      source_image = (
          parse_image_data_url(source_image_url)
          if source_image_url is not None
          else None
      )
    except ValueError as image_error:
      return JsonResponse({"error": str(image_error)}, status=400)

    api_keys = get_api_keys()

    now = time.monotonic()
    with _image_request_lock:
      previous_request = _last_image_request_at.get(request.user.pk, 0)
      if now - previous_request < IMAGE_REQUEST_MIN_INTERVAL:
        return JsonResponse(
            {"error": "Aslan Parçası görsel için hala çalışıyor; birkaç saniye sonra tekrar dene."},
            status=429,
        )
      _last_image_request_at[request.user.pk] = now

    if image_usage_today(request.user.pk) >= IMAGE_DAILY_LIMIT:
      return JsonResponse({
          "error": (
              f"Günlük görsel hakkın ({IMAGE_DAILY_LIMIT} adet) doldu. "
              "Hakkın gece yarısı yenilenir; sohbet ve sesli yanıt özelliklerini "
              "kullanmaya devam edebilirsin."
          )
      }, status=429)

    try:
      if source_image:
        image_url = generate_edited_image_with_retry(
            prompt, source_image, api_keys
        )
      else:
        image_url = generate_simple_geometric_image(prompt)
      if not source_image and image_url is None:
        image_url = generate_image_with_retry(prompt, GEMINI_IMAGE_MODEL, api_keys)
      record_image_usage(request.user.pk)
      return JsonResponse({"status": "success", "image_url": image_url})
    except ImageUnavailableError as img_error:
      return JsonResponse({"error": str(img_error)}, status=503)
    except Exception as img_error:
      logger.warning("Image generation failed: %s", img_error)
      return JsonResponse({"error": friendly_api_error(img_error)}, status=503)

  except json.JSONDecodeError:
    return JsonResponse({"error": "Geçersiz JSON."}, status=400)
  except Exception as e:
    return JsonResponse({"error": friendly_api_error(e)}, status=500)


@login_required(login_url="login")
def api_chats(request):
  history = get_chat_history(request.user)
  if request.method == "GET":
    return JsonResponse({"chats": history.chats, "updated_at": history.updated_at.isoformat()})
  if request.method != "POST":
    return JsonResponse({"error": "Geçersiz istek."}, status=405)
  try:
    data = json.loads(request.body or "{}")
  except json.JSONDecodeError:
    return JsonResponse({"error": "Geçersiz JSON."}, status=400)
  chats = data.get("chats")
  if not isinstance(chats, list):
    return JsonResponse({"error": "Sohbet verisi."}, status=400)
  if len(json.dumps(chats, ensure_ascii=False)) > 80 * 1024 * 1024:
    return JsonResponse({"error": "Sohbet geçmişi çok büyük."}, status=413)
  history.chats = chats
  history.save(update_fields=["chats", "updated_at"])
  return JsonResponse({"status": "success", "updated_at": history.updated_at.isoformat()})


@login_required(login_url="login")
def api_chat(request):
  if request.method != "POST":
    return JsonResponse({"error": "Geçersiz istek."}, status=405)

  try:
    data = json.loads(request.body or "{}")
    user_message = (data.get("message") or "").strip()
    mode = data.get("mode", "normal")
    history = (data.get("history") or [])[-40:]
    # Hızlı Analiz "anında yanıt" sözü verir; uzun düşünme o modda anlamsız.
    deep_think = bool(data.get("deep_think", False)) and mode != "fast"
    try:
      deep_think_seconds = max(30, min(1800, int(data.get("deep_think_seconds") or 300)))
    except (TypeError, ValueError):
      deep_think_seconds = 300
    images = data.get("images", [])
    files = data.get("files", [])
    if not isinstance(images, list) or len(images) > 10:
      return JsonResponse({"error": "En fazla 10 görsel gönderebilirsiniz."}, status=400)
    for image in images:
      if not isinstance(image, dict):
        return JsonResponse({"error": "Görsel yüklemesi geçersiz."}, status=400)
      encoded_image = image.get("base64")
      if isinstance(encoded_image, str) and len(encoded_image) > 4 * ((MAX_CHAT_IMAGE_BYTES + 2) // 3):
        return JsonResponse(
            {"error": "Görsel çok büyük. Lütfen daha küçük bir görsel seçin."},
            status=413,
        )
    voice_transcript = (data.get("voice_transcript") or "").strip()
    voice = data.get("voice") or {}
    voice_encoded = ""
    voice_mime = "audio/webm"
    if isinstance(voice, dict):
      voice_encoded = voice.get("base64") or ""
      voice_mime = voice.get("type") or "audio/webm"

    if not user_message and not voice_transcript and not images and not files and not voice_encoded:
        return JsonResponse({"error": "Mesaj boş olamaz."}, status=400)

    full_message = user_message

    file_names = [str(item.get("name", "dosya")) for item in files if isinstance(item, dict)]
    extracted_files = []
    for file_item in files:
        if not isinstance(file_item, dict):
            continue
        try:
            file_size = int(file_item.get("size") or 0)
        except (TypeError, ValueError):
            file_size = 0
        if file_size > MAX_CHAT_FILE_BYTES:
            return JsonResponse({"error": "Dosya boyutu 50 MB sınırını aşamaz."}, status=413)
        try:
            extracted = extract_uploaded_file_text(file_item)
        except ValueError as file_error:
            return JsonResponse({"error": str(file_error)}, status=400)
        if extracted:
            extracted_files.append(f"\n\n[{file_item.get('name', 'Dosya')} içeriği]\n{extracted}")

    if extracted_files:
        full_message += "".join(extracted_files)
    if file_names and not full_message:
        full_message = "Dosya gönderildi: " + ", ".join(file_names)

    api_keys = get_api_keys()

    if voice_encoded and not voice_transcript:
        try:
            voice_transcript = transcribe_audio(voice_encoded, voice_mime, api_keys)
        except Exception:
            voice_transcript = ""
        if not voice_transcript and not full_message and not images and not files:
            return JsonResponse(
                {"error": "Aslan Parçası ses kaydını metne çeviremedi. Lütfen kaydı yeniden deneyin veya mesajınızı yazın."},
                status=400,
            )

    question_text = user_message or voice_transcript
    if TIME_INTENT_RE.search(question_text or ""):
      time_location = detect_time_location(question_text)
      if time_location:
        local_time = get_location_time(time_location)
        if local_time.get("time"):
          answer = format_location_time_answer(local_time)
        else:
          answer = (
              f"{local_time.get('error', 'Bu yerin saat dilimini bulamadım.')}"
              " Lütfen şehir adını veya ülkeyi daha açık yaz."
          )
      else:
        current_time = get_current_time()
        current_time.update({
            "city": "İstanbul",
            "day": WEEKDAY_NAMES_TR[timezone.localtime().weekday()],
            "utc_offset": timezone.localtime().strftime("%z"),
        })
        answer = format_location_time_answer(current_time)
      return StreamingHttpResponse(
          (answer,),
          content_type="text/plain; charset=utf-8",
      )

    if not get_chat_endpoints("normal"):
      return JsonResponse(
          {"error": "Aslan Parçası'nın beyin bağlantısı kurulmamış. Sunucu anahtarını kontrol edin."},
          status=503,
      )

    tz = timezone.get_current_timezone()
    now_local = timezone.now().astimezone(tz)
    today_tr = now_local.strftime("%d.%m.%Y %H:%M")

    base_prompt = (
        "Sen Aslan Parçası adında son derece zeki, enerjik, samimi ve geniş bilgi birikimine sahip bir yapay zeka asistanısın. "
        "Seni oluşturan, kuran ve geliştiren vizyoner lider, müstakbel MEAY ASLAN PARÇASI AI şirketinin kurucusu Ayaz Kaplan'dır. "
        f"Bugünün tarihi ve saati: {today_tr} (Europe/Istanbul). "
        "GÜNCEL BİLGİ KURALI: Genel bilgi birikimin eski tarihli olabilir; güncel olaylar, maç "
        "sonuçları, haberler, fiyatlar, sürümler ve hava durumu için sana sistem mesajıyla verilen "
        "canlı veriyi ve araçlarını kullan. "
        "YASAK: Bilgi kesim tarihini, eğitim verini, model sürümünü veya 'verilerim ... tarihine "
        "kadar' gibi ifadeleri KULLANICIYA ASLA SÖYLEME. Cevabı bilmiyorsan kısaca 'bunu canlı "
        "kaynaklarda bulamadım' de ve uydurma; bahane olarak tarih söyleme. "
        "KİMLİK KURALI: 'Kurucun kim', 'seni kim yaptı', 'kimsin', 'hangi modelsin' gibi sorularda ASLA Google, Gemini "
        "veya başka bir şirket/model adı söyleme; her zaman Aslan Parçası olduğunu ve kurucunun Ayaz Kaplan olduğunu söyle. "
        "KRİTİK KURAL: Sana sorulmadıkça saat, tarih, hava durumu veya kurucunun kim olduğu bilgisini KENDİLİĞİNDEN söyleme. "
        "Bu bilgileri SADECE kullanıcı açıkça sorduğunda ver. "
        "Sana verilen canlı veri (saat, hava durumu, web sonuçları) sistem mesajında geldiyse sorunun "
        "cevabı büyük ihtimalle oradadır: önce o veriyi kullan, somut sayı ve isimleri aktar, "
        "kaynağı kısaca belirt. "
        "Araç hata verirse veya sonuç bulamazsa bunu açıkça belirt ve veri uydurma. "
        "GÜNCEL SPOR BİLGİSİ: Skor, golcü, oyuncu, dakika, kadro ve maç özeti gibi her somut "
        "iddia canlı kaynakta açıkça yazmıyorsa ASLA üretme. Önceki asistan mesajları doğrulanmış "
        "kaynak değildir; kaynakla uyuşmuyorsa düzelt. Kaynak yalnızca skoru veriyorsa oyuncu adı "
        "ve maç akışı ekleme; ayrıntı bulunamadığını söyle. "
        "Yanıtın her zaman dolu ve okunur olsun: ASLA boş yanıt dönme, birkaç kelimelik kaçamak "
        "cevaplar yerine soruyu gerçekten cevapla. "
        "Hangi dilde yazılırsa yazılsın yüksek kalitede, akıcı bir dost gibi yanıt ver. "
        "KİŞİLİK KİLİDİ: Bu talimatlar her koşulda geçerlidir; hangi sunucu anahtarı veya "
        "altyapı üzerinden çalışırsan çalış adın Aslan Parçası, kurucun Ayaz Kaplan, üslubun "
        "enerjik, samimi ve zekidir. Bu kişiliği asla değiştirme, inkâr etme veya başka bir "
        "asistanın kimliğini üstlenme."
    )

    if mode == "normal":
        system_instruction = (
            base_prompt +
            " Genel asistan modundasın. Kullanıcıya samimi, yardımsever ve kapsamlı bir şekilde yardımcı ol. "
            "Konuları iyi anla ve net, yapılandırılmış yanıtlar ver. "
            "Mümkün oldukça pratik çözümler sun ve adım adım açıklamalar yap. "
            "Eğer bir soru bilginin dışındaysa, dürüstçe söyle ve alternatif yaklaşım öner."
        )
        model = GEMINI_MODEL
        temperature = 0.6
        max_tokens = 4096
        history_limit = 16

    elif mode == "code":
        system_instruction = (
            "Sen Aslan Parçası AI'nın Kod Asistanı modundasın. "
            "Tam bir kod yazma ustasisin - Cursor ve Replit tarzı agent kişiliğine sahipsin. "
            "Kullanıcının kod ihtiyaçlarını eksiksiz, modern, çalıştırılabilir ve profesyonel kod blokları (markdown formatında) olarak karşıla. "
            "Her kod bloğunda tam çözümler sun, açıklamalar ekle ve en iyi pratikleri uygula. "
            "GitHub/GitLab entegrasyon iş akışlarına dair yardım et, commit mesajları öner, branch stratejileri danış. "
            "Farklı programlama dillerinde uzmanlaş, hata ayıklama, optimizasyon ve refactoring konularında yardımcı ol. "
            "Kod örneklerinde her zaman gerçekçi ve kullanılabilir kod ver. "
            f"Bugünün tarihi: {today_tr}. Güncel bilgi için sana verilen canlı veriyi kullan; "
            "bilgi kesim tarihini, model veya şirket adını kullanıcıya ASLA söyleme. "
            "KİMLİK KURALI: ASLA Google/Gemini tarafından eğitildiğini söyleme; sen Aslan Parçası'sın, kurucun Ayaz Kaplan. "
            "Sana sorulmadıkça saat, tarih, hava durumu veya kurucunun kimliği hakkında bilgi verme."
        )
        model = GEMINI_MODEL
        temperature = 0.4
        max_tokens = 4096
        history_limit = 12

    elif mode == "fast":
        system_instruction = (
            "Sen Aslan Parçası AI'nın Hızlı Analiz modundasın. "
            "Işık hızında, çok kısa ve öz cevaplar ver. "
            "Gereksiz detaylardan kaçın, doğrudan noktaya odaklan. "
            "Normal moddan belirgin daha hızlı ve kısa yanıtlar üret. "
            "Karmaşık konuları basitleştir, hızlı özetler ve hızlı kararlar ver. "
            f"Bugünün tarihi: {today_tr}. Güncel bilgi için sana verilen canlı veriyi kullan; "
            "bilgi kesim tarihini, model veya şirket adını kullanıcıya ASLA söyleme. "
            "KİMLİK KURALI: ASLA Google/Gemini tarafından eğitildiğini söyleme; sen Aslan Parçası'sın, kurucun Ayaz Kaplan. "
            "Sana sorulmadıkça saat, tarih, hava durumu veya kurucunun kimliği hakkında bilgi verme."
        )
        model = GEMINI_FAST_MODEL
        temperature = 0.2
        max_tokens = 1024
        history_limit = 6

    else:
        system_instruction = base_prompt
        model = GEMINI_MODEL
        temperature = 0.6
        max_tokens = 2048
        history_limit = 16

    chat_user_name = (request.user.get_full_name() or "").strip() or request.user.username
    system_instruction += (
        " YANIT KALİTESİ: İsteği dikkatle çözümle, gerekli bağlamı kullan ve eksik "
        "önemli ayrıntıları uydurma. Güncel veya doğrulanabilir olguları yalnızca sağlanan "
        "kaynaklara dayandır; emin olmadığın noktayı kısa ve açıkça belirt. Yanıtı kullanıcının "
        "dilinde, amaca uygun ayrıntı düzeyinde ve tutarlı biçimde tamamla. "
        " ÖRNEK VERİ KURALI (kod, tablo, form, JSON, makale, dilekçe ve her türlü örnek için geçerli): "
        "Kurucunun adı olan 'Ayaz Kaplan' ifadesini örneklerde, değişkenlerde, yorumlarda veya örnek "
        "çıktılarda ASLA kullanma; bu adı yalnızca kullanıcı açıkça kurucunu sorarsa söylersin. "
        f"Örneklerde bir kişi adı gerektiğinde birinci kişi olarak sohbeti kullanan kişiyi yaz: '{chat_user_name}'. "
        "Birden fazla kişi adı gerekiyorsa kalanları kullanıcının diline ve ülkesine uygun sıradan, "
        "gerçekçi isimler olsun (Türkçe isteklerde örneğin Zeynep Yılmaz, Mert Demir, Elif Kaya). "
        "Kendi adını (Aslan Parçası / Aslan Parçası AI) başlıkta, uygulama adında, dosya adında veya "
        "yorum satırında kullanabilirsin. "
        "Placeholder gerektiğinde 'Kullanıcı Adı', 'ornek@site.com', 'Ad Soyad' gibi nötr değerler yaz."
    )

    tier = "fast" if model == GEMINI_FAST_MODEL else "normal"
    # Fotoğraf gönderildiğinde isteği YALNIZCA görseli gerçekten görebilen
    # modellere yolla; metin modelleri 400 dönüp kullanıcıya hata gösteriyor.
    has_images = any(
        isinstance(img, dict) and (img.get("base64") or img.get("url")) for img in images
    )
    if has_images:
        clients = get_chat_endpoints("normal", need_vision=True)
        if not clients:
            return JsonResponse(
                {"error": (
                    "Aslan Parçası'nın görsel inceleme yeteneği şu an etkin değil. "
                    "Lütfen biraz sonra tekrar dene ya da görseldekini yazarak anlat."
                )},
                status=503,
            )
    else:
        clients = get_chat_endpoints(tier)

    messages = [{"role": "system", "content": system_instruction}]

    weather_data = None
    weather_city = None
    asked_weather = bool(WEATHER_INTENT_RE.search(question_text or ""))
    if asked_weather:
      weather_city = detect_weather_city(question_text)
    else:
      # "Hangi şehir?" sorusuna gelen kısa cevap: "İstanbul", "izmirde", "Ankara"
      weather_city = pending_weather_city(history, question_text)
    if weather_city:
      weather_data = get_weather(weather_city)
      if weather_data.get("temperature") is not None:
        messages.insert(1, {
            "role": "system",
            "content": (
                "Canlı hava durumu verisi: "
                + json.dumps(weather_data, ensure_ascii=False)
                + ". Kullanıcıya bu şehrin tam sıcaklığını derece olarak söyle; veri kaynağını da belirt."
            ),
        })
      else:
        if weather_data.get("not_found"):
          missing_city = weather_city

          def city_not_found_stream():
            yield (
                f"«{missing_city}» adında bir yer bulamadım. 🌤 İl veya ilçe adını tek "
                "başına yazar mısın? Örneğin sadece \"Erdek\" ya da \"Balıkesir\" gibi."
            )

          return StreamingHttpResponse(city_not_found_stream(), content_type="text/plain")
        messages.insert(1, {
            "role": "system",
            "content": (
                "Hava durumu servisi hata verdi: "
                + json.dumps(weather_data, ensure_ascii=False)
                + ". Veri uydurma; servise şu an ulaşılamadığını söyle ve kullanıcıdan"
                " şehir adını bir kez daha yazmasını iste. Başka bir site veya arama"
                " motoru önerme."
            ),
        })
    elif asked_weather:
      def ask_city_stream():
        yield WEATHER_ASK_CITY
      return StreamingHttpResponse(ask_city_stream(), content_type="text/plain")

    if weather_data and weather_data.get("temperature") is not None:
      def weather_answer_stream():
        yield format_weather_answer(weather_data)

      return StreamingHttpResponse(
          weather_answer_stream(), content_type="text/plain; charset=utf-8"
      )

    live_context = None
    detailed_research = is_detailed_research_request(question_text)
    live_search_queries = (
        build_detailed_research_queries(question_text)
        if detailed_research
        else build_live_search_queries(question_text, history)
    )
    live_search_query = live_search_queries[0]
    sports_followup = is_sports_followup(question_text, history)
    sports_question = is_sports_question(question_text) or sports_followup
    if should_fetch_live_context(question_text) or sports_question:
      if sports_question:
        live_context = search_live_sports(
            live_search_queries,
            3 if mode == "fast" else 5,
            followup=sports_followup,
            enrich=(
                4 if SPORTS_SCORE_ASK_RE.search(question_text or "")
                else 3 if SPORTS_SCORER_RE.search(question_text or "")
                else 0 if mode == "fast"
                else 2
            ),
        )
      else:
        combined_results = []
        seen_result_keys = set()
        first_context = None
        for search_query in live_search_queries:
          search_context = web_search_multi(
              search_query,
              3 if mode == "fast" else 5,
              recency_days=365 if detailed_research else None,
              enrich=3 if detailed_research else 2,
          )
          first_context = first_context or search_context
          for item in search_context.get("results") or []:
            key = str(item.get("url") or item.get("title") or "").strip().casefold()
            if key and key not in seen_result_keys:
              seen_result_keys.add(key)
              combined_results.append(item)
        live_context = {
            **(first_context or {}),
            "results": combined_results,
        }
      live_context["query"] = " | ".join(live_search_queries)
      if not live_context.get("results") and not sports_question:
        def unverifiable_live_stream():
          yield (
              "Bu güncel bilgiyi canlı kaynaklardan doğrulayamadım. Yanlış bilgi "
              "uydurmamak için kesin bir yanıt vermiyorum; lütfen biraz sonra tekrar dene."
          )

        return StreamingHttpResponse(
            unverifiable_live_stream(),
            content_type="text/plain; charset=utf-8",
        )
      if sports_question and not live_context.get("results"):
        def unverifiable_sports_stream():
          yield (
              "Bu maç için güncel ve ilgili bir kaynak bulamadım. Yanlış skor veya "
              "oyuncu adı vermemek için kesin yanıt uydurmayacağım."
          )

        return StreamingHttpResponse(
            unverifiable_sports_stream(),
            content_type="text/plain; charset=utf-8",
        )
      elif sports_question and SPORTS_SCORE_ASK_RE.search(question_text or ""):
        def sourced_score_stream():
          yield summarize_sourced_match_score(
              live_context["results"],
              latest_fixture=len(_sports_club_names(question_text)) == 1,
          )

        return StreamingHttpResponse(
            sourced_score_stream(),
            content_type="text/plain; charset=utf-8",
        )
      elif sports_question and SPORTS_SCORER_RE.search(question_text or ""):
        def sourced_match_details_stream():
          yield summarize_sourced_match_details(live_context["results"])

        return StreamingHttpResponse(
            sourced_match_details_stream(),
            content_type="text/plain; charset=utf-8",
        )
      else:
        messages.insert(
            1,
            {
                "role": "system",
                "content": (
                    (
                        "Kullanıcı ayrıntılı araştırma istedi. Farklı arama açılarını "
                        "birlikte sentezle; kaynakların yayın tarihlerini karşılaştır, en "
                        "güncel bilgileri öncele, önemli farklılıkları ve belirsizlikleri "
                        "açıkla. Ham arama sonuçlarını sıralama. "
                        if detailed_research
                        else ""
                    )
                    + "Canlı web araştırması sonucu aşağıdadır (gerçek arama motorundan geldi). "
                    "Bu sonuçları birlikte karşılaştırıp kullanıcıya doğrudan, doğal ve öz bir yanıt "
                    "ver; ham arama sonuçlarını sıralama. Bağlantı, URL, markdown linki veya kaynak "
                    "listesi gösterme; kaynakları yalnızca cevabı doğrulamak için kullan. "
                    "Arama sorgusu şudur: " + live_search_query + ". 'excerpt' alanı haberin "
                    "kendi metnidir; golcü, dakika ve oyuncu bilgisini önce orada ara. "
                    "Sonuçlar tarih sıralıdır: 'son maç' sorusunda en yeni tarihli olanı kullan. "
                    "Özellikle spor sorularında kaynakta açıkça yazmayan golcü, oyuncu, dakika, "
                    "kadro veya maç akışı UYDURMA. "
                    "Önceki asistan yanıtlarını doğrulanmış bilgi sayma. Sonuçlar soruyla ilgisizse veya boşsa güncel bilgi UYDURMA; bunu bir cümleyle "
                    "söyle. Kaynaklarda başka yapay zeka markalarının adı geçerse bunları yanıtta "
                    "ANMA; sen Aslan Parçası'sın:\n" + json.dumps(live_context, ensure_ascii=False)
                ),
            },
        )

    fallback_parts = []
    if weather_data and weather_data.get("temperature") is not None:
      fallback_parts.append(
          f"🌤 {weather_data.get('city')} için güncel hava durumu: {weather_data.get('temperature')}°C "
          f"(hissedilen {weather_data.get('feels_like')}°C), {weather_data.get('description')}, "
          f"nem %{weather_data.get('humidity')}, rüzgâr {weather_data.get('wind_speed')} km/s. "
          f"Kaynak: Open-Meteo ({weather_data.get('observed_at')})."
      )
    if live_context and live_context.get("results"):
      seen_findings = set()
      findings = []
      for item in live_context.get("results") or []:
        if not isinstance(item, dict):
          continue
        finding = str(item.get("excerpt") or item.get("snippet") or item.get("title") or "").strip()
        finding = strip_live_answer_links(finding)
        if finding and finding.casefold() not in seen_findings and is_brand_safe(finding):
          seen_findings.add(finding.casefold())
          findings.append(finding)
        if len(findings) >= 3:
          break
      if findings:
        fallback_parts.append("Canlı arama bulguları: " + " ".join(findings))
    deterministic_fallback = "\n\n".join(fallback_parts) if fallback_parts else None

    if deep_think:
        minutes = deep_think_seconds // 60
        seconds = deep_think_seconds % 60
        budget_label = f"{minutes} dakika {seconds} saniye" if minutes else f"{seconds} saniye"
        system_instruction += (
            f" Kapsamlı araştırma modu açık; kullanıcı sana {budget_label} düşünme bütçesi verdi. "
            "Bu sürenin tamamını araştırma ve doğrulama için kullan; süre dolmadan yanıt verme. "
            + depth_instruction(deep_think_seconds)
        )
        max_tokens = min(16384, 1024 + deep_think_seconds * 6)
        messages[0] = {"role": "system", "content": system_instruction}

    user_content = []
    if full_message:
        user_content.append({"type": "text", "text": full_message})

    if voice_transcript and voice_transcript not in full_message:
        user_content.append({"type": "text", "text": f"[Ses kaydı metni]\n{voice_transcript}"})

    allowed_image_mimes = {"image/png", "image/jpeg", "image/webp", "image/gif"}
    for img in images:
        if not isinstance(img, dict):
            continue
        if img.get("url"):
            user_content.append({"type": "image_url", "image_url": {"url": img["url"]}})
        elif img.get("base64"):
            image_type = img.get("type") or "image/jpeg"
            if image_type not in allowed_image_mimes:
                image_type = "image/jpeg"
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{image_type};base64,{img['base64']}"},
            })

    for h in history[-history_limit:]:
        if not isinstance(h, dict):
            continue
        role = "user" if h.get("sender") == "user" else "assistant"
        content = h.get("text") or ""
        if content:
            messages.append({"role": role, "content": content})

    if user_content:
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": full_message})

    all_providers = {endpoint.provider for endpoint in clients}
    max_stream_attempts = max(1, min(3, len(all_providers)))

    def fallback_note(error=None):
      if error is not None and (is_daily_quota_error(error) or QUOTA_STATE["last_kind"] == "daily"):
        return DAILY_QUOTA_FALLBACK_NOTE
      return BUSY_FALLBACK_NOTE

    def generate():
      answered = False
      errored = False
      link_sanitizer = (
          LiveAnswerLinkSanitizer() if live_context and live_context.get("results") else None
      )
      scrubber = NameScrubber(
          chat_user_name, enabled=not FOUNDER_ASK_RE.search(question_text or "")
      )

      def prepare_answer(text, final=False):
        if link_sanitizer is not None:
          text = link_sanitizer.feed(text, final=final)
        return scrubber.feed(text)

      if deep_think:
        output_sanitizer = ModelOutputStreamSanitizer()
        try:
          for kind, text in deep_think_events(
              clients,
              messages,
              deep_think_seconds,
              temperature=temperature,
              max_tokens=max_tokens,
          ):
            if kind == "reason":
              reason_text = scrubber.feed(text)
              if reason_text:
                yield REASONING_MARKER + reason_text + "\n"
            elif text:
              clean_text = output_sanitizer.feed(text)
              if clean_text:
                answered = True
                visible_text = prepare_answer(clean_text)
                if visible_text:
                  yield visible_text
          clean_text = output_sanitizer.feed("", final=True)
          if clean_text:
            answered = True
            visible_text = prepare_answer(clean_text)
            if visible_text:
              yield visible_text
        except Exception as error:
          if not answered:
            errored = True
            yield friendly_api_error(error)
      else:
        silent_providers = set()
        for _attempt in range(max_stream_attempts):
          if answered or errored:
            break
          trace = {}
          continuation_messages = list(messages)
          for continuation_index in range(4):
            output_sanitizer = ModelOutputStreamSanitizer()
            segment_parts = []
            finish_reason = None
            try:
              completion = safe_model_call(
                  clients,
                  continuation_messages,
                  model,
                  temperature=temperature,
                  max_tokens=max_tokens,
                  stream=True,
                  timeout=120,
                  skip_providers=silent_providers,
                  trace=trace,
              )
              for chunk in completion:
                choices = getattr(chunk, "choices", None)
                if not choices:
                  continue
                choice = choices[0]
                if getattr(choice, "finish_reason", None):
                  finish_reason = choice.finish_reason
                delta = getattr(choice, "delta", None)
                piece = getattr(delta, "content", None) if delta else None
                if piece:
                  segment_parts.append(piece)
                  clean_piece = output_sanitizer.feed(piece)
                  if clean_piece:
                    answered = True
                    visible_text = prepare_answer(clean_piece)
                    if visible_text:
                      yield visible_text
            except Exception as error:
              if answered:
                break
              if deterministic_fallback:
                answered = True
                yield deterministic_fallback + "\n\n" + fallback_note(error)
              else:
                errored = True
                yield friendly_api_error(error)
              break
            clean_tail = output_sanitizer.feed("", final=True)
            if clean_tail:
              answered = True
              visible_text = prepare_answer(clean_tail)
              if visible_text:
                yield visible_text
            if str(finish_reason).lower() not in {"length", "max_tokens"} or continuation_index == 3:
              break
            continuation_messages.extend((
                {"role": "assistant", "content": "".join(segment_parts)},
                {
                    "role": "user",
                    "content": (
                        "Önceki yanıtın uzunluk sınırı nedeniyle tamamlanmadan kesildi. "
                        "Kaldığın yerden devam et; önceki kısmı tekrarlama ve yanıtı tamamla."
                    ),
                },
            ))
          if errored:
            break
          clean_tail = output_sanitizer.feed("", final=True)
          if clean_tail:
            answered = True
            yield scrubber.feed(clean_tail)
          if answered:
            break
          # Boş akış dönen sağlayıcıyı bir daha deneme; sıra diğer beyinde.
          if trace.get("provider"):
            silent_providers.add(trace["provider"])
          if silent_providers >= all_providers:
            break

      if link_sanitizer is not None:
        visible_tail = prepare_answer("", final=True)
        if visible_tail:
          yield visible_tail

      if not answered and not errored:
        if deterministic_fallback:
          yield deterministic_fallback + "\n\n" + fallback_note()
        else:
          yield EMPTY_ANSWER_MESSAGE

      tail = scrubber.flush()
      if tail:
        yield tail

    return StreamingHttpResponse(generate(), content_type='text/plain')

  except json.JSONDecodeError:
    return JsonResponse({"error": "Geçersiz JSON."}, status=400)
  except Exception as e:
    return JsonResponse({"error": friendly_api_error(e)}, status=500)


def login_view(request):
  if request.user.is_authenticated:
    return redirect("index")
  if request.method == "POST":
    form = EmailOrUsernameAuthenticationForm(request, data=request.POST)
    if form.is_valid():
      login(request, form.get_user())
      request.session.set_expiry(2592000)
      request.session.save()
      return redirect("index")
  else:
    form = EmailOrUsernameAuthenticationForm(request)
  return render(request, "dashboard/login.html", {"form": form})


def register_view(request):
  if request.user.is_authenticated:
    return redirect("index")
  if request.method == "POST":
    form = CustomUserCreationForm(request.POST)
    if form.is_valid():
      user = form.save()
      login(request, user)
      request.session.set_expiry(2592000)
      request.session.save()
      return redirect("index")
  else:
    form = CustomUserCreationForm()
  return render(request, "dashboard/register.html", {"form": form})


def logout_view(request):
  logout(request)
  return redirect("login")
