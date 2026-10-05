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

  if re.search(
      r"\b(brand(?:ed)?|bottle|can|soda|cola|şişe\w*|kutu\w*|ambalaj\w*|"
      r"markalı|markali|packaging|label)\b",
      raw,
      re.IGNORECASE,
  ):
    topic = f"{topic}, clearly visible branded product packaging, readable label, actual bottle/can design, not a generic cup, centered in frame"

  descriptors = [
      "photorealistic",
      "highly detailed",
      "sharp focus",
      "cinematic lighting",
      "ultra realistic texture",
      "professional composition",
      "8k",
      "natural anatomy",
      "realistic shadows",
      "studio quality",
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
      "enhance": "true",
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


TURKISH_ASCII_MAP = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


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
    """Haber/spor sorularında skor başlığın içinde gelir; tarih de kaynağa eklenir.

    Bağlantılar arama motorunun kendi yönlendirmesi olduğu için bilinçli olarak
    boş bırakılır: başlık + kaynak adı + tarih kullanıcıya ve modele yeter.
    """
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
        date_match = re.search(r"<pubDate>(.*?)</pubDate>", item, re.S)
        meta = " · ".join(part for part in (
            _strip_tags(date_match.group(1)) if date_match else "",
            _strip_tags(source_match.group(1)) if source_match else "",
        ) if part)
        results.append({"title": title, "url": "", "snippet": meta})
        if len(results) >= limit:
            break
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


def _is_relevant(result, tokens, allow_chatter):
    """On-topic and, unless the user asked for it, not entertainment noise."""
    if tokens:
        minimum_matches = 1 if len(tokens) <= 2 else (len(tokens) + 1) // 2
        if _relevance(result, tokens) < minimum_matches:
            return False
    if allow_chatter:
        return True
    return not CHATTER_TITLE_RE.search(str(result.get("title") or ""))


def _search_attempts(query, duckduckgo_only=False):
    """News engine first only for news-like queries; it is junk for general topics."""
    general = (
        ("ddg-html", "post", "https://html.duckduckgo.com/html/", {"kl": "tr-tr"}, _parse_ddg_results),
        ("ddg-lite", "post", "https://lite.duckduckgo.com/lite/", {"kl": "tr-tr"}, _parse_ddg_results),
        ("ddg-get", "get", "https://html.duckduckgo.com/html/", {}, _parse_ddg_results),
        ("bing", "get", "https://www.bing.com/search", {"setmkt": "tr-TR", "setlang": "tr"}, _parse_bing_results),
    )
    news = (
        ("haber-rss", "get", "https://news.google.com/rss/search",
         {"hl": "tr", "gl": "TR", "ceid": "TR:tr"}, _parse_google_news_rss),
    )
    if duckduckgo_only:
        return tuple(attempt for attempt in general if attempt[0].startswith("ddg-"))
    return news + general if NEWS_HINT_RE.search(query or "") else general + news


def web_search(query, num_results=5, *, duckduckgo_only=False, deadline=None):
    """Current web results with no API key, filtered for relevance to the query."""
    limit = max(1, min(int(num_results), 10))
    clean_query = (query or "").strip()
    if not clean_query:
        return {"error": "Web search failed: empty query", "results": [], "query": query}
    allow_chatter = bool(
        NEWS_HINT_RE.search(clean_query) or ENTERTAINMENT_ASK_RE.search(clean_query)
    )
    last_error = None
    partial = None
    for variant in _query_variants(clean_query):
        if deadline is not None and time.monotonic() >= deadline:
            break
        tokens = _query_tokens(variant)
        for engine, method, url, extra, parser in _search_attempts(variant, duckduckgo_only):
            if deadline is not None and time.monotonic() >= deadline:
                break
            try:
                timeout = 4 if duckduckgo_only else 10
                if deadline is not None:
                    timeout = min(timeout, max(0.5, deadline - time.monotonic()))
                if method == "post":
                    response = requests.post(
                        url,
                        data={"q": variant, **extra},
                        headers=SEARCH_HEADERS,
                        timeout=timeout,
                    )
                else:
                    response = requests.get(
                        url,
                        params={"q": variant, **extra},
                        headers=SEARCH_HEADERS,
                        timeout=timeout,
                    )
                if duckduckgo_only and (
                    response.status_code == 202
                    or "anomalyDetectionBlock" in response.text
                ):
                    return {
                        "error": "DuckDuckGo requested a security check for this server.",
                        "results": [],
                        "query": variant,
                        "engine": engine,
                        "blocked": True,
                    }
                if response.status_code >= 400:
                    last_error = f"{engine} HTTP {response.status_code}"
                    continue
                results = parser(response.text, limit * 2)
                if not results:
                    last_error = f"{engine}: no results parsed"
                    continue
                relevant = [item for item in results if _is_relevant(item, tokens, allow_chatter)]
                relevant.sort(key=lambda item: -_relevance(item, tokens))
                # Tek sonuç tesadüf olabilir; yalın hâl varyantı genelde çok daha isabetli.
                if len(relevant) >= 2:
                    return {"query": variant, "results": relevant[:limit], "engine": engine}
                if relevant and partial is None:
                    partial = relevant[:limit]
                last_error = f"{engine}: sonuç soruyla ilgili değil"
            except Exception as error:
                last_error = f"{engine}: {error}"
    if partial:
        return {"query": clean_query, "results": partial, "engine": "weak-match"}
    return {"error": f"Web search failed: {last_error}", "results": [], "query": clean_query}


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
  """Identify requests where a stale model answer would be misleading."""
  normalized = (text or "").lower()
  hints = (
      "internetten", "güncel", "bugün", "şu an", "şuan", "son dakika",
      "haber", "maç", "mac ", "skor", "sonuç", "spor", "fiyat", "kur", "döviz",
      "kim kazandı", "ne zaman", "araştır", "web'de", "webte", "bitti",
      "kazandı", "puan durumu", "transfer", "deprem", "seçim", "altın",
      "dolar", "euro", "bitcoin", "hisse", "sürüm", "versiyon", "yeni çıkan",
  )
  return any(hint in normalized for hint in hints)


SPORTS_FACTS_RE = re.compile(
    r"\b(maç|mac|futbol|gol|skor|puan|derbi|lig|transfer|kadrosu|golcü|golcu)\b|"
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


def is_sports_question(text):
    return bool(SPORTS_FACTS_RE.search(text or ""))


def build_live_search_query(text, history=None):
    """Resolve a sports follow-up from prior user questions, never prior AI claims."""
    current = (text or "").strip()
    if not is_sports_question(current) or not SPORTS_FOLLOWUP_RE.search(current):
        return build_search_query(current)
    for message in reversed(history or []):
        if not isinstance(message, dict) or message.get("sender") != "user":
            continue
        previous_question = str(message.get("text") or "").strip()
        if is_sports_question(previous_question):
            return build_search_query(f"{previous_question} {current}")
    return build_search_query(current)


def should_retry_live_search_with_general_results(question, live_context):
    """Sports and live-search asks should not fail hard when DDG is blocked."""
    if not is_sports_question(question) or not isinstance(live_context, dict):
        return False
    if live_context.get("blocked"):
        return True
    if live_context.get("engine") == "weak-match":
        return True
    results = live_context.get("results") or []
    return not results


TIME_INTENT_RE = re.compile(
    r"(saat\s+kaç|saat\s+kaçtır|şu\s+an\s+saat|saati\s+söyle|tarih\s+(ne|kaç)|"
    r"bugün\s+(ayın\s+kaçı|ne\s+günü|hangi\s+gün)|günlerden\s+(ne|hangi)|"
    r"hangi\s+gündeyiz|bugün\s+günlerden)",
    re.I,
)
WEATHER_INTENT_RE = re.compile(
    r"(hava\s+durumu|hava\s+nasıl|havası\s+nasıl|kaç\s+derece|sıcaklık\s+kaç|hava\s+kaç\s+derece)",
    re.I,
)
CITY_PATTERNS = (
    r"([A-Za-zÇĞİÖŞÜçğıöşü\-\.]+)['’]?(?:de|da|te|ta|nde|nda|nte|nta)\s+(?:için\s+)?hava",
    r"([A-Za-zÇĞİÖŞÜçğıöşü\-\.]+)\s+(?:için\s+)?hava\s+durumu",
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


def generate_image_with_gemini(prompt, image_model, api_key):
  """Generate an image through the generateContent endpoint."""
  image_size = "1K" if image_model == "gemini-2.5-flash-image" else "4K"
  image_prompt = (
      "Create exactly one exceptionally detailed image that follows the user's brief. "
      "Unless the brief explicitly requests an illustration, cartoon, logo, or another "
      "non-photographic style, render it as an ultra-photorealistic photograph captured "
      "with a professional full-frame camera. Use physically plausible light, natural "
      "skin and material textures, accurate anatomy and perspective, realistic depth of "
      "field, crisp focus on the subject, nuanced shadows, and restrained true-to-life "
      "color grading. Preserve the requested subject, count, action, and composition; "
      "do not invent unrelated objects. Do not add any text, watermark, signature, logo, "
      "border, or provider branding unless the user explicitly asks for it. Follow any "
        "style explicitly requested by the user instead of forcing photorealism. Exact user "
        "brief: "
        + prompt
        + "\nPhotographic translation and detail cues: "
        + enhance_image_prompt(prompt)
  )
  response = requests.post(
      f"https://generativelanguage.googleapis.com/v1beta/models/{image_model}:generateContent",
      headers={
          "Content-Type": "application/json",
      },
      params={"key": api_key},
      json={
          "contents": [{
            "parts": [{"text": image_prompt}],
          }],
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
      f"https://image.pollinations.ai/prompt/{quote(params['prompt'][:500], safe='')}",
      params={
          "width": params["width"],
          "height": params["height"],
          "nologo": params["nologo"],
          "safe": params["safe"],
          "model": params["model"],
          "seed": params["seed"],
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
  ddg_blocked = False

  def add_findings(result):
    for item in (result.get("results") or []):
      if not isinstance(item, dict) or not (item.get("title") or item.get("url")):
        continue
      line = f"- {item.get('title', '')}"
      if item.get("url"):
        line += f" ({item['url']})"
      if item.get("snippet"):
        line += f": {item['snippet']}"
      if line not in findings:
        findings.append(line)

  seen_lines = set()

  def report_sources(result, per_engine_limit=5):
    """Show only brand-safe, not-yet-seen sources; returns how many were new."""
    fresh = 0
    for line in format_source_lines(result, per_engine_limit * 2):
      if line in seen_lines:
        continue
      seen_lines.add(line)
      fresh += 1
      if fresh <= per_engine_limit:
        yield ("reason", f"SRC:🌐 Kaynak: {line}")

  def search_with_fallback(query, num_results):
    result = web_search(
      query, num_results, duckduckgo_only=True, deadline=deadline - reserve
    )
    if result.get("blocked"):
      yield ("reason", "DuckDuckGo güvenlik doğrulaması istedi; genel arama ile yeniden deneniyor.")
      result = web_search(
        query, num_results, duckduckgo_only=False, deadline=deadline - reserve
      )
    return result

  def facet_search(index):
    nonlocal ddg_blocked
    """Server-driven research step so the whole budget is used, never idled."""
    facet = RESEARCH_FACETS[index % len(RESEARCH_FACETS)]
    facet_query = f"{search_topic(question)} {facet}".strip()
    yield ("reason", f"SRC:🔎 Ek araştırma ({index + 1}. tur): {facet_query[:90]}")
    result = yield from search_with_fallback(facet_query, 5)
    left = max(0, int(deadline - time.monotonic()))
    if result.get("blocked"):
      ddg_blocked = True
      yield ("reason", "Genel arama da doğrulanabilir kaynak vermedi; araştırma dürüstçe durduruluyor.")
      return
    if result.get("engine") == "weak-match" or not result.get("results"):
      yield ("reason", f"Bu açıdan güvenilir kaynak çıkmadı; {left} sn kaldı, farklı bir açı deneniyor.")
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
            + "\n".join(f"- {item.get('title')}" for item in (result.get("results") or [])[:5])
        ),
    })

  if question:
    yield ("reason", f"SRC:🔎 Canlı arama: {build_search_query(question)[:90]}")
    seeded = yield from search_with_fallback(build_search_query(question), 6)
    if seeded.get("blocked") or not seeded.get("results"):
      yield ("answer", (
        "Bu oturumda doğrulanabilir web kaynaklarına erişemedim; bu yüzden araştırma yaptığımı "
        "iddia etmeyeceğim. Daha sonra yeniden deneyebilirsin."
      ))
      return
    if seeded.get("engine") == "weak-match":
      yield ("reason", "İlk arama konuyla ilgili güçlü kaynak vermedi; ek turlarda yeniden denenecek.")
    else:
      add_findings(seeded)
      for event in report_sources(seeded, 6):
        yield event
      if seeded.get("error") and not seeded.get("results"):
        yield ("reason", "İlk arama sonuç vermedi; ek araştırma turlarıyla yeniden denenecek.")
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
        if result.get("blocked"):
          ddg_blocked = True
          yield ("reason", "Genel arama da doğrulanabilir kaynak vermedi; araştırma dürüstçe durduruluyor.")
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
    if ddg_blocked:
      break

  # Bütçe bitene kadar boş durma: her turda yeni bir açıdan gerçekten araştır.
  while time.monotonic() < deadline - reserve and question and not ddg_blocked:
    yield from facet_search(facet_index)
    facet_index += 1
    remaining = max(0.0, deadline - time.monotonic())
    pause = min(8, max(1, remaining))
    if pause > 0:
      yield ("reason", f"Toplam {len(findings)} bulgu birikti; {int(remaining)} sn kaldı, yeni tur hazırlanıyor...")
      time.sleep(min(pause, 5))

  if ddg_blocked and not findings:
    yield ("answer", (
        "Bu oturumda DuckDuckGo kaynaklarına erişemedim; araştırma yaptığımı iddia "
        "etmeyeceğim. Güvenlik doğrulaması kalktığında aynı soruyu yeniden deneyebilirsin."
    ))
    return

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
    prompt = (data.get("prompt") or "").strip()

    if not prompt:
      return JsonResponse({"error": "Prompt boş olamaz."}, status=400)

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
    if not get_chat_endpoints("normal"):
        return JsonResponse(
            {"error": "Aslan Parçası'nın beyin bağlantısı kurulmamış. Sunucu anahtarını kontrol edin."},
            status=503,
        )

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
        max_tokens = 1536
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
        max_tokens = 2048
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
        max_tokens = 512
        history_limit = 6

    else:
        system_instruction = base_prompt
        model = GEMINI_MODEL
        temperature = 0.6
        max_tokens = 2048
        history_limit = 16

    chat_user_name = (request.user.get_full_name() or "").strip() or request.user.username
    system_instruction += (
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

    question_text = user_message or voice_transcript
    clock_data = None
    if TIME_INTENT_RE.search(question_text or ""):
      clock_data = get_current_time()
      messages.insert(1, {
          "role": "system",
          "content": (
              "Sunucunun kesin saati (Europe/Istanbul): "
              + json.dumps(clock_data, ensure_ascii=False)
              + ". Kullanıcı saat/tarih sordu; bu veriyi kullanarak doğrudan ve kısa yanıtla."
          ),
      })

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
    live_search_query = build_live_search_query(question_text, history)
    sports_question = is_sports_question(question_text)
    if should_fetch_live_context(question_text) or (
      sports_question and live_search_query != build_search_query(question_text)
    ):
      live_context = web_search(
        live_search_query,
        3 if mode == "fast" else 5,
        duckduckgo_only=sports_question,
      )
      if should_retry_live_search_with_general_results(question_text, live_context):
        live_context = web_search(
            live_search_query,
            3 if mode == "fast" else 5,
            duckduckgo_only=False,
        )
      if sports_question and not live_context.get("results"):
        def unverifiable_sports_stream():
          yield (
              "Bu maç ayrıntılarını canlı kaynaklarda doğrulayamadım. "
              "Oyuncu, golcü veya maç akışı uydurmak yerine doğrulanabilir kaynak bekliyorum."
          )

        return StreamingHttpResponse(unverifiable_sports_stream(), content_type="text/plain")
      messages.insert(
          1,
          {
              "role": "system",
              "content": (
                  "Canlı web araştırması sonucu aşağıdadır (gerçek arama motorundan geldi). "
                  "Kullanıcının sorusunun cevabı büyük olasılıkla bu sonuçlardadır: önce bunları "
                  "oku, somut bilgiyi (skor, isim, tarih, sayı) doğrudan aktar ve kaynağı belirt. "
                  "Arama sorgusu şudur: " + live_search_query + ". Özellikle spor sorularında "
                  "kaynakta açıkça yazmayan golcü, oyuncu, dakika, kadro veya maç akışı UYDURMA. "
                  "Önceki asistan yanıtlarını doğrulanmış bilgi sayma. Sonuçlar soruyla ilgisizse veya boşsa güncel bilgi UYDURMA; bunu bir cümleyle "
                  "söyle. Kaynaklarda başka yapay zeka markalarının adı geçerse bunları yanıtta "
                  "ANMA; sen Aslan Parçası'sın:\n" + json.dumps(live_context, ensure_ascii=False)
              ),
          },
      )

    fallback_parts = []
    if clock_data:
      fallback_parts.append(
          f"🕒 Şu anki saat: {clock_data.get('time')} · tarih: {clock_data.get('date')} "
          f"{clock_data.get('day')} (Europe/Istanbul)."
      )
    if weather_data and weather_data.get("temperature") is not None:
      fallback_parts.append(
          f"🌤 {weather_data.get('city')} için güncel hava durumu: {weather_data.get('temperature')}°C "
          f"(hissedilen {weather_data.get('feels_like')}°C), {weather_data.get('description')}, "
          f"nem %{weather_data.get('humidity')}, rüzgâr {weather_data.get('wind_speed')} km/s. "
          f"Kaynak: Open-Meteo ({weather_data.get('observed_at')})."
      )
    if live_context and live_context.get("results"):
      source_lines = format_source_lines(live_context, 8, with_snippets=True)
      if source_lines:
        lines = [f"🔎 \"{live_context.get('query')}\" için canlı web araştırması sonuçları:"]
        for index, line in enumerate(source_lines, 1):
          lines.append(f"{index}. {line}")
        fallback_parts.append("\n".join(lines))
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
      scrubber = NameScrubber(
          chat_user_name, enabled=not FOUNDER_ASK_RE.search(question_text or "")
      )

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
                yield scrubber.feed(clean_text)
          clean_text = output_sanitizer.feed("", final=True)
          if clean_text:
            answered = True
            yield scrubber.feed(clean_text)
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
          output_sanitizer = ModelOutputStreamSanitizer()
          try:
            completion = safe_model_call(
                clients,
                messages,
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
              delta = getattr(choices[0], "delta", None)
              piece = getattr(delta, "content", None) if delta else None
              if piece:
                clean_piece = output_sanitizer.feed(piece)
                if clean_piece:
                  answered = True
                  yield scrubber.feed(clean_piece)
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
            yield scrubber.feed(clean_tail)
          if answered:
            break
          # Boş akış dönen sağlayıcıyı bir daha deneme; sıra diğer beyinde.
          if trace.get("provider"):
            silent_providers.add(trace["provider"])
          if silent_providers >= all_providers:
            break

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
