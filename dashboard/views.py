import json
import os
import base64
import io
import re
import threading
import time
import requests
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

# Dosya okuma kütüphaneleri en tepeye taşındı
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


MAX_CHAT_FILE_BYTES = 50 * 1024 * 1024
MAX_EXTRACTED_TEXT = 300_000
IMAGE_REQUEST_MIN_INTERVAL = 8
_image_request_lock = threading.Lock()
_last_image_request_at = {}


PROVIDER_GROQ = "groq"
PROVIDER_COHERE = "cohere"
PROVIDER_MISTRAL = "mistral"
PROVIDER_OPENROUTER = "openrouter"

PROVIDER_BASE_URLS = {
    PROVIDER_GROQ: "https://api.groq.com/openai/v1",
    PROVIDER_COHERE: "https://api.cohere.ai/v2",
    PROVIDER_MISTRAL: "https://api.mistral.ai/v1",
    PROVIDER_OPENROUTER: "https://openrouter.ai/api/v1",
}

PROVIDER_DEFAULT_MODELS = {
    PROVIDER_GROQ: os.environ.get("GROQ_DEFAULT_MODEL", "").strip() or "llama-3.3-70b-versatile",
    PROVIDER_COHERE: os.environ.get("COHERE_DEFAULT_MODEL", "").strip() or "command-r-plus-08-2024",
    PROVIDER_MISTRAL: os.environ.get("MISTRAL_DEFAULT_MODEL", "").strip() or "mistral-large-latest",
    PROVIDER_OPENROUTER: os.environ.get("OPENROUTER_DEFAULT_MODEL", "").strip() or "openrouter/auto",
}

PROVIDER_STREAM_SUPPORTED = {
    PROVIDER_GROQ: True,
    PROVIDER_COHERE: True,
    PROVIDER_MISTRAL: True,
    PROVIDER_OPENROUTER: True,
}

PROVIDER_TOOLS_SUPPORTED = {
    PROVIDER_GROQ: True,
    PROVIDER_COHERE: False,
    PROVIDER_MISTRAL: True,
    PROVIDER_OPENROUTER: True,
}


def get_provider_from_env_name(env_name):
  upper = (env_name or "").upper()
  if "GROQ" in upper:
    return PROVIDER_GROQ
  if "COHERE" in upper:
    return PROVIDER_COHERE
  if "MISTRAL" in upper:
    return PROVIDER_MISTRAL
  return PROVIDER_OPENROUTER


def get_provider_api_keys():
  entries = []

  def maybe_add(provider, env_name):
    val = (os.environ.get(env_name) or "").strip().strip("\"'")
    if not val or val == "gecici_anahtar":
      return
    if len(val) < 6:
      return
    entries.append({
      "provider": provider,
      "env": env_name,
      "api_key": val,
      "base_url": PROVIDER_BASE_URLS[provider],
    })

  ordered_envs = [
      (PROVIDER_OPENROUTER, "OPENROUTER_API_KEY"),
      (PROVIDER_GROQ, "GROQ_API_KEY"),
      (PROVIDER_MISTRAL, "MISTRAL_API_KEY"),
      (PROVIDER_COHERE, "COHERE_API_KEY"),
  ]
  for provider, env_name in ordered_envs:
    maybe_add(provider, env_name)

  numbered_idx = 1
  while numbered_idx <= 20:
    added_any = False
    for provider, prefix in [
        (PROVIDER_OPENROUTER, "OPENROUTER_API_KEY"),
        (PROVIDER_GROQ, "GROQ_API_KEY"),
        (PROVIDER_MISTRAL, "MISTRAL_API_KEY"),
        (PROVIDER_COHERE, "COHERE_API_KEY"),
    ]:
      env_name = f"{prefix}_{numbered_idx}"
      val = (os.environ.get(env_name) or "").strip().strip("\"'")
      if val and val != "gecici_anahtar" and len(val) >= 6:
        entries.append({
          "provider": provider,
          "env": env_name,
          "api_key": val,
          "base_url": PROVIDER_BASE_URLS[provider],
        })
        added_any = True
    if not added_any and numbered_idx > 3:
      pass
    if numbered_idx > 3 and not any(
        (os.environ.get(f"{p}_{numbered_idx}") or "").strip() for p in ("GROQ_API_KEY", "COHERE_API_KEY", "MISTRAL_API_KEY", "OPENROUTER_API_KEY")
    ):
      break
    numbered_idx += 1

  list_envs = [
      (PROVIDER_OPENROUTER, "OPENROUTER_API_KEYS"),
      (PROVIDER_OPENROUTER, "OPENROUTER_API_KEY_POOL"),
      (PROVIDER_OPENROUTER, "OPENROUTER_KEYS"),
      (PROVIDER_OPENROUTER, "OPENROUTER_POOL"),
      (PROVIDER_OPENROUTER, "OPENROUTER_BACKUP_KEYS"),
      (PROVIDER_OPENROUTER, "OPENROUTER_RESERVE_KEYS"),
      (PROVIDER_GROQ, "GROQ_API_KEYS"),
      (PROVIDER_GROQ, "GROQ_BACKUP_KEYS"),
      (PROVIDER_MISTRAL, "MISTRAL_API_KEYS"),
      (PROVIDER_MISTRAL, "MISTRAL_BACKUP_KEYS"),
      (PROVIDER_COHERE, "COHERE_API_KEYS"),
      (PROVIDER_COHERE, "COHERE_BACKUP_KEYS"),
  ]
  for provider, env_name in list_envs:
    raw = (os.environ.get(env_name) or "").strip()
    if not raw:
      continue
    for piece in raw.split(","):
      val = piece.strip().strip("\"'")
      if not val or val == "gecici_anahtar" or len(val) < 6:
        continue
      entries.append({
        "provider": provider,
        "env": env_name,
        "api_key": val,
        "base_url": PROVIDER_BASE_URLS[provider],
      })

  numbered_idx = 1
  while numbered_idx <= 20:
    patterns = [
        (PROVIDER_GROQ, f"GROQ_BACKUP_{numbered_idx}"),
        (PROVIDER_GROQ, f"GROQ_RESERVE_{numbered_idx}"),
        (PROVIDER_GROQ, f"GROQ_ALT_{numbered_idx}"),
        (PROVIDER_MISTRAL, f"MISTRAL_BACKUP_{numbered_idx}"),
        (PROVIDER_MISTRAL, f"MISTRAL_RESERVE_{numbered_idx}"),
        (PROVIDER_MISTRAL, f"MISTRAL_ALT_{numbered_idx}"),
        (PROVIDER_COHERE, f"COHERE_BACKUP_{numbered_idx}"),
        (PROVIDER_COHERE, f"COHERE_RESERVE_{numbered_idx}"),
        (PROVIDER_COHERE, f"COHERE_ALT_{numbered_idx}"),
        (PROVIDER_OPENROUTER, f"OPENROUTER_BACKUP_{numbered_idx}"),
        (PROVIDER_OPENROUTER, f"OPENROUTER_RESERVE_{numbered_idx}"),
        (PROVIDER_OPENROUTER, f"OPENROUTER_ALT_{numbered_idx}"),
    ]
    any_added = False
    for provider, env_name in patterns:
      val = (os.environ.get(env_name) or "").strip().strip("\"'")
      if val and val != "gecici_anahtar" and len(val) >= 6:
        entries.append({
          "provider": provider,
          "env": env_name,
          "api_key": val,
          "base_url": PROVIDER_BASE_URLS[provider],
        })
        any_added = True
    if numbered_idx > 5 and not any_added:
      break
    numbered_idx += 1

  seen_keys = set()
  unique_entries = []
  for entry in entries:
    sig = (entry["provider"], entry["api_key"])
    if sig in seen_keys:
      continue
    seen_keys.add(sig)
    unique_entries.append(entry)
  return unique_entries


def get_all_openrouter_api_keys():
  return [e["api_key"] for e in get_provider_api_keys() if e["provider"] == PROVIDER_OPENROUTER]


def get_openai_client():
  providers = get_provider_api_keys()
  if not providers:
    return None
  first = providers[0]
  return OpenAI(
      base_url=first["base_url"],
      api_key=first["api_key"],
  )


def build_provider_client(entry):
  return OpenAI(
      base_url=entry["base_url"],
      api_key=entry["api_key"],
  )


def build_openrouter_client(api_key):
  return OpenAI(
      base_url=PROVIDER_BASE_URLS[PROVIDER_OPENROUTER],
      api_key=api_key,
  )


def provider_resolve_model(entry, requested_model, fallback_overrides=None):
  provider = entry["provider"]
  default = PROVIDER_DEFAULT_MODELS[provider]
  if fallback_overrides and provider in fallback_overrides and fallback_overrides[provider]:
    default = fallback_overrides[provider]

  if provider == PROVIDER_OPENROUTER:
    if not requested_model:
      return default
    return requested_model

  return default


def provider_supports_stream(entry):
  return PROVIDER_STREAM_SUPPORTED.get(entry["provider"], True)


def _error_http_code(error):
  code = None
  if hasattr(error, "status_code"):
    code = str(getattr(error, "status_code", "") or "") or None
  if code is None and hasattr(error, "response") and error.response is not None:
    try:
      code = str(error.response.status_code)
    except Exception:
      code = None
  return code


def _error_code_attr(error):
  code = None
  try:
    code = str(getattr(error, "code", "") or "")
  except Exception:
    code = None
  if not code:
    try:
      body = getattr(error, "body", None) or {}
      if isinstance(body, dict):
        nested = body.get("error") or body.get("code")
        if isinstance(nested, dict):
          code = str(nested.get("code") or "")
        elif nested is not None:
          code = str(nested)
    except Exception:
      pass
  return (code or "").lower() or None


def _error_type_name(error):
  cls_name = type(error).__name__.lower()
  module = (getattr(type(error), "__module__", "") or "").lower()
  return f"{module}.{cls_name}" if module else cls_name


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
    return str(text_content)[:MAX_EXTRACTED_TEXT]
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
      return raw.decode("utf-8", errors="replace")[:MAX_EXTRACTED_TEXT]
    if (extension == ".pdf" or content_type == "application/pdf") and PdfReader:
      pages = PdfReader(io.BytesIO(raw)).pages
      extracted = "\n\n".join((page.extract_text() or "") for page in pages)
      if not extracted.strip():
        raise ValueError(
            f"{filename} içinde seçilebilir metin yok. Taranmış PDF'ler şu anda okunamıyor."
        )
      return extracted[:MAX_EXTRACTED_TEXT]
    if (extension == ".docx" or content_type.endswith("wordprocessingml.document")) and Document:
      document = Document(io.BytesIO(raw))
      extracted = "\n".join(paragraph.text for paragraph in document.paragraphs)
      if not extracted.strip():
        raise ValueError(f"{filename} içinde okunabilir metin bulunamadı.")
      return extracted[:MAX_EXTRACTED_TEXT]
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


def get_weather(city, country="TR"):
    """Get current weather from Open-Meteo without a paid API key."""
    city = (city or "").strip()
    country = (country or "TR").strip().upper()
    if not city:
        return {"error": "Hava durumunu bulmak için bir şehir adı gerekli."}

    try:
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 10, "language": "tr", "format": "json"},
            timeout=10,
        )
        geo_response.raise_for_status()
        matches = geo_response.json().get("results") or []
        if country:
            matches = sorted(
                matches,
                key=lambda result: result.get("country_code", "").upper() != country,
            )
        if not matches:
            return {"error": f"City '{city}' not found"}
        location = matches[0]
        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "weather_code,wind_speed_10m"
                ),
                "timezone": "auto",
            },
            timeout=10,
        )
        weather_response.raise_for_status()
        current = weather_response.json().get("current") or {}
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
        return {
            "city": location.get("name", city),
            "country": location.get("country", country),
            "temperature": current.get("temperature_2m"),
            "feels_like": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "description": weather_descriptions.get(weather_code, "Güncel hava durumu"),
            "wind_speed": current.get("wind_speed_10m"),
            "observed_at": current.get("time"),
            "source": "Open-Meteo",
        }
    except requests.Timeout:
        return {"error": "Weather service timeout"}
    except requests.RequestException as e:
        return {"error": f"Weather service error: {str(e)}"}
    except Exception as e:
        return {"error": f"Weather error: {str(e)}"}


def web_search(query, num_results=5):
    """Search current web results using DuckDuckGo HTML (no API key needed)."""
    try:
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        response = requests.post(url, data=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        for result in soup.select(".result")[:max(1, min(int(num_results), 10))]:
            title = result.select_one(".result__a")
            snippet = result.select_one(".result__snippet")
            if title or snippet:
                results.append({
                    "title": title.get_text(" ", strip=True) if title else "",
                    "url": title.get("href", "") if title else "",
                    "snippet": snippet.get_text(" ", strip=True) if snippet else "",
                })
        
        return {
            "query": query,
            "results": results if results else ["Arama sonucu bulunamadı."]
        }
    except Exception as e:
        return {"error": f"Web search failed: {str(e)}", "results": []}


def should_fetch_live_context(text):
  """Identify requests where a stale model answer would be misleading."""
  normalized = (text or "").lower()
  hints = (
      "internetten", "güncel", "bugün", "şu an", "şuan", "son dakika",
      "haber", "maç", "skor", "sonuç", "spor", "hava durumu", "sıcaklık",
      "fiyat", "kur", "döviz", "kim kazandı", "ne zaman", "kaçta",
  )
  return any(hint in normalized for hint in hints)


def profile_payload(user):
  profile = get_user_profile(user)
  return {
      "username": user.username,
      "avatar": profile.avatar,
      "theme": profile.theme,
      "pattern": profile.pattern,
  }


def is_api_key_retriable_error(error):
  http_code = _error_http_code(error)
  attr_code = _error_code_attr(error)
  type_name = _error_type_name(error)
  error_text = str(error).lower()

  if http_code in {"401", "402", "403"}:
    return True

  auth_quota_codes = (
      "insufficient_quota", "quota_exceeded", "billing_not_active",
      "payment_required", "invalid_api_key", "api_key_expired",
      "forbidden", "unauthorized", "key_expired", "key_revoked",
      "key_disabled", "account_disabled",
  )
  if attr_code and attr_code in auth_quota_codes:
    return True

  auth_quota_types = (
      "insufficientquotaerror", "authenticationerror", "permissiondeniederror",
      "ratelimiterror", "openaierror.authenticationerror",
      "openaierror.insufficientquota", "unauthenticatederror",
      "paymentrequirederror",
  )
  if any(tok in type_name for tok in auth_quota_types):
    return True

  explicit_markers = (
      "insufficient_quota", "quota_exceeded", "quota exceeded",
      "billing_not_active", "payment required", "payment_required",
      "out of credit", "no credits remaining", "credits exhausted",
      "balance exceeded", "credit limit reached",
      "invalid_api_key", "api key expired", "your api key is expired",
      "key is invalid", "unauthorized access", "invalid authentication",
      "revoked", "disabled key", "account is disabled",
  )
  if any(marker in error_text for marker in explicit_markers):
    return True

  if "credit" in error_text and (
      "out of" in error_text or "no credit" in error_text or
      "not enough" in error_text or "insufficient" in error_text or
      "exceeded your" in error_text or "run out" in error_text or
      "remaining" in error_text or "limit" in error_text or
      "add more" in error_text or "balance" in error_text or
      "quota" in error_text or "billing" in error_text or
      "payment" in error_text or "purchase" in error_text or
      "top up" in error_text or "top-up" in error_text
  ):
    return True

  if "401" in error_text or "403" in error_text or "402" in error_text:
    if (
        "unauthorized" in error_text or "forbidden" in error_text or
        "payment" in error_text or "quota" in error_text or "credit" in error_text or
        "billing" in error_text or "key" in error_text or "auth" in error_text
    ):
      return True

  return False


def is_context_length_error(error):
  http_code = _error_http_code(error)
  attr_code = _error_code_attr(error)
  type_name = _error_type_name(error)
  error_text = str(error).lower()

  if attr_code and attr_code in ("context_length_exceeded", "string_above_max_length", "max_length"):
    return True

  if "toolarge" in type_name or "toolong" in type_name:
    return True

  markers = (
      "context_length_exceeded",
      "context length exceeded",
      "prompt is too long",
      "content length exceeded",
      "maximum content length",
      "input too long",
      "too many input tokens",
      "prompt tokens exceed",
      "token count exceeds model's maximum",
      "your input is too long",
      "exceeds the model's maximum context length",
      "the message you submitted was too long",
      "maximum context length is",
      "request exceeds max content length",
      "input length exceeded",
  )
  if any(marker in error_text for marker in markers):
    return True

  if http_code == "400" and ("context" in error_text and "length" in error_text):
    return True

  return False


def is_rate_limit_error(error):
  http_code = _error_http_code(error)
  attr_code = _error_code_attr(error)
  type_name = _error_type_name(error)
  error_text = str(error).lower()

  if http_code == "429":
    return True
  if attr_code and ("rate_limit" in attr_code or attr_code == "429"):
    return True
  if "ratelimiterror" in type_name or "toomanyrequestserror" in type_name:
    return True

  rate_markers = ("429", "rate limit", "too many requests", "requests per minute", "per second", "throttled", "rate-limit")
  return any(marker in error_text for marker in rate_markers)


def friendly_api_error(error):
  http_code = _error_http_code(error)
  error_text = str(error).lower()

  if http_code == "429" or is_rate_limit_error(error):
    return "Yapay zekâ servisi şu anda yoğun. Birkaç saniye sonra tekrar deneyin."

  if http_code == "404":
    return "Seçili yapay zekâ modeli kullanılamıyor. Sunucu ayarlarından geçerli bir model seçin."
  if "404" in error_text or "not found" in error_text or "model_not_found" in error_text:
    if not is_context_length_error(error):
      return "Seçili yapay zekâ modeli kullanılamıyor. Sunucu ayarlarından geçerli bir model seçin."

  if http_code in {"401", "403"}:
    return "Yapay zekâ servisi yetkilendirmeyi reddetti. API anahtarını kontrol edin."
  if "401" in error_text or "403" in error_text:
    if "unauthorized" in error_text or "forbidden" in error_text or "invalid" in error_text or "key" in error_text or "auth" in error_text:
      return "Yapay zekâ servisi yetkilendirmeyi reddetti. API anahtarını kontrol edin."

  if http_code == "402":
    return "Yapay zekâ servisi için yeterli API kredisi yok veya istek çok uzun. Daha kısa bir mesaj deneyin ya da API kredisi ekleyin."
  attr_code = _error_code_attr(error)
  if attr_code and attr_code in ("insufficient_quota", "quota_exceeded", "billing_not_active", "payment_required"):
    return "Yapay zekâ servisi için yeterli API kredisi yok veya istek çok uzun. Daha kısa bir mesaj deneyin ya da API kredisi ekleyin."

  explicit_quota_phrases = (
      "insufficient_quota", "quota exceeded", "out of credit",
      "no credits remaining", "credits exhausted",
      "balance exceeded", "credit limit reached",
      "billing not active", "payment required",
      "add more credits", "purchase credits", "top up credits",
  )
  if any(phrase in error_text for phrase in explicit_quota_phrases):
    return "Yapay zekâ servisi için yeterli API kredisi yok veya istek çok uzun. Daha kısa bir mesaj deneyin ya da API kredisi ekleyin."

  if is_context_length_error(error):
    return "Mesaj veya sohbet geçmişi çok uzun. Daha kısa bir mesaj deneyin ya da sohbeti sıfırlayın."

  if "credit" in error_text and (
      "out of" in error_text or "no credit" in error_text or
      "not enough" in error_text or "insufficient" in error_text or
      "exceeded your" in error_text or "run out" in error_text or
      "remaining" in error_text or "limit" in error_text or
      "add more" in error_text or "balance" in error_text or
      "quota" in error_text or "billing" in error_text or
      "payment" in error_text or "purchase" in error_text or
      "top up" in error_text or "top-up" in error_text
  ):
    return "Yapay zekâ servisi için yeterli API kredisi yok veya istek çok uzun. Daha kısa bir mesaj deneyin ya da API kredisi ekleyin."

  return "Yapay zekâ yanıtı alınamadı. Lütfen biraz sonra tekrar deneyin."


def generate_image_with_openrouter(prompt, image_model, api_key):
  """Use OpenRouter's dedicated image API and return a data URL."""
  response = requests.post(
      "https://openrouter.ai/api/v1/images",
      headers={
          "Authorization": f"Bearer {api_key}",
          "Content-Type": "application/json",
          "HTTP-Referer": "https://aslan-parcasi-ai.onrender.com",
          "X-Title": "Aslan Parçası AI",
      },
      json={
          "model": image_model,
          "prompt": (
              "Create exactly one image that follows the user's request literally. "
              "The named subject, object, place, count, action and composition are mandatory. "
              "Never replace an airplane, city, building or other requested subject with a "
              "generic animal, stock photo or unrelated scene. If the user asks for a "
              "realistic image, make it photorealistic with natural lighting, accurate "
              "materials, perspective and fine detail. Do not add unrelated objects. "
              "User request, verbatim: "
              + prompt
          ),
      },
      timeout=180,
  )
  response.raise_for_status()
  payload = response.json()
  image = (payload.get("data") or [{}])[0]
  if image.get("url"):
    return image["url"]
  encoded = image.get("b64_json")
  if encoded:
    media_type = image.get("media_type") or "image/png"
    return f"data:{media_type};base64,{encoded}"
  raise ValueError("Görsel servisi yanıtında görsel verisi bulunamadı.")


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
):
    all_entries = get_provider_api_keys()
    if not all_entries:
        all_entries = [None]

    configured_fallback_openrouter = os.environ.get("OPENROUTER_FALLBACK_MODEL", "").strip()

    last_error = None
    context_length_hit = False
    used_original = False

    for entry in all_entries:
        current_client = client
        provider = PROVIDER_OPENROUTER
        actually_stream = stream
        if entry is None:
            if used_original:
                continue
            used_original = True
            provider = PROVIDER_OPENROUTER
            actually_stream = stream
            model_choices = list(dict.fromkeys(item for item in [model, configured_fallback_openrouter, "openrouter/auto"] if item))
        else:
            current_client = build_provider_client(entry)
            provider = entry["provider"]
            fallback_overrides = {
                PROVIDER_GROQ: os.environ.get("GROQ_FALLBACK_MODEL", "").strip() or None,
                PROVIDER_COHERE: os.environ.get("COHERE_FALLBACK_MODEL", "").strip() or None,
                PROVIDER_MISTRAL: os.environ.get("MISTRAL_FALLBACK_MODEL", "").strip() or None,
            }
            resolved_main = provider_resolve_model(entry, model, fallback_overrides)
            if provider == PROVIDER_OPENROUTER:
                model_choices = list(dict.fromkeys(x for x in [
                    resolved_main,
                    configured_fallback_openrouter,
                    fallback_overrides.get(provider),
                    PROVIDER_DEFAULT_MODELS[provider],
                ] if x))
            else:
                model_choices = list(dict.fromkeys(x for x in [
                    resolved_main,
                    fallback_overrides.get(provider),
                    PROVIDER_DEFAULT_MODELS[provider],
                ] if x))
            actually_stream = stream and provider_supports_stream(entry)

        use_tools = tools if PROVIDER_TOOLS_SUPPORTED.get(provider, False) else None

        for attempt_model in model_choices:
            try:
                kwargs = {
                    "model": attempt_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": actually_stream,
                }
                if timeout is not None:
                    kwargs["timeout"] = max(1, timeout)
                if use_tools:
                    kwargs["tools"] = use_tools
                    kwargs["tool_choice"] = "auto"

                completion = current_client.chat.completions.create(**kwargs)
                return completion
            except Exception as e:
                last_error = e

                if is_rate_limit_error(e):
                    for retry in range(2):
                        time.sleep(1 + retry)
                        try:
                            completion = current_client.chat.completions.create(**kwargs)
                            return completion
                        except Exception as retry_e:
                            last_error = retry_e
                            continue

                if is_context_length_error(e):
                    context_length_hit = True
                    break

                break

    if context_length_hit and is_context_length_error(last_error):
        raise last_error

    raise last_error or Exception("Tüm sağlayıcılar, modeller ve API anahtarları başarısız oldu")


def deep_think_call(
    client,
    messages,
    model,
    deep_think_seconds,
    temperature=0.7,
    max_tokens=4096,
    tools=None,
):
    """Research with available tools, then stream a concise answer.

    The selected duration is a research budget and API timeout ceiling. The
    assistant may finish early when its research is complete rather than
    wasting the user's time or spending credits on idle calls.
    """
    deadline = time.monotonic() + max(30, min(1800, int(deep_think_seconds)))
    research_messages = [*messages]
    research_messages[0] = {
        **messages[0],
        "content": (
            str(messages[0].get("content", ""))
            + " Araştırma modunda web_search ve get_weather araçlarını kullan. "
            "Güncel sonuç, hava durumu, spor skoru veya değişken bilgi sorularında "
            "kaynakları kontrol et. Gizli düşünce zincirini kullanıcıya yazma; "
            "bulguları ve gerekli kısa gerekçeyi final yanıtta özetle."
        ),
    }

    rounds = 0
    max_rounds = max(1, min(6, (deep_think_seconds + 89) // 90))
    while tools and rounds < max_rounds:
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            break
        research_completion = safe_model_call(
            client,
            research_messages,
            model,
            temperature=0.3,
            max_tokens=min(max_tokens, 2048),
            stream=False,
            tools=tools,
            timeout=min(120, remaining),
        )
        choice = research_completion.choices[0] if research_completion.choices else None
        assistant_message = choice.message if choice else None
        tool_calls = getattr(assistant_message, "tool_calls", None) or []
        if not tool_calls:
            break

        research_messages.append({
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
            if time.monotonic() >= deadline:
                break
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except (TypeError, json.JSONDecodeError):
                arguments = {}
            result = execute_function(call.function.name, arguments)
            research_messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
        rounds += 1

    final_messages = [
        *research_messages,
        {
            "role": "system",
            "content": (
                "Yanıtını araştırma bulgularına dayandır. Güncel bilgi için "
                "araçlardan gelen veriyi kullan; araç hata verdiyse veri uydurma. "
                "Gizli düşünme sürecini paylaşma."
            ),
        },
    ]
    remaining = max(1, int(deadline - time.monotonic()))
    return safe_model_call(
        client,
        final_messages,
        model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        timeout=min(120, remaining),
    )


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
  return JsonResponse({"status": "ok", "service": "aslan-parcasi-ai"})


@login_required(login_url="login")
@require_POST  
def api_image_generate(request):
  try:
    data = json.loads(request.body or "{}")
    prompt = (data.get("prompt") or "").strip()
    
    if not prompt:
      return JsonResponse({"error": "Prompt boş olamaz."}, status=400)
    
    all_keys = get_all_openrouter_api_keys()
    if not all_keys:
      return JsonResponse(
          {"error": "OPENROUTER_API_KEY tanımlı değil. Lütfen API anahtarını ayarlayın."},
          status=503,
      )

    now = time.monotonic()
    with _image_request_lock:
      previous_request = _last_image_request_at.get(request.user.pk, 0)
      if now - previous_request < IMAGE_REQUEST_MIN_INTERVAL:
        return JsonResponse(
            {"error": "Görsel üretimi için birkaç saniye bekleyin; aynı anda birden fazla istek gönderilemez."},
            status=429,
        )
      _last_image_request_at[request.user.pk] = now

    image_model = (
        os.environ.get("OPENROUTER_IMAGE_MODEL", "").strip()
        or "google/gemini-2.5-flash-image-preview"
    )

    last_image_error = None
    img_context_hit = False
    for img_api_key in all_keys:
      try:
        image_url = generate_image_with_openrouter(prompt, image_model, img_api_key)
        return JsonResponse({"status": "success", "image_url": image_url})
      except Exception as img_error:
        last_image_error = img_error
        if is_context_length_error(img_error):
          img_context_hit = True
          continue
        if not is_api_key_retriable_error(img_error) and not is_rate_limit_error(img_error):
          return JsonResponse({"error": friendly_api_error(img_error)}, status=503)
        continue

    if img_context_hit and last_image_error and is_context_length_error(last_image_error):
      return JsonResponse({"error": friendly_api_error(last_image_error)}, status=503)
    if last_image_error:
      return JsonResponse({"error": friendly_api_error(last_image_error)}, status=503)
    return JsonResponse({"error": "Görsel üretilemedi."}, status=503)
      
  except json.JSONDecodeError:
    return JsonResponse({"error": "Geçersiz JSON."}, status=400)
  except Exception as e:
    return JsonResponse({"error": str(e)}, status=500)


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
    return JsonResponse({"error": "Geçersiz sohbet verisi."}, status=400)
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
    deep_think = data.get("deep_think", False)
    try:
      deep_think_seconds = max(30, min(1800, int(data.get("deep_think_seconds") or 300)))
    except (TypeError, ValueError):
      deep_think_seconds = 300
    images = data.get("images", [])
    files = data.get("files", [])
    voice_transcript = data.get("voice_transcript", "")
    voice = data.get("voice") or {}

    if not user_message and not voice_transcript and not images and not files and not (isinstance(voice, dict) and voice.get("base64")):
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

    client = get_openai_client()
    if client is None:
        return JsonResponse(
            {"error": "API anahtarı tanımlı değil. Lütfen OPENROUTER_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY veya COHERE_API_KEY'den en az birini ayarlayın."},
            status=503,
        )

    base_prompt = (
        "Sen Aslan Parçası adında son derece zeki, enerjik, samimi ve geniş bilgi birikimine sahip bir yapay zeka asistanısın. "
        "Seni oluşturan, kuran ve geliştiren vizyoner lider, müstakbel MEAY ASLAN PARÇASI AI şirketinin kurucusu Ayaz Kaplan'dır. "
        "KRİTİK KURAL: Sana sorulmadıkça saat, tarih, hava durumu veya kurucunun kim olduğu bilgisini KENDİLİĞİNDEN söyleme. "
        "Bu bilgileri SADECE kullanıcı açıkça sorduğunda ver. "
        "Gerçek zamanlı web araması ve güncel hava durumu için araçların var. "
        "Güncel skor, sonuç, hava durumu, haber veya değişken bilgi sorularında ilgili aracı çağır. "
        "Araç hata verirse veya sonuç bulamazsa bunu açıkça belirt ve veri uydurma. "
        "Hangi dilde yazılırsa yazılsın yüksek kalitede, akıcı bir dost gibi yanıt ver."
    )

    if mode == "normal":
        system_instruction = (
            base_prompt +
            " Genel asistan modundasın. Kullanıcıya samimi, yardımsever ve kapsamlı bir şekilde yardımcı ol. "
            "Konuları derinlemesine ara, bağlamı iyi anla ve net, yapılandırılmış yanıtlar ver. "
            "Mümkün oldukça pratik çözümler sun ve adım adım açıklamalar yap. "
            "Eğer bir soru bilginin dışındaysa, dürüstçe söyle ve alternatif yaklaşım öner."
        )
        model = "openai/gpt-4o"
        temperature = 0.7
        max_tokens = 2048

    elif mode == "code":
        system_instruction = (
            "Sen Aslan Parçası AI'nın Kod Asistanı modundasın. "
            "Tam bir kod yazma ustasisin - Cursor ve Replit tarzı agent kişiliğine sahipsin. "
            "Kullanıcının kod ihtiyaçlarını eksiksiz, modern, çalıştırılabilir ve profesyonel kod blokları (markdown formatında) olarak karşıla. "
            "Her kod bloğunda tam çözümler sun, açıklamalar ekle ve en iyi pratikleri uygula. "
            "GitHub/GitLab entegrasyon iş akışlarına dair yardım et, commit mesajları öner, branch stratejileri danış. "
            "Farklı programlama dillerinde uzmanlaş, hata ayıklama, optimizasyon ve refactoring konularında yardımcı ol. "
            "Kod örneklerinde her zaman gerçekçi ve kullanılabilir kod ver. "
            "Sana sorulmadıkça saat, tarih, hava durumu veya kurucunun kimliği hakkında bilgi verme."
        )
        model = "openai/gpt-4o"
        temperature = 0.3
        max_tokens = 4096

    elif mode == "fast":
        system_instruction = (
            "Sen Aslan Parçası AI'nın Hızlı Analiz modundasın. "
            "Işık hızında, çok kısa ve öz cevaplar ver. "
            "Gereksiz detaylardan kaçın, doğrudan noktaya odaklan. "
            "Normal moddan belirgin daha hızlı ve kısa yanıtlar üret. "
            "Karmaşık konuları basitleştir, hızlı özetler ve hızlı kararlar ver. "
            "Sana sorulmadıkça saat, tarih, hava durumu veya kurucunun kimliği hakkında bilgi verme."
        )
        model = "openai/gpt-4o-mini"
        temperature = 0.5
        max_tokens = 2048

    else:
        system_instruction = base_prompt
        model = "openai/gpt-4o"
        temperature = 0.7
        max_tokens = 4096

    system_instruction += (
        " Güncel skor, hava durumu, haber veya değişken bilgi istenirse web_search ya da "
        "get_weather aracını kullan; araç sonucu yoksa güncel veri uydurma."
    )

    if deep_think:
        minutes = deep_think_seconds // 60
        seconds = deep_think_seconds % 60
        budget_label = f"{minutes} dakika {seconds} saniye" if minutes else f"{seconds} saniye"
        system_instruction += (
            f" Kapsamlı araştırma modu açık; kullanıcı sana {budget_label} düşünme bütçesi verdi. "
            "Bu süreyi araştırma bütçesi olarak kullan. Güncel bilgi gerekiyorsa web_search veya "
            "get_weather aracını çağır. Araç sonuçlarını kontrol et; başarısız olursa veri uydurma. "
            "Gizli düşünce zincirini kullanıcıya yazma; bulguları ve gerekli kısa gerekçeyi özetle."
        )
        max_tokens = min(max_tokens, 4096)

    messages = [{"role": "system", "content": system_instruction}]

    user_content = []
    if full_message:
        user_content.append({"type": "text", "text": full_message})
    
    if voice_transcript and voice_transcript not in full_message:
        user_content.append({"type": "text", "text": f"[Ses kaydı metni]\n{voice_transcript}"})

    for img in images:
        if img.get("url"):
            user_content.append({"type": "image_url", "image_url": {"url": img["url"]}})
        elif img.get("base64"):
            image_type = img.get("type") or "image/jpeg"
            user_content.append({"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{img['base64']}"}})

    if isinstance(voice, dict) and voice.get("base64") and not voice_transcript:
        return JsonResponse(
            {"error": "Ses kaydı metne çevrilemedi. Lütfen kaydı yeniden deneyin veya mesajınızı yazın."},
            status=400,
        )

    for h in history:
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

    if should_fetch_live_context(full_message):
        live_context = web_search(full_message, 5)
        messages.insert(
            1,
            {
                "role": "system",
                "content": (
                    "Canlı web araştırması sonucu aşağıdadır. Bu veriyi yalnızca "
                    "kullanıcının sorusuyla ilgiliyse kullan; sonuç yoksa veya hata varsa "
                    "güncel bilgi uydurma:\n"
                    + json.dumps(live_context, ensure_ascii=False)
                ),
            },
        )

    try:
        request_model = model
        completion = None
        if not deep_think:
            completion = safe_model_call(
                client,
                messages,
                request_model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                deep_think=False,
                tools=TOOLS,
            )
    except Exception as api_error:
      return JsonResponse({"error": friendly_api_error(api_error)}, status=503)

    def run_initial_chat_round():
      last_stream_error = None
      all_entries = get_provider_api_keys() or [None]
      total_attempts = 0
      used_original = False
      for entry in all_entries:
        total_attempts += 1
        if total_attempts > 3 * max(1, len(all_entries)):
          break
        attempt_client = client
        provider = PROVIDER_OPENROUTER
        actually_stream = True
        attempt_model_to_use = request_model
        use_tools = TOOLS
        if entry is None:
          if used_original:
            continue
          used_original = True
          provider = PROVIDER_OPENROUTER
          actually_stream = True
          attempt_model_to_use = request_model
          use_tools = TOOLS if PROVIDER_TOOLS_SUPPORTED.get(PROVIDER_OPENROUTER, True) else None
        else:
          attempt_client = build_provider_client(entry)
          provider = entry["provider"]
          fallback_overrides = {
              PROVIDER_GROQ: os.environ.get("GROQ_FALLBACK_MODEL", "").strip() or None,
              PROVIDER_COHERE: os.environ.get("COHERE_FALLBACK_MODEL", "").strip() or None,
              PROVIDER_MISTRAL: os.environ.get("MISTRAL_FALLBACK_MODEL", "").strip() or None,
          }
          attempt_model_to_use = provider_resolve_model(entry, request_model, fallback_overrides)
          actually_stream = provider_supports_stream(entry)
          use_tools = TOOLS if PROVIDER_TOOLS_SUPPORTED.get(provider, False) else None

        buffered_text = []
        buffered_tool_calls = {}
        try:
          tool_calls_buffer = []
          response_completion = None
          if deep_think:
            response_completion = deep_think_call(
                attempt_client,
                messages,
                attempt_model_to_use,
                deep_think_seconds,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=use_tools,
            )
          else:
            try:
              response_completion = safe_model_call(
                  attempt_client,
                  messages,
                  attempt_model_to_use,
                  temperature=temperature,
                  max_tokens=max_tokens,
                  stream=actually_stream,
                  deep_think=False,
                  tools=use_tools,
              )
            except Exception as pre_err:
              last_stream_error = pre_err
              if is_context_length_error(pre_err):
                return {
                  "ok": False,
                  "error": pre_err,
                  "tool_calls": [],
                  "finalize_partial": False,
                }
              if is_rate_limit_error(pre_err):
                try:
                  time.sleep(2)
                except Exception:
                  pass
              continue

          for chunk in response_completion:
            if chunk.choices and chunk.choices[0].delta.content:
              buffered_text.append(chunk.choices[0].delta.content)

            if chunk.choices and chunk.choices[0].delta.tool_calls:
              for tool_call in chunk.choices[0].delta.tool_calls:
                idx = int(getattr(tool_call, "index", 0) or 0)
                if idx not in buffered_tool_calls:
                  buffered_tool_calls[idx] = {
                    "id": getattr(tool_call, "id", None),
                    "name": "",
                    "arguments": "",
                  }
                slot = buffered_tool_calls[idx]
                if not slot["id"] and getattr(tool_call, "id", None):
                  slot["id"] = tool_call.id
                func = getattr(tool_call, "function", None)
                if func is not None:
                  if getattr(func, "name", None):
                    slot["name"] = slot["name"] or (func.name or "")
                  if getattr(func, "arguments", None):
                    slot["arguments"] = slot["arguments"] + (func.arguments or "")

          final_tool_calls = []
          for idx in sorted(buffered_tool_calls.keys()):
            item = buffered_tool_calls[idx]
            if item and (item["name"] or item["arguments"] or item.get("id")):
              final_tool_calls.append({
                "index": idx,
                "id": item.get("id"),
                "name": item.get("name") or "",
                "arguments": item.get("arguments") or "",
              })
          return {
            "ok": True,
            "text": "".join(buffered_text),
            "tool_calls": final_tool_calls,
            "error": None,
          }
        except Exception as stream_e:
          last_stream_error = stream_e
          total_yielded_so_far = len(buffered_text)
          if total_yielded_so_far > 0:
            return {
              "ok": False,
              "partial_text": "".join(buffered_text),
              "error": last_stream_error,
              "tool_calls": [],
              "finalize_partial": True,
            }
          if is_context_length_error(stream_e):
            return {
              "ok": False,
              "error": stream_e,
              "tool_calls": [],
              "finalize_partial": False,
            }
          if is_rate_limit_error(stream_e):
            try:
              time.sleep(2)
            except Exception:
              pass
          continue

      return {
        "ok": False,
        "error": last_stream_error or Exception("Akış başarısız"),
        "tool_calls": [],
        "finalize_partial": False,
      }

    def run_final_text_round(local_messages):
      last_stream_error = None
      all_entries = get_provider_api_keys() or [None]
      total_attempts = 0
      used_original = False
      for entry in all_entries:
        total_attempts += 1
        if total_attempts > 3 * max(1, len(all_entries)):
          break
        attempt_client = client
        provider = PROVIDER_OPENROUTER
        actually_stream = True
        attempt_model_to_use = request_model
        if entry is None:
          if used_original:
            continue
          used_original = True
        else:
          attempt_client = build_provider_client(entry)
          provider = entry["provider"]
          fallback_overrides = {
              PROVIDER_GROQ: os.environ.get("GROQ_FALLBACK_MODEL", "").strip() or None,
              PROVIDER_COHERE: os.environ.get("COHERE_FALLBACK_MODEL", "").strip() or None,
              PROVIDER_MISTRAL: os.environ.get("MISTRAL_FALLBACK_MODEL", "").strip() or None,
          }
          attempt_model_to_use = provider_resolve_model(entry, request_model, fallback_overrides)
          actually_stream = provider_supports_stream(entry)

        buffered_text = []
        try:
          final_completion = None
          try:
            final_completion = safe_model_call(
                attempt_client,
                local_messages,
                attempt_model_to_use,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=actually_stream,
                deep_think=False,
            )
          except Exception as pre_err:
            last_stream_error = pre_err
            if is_context_length_error(pre_err):
              return {"ok": False, "error": pre_err, "finalize_partial": False}
            if is_rate_limit_error(pre_err):
              try:
                time.sleep(2)
              except Exception:
                pass
            continue
          for chunk in final_completion:
            if chunk.choices and chunk.choices[0].delta.content:
              buffered_text.append(chunk.choices[0].delta.content)
          return {
            "ok": True,
            "text": "".join(buffered_text),
            "error": None,
          }
        except Exception as stream_e:
          last_stream_error = stream_e
          total_yielded_so_far = len(buffered_text)
          if total_yielded_so_far > 0:
            return {
              "ok": False,
              "partial_text": "".join(buffered_text),
              "error": last_stream_error,
              "finalize_partial": True,
            }
          if is_context_length_error(stream_e):
            return {"ok": False, "error": stream_e, "finalize_partial": False}
          if is_rate_limit_error(stream_e):
            try:
              time.sleep(2)
            except Exception:
              pass
          continue

      return {
        "ok": False,
        "error": last_stream_error or Exception("Final akış başarısız"),
        "finalize_partial": False,
      }

    def generate():
      overall_yielded = 0
      try:
        round_result = run_initial_chat_round()
        if not round_result.get("ok"):
          if round_result.get("finalize_partial") and round_result.get("partial_text"):
            overall_yielded += len(round_result["partial_text"])
            for part in round_result["partial_text"]:
              yield part
            yield friendly_api_error(round_result["error"])
            overall_yielded += 1
            return
          if round_result.get("error"):
            yield friendly_api_error(round_result["error"])
            overall_yielded += 1
          return

        initial_text = round_result.get("text") or ""
        if initial_text:
          overall_yielded += len(initial_text)
          for char in initial_text:
            yield char

        tool_calls_payload = round_result.get("tool_calls") or []
        if tool_calls_payload:
          assistant_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": []
          }
          for tc in tool_calls_payload:
            assistant_message["tool_calls"].append({
              "id": tc.get("id"),
              "type": "function",
              "function": {
                "name": tc.get("name") or "",
                "arguments": tc.get("arguments") or "",
              }
            })
          messages.append(assistant_message)

          for tc in tool_calls_payload:
            try:
              args = json.loads(tc["arguments"]) if tc.get("arguments") else {}
            except json.JSONDecodeError:
              args = {}
            result = execute_function(tc.get("name") or "", args)
            messages.append({
              "role": "tool",
              "tool_call_id": tc.get("id"),
              "content": json.dumps(result, ensure_ascii=False)
            })

          final_round = run_final_text_round(messages)
          if final_round.get("ok"):
            final_text = final_round.get("text") or ""
            if final_text:
              overall_yielded += len(final_text)
              for char in final_text:
                yield char
          else:
            if final_round.get("finalize_partial") and final_round.get("partial_text"):
              partial = final_round["partial_text"]
              overall_yielded += len(partial)
              for char in partial:
                yield char
              yield friendly_api_error(final_round["error"])
              overall_yielded += 1
            elif final_round.get("error"):
              yield friendly_api_error(final_round["error"])
              overall_yielded += 1

      except Exception as e:
        if overall_yielded == 0:
          yield friendly_api_error(e)
          overall_yielded += 1
        else:
          yield friendly_api_error(e)
          overall_yielded += 1
        return

      if overall_yielded == 0:
        yield "Yapay zekâ boş yanıt verdi. Lütfen mesajınızı yeniden gönderin."

    return StreamingHttpResponse(generate(), content_type='text/plain')

  except json.JSONDecodeError:
    return JsonResponse({"error": "Geçersiz JSON."}, status=400)
  except Exception as e:
    return JsonResponse({"error": str(e)}, status=500)


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