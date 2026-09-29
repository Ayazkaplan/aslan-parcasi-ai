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
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_FAST_MODEL = "gemini-2.5-flash-lite"
GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"
KNOWLEDGE_CUTOFF = "29 Eylül 2026"
REASONING_MARKER = "\x00R\x00"
_image_request_lock = threading.Lock()
_last_image_request_at = {}


def get_gemini_client():
  api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
  if not api_key:
    return None
  return OpenAI(
      base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
      api_key=api_key,
  )


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


def _parse_ddg_results(html_text, limit):
    """Parse DuckDuckGo result pages without depending on BeautifulSoup."""
    results = []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_text, "html.parser")
        for result in soup.select(".result")[:limit]:
            title = result.select_one(".result__a")
            snippet = result.select_one(".result__snippet")
            if title or snippet:
                results.append({
                    "title": title.get_text(" ", strip=True) if title else "",
                    "url": title.get("href", "") if title else "",
                    "snippet": snippet.get_text(" ", strip=True) if snippet else "",
                })
        return results
    except ImportError:
        pass
    blocks = re.findall(
        r'<a[^>]+class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        html_text,
        re.S,
    )
    snippets = re.findall(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', html_text, re.S)
    for index, (href, title_html) in enumerate(blocks[:limit]):
        snippet_html = snippets[index] if index < len(snippets) else ""
        results.append({
            "title": re.sub(r"<[^>]+>", "", title_html).strip(),
            "url": href.strip(),
            "snippet": re.sub(r"<[^>]+>", "", snippet_html).strip(),
        })
    return results


def web_search(query, num_results=5):
    """Search current web results using DuckDuckGo (no API key needed)."""
    limit = max(1, min(int(num_results), 10))
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    endpoints = (
        ("post", "https://html.duckduckgo.com/html/"),
        ("post", "https://lite.duckduckgo.com/lite/"),
    )
    last_error = None
    for method, url in endpoints:
        try:
            if method == "post":
                response = requests.post(url, data={"q": query}, headers=headers, timeout=8)
            else:
                response = requests.get(url, params={"q": query}, headers=headers, timeout=8)
            response.raise_for_status()
            results = _parse_ddg_results(response.text, limit)
            if results:
                return {"query": query, "results": results}
            last_error = "no results parsed"
        except Exception as error:
            last_error = str(error)
    return {"error": f"Web search failed: {last_error}", "results": [], "query": query}


def should_fetch_live_context(text):
  """Identify requests where a stale model answer would be misleading."""
  normalized = (text or "").lower()
  hints = (
      "internetten", "güncel", "bugün", "şu an", "şuan", "son dakika",
      "haber", "maç", "skor", "sonuç", "spor", "fiyat", "kur", "döviz",
      "kim kazandı", "ne zaman", "araştır", "web'de", "webte",
  )
  return any(hint in normalized for hint in hints)


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


def generate_image_with_gemini(prompt, image_model, api_key):
  """Generate an image through the generateContent endpoint."""
  response = requests.post(
      f"https://generativelanguage.googleapis.com/v1beta/models/{image_model}:generateContent",
      headers={
          "Content-Type": "application/json",
      },
      params={"key": api_key},
      json={
          "contents": [{
              "parts": [{
                  "text": (
                      "Create exactly one image that follows the user's request literally. "
                      "The named subject, object, place, count, action and composition are "
                      "mandatory. If the user asks for a realistic image, make it "
                      "photorealistic. Do not add unrelated objects. User request: "
                      + prompt
                  ),
              }],
          }],
          "generationConfig": {"responseModalities": ["IMAGE"]},
      },
      timeout=180,
  )
  response.raise_for_status()
  payload = response.json()
  for candidate in payload.get("candidates") or []:
    for part in (candidate.get("content") or {}).get("parts") or []:
      inline_data = part.get("inlineData") or part.get("inline_data")
      if inline_data and inline_data.get("data"):
        media_type = inline_data.get("mimeType") or inline_data.get("mime_type") or "image/png"
        return f"data:{media_type};base64,{inline_data['data']}"
  raise ValueError("Görsel servisi yanıtında görsel verisi bulunamadı.")


def generate_image_with_retry(prompt, image_model, api_key):
  last_error = None
  for attempt in range(3):
    try:
      return generate_image_with_gemini(prompt, image_model, api_key)
    except Exception as error:
      last_error = error
      if attempt < 2:
        time.sleep(2 * (attempt + 1))
  raise last_error


def transcribe_audio(encoded, mime_type, api_key):
  """Server-side speech-to-text so voice notes work on every browser."""
  response = requests.post(
      f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
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
  response.raise_for_status()
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
    kwargs = {
        "model": model or GEMINI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if timeout is not None:
        kwargs["timeout"] = max(1, timeout)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    last_error = None
    for attempt in range(3):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as error:
            last_error = error
            error_text = str(error).lower()
            transient = (
                "429" in error_text
                or "rate limit" in error_text
                or "503" in error_text
                or "unavailable" in error_text
                or "overloaded" in error_text
            )
            if not transient or attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise last_error


def depth_instruction(seconds):
    if seconds <= 60:
        return (
            "Kısa düşünme bütçesi kullanıldı: net, öz ama gerekçeli bir yanıt ver; "
            "en fazla 3 madde."
        )
    if seconds <= 300:
        return (
            "Orta düzey düşünme bütçesi kullanıldı: başlıklarla yapılandırılmış, örnekli "
            "ve adım adım açıklanan bir yanıt ver."
        )
    if seconds <= 900:
        return (
            "Derin düşünme bütçesi kullanıldı: bölümler halinde ayrıntılı analiz yap, "
            "karşıt görüşleri ve riskleri değerlendir, adım adım akıl yürüt, sonunda net "
            "bir sonuç bölümü ver."
        )
    return (
        "Uzman düzey düşünme bütçesi kullanıldı: kapsamlı bir rapor yaz: yönetici özeti, "
        "yöntem, ayrıntılı bölümler, karşıt görüşler, riskler, kaynak değerlendirmesi ve "
        "sonuç önerileri. Bulabildiğin her ayrıntıyı işle."
    )


def deep_think_events(client, messages, seconds, temperature, max_tokens):
  """Research for the full selected budget, then stream the final answer.

  Yields ("reason", line) events for the thinking box and ("answer", text)
  chunks for the reply. The loop keeps working until the deadline minus a
  reserve for the final answer, so 30 s and 30 min budgets differ in depth.
  """
  deadline = time.monotonic() + max(30, min(1800, int(seconds)))
  reserve = max(20, min(90, int(seconds * 0.25)))
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
  max_rounds = max(2, min(24, int(seconds) // 45))
  yield ("reason", f"Derin düşünme başladı · bütçe {int(seconds)} sn")
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
          tools=TOOLS,
          timeout=max(5, min(60, int(remaining - reserve))),
      )
    except Exception as error:
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
          "noktaları belirle; gerekirse araç çağır, değilse bulgularını maddele."
      )
      research.append({"role": "user", "content": gap_note})
      yield ("reason", "Bulgular gözden geçiriliyor, eksikler aranıyor...")
      rounds += 1
      time.sleep(min(6, max(0, deadline - time.monotonic() - reserve)))
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
      result = execute_function(call.function.name, arguments)
      research.append({
          "role": "tool",
          "tool_call_id": call.id,
          "content": json.dumps(result, ensure_ascii=False),
      })
      if call.function.name == "web_search":
        for item in (result.get("results") or [])[:5]:
          if isinstance(item, dict) and item.get("title"):
            yield ("reason", f"SRC:🌐 Kaynak: {item['title']} — {item.get('url', '')}")
      elif call.function.name == "get_weather":
        if result.get("temperature") is not None:
          yield ("reason", f"SRC:🌤 Hava verisi: {result.get('city')} {result.get('temperature')}°C")
      elif call.function.name == "get_current_time":
        yield ("reason", f"SRC:🕒 Saat verisi: {result.get('time')} {result.get('date')}")
    rounds += 1

  while time.monotonic() < deadline - reserve:
    yield ("reason", "Derin analiz sürüyor, bulgular olgunlaştırılıyor...")
    time.sleep(min(5, max(1, deadline - reserve - time.monotonic())))

  final_messages = [
      *research,
      {
          "role": "system",
          "content": (
              "Yanıtını araştırma bulgularına dayandır. Araç hata verdiyse veri uydurma. "
              + depth_instruction(int(seconds))
          ),
      },
  ]
  yield ("reason", "Yanıt yazılıyor...")
  final_completion = safe_model_call(
      client,
      final_messages,
      GEMINI_MODEL,
      temperature=temperature,
      max_tokens=max_tokens,
      stream=True,
      timeout=max(60, min(300, int(seconds * 0.5) + 60)),
  )
  for chunk in final_completion:
    if chunk.choices and chunk.choices[0].delta.content:
      yield ("answer", chunk.choices[0].delta.content)


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
    
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
      return JsonResponse(
          {"error": "Aslan Parçası'nın beyin bağlantısı kurulmamış. Sunucu anahtarını kontrol edin."},
          status=503,
      )

    now = time.monotonic()
    with _image_request_lock:
      previous_request = _last_image_request_at.get(request.user.pk, 0)
      if now - previous_request < IMAGE_REQUEST_MIN_INTERVAL:
        return JsonResponse(
            {"error": "Aslan Parçası görsel için hala çalışıyor; birkaç saniye sonra tekrar dene."},
            status=429,
        )
      _last_image_request_at[request.user.pk] = now

    try:
      image_url = generate_image_with_retry(prompt, GEMINI_IMAGE_MODEL, api_key)
      return JsonResponse({"status": "success", "image_url": image_url})
    except Exception as img_error:
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
    deep_think = data.get("deep_think", False)
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

    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    client = get_gemini_client()
    if client is None:
        return JsonResponse(
            {"error": "Aslan Parçası'nın beyin bağlantısı kurulmamış. Sunucu anahtarını kontrol edin."},
            status=503,
        )

    if voice_encoded and not voice_transcript:
        try:
            voice_transcript = transcribe_audio(voice_encoded, voice_mime, api_key)
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
        f"Bilgi birikimin {KNOWLEDGE_CUTOFF} tarihine kadar günceldir; bu tarihten sonrası için araçlarını kullanırsın. "
        "KİMLİK KURALI: 'Kurucun kim', 'seni kim yaptı', 'kimsin', 'hangi modelsin' gibi sorularda ASLA Google, Gemini "
        "veya başka bir şirket/model adı söyleme; her zaman Aslan Parçası olduğunu ve kurucunun Ayaz Kaplan olduğunu söyle. "
        "KRİTİK KURAL: Sana sorulmadıkça saat, tarih, hava durumu veya kurucunun kim olduğu bilgisini KENDİLİĞİNDEN söyleme. "
        "Bu bilgileri SADECE kullanıcı açıkça sorduğunda ver. "
        "Sana verilen canlı veri (saat, hava durumu, web sonuçları) sistem mesajında geldiyse onu doğrudan kullan. "
        "Araç hata verirse veya sonuç bulamazsa bunu açıkça belirt ve veri uydurma. "
        "Hangi dilde yazılırsa yazılsın yüksek kalitede, akıcı bir dost gibi yanıt ver."
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
            f"Bugünün tarihi: {today_tr}. Bilgi birikimin {KNOWLEDGE_CUTOFF} tarihine kadar günceldir. "
            "KİMLİK KURALI: ASLA Google/Gemini tarafından eğitildiğini söyleme; sen Aslan Parçası'sın, kurucun Ayaz Kaplan. "
            "Sana sorulmadıkça saat, tarih, hava durumu veya kurucunun kimliği hakkında bilgi verme."
        )
        model = GEMINI_MODEL
        temperature = 0.3
        max_tokens = 4096
        history_limit = 16

    elif mode == "fast":
        system_instruction = (
            "Sen Aslan Parçası AI'nın Hızlı Analiz modundasın. "
            "Işık hızında, çok kısa ve öz cevaplar ver. "
            "Gereksiz detaylardan kaçın, doğrudan noktaya odaklan. "
            "Normal moddan belirgin daha hızlı ve kısa yanıtlar üret. "
            "Karmaşık konuları basitleştir, hızlı özetler ve hızlı kararlar ver. "
            f"Bugünün tarihi: {today_tr}. Bilgi birikimin {KNOWLEDGE_CUTOFF} tarihine kadar günceldir. "
            "KİMLİK KURALI: ASLA Google/Gemini tarafından eğitildiğini söyleme; sen Aslan Parçası'sın, kurucun Ayaz Kaplan. "
            "Sana sorulmadıkça saat, tarih, hava durumu veya kurucunun kimliği hakkında bilgi verme."
        )
        model = GEMINI_FAST_MODEL
        temperature = 0.2
        max_tokens = 512
        history_limit = 8

    else:
        system_instruction = base_prompt
        model = GEMINI_MODEL
        temperature = 0.6
        max_tokens = 2048
        history_limit = 16

    messages = [{"role": "system", "content": system_instruction}]

    question_text = user_message or voice_transcript
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
    if WEATHER_INTENT_RE.search(question_text or ""):
      city = detect_weather_city(question_text)
      if city:
        weather_data = get_weather(city)
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
          messages.insert(1, {
              "role": "system",
              "content": (
                  "Hava durumu servisi hata verdi: "
                  + json.dumps(weather_data, ensure_ascii=False)
                  + ". Veri uydurma; servise şu an ulaşılamadığını söyle."
              ),
          })
      else:
        def ask_city_stream():
          yield (
              "Hangi şehir için hava durumu öğrenmek istiyorsun? 🌤 Şehri yazman yeterli, "
              "anlık sıcaklığı derece derece hemen getireyim."
          )
        return StreamingHttpResponse(ask_city_stream(), content_type="text/plain")

    if should_fetch_live_context(question_text) and not deep_think:
      live_context = web_search(question_text, 3 if mode == "fast" else 4)
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

    def generate():
      answered = False
      errored = False
      try:
        if deep_think:
          for kind, text in deep_think_events(
              client,
              messages,
              deep_think_seconds,
              temperature=temperature,
              max_tokens=max_tokens,
          ):
            if kind == "reason":
              yield REASONING_MARKER + text + "\n"
            else:
              answered = True
              yield text
        else:
          completion = safe_model_call(
              client,
              messages,
              model,
              temperature=temperature,
              max_tokens=max_tokens,
              stream=True,
          )
          for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
              answered = True
              yield chunk.choices[0].delta.content
      except Exception as error:
        errored = True
        yield friendly_api_error(error)

      if not answered and not errored:
        yield "Aslan Parçası yanıt üretemedi. Lütfen mesajını yeniden gönderin."

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
