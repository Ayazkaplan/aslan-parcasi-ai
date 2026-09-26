import json
import os
import base64
import io
import re
import time
import logging
import requests
from urllib.parse import quote
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST
from openai import OpenAI
from .forms import CustomUserCreationForm, EmailOrUsernameAuthenticationForm
from .middleware import clear_remember_cookie, set_remember_cookie
from .models import ChatHistory, UserProfile

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

logger = logging.getLogger(__name__)


def get_openai_client():
  api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
  if not api_key or api_key == "gecici_anahtar":
    return None
  return OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=api_key,
  )


def get_user_profile(user):
  profile, _ = UserProfile.objects.get_or_create(user=user)
  return profile


def get_chat_history(user):
  history, _ = ChatHistory.objects.get_or_create(user=user)
  return history


def extract_uploaded_file_text(file_item):
  """Extract useful text from common uploaded formats before sending to AI."""
  if not isinstance(file_item, dict):
    return ""
  encoded = file_item.get("base64") or ""
  if file_item.get("text"):
    return str(file_item.get("text"))[:MAX_EXTRACTED_TEXT]
  if not encoded:
    return ""
  try:
    raw = base64.b64decode(encoded, validate=True)
  except (ValueError, TypeError):
    return ""

  name = str(file_item.get("name") or "").lower()
  content_type = str(file_item.get("type") or "").lower()
  try:
    if content_type.startswith("text/") or re.search(
        r"\.(txt|md|json|csv|py|js|ts|html|css|java|c|cpp|sql|xml|yaml|yml|log)$",
        name,
    ):
      return raw.decode("utf-8", errors="replace")[:MAX_EXTRACTED_TEXT]
    if (name.endswith(".pdf") or content_type == "application/pdf") and PdfReader:
      pages = PdfReader(io.BytesIO(raw)).pages
      return "\n\n".join((page.extract_text() or "") for page in pages)[:MAX_EXTRACTED_TEXT]
    if (name.endswith(".docx") or content_type.endswith("wordprocessingml.document")) and Document:
      document = Document(io.BytesIO(raw))
      return "\n".join(paragraph.text for paragraph in document.paragraphs)[:MAX_EXTRACTED_TEXT]
  except Exception:
    return ""
  return ""


def extract_image_url(message):
  """Accept the different image shapes returned by OpenRouter models."""
  images = getattr(message, "images", None) if message else None
  if images:
    for image in images:
      image_url = getattr(image, "image_url", None)
      if isinstance(image_url, dict):
        image_url = image_url.get("url")
      elif image_url:
        image_url = getattr(image_url, "url", image_url)
      if image_url:
        return image_url

  content = getattr(message, "content", None) if message else None
  if isinstance(content, list):
    for part in content:
      if not isinstance(part, dict):
        continue
      if part.get("type") == "image_url":
        image_url = part.get("image_url")
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
    tz = timezone.get_current_timezone()
    now = timezone.now().astimezone(tz)
    return {
        "time": now.strftime("%H:%M"),
        "date": now.strftime("%d.%m.%Y"),
        "day": now.strftime("%A"),
        "timezone": "Europe/Istanbul (UTC+3)"
    }


def _weather_from_wttr(city):
    """Open-Meteo erişilemezse (veri merkezi IP'lerinde oran sınırı olabiliyor) wttr.in yedeği."""
    try:
        resp = requests.get(
            f"https://wttr.in/{quote(city)}",
            params={"format": "j1"},
            headers={"User-Agent": "curl/8.5", "Accept-Language": "tr-TR,tr"},
            timeout=12,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        cur = (data.get("current_condition") or [{}])[0]
        if not cur:
            return None
        desc = ((cur.get("lang_tr") or [{}])[0].get("value")
                or (cur.get("weatherDesc") or [{}])[0].get("value")
                or "Bilinmeyen")
        area = (data.get("nearest_area") or [{}])[0]
        return {
            "city": ((area.get("areaName") or [{}])[0].get("value") or city),
            "country": ((area.get("country") or [{}])[0].get("value") or ""),
            "temperature": int(cur.get("temp_C", 0)),
            "feels_like": int(cur.get("FeelsLikeC", 0)),
            "humidity": cur.get("humidity"),
            "pressure": cur.get("pressure"),
            "description": desc,
            "wind_speed": cur.get("windspeedKmph"),
            "wind_deg": cur.get("winddirDegree", 0),
        }
    except Exception as e:
        logger.warning("wttr.in yedeği başarısız: %s", e)
        return None


def get_weather(city, country="TR"):
    """Get weather information using Open-Meteo (no API key required), wttr.in fallback."""
    weather_codes = {
        0: "Açık", 1: "Çoğunlukla açık", 2: "Parçalı bulutlu", 3: "Kapalı",
        45: "Sisli", 48: "Sisli", 51: "Hafif çisenti", 53: "Çisenti", 55: "Yoğun çisenti",
        61: "Hafif yağmur", 63: "Yağmur", 65: "Şiddetli yağmur",
        71: "Hafif kar", 73: "Kar", 75: "Yoğun kar",
        80: "Sağanak yağış", 81: "Sağanak yağış", 82: "Şiddetli sağanak",
        95: "Gök gürültülü fırtına", 96: "Gök gürültülü fırtına", 99: "Gök gürültülü fırtına",
    }
    last_error = None
    for attempt in range(2):
        try:
            geo_response = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "tr", "format": "json"},
                timeout=10,
            )
            geo_results = (geo_response.json() or {}).get("results") or []
            if not geo_results:
                fallback = _weather_from_wttr(city)
                if fallback:
                    return fallback
                return {"error": f"City '{city}' not found"}
            place = geo_results[0]
            lat, lon = place["latitude"], place["longitude"]

            weather_response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,apparent_temperature,pressure_msl,wind_speed_10m,wind_direction_10m,weather_code",
                    "timezone": "auto",
                },
                timeout=10,
            )
            if weather_response.status_code != 200:
                last_error = f"Weather API error: {weather_response.status_code}"
                logger.warning("open-meteo forecast HTTP %s (deneme %s)", weather_response.status_code, attempt + 1)
                time.sleep(1)
                continue
            current = (weather_response.json() or {}).get("current") or {}

            return {
                "city": place.get("name", city),
                "country": place.get("country", country),
                "temperature": round(current.get("temperature_2m", 0)),
                "feels_like": round(current.get("apparent_temperature", 0)),
                "humidity": current.get("relative_humidity_2m"),
                "pressure": current.get("pressure_msl"),
                "description": weather_codes.get(current.get("weather_code"), "Bilinmeyen"),
                "wind_speed": current.get("wind_speed_10m"),
                "wind_deg": current.get("wind_direction_10m", 0),
            }
        except requests.Timeout:
            last_error = "Weather service timeout"
        except requests.RequestException as e:
            last_error = f"Weather service error: {str(e)}"
        except Exception as e:
            last_error = f"Weather error: {str(e)}"
        time.sleep(1)
    fallback = _weather_from_wttr(city)
    if fallback:
        return fallback
    logger.error("get_weather tüm kaynaklar başarısız (%s): %s", city, last_error)
    return {"error": last_error or "Weather service unavailable"}


def web_search(query, num_results=5):
    """Anahtarsız web araması: önce Bing, olmazsa DuckDuckGo (html/lite)."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        BeautifulSoup = None

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
    }

    def from_bing():
        response = requests.get(
            "https://www.bing.com/search",
            params={"q": query},
            headers=headers,
            timeout=12,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for item in soup.select("li.b_algo"):
            title_el = item.select_one("h2")
            snippet_el = item.select_one(".b_caption p") or item.select_one("p")
            title = title_el.get_text(" ", strip=True) if title_el else ""
            snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
            text = f"{title} - {snippet}" if title and snippet else (title or snippet)
            if text and text not in results:
                results.append(text)
            if len(results) >= num_results:
                break
        return results

    def from_duckduckgo(base_url):
        response = requests.post(base_url, data={"q": query}, headers=headers, timeout=12)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for selector in (".result__snippet", ".web-result-description", ".result-snippet", "td.result-snippet"):
            for element in soup.select(selector):
                text = element.get_text(" ", strip=True)
                if text and text not in results:
                    results.append(text)
            if results:
                break
        return results[:num_results]

    def from_generic(url, params, title_selectors, snippet_selectors):
        response = requests.get(url, params=params, headers=headers, timeout=12)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for container in soup.select("li, div.result, div.snippet"):
            title = ""
            for sel in title_selectors:
                el = container.select_one(sel)
                if el:
                    title = el.get_text(" ", strip=True)
                    break
            snippet = ""
            for sel in snippet_selectors:
                el = container.select_one(sel)
                if el:
                    snippet = el.get_text(" ", strip=True)
                    break
            text = f"{title} - {snippet}" if title and snippet else (title or snippet)
            if text and len(text) > 40 and text not in results:
                results.append(text)
            if len(results) >= num_results:
                break
        return results[:num_results]

    def from_yahoo():
        return from_generic(
            "https://search.yahoo.com/search",
            {"p": query},
            ("h3.title", ".compTitle a", "a h3"),
            (".compText p", ".compText a p", "p"),
        )

    def from_brave():
        return from_generic(
            "https://search.brave.com/search",
            {"q": query},
            (".snippet .title", "div.title", "h3"),
            (".snippet-content p", ".snippet p", "p"),
        )

    attempts = []
    if BeautifulSoup:
        attempts.append(("bing", from_bing))
        attempts.append(("yahoo", from_yahoo))
        attempts.append(("brave", from_brave))
        attempts.append(("ddg-html", lambda: from_duckduckgo("https://html.duckduckgo.com/html/")))
        attempts.append(("ddg-lite", lambda: from_duckduckgo("https://lite.duckduckgo.com/lite/")))

    for name, attempt in attempts:
        try:
            results = attempt()
            if results:
                logger.info("web_search '%s' -> %s (%s sonuç)", query, name, len(results))
                return {"query": query, "results": results}
            logger.warning("web_search '%s' -> %s boş sonuç", query, name)
        except Exception as e:
            logger.warning("web_search '%s' -> %s hata: %s", query, name, e)
            continue

    logger.error("web_search '%s': tüm motorlar başarısız", query)
    return {"query": query, "results": ["[Sistem] Web arama servislerine şu an ulaşılamadı, sonuç bulunamadı. Kullanıcıya arama yapılamadığını dürüstçe belirt."]}


def profile_payload(user):
  profile = get_user_profile(user)
  return {
      "username": user.username,
      "avatar": profile.avatar,
      "theme": profile.theme,
      "pattern": profile.pattern,
  }


def friendly_api_error(error):
  error_text = str(error).lower()
  if "402" in error_text or "credit" in error_text or "max_tokens" in error_text:
    return "Yapay zekâ servisi için yeterli API kredisi yok veya istek çok uzun. Daha kısa bir mesaj deneyin ya da API kredisi ekleyin."
  if "401" in error_text or "403" in error_text or "unauthorized" in error_text:
    return "Yapay zekâ servisi yetkilendirmeyi reddetti. API anahtarını kontrol edin."
  if "404" in error_text or "not found" in error_text:
    return "Seçili yapay zekâ modeli kullanılamıyor. Sunucu ayarlarından geçerli bir model seçin."
  if "429" in error_text or "rate limit" in error_text:
    return "Yapay zekâ servisi şu anda yoğun. Birkaç saniye sonra tekrar deneyin."
  return "Yapay zekâ yanıtı alınamadı. Lütfen biraz sonra tekrar deneyin."


def build_provider_pool(openrouter_models=None, only_openrouter=False, vision=False):
    """Yedeklemeli sağlayıcı havuzu: hata veren model/sağlayıcı atlanıp sıradakine geçilir.

    OpenRouter en sonda tutulur ki ücretli kota yalnızca diğerleri tükenince harcansın.
    vision=True ise yalnızca görsel anlayabilen modeller kullanılır.
    """
    pool = []
    if not only_openrouter:
        groq_key = (os.environ.get("GROQ_API_KEY") or "").strip()
        if groq_key and not vision:
            pool.append({
                "name": "groq",
                "client": OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key),
                "models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
            })
        mistral_key = (os.environ.get("MISTRAL_API_KEY") or "").strip()
        if mistral_key:
            pool.append({
                "name": "mistral",
                "client": OpenAI(base_url="https://api.mistral.ai/v1", api_key=mistral_key),
                "models": ["pixtral-12b-2409"] if vision else ["open-mistral-nemo", "mistral-small-latest"],
            })
        cohere_key = (os.environ.get("COHERE_API_KEY") or "").strip()
        if cohere_key:
            pool.append({
                "name": "cohere",
                "client": OpenAI(base_url="https://api.cohere.com/compatibility/v1", api_key=cohere_key),
                "models": ["command-a-vision-07-2025"] if vision else ["command-r-plus-08-2024", "command-r-08-2024"],
            })
    openrouter_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if openrouter_key and openrouter_key != "gecici_anahtar":
        if vision:
            models = ["google/gemini-2.0-flash-001", "openai/gpt-4o-mini"]
        else:
            models = list(dict.fromkeys(item for item in (openrouter_models or []) if item))
            models.append("openrouter/auto")
            models = list(dict.fromkeys(models))
        pool.append({
            "name": "openrouter",
            "client": OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key),
            "models": models,
        })
    return pool


def pool_chat_completion(messages, temperature=0.7, max_tokens=4096, stream=False, tools=None, openrouter_models=None, only_openrouter=False, vision=False):
    last_error = None
    for provider in build_provider_pool(openrouter_models, only_openrouter, vision=vision):
        for model in provider["models"]:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            try:
                return provider["client"].chat.completions.create(**kwargs)
            except Exception as e:
                last_error = e
                logger.warning("pool_chat_completion: %s/%s failed: %s", provider.get("name"), model, e)
                continue
    logger.error("pool_chat_completion: all providers failed. last_error=%s", last_error)
    raise last_error or Exception("Tüm sağlayıcılar başarısız oldu")


def transcribe_audio(voice_item):
    """Groq whisper-large-v3 ile ses kaydını metne çevirir. Başarısızsa boş string."""
    if not isinstance(voice_item, dict):
        return ""
    encoded = voice_item.get("base64") or ""
    if not encoded:
        return ""
    groq_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not groq_key:
        return ""
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return ""
    audio_type = str(voice_item.get("type") or "audio/webm").split(";")[0]
    fmt = audio_type.split("/")[-1] or "webm"
    audio_buffer = io.BytesIO(raw)
    audio_buffer.name = f"voice.{fmt}"
    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
        result = client.audio.transcriptions.create(
            file=audio_buffer,
            model="whisper-large-v3",
        )
        return (getattr(result, "text", "") or "").strip()
    except Exception:
        return ""


REASON_STREAM_PREFIX = "\x01R\x01"


def deep_think_stream(messages, deep_think_seconds, temperature=0.7, max_tokens=4096, vision=False):
    """Verilen süre boyunca araçlarla araştırır; ilerlemeyi canlı yayınlar (bağlantı kopmaz),
    sonunda bulguları birleştirip akıcı yanıt üretir.
    """
    deadline = time.time() + max(5, deep_think_seconds)
    max_iterations = 12

    base_messages = list(messages)
    research_messages = list(base_messages)
    research_messages.append({
        "role": "system",
        "content": (
            "Derin düşünme ve araştırma modundasın. Kullanıcının sorusunu yanıtlamak için "
            "gerekiyorsa web_search, get_weather veya get_current_time araçlarını KULLAN. "
            "Her turda kısa bir analiz yap ve gerekiyorsa yeni bir araç çağır. "
            "Yeterli bilgiye ulaşınca araç çağırmayı bırakıp 'HAZIRIM' yaz."
        ),
    })

    notes = []
    yield REASON_STREAM_PREFIX + "Araştırma başlatıldı, soru parçalara ayrılıyor...\n"

    for _ in range(max_iterations):
        if time.time() >= deadline:
            yield REASON_STREAM_PREFIX + "Düşünme bütçesi doldu, bulgular derleniyor...\n"
            break
        try:
            completion = pool_chat_completion(
                research_messages,
                temperature=0.3,
                max_tokens=min(max_tokens, 2048),
                stream=False,
                tools=TOOLS,
                vision=vision,
            )
        except Exception:
            break
        if not completion.choices:
            break
        message = completion.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
        content = (message.content or "").strip()

        if tool_calls:
            research_messages.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                    }
                    for tc in tool_calls
                ],
            })
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                result = execute_function(tc.function.name, args)
                research_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
                note = f"{tc.function.name} → {json.dumps(result, ensure_ascii=False)}"
                notes.append(note)
                yield REASON_STREAM_PREFIX + note[:400] + "\n"
        else:
            if content:
                notes.append(content)
                yield REASON_STREAM_PREFIX + content[:400] + "\n"
            if content and "HAZIRIM" in content.upper():
                break
            research_messages.append({"role": "assistant", "content": content or ""})
            research_messages.append({
                "role": "user",
                "content": "Devam et. Gerekiyorsa araç kullanarak daha fazla araştır, değilse HAZIRIM yaz.",
            })

    research_summary = "\n\n".join(notes) if notes else "(Ek bulgu toplanmadı.)"

    last_user = base_messages[-1] if base_messages else {"role": "user", "content": ""}
    last_content = last_user.get("content")
    if isinstance(last_content, str):
        final_last = {"role": "user", "content": f"{last_content}\n\n[Araştırma Bulguları]\n{research_summary}"}
    elif isinstance(last_content, list):
        final_last = {"role": "user", "content": list(last_content) + [{"type": "text", "text": f"\n\n[Araştırma Bulguları]\n{research_summary}"}]}
    else:
        final_last = {"role": "user", "content": f"[Araştırma Bulguları]\n{research_summary}"}

    final_messages = base_messages[:-1] + [final_last] if base_messages else [final_last]

    answer_instruction = (
        " Araştırma aşaması BİTTİ. Şimdi araç ÇAĞIRMA; yalnızca yukarıdaki [Araştırma Bulguları] bölümünü "
        "kullanarak kullanıcıya doğrudan, akıcı ve nihai yanıtı ver. Bulgu yetersizse dürüstçe belirt."
    )
    if final_messages and final_messages[0].get("role") == "system":
        final_messages[0] = {"role": "system", "content": str(final_messages[0].get("content") or "") + answer_instruction}
    else:
        final_messages.insert(0, {"role": "system", "content": answer_instruction.strip()})

    streamed_any = False
    try:
        stream = pool_chat_completion(
            final_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                streamed_any = True
                yield chunk.choices[0].delta.content
    except Exception:
        streamed_any = False

    if streamed_any:
        return

    stream_messages = list(final_messages)
    for _round in range(2):
        try:
            stream = pool_chat_completion(
                stream_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                tools=TOOLS,
                vision=vision,
            )
        except Exception as e:
            yield friendly_api_error(e)
            return

        tool_calls_buffer = []
        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    streamed_any = True
                    yield delta.content
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if len(tool_calls_buffer) <= tool_call.index:
                            tool_calls_buffer.extend([None] * (tool_call.index + 1 - len(tool_calls_buffer)))
                        if tool_calls_buffer[tool_call.index] is None:
                            tool_calls_buffer[tool_call.index] = {
                                "id": tool_call.id,
                                "name": tool_call.function.name if tool_call.function else "",
                                "arguments": tool_call.function.arguments if tool_call.function else "",
                            }
                        elif tool_call.function and tool_call.function.arguments:
                            tool_calls_buffer[tool_call.index]["arguments"] += tool_call.function.arguments
        except Exception as e:
            yield friendly_api_error(e)
            return

        if not tool_calls_buffer or not any(tc is not None for tc in tool_calls_buffer):
            break

        stream_messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls_buffer if tc
            ],
        })
        for tc in tool_calls_buffer:
            if not tc:
                continue
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            result = execute_function(tc["name"], args)
            stream_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result, ensure_ascii=False),
            })

    if not streamed_any:
        if notes:
            yield "Araştırdım ancak nihai yanıtı üretemedim. Topladığım bulgular:\n\n" + "\n\n".join(notes[:6])
        else:
            yield "Derin düşünme sırasında yanıt üretilemedi. Lütfen tekrar deneyin."


@login_required(login_url="login")
def index(request):
  if not request.user.email:
    return redirect("update_email")
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
  return clear_remember_cookie(JsonResponse({"status": "success"}))


def _pollinations_data_url(prompt_text):
  """Pollinations yedek görseli: önbelleği kır, görseli indir, alt filigran şeridini
  kırp ve data URL olarak döndür. OpenRouter görsel modeli kullanılamadığında devreye girer."""
  import random
  # Önbellek prompt'a göre anahtarlanıyor ve seed yoksayılıyor; bu yüzden prompt'u
  # benzersiz kılmak için anlamsal olmayan bir özel anahtar ekliyoruz ve private istiyoruz.
  nonce = random.randint(100000, 999999)
  varyant = f"{prompt_text} (unique {nonce})"
  url = (
      f"https://image.pollinations.ai/prompt/{quote(varyant)}"
      f"?width=1024&height=1024&nologo=true&private=true&model=flux&enhance=true&seed={nonce}"
  )
  try:
    resp = requests.get(url, timeout=60, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "Referer": "https://aslan-parcasi-ai.onrender.com/",
    })
    resp.raise_for_status()
    raw = resp.content
    if not raw:
      return url
    try:
      from PIL import Image
      img = Image.open(io.BytesIO(raw))
      img = img.convert("RGB")
      w, h = img.size
      # "pollinations.ai" filigranı alt kısımda; alt ~%7'lik şeridi kırp.
      crop_h = max(1, int(h * 0.07))
      if h - crop_h > 100:
        img = img.crop((0, 0, w, h - crop_h))
      buf = io.BytesIO()
      img.save(buf, format="JPEG", quality=92)
      return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
      return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
  except Exception:
    return url


OPENROUTER_IMAGE_MODELS = [
    "google/gemini-2.5-flash-image",
    "google/gemini-3-pro-image-preview",
    "google/gemini-2.5-flash-image-preview",
]


@login_required(login_url="login")
@require_POST
def api_image_generate(request):
  try:
    data = json.loads(request.body or "{}")
    prompt = (data.get("prompt") or "").strip()

    if not prompt:
      return JsonResponse({"error": "Prompt boş olamaz."}, status=400)

    client = get_openai_client()
    if client is not None:
      env_model = (os.environ.get("OPENROUTER_IMAGE_MODEL") or "").strip()
      models = [env_model] if env_model else list(OPENROUTER_IMAGE_MODELS)
      for image_model in models:
        if not image_model:
          continue
        try:
          response = client.chat.completions.create(
              model=image_model,
              messages=[{"role": "user", "content": prompt}],
              modalities=["text", "image"],
          )
          message = response.choices[0].message if response.choices else None
          image_url = extract_image_url(message)
          if image_url:
            return JsonResponse({"status": "success", "image_url": image_url})
        except Exception:
          continue

    # OpenRouter kullanılamadı / görsel üretmedi: filigransız pollinations yedeği.
    return JsonResponse({"status": "success", "image_url": _pollinations_data_url(prompt)})

  except json.JSONDecodeError:
    return JsonResponse({"error": "Geçersiz JSON."}, status=400)
  except Exception as e:
    return JsonResponse({"error": str(e)}, status=500)


@login_required(login_url="login")
def api_debug(request):
  """Canlı sunucuda (Render) neyin çalışıp çalışmadığını görmek için teşhis ucu.
  Yalnızca giriş yapmış kullanıcı; API anahtarı DEĞERLERİ asla döndürülmez."""
  report = {"keys": {}, "pool": [], "checks": {}}
  for key in ("GROQ_API_KEY", "MISTRAL_API_KEY", "COHERE_API_KEY", "OPENROUTER_API_KEY", "DATABASE_URL"):
    val = (os.environ.get(key) or "").strip()
    report["keys"][key] = bool(val) and val != "gecici_anahtar"
  try:
    for provider in build_provider_pool():
      report["pool"].append({"name": provider.get("name"), "models": provider.get("models")})
  except Exception as e:
    report["checks"]["pool_error"] = str(e)
  try:
    import bs4  # noqa: F401
    report["checks"]["bs4"] = True
  except Exception as e:
    report["checks"]["bs4"] = f"MISSING: {e}"
  try:
    w = get_weather("Istanbul", "TR")
    report["checks"]["weather"] = w.get("error") or f"OK {w.get('temperature')}°C {w.get('description')}"
  except Exception as e:
    report["checks"]["weather"] = f"ERROR: {e}"
  try:
    s = web_search("İstanbul hava durumu", 3)
    res = s.get("results") or []
    report["checks"]["web_search"] = s.get("error") or f"OK {len(res)} sonuç"
  except Exception as e:
    report["checks"]["web_search"] = f"ERROR: {e}"
  try:
    completion = pool_chat_completion(
        [{"role": "user", "content": "Tek kelimeyle selam ver."}],
        temperature=0.2, max_tokens=16, stream=False,
    )
    text = completion.choices[0].message.content if completion.choices else ""
    report["checks"]["chat"] = f"OK: {(text or '')[:40]!r}"
  except Exception as e:
    report["checks"]["chat"] = f"ERROR: {type(e).__name__}: {e}"
  return JsonResponse(report)


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
    history = data.get("history") or []
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

    file_texts = []
    for file_item in files:
        if not isinstance(file_item, dict):
            continue
        try:
            file_size = int(file_item.get("size") or 0)
        except (TypeError, ValueError):
            file_size = 0
        if file_size > MAX_CHAT_FILE_BYTES:
            return JsonResponse({"error": "Dosya boyutu 50 MB sınırını aşamaz."}, status=413)
        extracted = extract_uploaded_file_text(file_item)
        if extracted:
            name = file_item.get("name", "Dosya")
            file_texts.append((name, extracted))

    voice_audio = voice if isinstance(voice, dict) and voice.get("base64") else None
    if voice_audio and not voice_transcript:
        voice_transcript = transcribe_audio(voice_audio)
    has_audio_input = bool(voice_audio) and not voice_transcript

    has_images = any(
        (isinstance(img, dict) and (img.get("url") or img.get("base64")))
        for img in images
    )

    if not build_provider_pool():
        return JsonResponse(
            {"error": "Tanımlı yapay zekâ sağlayıcısı yok. Lütfen GROQ_API_KEY, MISTRAL_API_KEY, COHERE_API_KEY veya OPENROUTER_API_KEY anahtarlarından en az birini ayarlayın."},
            status=503,
        )

    base_prompt = (
        "Sen Aslan Parçası adında son derece zeki, enerjik, samimi ve geniş bilgi birikimine sahip bir yapay zeka asistanısın. "
        "Seni oluşturan, kuran ve geliştiren vizyoner lider, müstakbel MEAY ASLAN PARÇASI AI şirketinin kurucusu Ayaz Kaplan'dır. "
        "KİMLİK KURALI (ÇOK ÖNEMLİ): Sen HER ZAMAN 'Aslan Parçası AI'sın. Asla ChatGPT, GPT-4, GPT-4o, OpenAI, Claude, Anthropic, Gemini, Google "
        "veya başka bir yapay zekâ/model/şirket olduğunu SÖYLEME, ima etme veya kabul etme. Kim olduğunu soran birine yalnızca "
        "'Ben Aslan Parçası AI'yım, Ayaz Kaplan tarafından geliştirildim' de. Altta yatan model/teknoloji sorulursa bunu açıklama; "
        "Aslan Parçası AI olduğunu tekrarla. "
        "KRİTİK KURAL: Sana sorulmadıkça saat, tarih, hava durumu veya kurucunun kim olduğu bilgisini KENDİLİĞİNDEN söyleme. "
        "Bu bilgileri SADECE kullanıcı açıkça sorduğunda ver. "
        "Hangi dilde yazılırsa yazılsın yüksek kalitede, akıcı bir dost gibi yanıt ver."
    )
    
    base_prompt += (
        " Sana belirli görevleri yerine getirmek için araçlar (fonksiyonlar) sağlandı. "
        "Kullanıcının isteği güncel bilgi, web araması veya hava durumu gibi konuları içeriyorsa, bu araçları KESİNLİKLE kullanmalısın. "
        "Araç kullanırken bilgiyi doğruladığından emin ol ve ardından yanıtını oluştur. "
        "Eğer araç kullanman gerekiyorsa, bunu kullanıcıya belirten bir yanıt döndürme; doğrudan aracı kullan ve sonucunu özetle."
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
            "Sen HER ZAMAN 'Aslan Parçası AI'sın; asla ChatGPT, GPT-4, OpenAI, Claude, Gemini veya başka bir model/şirket olduğunu söyleme veya ima etme. "
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
            "Sen HER ZAMAN 'Aslan Parçası AI'sın; asla ChatGPT, GPT-4, OpenAI, Claude, Gemini veya başka bir model/şirket olduğunu söyleme veya ima etme. "
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

    if deep_think:
        minutes = deep_think_seconds // 60
        seconds = deep_think_seconds % 60
        budget_label = f"{minutes} dakika {seconds} saniye" if minutes else f"{seconds} saniye"
        system_instruction += (
            f" Kapsamlı araştırma modu açık; kullanıcı sana {budget_label} düşünme bütçesi verdi. "
            "Bu süreyi boş beklemek için değil, soruyu parçalara ayırmak, varsayımları kontrol etmek, "
            "kanıt ve karşı örnekleri değerlendirmek ve sonunda net bir araştırma özeti üretmek için kullan. "
            "Canlı internet erişimin yoksa bunu dürüstçe belirt; kaynak uydurma. "
            "Yanıt vermeden önce kısa bir araştırma planı ve bulgularını zihinsel olarak kontrol et."
        )
        max_tokens = min(max_tokens * 2, 8192)

    messages = [{"role": "system", "content": system_instruction}]

    user_content = []
    if full_message:
        user_content.append({"type": "text", "text": full_message})

    if voice_transcript:
        label = "[Kullanıcı yazı yerine SES KAYDI gönderdi; aşağıdaki metin bu ses kaydının yazıya dökülmüş halidir, kullanıcının kendi yazdığı bir mesaj DEĞİLDİR]\n"
        user_content.append({"type": "text", "text": f"{label}{voice_transcript}"})

    for name, text in file_texts:
        user_content.append({"type": "text", "text": f"[{name} içeriği]\n{text}"})

    for img in images:
        if not isinstance(img, dict):
            continue
        if img.get("url"):
            user_content.append({"type": "image_url", "image_url": {"url": img["url"]}})
        elif img.get("base64"):
            image_type = img.get("type") or "image/jpeg"
            user_content.append({"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{img['base64']}"}})

    for h in history:
        if not isinstance(h, dict):
            continue
        role = "user" if h.get("sender") == "user" else "assistant"
        content = h.get("text") or ""
        if h.get("files") and isinstance(h["files"], list):
            file_parts = []
            for file_item in h["files"]:
                extracted = extract_uploaded_file_text(file_item) if isinstance(file_item, dict) else ""
                if extracted:
                    file_parts.append({"type": "text", "text": f"[{file_item.get('name', 'Dosya')} içeriği]\n{extracted}"})
                elif isinstance(file_item, dict) and file_item.get("text"):
                    file_parts.append({"type": "text", "text": f"[{file_item.get('name', 'Dosya')} içeriği]\n{file_item['text']}"})
            if content:
                file_parts.insert(0, {"type": "text", "text": content})
            messages.append({"role": role, "content": file_parts if file_parts else content})
        elif content:
            messages.append({"role": role, "content": content})

    if user_content:
        messages.append({"role": "user", "content": user_content})
    elif has_audio_input:
        audio_type = str(voice_audio.get("type") or "audio/webm").split(";")[0]
        audio_format = audio_type.split("/")[-1] or "webm"
        messages.append({"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": voice_audio["base64"], "format": audio_format}}
        ]})
    else:
        messages.append({"role": "user", "content": full_message})

    try:
        request_model = model
        
        if deep_think:
            def generate_deep():
                try:
                    for piece in deep_think_stream(
                        messages,
                        deep_think_seconds,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        vision=has_images,
                    ):
                        yield piece
                except Exception as e:
                    yield friendly_api_error(e)
            return StreamingHttpResponse(generate_deep(), content_type='text/plain')

        completion = pool_chat_completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            tools=TOOLS,
            openrouter_models=[request_model],
            only_openrouter=has_audio_input,
            vision=has_images,
        )
    except Exception as api_error:
      return JsonResponse({"error": friendly_api_error(api_error)}, status=503)

    def generate():
      yielded_count = 0
      try:
        current_completion = completion
        for _round in range(4):
            tool_calls_buffer = []
            for chunk in current_completion:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    
                    if delta.content:
                        yielded_count += 1
                        yield delta.content
                    
                    if delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            if len(tool_calls_buffer) <= tool_call.index:
                                tool_calls_buffer.extend([None] * (tool_call.index + 1 - len(tool_calls_buffer)))
                            
                            if tool_calls_buffer[tool_call.index] is None:
                                tool_calls_buffer[tool_call.index] = {
                                    "id": tool_call.id,
                                    "name": tool_call.function.name if tool_call.function else "",
                                    "arguments": tool_call.function.arguments if tool_call.function else ""
                                }
                            else:
                                if tool_call.function and tool_call.function.arguments:
                                    tool_calls_buffer[tool_call.index]["arguments"] += tool_call.function.arguments
            
            if not tool_calls_buffer or not any(tc is not None for tc in tool_calls_buffer):
                break
            
            assistant_message = {
                "role": "assistant",
                "content": "",
                "tool_calls": []
            }
            
            for tc in tool_calls_buffer:
                if tc:
                    assistant_message["tool_calls"].append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"]
                        }
                    })
            
            messages.append(assistant_message)
            
            for tc in tool_calls_buffer:
                if tc:
                    try:
                        args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    
                    result = execute_function(tc["name"], args)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, ensure_ascii=False)
                    })
            
            try:
                current_completion = pool_chat_completion(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    tools=TOOLS,
                    openrouter_models=[request_model],
                    only_openrouter=has_audio_input,
                    vision=has_images,
                )
            except Exception as e:
                yield friendly_api_error(e)
                return
        
        if yielded_count == 0:
            try:
                messages.append({"role": "user", "content": "Araç çağırma; elindeki bilgilerle doğrudan yanıt ver."})
                last_stream = pool_chat_completion(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    openrouter_models=[request_model],
                    only_openrouter=has_audio_input,
                    vision=has_images,
                )
                for chunk in last_stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yielded_count += 1
                        yield chunk.choices[0].delta.content
            except Exception:
                pass
        
        if yielded_count == 0:
            yield "Yapay zekâ boş yanıt verdi veya araçlar kullanılamadı. Lütfen mesajınızı yeniden gönderin."
      except Exception as e:
        yield friendly_api_error(e)

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
      response = redirect("index")
      return set_remember_cookie(response, request.user)
  else:
    form = EmailOrUsernameAuthenticationForm()
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
      response = redirect("index")
      return set_remember_cookie(response, user)
  else:
    form = CustomUserCreationForm()
  return render(request, "dashboard/register.html", {"form": form})


def logout_view(request):
  logout(request)
  response = redirect("login")
  return clear_remember_cookie(response)