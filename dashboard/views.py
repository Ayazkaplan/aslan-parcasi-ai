import json
import os
import base64
import io
import re
import time
import requests
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST
from openai import OpenAI
from .forms import CustomUserCreationForm
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


def get_weather(city, country="TR"):
    """Get weather information using OpenWeatherMap API"""
    api_key = os.environ.get("OPENWEATHER_API_KEY", "").strip()
    if not api_key:
        return {"error": "Weather API key not configured. Please set OPENWEATHER_API_KEY environment variable."}
    
    try:
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city},{country}&limit=1&appid={api_key}"
        geo_response = requests.get(geo_url, timeout=10)
        geo_data = geo_response.json()
        
        if not geo_data:
            return {"error": f"City '{city}' not found"}
        
        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]
        city_name = geo_data[0].get("name", city)
        country_name = geo_data[0].get("country", country)
        
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=tr"
        weather_response = requests.get(weather_url, timeout=10)
        weather_data = weather_response.json()
        
        if weather_response.status_code != 200:
            return {"error": f"Weather API error: {weather_data.get('message', 'Unknown error')}"}
        
        return {
            "city": city_name,
            "country": country_name,
            "temperature": round(weather_data["main"]["temp"]),
            "feels_like": round(weather_data["main"]["feels_like"]),
            "humidity": weather_data["main"]["humidity"],
            "pressure": weather_data["main"]["pressure"],
            "description": weather_data["weather"][0]["description"].capitalize(),
            "wind_speed": weather_data["wind"]["speed"],
            "wind_deg": weather_data["wind"].get("deg", 0),
            "icon": weather_data["weather"][0]["icon"]
        }
    except requests.Timeout:
        return {"error": "Weather service timeout"}
    except requests.RequestException as e:
        return {"error": f"Weather service error: {str(e)}"}
    except Exception as e:
        return {"error": f"Weather error: {str(e)}"}


def web_search(query, num_results=5):
    """Search the web using DuckDuckGo HTML scraping (no API key needed)"""
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
        for result in soup.select('.result__snippet')[:num_results]:
            text = result.get_text(strip=True)
            if text:
                results.append(text)
        
        if not results:
            for result in soup.select('.web-result-description')[:num_results]:
                text = result.get_text(strip=True)
                if text:
                    results.append(text)
        
        return {
            "query": query,
            "results": results[:num_results] if results else ["Arama sonucu bulunamadı."]
        }
    except Exception as e:
        return {"error": f"Web search failed: {str(e)}", "results": []}


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


def safe_model_call(client, messages, model, temperature=0.7, max_tokens=4096, stream=False, deep_think=False, tools=None):
    configured_fallback = os.environ.get("OPENROUTER_FALLBACK_MODEL", "").strip()
    models = [model, configured_fallback, "openrouter/auto"]
    models = list(dict.fromkeys(item for item in models if item))
    
    last_error = None
    
    for attempt_model in models:
        try:
            kwargs = {
                "model": attempt_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            completion = client.chat.completions.create(**kwargs)
            return completion
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            if "429" in error_str or "rate limit" in error_str:
                for retry in range(2):
                    time.sleep(1 + retry)
                    try:
                        completion = client.chat.completions.create(**kwargs)
                        return completion
                    except Exception as retry_e:
                        last_error = retry_e
                        continue
            continue
    
    raise last_error or Exception("Tüm modeller başarısız oldu")


def deep_think_call(client, messages, model, deep_think_seconds, base_prompt, temperature=0.7, max_tokens=4096):
    thinking_prompt = (
        "Sen derin düşünme modundasın. Kullanıcının sorusunu/isteğini dikkatlice analiz et. "
        "Soruyu parçalara ayır, varsayımları kontrol et, kanıt ve karşı örnekleri değerlendir. "
        "Düşünce sürecini adım adım yaz. SADECE düşünce sürecini yaz, final cevabı verme. "
        "Türkçe düşün ve yaz. Cevabın sonunda kullanıcının orijinal isteğine atıfta bulunarak düşünme sürecini tamamla."
    )
    
    # Derin düşünme için sistem mesajını en başa ekleyelim
    # Kullanıcının mevcut mesajlarını da almalı
    # Eğer messages listesi multimodal içerik içeriyorsa bunu da uygun şekilde aktarmalıyız.
    
    # Kullanıcının son isteği
    user_request_message = messages[-1] if messages else {"role": "user", "content": ""}

    # base_prompt'u kontrol ederken, eğer messages listesindeki ilk mesajın içeriği ile eşleşiyorsa onu filtrele.
    # Ancak deep_think_call'a gelen messages listesi zaten api_chat'taki işlemden geçmiş olmalı, 
    # yani base_prompt zaten ilk sistem mesajı olarak eklenmiş olmalı.
    # Bu nedenle, thinking_messages oluşturulurken mevcut sistem mesajını (base_prompt'u) korumak yerine,
    # düşünme prompt'unu en başa koyup, orijinal base_prompt'u temizleyip
    # kullanıcının önceki mesajlarını ve şimdiki isteğini eklemeliyiz.
    thinking_messages = [
        {"role": "system", "content": thinking_prompt}, # Derin düşünme sistem prompt'u
    ] + [msg for msg in messages if msg.get('role') != 'system'] + [ # Orijinal sistem mesajını filtrele
        {"role": user_request_message['role'], "content": user_request_message['content']} # Kullanıcının mevcut isteği
    ]
    
    # Ancak, yukarıdaki filtreleme, eğer geçmiş mesajlar arasında başka sistem mesajları varsa onları da kaldırabilir.
    # En güvenli yaklaşım, sadece ana sistem mesajını ele almaktır.
    # Orijinal base_prompt'un ilk eleman olarak geldiğini varsayarak, onu filtreleyip yerine thinking_prompt'u koyarız.
    # Eğer orijinal `messages` listesinin ilk elemanı bir sistem mesajıysa ve `base_prompt`'a eşitse,
    # onu `thinking_prompt` ile değiştir. Aksi takdirde, `thinking_prompt`'u en başa ekle.

    # Daha basit bir yaklaşımla, sadece deep_think_call'ın kendi düşünme prompt'unu ekleyip,
    # diğer tüm mesajları korumak, `base_prompt`'u mesajlar listesinden filtreleme ihtiyacını ortadan kaldırır.
    # `deep_think_call` içine girmeden önce `messages` listesine `base_prompt` zaten eklenmişti.
    # `thinking_messages` oluştururken `base_prompt`'u bilerek filtrelemeyeceğiz,
    # sadece `thinking_prompt`'u ekleyeceğiz ve AI'ın hem `base_prompt` hem de `thinking_prompt` ile düşünmesini sağlayacağız.
    # Veya daha iyi bir yaklaşım: `thinking_prompt`'u ana sistem prompt'unun yerine koyup, sonra AI'ın düşünme süreci bittikten sonra
    # orijinal `base_prompt`'u tekrar yerine koymak. Mevcut kod bu ikinci yaklaşıma daha yakın.

    # Düzeltilmiş düşünce mesajları listesi:
    # Düşünme modu için ana sistem mesajı (thinking_prompt)
    # Mevcut mesaj geçmişi (ilk sistem mesajı hariç, çünkü onu değiştireceğiz)
    # Kullanıcının son isteği
    
    # Eğer messages listesi, `base_prompt` ile başlayan bir sistem mesajı içeriyorsa,
    # onu `thinking_prompt` ile değiştirerek başlayın.
    temp_messages = []
    if messages and messages[0].get('role') == 'system' and messages[0].get('content') == base_prompt:
        temp_messages.append({"role": "system", "content": thinking_prompt})
        temp_messages.extend(messages[1:]) # Geri kalan mesajları ekle
    else: # Eğer ilk mesaj system base_prompt değilse, sadece thinking_prompt'u ekle
        temp_messages.append({"role": "system", "content": thinking_prompt})
        temp_messages.extend(messages) # Tüm orijinal mesajları ekle

    thinking_messages = temp_messages
    
    try:
        thinking_completion = client.chat.completions.create(
            model=model,
            messages=thinking_messages,
            temperature=0.3, # Daha deterministik bir düşünme süreci için düşük sıcaklık
            max_tokens=min(max_tokens, 4096),
            stream=False
        )
        reasoning = thinking_completion.choices[0].message.content if thinking_completion.choices else "Düşünme süreci boş."
    except Exception as e:
        reasoning = f"Düşünme sürecinde hata: {str(e)}"
    
    # Orijinal sistem mesajını geri ekle ve düşünme sürecini son kullanıcı mesajına dahil et
    final_messages = [
        messages[0], # Orijinal sistem mesajı
    ] + [msg for msg in messages[1:-1]] + [ # Geçmiş mesajlar (son kullanıcı mesajı hariç)
        {"role": "user", "content": f"[Düşünme Süreci]\n{reasoning}\n\n[Orijinal İstek]\n" + (user_request_message.get('content') if isinstance(user_request_message.get('content'), str) else json.dumps(user_request_message.get('content'), ensure_ascii=False))}
    ]
    
    return client.chat.completions.create(
        model=model,
        messages=final_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        tools=TOOLS, # Derin düşünme modunda da araçları kullanabilmeli
        tool_choice="auto",
    )


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
  return JsonResponse({"status": "success"})


@login_required(login_url="login")
@require_POST  
def api_image_generate(request):
  try:
    data = json.loads(request.body or "{}")
    prompt = (data.get("prompt") or "").strip()
    
    if not prompt:
      return JsonResponse({"error": "Prompt boş olamaz."}, status=400)
    
    client = get_openai_client()
    if client is None:
      return JsonResponse(
          {"error": "OPENROUTER_API_KEY tanımlı değil. Lütfen API anahtarını ayarlayın."},
          status=503,
      )
    
    image_model = os.environ.get(
        "OPENROUTER_IMAGE_MODEL",
        "google/gemini-pro-vision",
    ).strip()
    try:
      response = client.chat.completions.create(
          model=image_model,
          messages=[{"role": "user", "content": prompt}],
          modalities=["text", "image"], # Modalities parametresi, modelin görsel oluşturabilmesi için gerekli olabilir.
      )
      message = response.choices[0].message if response.choices else None
      image_url = extract_image_url(message)
      if image_url:
        return JsonResponse({"status": "success", "image_url": image_url})
      return JsonResponse({"error": "Görsel modeli yanıtında görsel bulunamadı."}, status=502)
    except Exception as img_error:
      # API hataları friendly_api_error fonksiyonu ile düzgün bir şekilde işleniyor.
      return JsonResponse({"error": friendly_api_error(img_error)}, status=503)
      
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
        extracted = extract_uploaded_file_text(file_item)
        if extracted:
            extracted_files.append(f"\n\n[{file_item.get('name', 'Dosya')} içeriği]\n{extracted}")

    # Extracted files are now handled as separate content parts, not appended to full_message
    # if extracted_files:
    #     full_message += "".join(extracted_files)
    # if file_names and not full_message:
    #     full_message = "Dosya gönderildi: " + ", ".join(file_names)

    client = get_openai_client()
    if client is None:
        return JsonResponse(
            {"error": "OPENROUTER_API_KEY tanımlı değil. Lütfen API anahtarını ayarlayın."},
            status=503,
        )

    base_prompt = (
        "Sen Aslan Parçası adında son derece zeki, enerjik, samimi ve geniş bilgi birikimine sahip bir yapay zeka asistanısın. "
        "Seni oluşturan, kuran ve geliştiren vizyoner lider, müstakbel MEAY ASLAN PARÇASI AI şirketinin kurucusu Ayaz Kaplan'dır. "
        "KRİTİK KURAL: Sana sorulmadıkça saat, tarih, hava durumu veya kurucunun kim olduğu bilgisini KENDİLİĞİNDEN söyleme. "
        "Bu bilgileri SADECE kullanıcı açıkça sorduğunda ver. "
        "Hangi dilde yazılırsa yazılsın yüksek kalitede, akıcı bir dost gibi yanıt ver."
    )
    
    # AI'ın araçları kullanma konusunda daha bilinçli olması için prompt'u güncelleyelim.
    # Kullanıcıdan gelen bir soruya doğrudan cevap vermek yerine, uygun araçları kullanarak doğruluk ve güncellik sağlamalıdır.
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
        if full_message:
            user_content.append({"type": "text", "text": f"[Ses kaydı metni]\n{voice_transcript}"})
        else:
            user_content.append({"type": "text", "text": voice_transcript})

    for img in images:
        if img.get("url"):
            user_content.append({"type": "image_url", "image_url": {"url": img["url"]}})
        elif img.get("base64"):
            image_type = img.get("type") or "image/jpeg"
            user_content.append({"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{img['base64']}"}})

    for file_item in files:
        if not isinstance(file_item, dict):
            continue
        if file_item.get("base64"):
            file_type = file_item.get("type") or "application/octet-stream"
            user_content.append({
                "type": "file",
                "file": {
                    "filename": file_item.get("name", "dosya"),
                    "file_data": f"data:{file_type};base64,{file_item['base64']}"
                }
            })
        elif file_item.get("text"):
            user_content.append({"type": "text", "text": f"[{file_item.get('name', 'Dosya')} içeriği]\n{file_item['text']}"})

    for h in history:
        if not isinstance(h, dict):
            continue
        role = "user" if h.get("sender") == "user" else "assistant"
        content = h.get("text") or ""
        # Geçmiş mesajlarda ekli dosyaları da ekle
        if h.get("files") and isinstance(h["files"], list):
            file_parts = []
            for file_item in h["files"]:
                if file_item.get("base64"):
                    file_type = file_item.get("type") or "application/octet-stream"
                    file_parts.append({
                        "type": "file",
                        "file": {
                            "filename": file_item.get("name", "dosya"),
                            "file_data": f"data:{file_type};base64,{file_item['base64']}"
                        }
                    })
                elif file_item.get("text"):
                    file_parts.append({"type": "text", "text": f"[{file_item.get('name', 'Dosya')} içeriği]\n{file_item['text']}"})
            if content:
                file_parts.insert(0, {"type": "text", "text": content})
            messages.append({"role": role, "content": file_parts if file_parts else content})
        elif content:
            messages.append({"role": role, "content": content})


    if user_content:
        if isinstance(voice, dict) and voice.get("base64"):
            audio_type = str(voice.get("type") or "audio/webm").split(";")[0]
            audio_format = audio_type.split("/")[-1] or "webm"
            user_content.append({
                "type": "input_audio",
                "input_audio": {"data": voice["base64"], "format": audio_format},
            })
        messages.append({"role": "user", "content": user_content})
    else:
        # If user_content is empty (e.g. only voice was sent without explicit text)
        if isinstance(voice, dict) and voice.get("base64"):
            audio_type = str(voice.get("type") or "audio/webm").split(";")[0]
            audio_format = audio_type.split("/")[-1] or "webm"
            messages.append({"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": voice["base64"], "format": audio_format}}
            ]})
        else:
            messages.append({"role": "user", "content": full_message})

    try:
        # Eğer mesajda input_audio varsa, multimodal audio modelini kullan
        has_audio_input = any(
            part.get("type") == "input_audio"
            for part in (user_content if user_content and isinstance(user_content, list) else [])
        ) or (isinstance(voice, dict) and voice.get("base64"))
        
        request_model = "openai/gpt-4o-audio-preview" if has_audio_input else model
        
        if deep_think:
            completion = deep_think_call(
                client,
                messages,
                request_model,
                deep_think_seconds,
                base_prompt, # base_prompt'u buraya ekledik
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
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

    def generate():
      yielded_count = 0
      try:
        tool_calls_buffer = []
        # Düşünme modunda veya normal modda gelen chunk'ları işle
        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                
                # Metin içeriği
                if delta.content:
                    yielded_count += 1
                    yield delta.content
                
                # Araç çağrıları
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
        
        # Eğer araç çağrıları varsa, bunları işle ve modeli tekrar çağır
        if tool_calls_buffer and any(tc is not None for tc in tool_calls_buffer):
            # İlk olarak, AI'ın araç çağrısı mesajını messages listesine ekleyelim
            assistant_message = {
                "role": "assistant",
                "content": "", # Araç çağrısı yapılırken içerik boş olabilir
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
            
            # Her bir araç çağrısını sırayla yürüt
            for tc in tool_calls_buffer:
                if tc:
                    try:
                        args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {} # JSON ayrıştırma hatası durumunda boş argümanlar
                    
                    # Fonksiyonu yürüt
                    result = execute_function(tc["name"], args)
                    
                    # Fonksiyon sonucunu messages listesine "tool" rolüyle ekle
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, ensure_ascii=False) # Sonucu JSON olarak stringleştir
                    })
            
            # Fonksiyon yürütme sonuçları ile modeli tekrar çağır
            try:
                final_completion = safe_model_call(
                    client,
                    messages,
                    request_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    deep_think=False, # Zaten derin düşünme yapıldıysa burada tekrar yapma
                    tools=TOOLS, # Araçları tekrar sun
                    tool_choice="auto",
                )
                
                # Tekrar çağrılan modelin yanıtını akışa dahil et
                for chunk in final_completion:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yielded_count += 1
                        yield chunk.choices[0].delta.content
                    # İkinci çağrıda tekrar araç çağrısı gelirse de işlenecektir.
            except Exception as e:
                yield friendly_api_error(e)
        
        # Eğer hiçbir şey döndürülmediyse, varsayılan hata mesajı
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
    form = AuthenticationForm(request, data=request.POST)
    if form.is_valid():
      login(request, form.get_user())
      request.session.set_expiry(2592000)
      request.session.save()
      return redirect("index")
    else:
      print(f"Login form errors: {form.errors}")
      print(f"POST data: {request.POST}")
  else:
    form = AuthenticationForm()
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
