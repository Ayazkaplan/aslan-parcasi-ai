import json
import os
import base64
import io
import re
import requests
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


def build_provider_pool(openrouter_models=None, only_openrouter=False):
    """Yedeklemeli sağlayıcı havuzu: hata veren model/sağlayıcı atlanıp sıradakine geçilir.

    OpenRouter en sonda tutulur ki ücretli kota yalnızca diğerleri tükenince harcansın.
    """
    pool = []
    if not only_openrouter:
        groq_key = (os.environ.get("GROQ_API_KEY") or "").strip()
        if groq_key:
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
                "models": ["open-mistral-nemo", "mistral-small-latest"],
            })
        cohere_key = (os.environ.get("COHERE_API_KEY") or "").strip()
        if cohere_key:
            pool.append({
                "name": "cohere",
                "client": OpenAI(base_url="https://api.cohere.com/compatibility/v1", api_key=cohere_key),
                "models": ["command-r-plus-08-2024", "command-r-08-2024"],
            })
    openrouter_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if openrouter_key and openrouter_key != "gecici_anahtar":
        models = list(dict.fromkeys(item for item in (openrouter_models or []) if item))
        models.append("openrouter/auto")
        models = list(dict.fromkeys(models))
        pool.append({
            "name": "openrouter",
            "client": OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key),
            "models": models,
        })
    return pool


def pool_chat_completion(messages, temperature=0.7, max_tokens=4096, stream=False, tools=None, openrouter_models=None, only_openrouter=False):
    last_error = None
    for provider in build_provider_pool(openrouter_models, only_openrouter):
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
                continue
    raise last_error or Exception("Tüm sağlayıcılar başarısız oldu")


def deep_think_call(messages, deep_think_seconds, base_prompt, temperature=0.7, max_tokens=4096):
    thinking_prompt = (
        "Sen derin düşünme modundasın. Kullanıcının sorusunu/isteğini dikkatlice analiz et. "
        "Soruyu parçalara ayır, varsayımları kontrol et, kanıt ve karşı örnekleri değerlendir. "
        "Düşünce sürecini adım adım yaz. SADECE düşünce sürecini yaz, final cevabı verme. "
        "Türkçe düşün ve yaz. Cevabın sonunda kullanıcının orijinal isteğine atıfta bulunarak düşünme sürecini tamamla."
    )
    
    user_request_message = messages[-1] if messages else {"role": "user", "content": ""}

    temp_messages = []
    if messages and messages[0].get('role') == 'system' and messages[0].get('content') == base_prompt:
        temp_messages.append({"role": "system", "content": thinking_prompt})
        temp_messages.extend(messages[1:])
    else:
        temp_messages.append({"role": "system", "content": thinking_prompt})
        temp_messages.extend(messages)

    thinking_messages = temp_messages
    
    try:
        thinking_completion = pool_chat_completion(
            thinking_messages,
            temperature=0.3,
            max_tokens=min(max_tokens, 4096),
            stream=False,
        )
        reasoning = thinking_completion.choices[0].message.content if thinking_completion.choices else "Düşünme süreci boş."
    except Exception as e:
        reasoning = f"Düşünme sürecinde hata: {str(e)}"
    
    final_messages = [
        messages[0],
    ] + [msg for msg in messages[1:-1]] + [
        {"role": "user", "content": f"[Düşünme Süreci]\n{reasoning}\n\n[Orijinal İstek]\n" + (user_request_message.get('content') if isinstance(user_request_message.get('content'), str) else json.dumps(user_request_message.get('content'), ensure_ascii=False))}
    ]
    
    return pool_chat_completion(
        final_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        tools=TOOLS,
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
  return clear_remember_cookie(JsonResponse({"status": "success"}))


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
          modalities=["text", "image"],
      )
      message = response.choices[0].message if response.choices else None
      image_url = extract_image_url(message)
      if image_url:
        return JsonResponse({"status": "success", "image_url": image_url})
      return JsonResponse({"error": "Görsel modeli yanıtında görsel bulunamadı."}, status=502)
    except Exception as img_error:
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
            pass

    if not build_provider_pool():
        return JsonResponse(
            {"error": "Tanımlı yapay zekâ sağlayıcısı yok. Lütfen GROQ_API_KEY, MISTRAL_API_KEY, COHERE_API_KEY veya OPENROUTER_API_KEY anahtarlarından en az birini ayarlayın."},
            status=503,
        )

    base_prompt = (
        "Sen Aslan Parçası adında son derece zeki, enerjik, samimi ve geniş bilgi birikimine sahip bir yapay zeka asistanısın. "
        "Seni oluşturan, kuran ve geliştiren vizyoner lider, müstakbel MEAY ASLAN PARÇASI AI şirketinin kurucusu Ayaz Kaplan'dır. "
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
        if isinstance(voice, dict) and voice.get("base64"):
            audio_type = str(voice.get("type") or "audio/webm").split(";")[0]
            audio_format = audio_type.split("/")[-1] or "webm"
            messages.append({"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": voice["base64"], "format": audio_format}}
            ]})
        else:
            messages.append({"role": "user", "content": full_message})

    try:
        has_audio_input = any(
            part.get("type") == "input_audio"
            for part in (user_content if user_content and isinstance(user_content, list) else [])
        ) or (isinstance(voice, dict) and voice.get("base64"))
        
        request_model = "openai/gpt-4o-audio-preview" if has_audio_input else model
        
        if deep_think:
            completion = deep_think_call(
                messages,
                deep_think_seconds,
                base_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            completion = pool_chat_completion(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                tools=TOOLS,
                openrouter_models=[request_model],
                only_openrouter=has_audio_input,
            )
    except Exception as api_error:
      return JsonResponse({"error": friendly_api_error(api_error)}, status=503)

    def generate():
      yielded_count = 0
      try:
        tool_calls_buffer = []
        for chunk in completion:
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
        
        if tool_calls_buffer and any(tc is not None for tc in tool_calls_buffer):
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
                final_completion = pool_chat_completion(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    tools=TOOLS,
                    openrouter_models=[request_model],
                    only_openrouter=has_audio_input,
                )
                
                for chunk in final_completion:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yielded_count += 1
                        yield chunk.choices[0].delta.content
            except Exception as e:
                yield friendly_api_error(e)
        
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