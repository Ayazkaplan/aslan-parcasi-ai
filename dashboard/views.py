import json
import os
import time
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
from .models import UserProfile


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


def safe_model_call(client, messages, model, temperature=0.7, max_tokens=4096, stream=False, deep_think=False):
    """
    Safe model call with retry logic and fallback models.
    Handles 404 (model not found) and 429 (rate limit) errors.
    """
    # Primary model and fallback models
    configured_fallback = os.environ.get("OPENROUTER_FALLBACK_MODEL", "").strip()
    models = [model, configured_fallback, "openrouter/auto"]
    models = list(dict.fromkeys(item for item in models if item))
    
    last_error = None
    
    for attempt_model in models:
        try:
            completion = client.chat.completions.create(
                model=attempt_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            return completion
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Handle rate limit (429) with retry
            if "429" in error_str or "rate limit" in error_str:
                for retry in range(2):  # Retry 1-2 times
                    time.sleep(1 + retry)  # Exponential backoff
                    try:
                        completion = client.chat.completions.create(
                            model=attempt_model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=stream
                        )
                        return completion
                    except Exception as retry_e:
                        last_error = retry_e
                        continue
            
            # The next configured model is attempted for provider/model errors.
            continue
    
    # All models failed
    raise last_error or Exception("Tüm modeller başarısız oldu")


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
  """Image generation endpoint"""
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
        "google/gemini-2.5-flash-image-preview",
    ).strip()
    try:
      # OpenRouter exposes current image models through chat completions. The
      # old openai/dall-e-3 name returned 404 on this deployment.
      response = client.chat.completions.create(
          model=image_model,
          messages=[{"role": "user", "content": prompt}],
          modalities=["text", "image"],
      )
      message = response.choices[0].message if response.choices else None
      images = getattr(message, "images", None) if message else None
      if images:
        image = images[0]
        image_url = getattr(image, "image_url", None)
        if isinstance(image_url, dict):
          image_url = image_url.get("url")
        elif image_url:
          image_url = getattr(image_url, "url", image_url)
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
def api_chat(request):
  if request.method != "POST":
    return JsonResponse({"error": "Geçersiz istek."}, status=405)

  try:
    data = json.loads(request.body or "{}")
    user_message = (data.get("message") or "").strip()
    mode = data.get("mode", "normal")
    history = data.get("history") or []
    deep_think = data.get("deep_think", False)
    images = data.get("images", [])
    files = data.get("files", [])
    voice_transcript = data.get("voice_transcript", "")

    if not user_message and not voice_transcript and not images and not files:
      return JsonResponse({"error": "Mesaj boş olamaz."}, status=400)

    # Combine voice transcript with text message
    full_message = user_message
    if voice_transcript:
      if full_message:
        full_message = f"{voice_transcript} {full_message}"
      else:
        full_message = voice_transcript
    file_names = [str(item.get("name", "dosya")) for item in files if isinstance(item, dict)]
    if file_names and not full_message:
      full_message = "Dosya gönderildi: " + ", ".join(file_names)

    client = get_openai_client()
    if client is None:
      return JsonResponse(
          {"error": "OPENROUTER_API_KEY tanımlı değil. Lütfen API anahtarını ayarlayın."},
          status=503,
      )

    current_time = timezone.localtime(timezone.now()).strftime("%d %B %Y, %A - %H:%M")

    # Mode-specific system prompts
    base_prompt = (
        "Sen Aslan Parçası adında son derece zeki, enerjik, samimi ve geniş bilgi birikimine sahip bir yapay zeka asistanısın. "
        "Seni oluşturan, kuran ve geliştiren vizyoner lider, müstakbel MEAY ASLAN PARÇASI AI şirketinin kurucusu Ayaz Kaplan'dır. "
        "KRİTİK KURAL: Sana sorulmadıkça saat, tarih, hava durumu veya kurucunun kim olduğu bilgisini KENDİLİĞİNDEN söyleme. "
        "Bu bilgileri SADECE kullanıcı açıkça sorduğunda ver. "
        "Gerçek zamanlı internet veya hava durumu erişimin yok; uydurma güncel veri verme. "
        "Hangi dilde yazılırsa yazılsın yüksek kalitede, akıcı bir dost gibi yanıt ver."
    )

    if mode == "normal":
      system_instruction = (
        base_prompt +
        " "
        "Genel asistan modundasın. Kullanıcıya samimi, yardımsever ve kapsamlı bir şekilde yardımcı ol. "
        "Konuları derinlemesine ara, bağlamı iyi anla ve net, yapılandırılmış yanıtlar ver. "
        "Mümkün olduğunca pratik çözümler sun ve adım adım açıklamalar yap. "
        "Eğer bir soru bilginin dışındaysa, dürüstçe söyle ve alternatif yaklaşım öner."
      )
      model = "openai/gpt-4o"
      temperature = 0.7
      max_tokens = 2048

    elif mode == "code":
      system_instruction = (
        "Sen Aslan Parçası AI'nın Kod Asistanı modusundasın. "
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

    # Keep requests within the available provider limits.
    if deep_think:
      system_instruction += " Kapsamlı düşünme modundasın. Konuyu dikkatle analiz et ve detaylı ama gereksiz tekrarsız bir yanıt ver."
      max_tokens = min(max_tokens, 3072)

    messages = [{"role": "system", "content": system_instruction}]

    # Handle multimodal content (images)
    user_content = []
    if full_message:
      user_content.append({"type": "text", "text": full_message})
    
    for img in images:
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
      if content:
        messages.append({"role": role, "content": content})

    if user_content:
      messages.append({"role": "user", "content": user_content})
    else:
      messages.append({"role": "user", "content": full_message})

    try:
      completion = safe_model_call(
        client,
        messages,
        model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        deep_think=deep_think,
      )
    except Exception as api_error:
      return JsonResponse({"error": friendly_api_error(api_error)}, status=503)

    # Use streaming response after the provider accepts the request. This
    # prevents an API exception from becoming a fake successful blank bubble.
    def generate():
      try:
        yielded = False
        for chunk in completion:
          if chunk.choices and chunk.choices[0].delta.content:
            yielded = True
            yield chunk.choices[0].delta.content
      except Exception as e:
        yield friendly_api_error(e)
      else:
        if not yielded:
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
    form = AuthenticationForm(request, data=request.POST)
    if form.is_valid():
      login(request, form.get_user())
      request.session.set_expiry(2592000)
      request.session.save()
      return redirect("index")
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
