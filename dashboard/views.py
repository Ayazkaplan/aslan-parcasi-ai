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


def get_openai_client():
  api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
  if not api_key or api_key == "gecici_anahtar":
    return None
  return OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=api_key,
  )


def safe_model_call(client, messages, model, temperature=0.7, max_tokens=4096, stream=False, deep_think=False):
    """
    Safe model call with retry logic and fallback models.
    Handles 404 (model not found) and 429 (rate limit) errors.
    """
    # Primary model and fallback models
    if deep_think:
        models = [model, "anthropic/claude-3.5-sonnet", "openai/gpt-4o"]
        max_tokens = 8192
    elif model == "fast":
        models = ["openai/gpt-4o-mini", "anthropic/claude-3-haiku", "openrouter/auto"]
        max_tokens = 2048
    else:
        models = [model, "openai/gpt-4o", "anthropic/claude-3.5-sonnet", "openrouter/auto"]
    
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
            
            # If model not found (404), try next model
            if "404" in error_str or "not found" in error_str or "model" in error_str:
                continue
            
            # For other errors, try next model
            continue
    
    # All models failed
    raise last_error or Exception("Tüm modeller başarısız oldu")


@login_required(login_url="login")
def index(request):
  if not request.user.email:
    return redirect("update_email")
  return render(request, "dashboard/index.html")


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
    
    try:
      # Try to use an image generation model via OpenRouter
      response = client.images.generate(
        model="openai/dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024"
      )
      
      if response.data and len(response.data) > 0:
        image_url = response.data[0].url
        return JsonResponse({"status": "success", "image_url": image_url})
      else:
        return JsonResponse({"error": "Görsel oluşturulamadı."}, status=500)
        
    except Exception as img_error:
      return JsonResponse(
          {"error": f"Görsel oluşturma hatası: {str(img_error)}. API erişimi olmayabilir."},
          status=500
      )
      
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
    voice_transcript = data.get("voice_transcript", "")

    if not user_message and not voice_transcript:
      return JsonResponse({"error": "Mesaj boş olamaz."}, status=400)

    # Combine voice transcript with text message
    full_message = user_message
    if voice_transcript:
      if full_message:
        full_message = f"{voice_transcript} {full_message}"
      else:
        full_message = voice_transcript

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
      max_tokens = 4096

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
      max_tokens = 8192

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

    # Adjust for deep think mode
    if deep_think:
      system_instruction += " Kapsamlı düşünme modundasın. Konuyu derinlemesine analiz et, farklı açıları değerlendir ve detaylı reasoning yap."
      max_tokens = 8192

    messages = [{"role": "system", "content": system_instruction}]

    # Handle multimodal content (images)
    user_content = []
    if full_message:
      user_content.append({"type": "text", "text": full_message})
    
    for img in images:
      if img.get("url"):
        user_content.append({"type": "image_url", "image_url": {"url": img["url"]}})
      elif img.get("base64"):
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img['base64']}"}})

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

    # Use streaming response
    def generate():
      try:
        completion = safe_model_call(
          client, 
          messages, 
          model, 
          temperature=temperature, 
          max_tokens=max_tokens, 
          stream=True,
          deep_think=deep_think
        )
        
        for chunk in completion:
          if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
      except Exception as e:
        error_msg = f"Hata: {str(e)}"
        yield error_msg

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
