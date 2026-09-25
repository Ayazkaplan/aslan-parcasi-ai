import json
import os
import base64
import io
import re
import time
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST
from openai import OpenAI
from .forms import CustomUserCreationForm, EmailOrUsernameAuthenticationForm
from .models import ChatHistory, UserProfile
from .web_tools import build_live_context, needs_live_data, pollinations_image_url


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


def transcribe_voice(client, voice):
  if not client or not isinstance(voice, dict) or not voice.get("base64"):
    return ""
  try:
    raw = base64.b64decode(voice.get("base64"), validate=False)
  except (ValueError, TypeError):
    return ""
  if not raw:
    return ""
  audio_file = io.BytesIO(raw)
  audio_type = str(voice.get("type") or "audio/webm").split(";")[0]
  ext = "webm" if "webm" in audio_type else ("mp4" if "mp4" in audio_type else "wav")
  audio_file.name = f"recording.{ext}"
  models = [
      os.environ.get("OPENROUTER_WHISPER_MODEL", "").strip(),
      "openai/whisper-large-v3",
      "openai/whisper-1",
  ]
  for model in dict.fromkeys(item for item in models if item):
    try:
      audio_file.seek(0)
      result = client.audio.transcriptions.create(model=model, file=audio_file)
      text = getattr(result, "text", "") or ""
      if text.strip():
        return text.strip()
    except Exception:
      continue
  return ""


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
    if name.endswith(".pdf") or content_type == "application/pdf":
      from pypdf import PdfReader
      pages = PdfReader(io.BytesIO(raw)).pages
      return "\n\n".join((page.extract_text() or "") for page in pages)[:MAX_EXTRACTED_TEXT]
    if name.endswith(".docx") or content_type.endswith("wordprocessingml.document"):
      from docx import Document
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
    configured_fallback = os.environ.get("OPENROUTER_FALLBACK_MODEL", "").strip()
    models = [model, configured_fallback, "openai/gpt-4o-mini", "openrouter/auto"]
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
            
            if "429" in error_str or "rate limit" in error_str:
                for retry in range(2):
                    time.sleep(1 + retry)
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
            continue
    
    raise last_error or Exception("Tüm modeller başarısız oldu")


@login_required(login_url="login")
def index(request):
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

    image_url = pollinations_image_url(prompt)
    client = get_openai_client()
    image_model = os.environ.get("OPENROUTER_IMAGE_MODEL", "").strip()
    if client and image_model:
      try:
        response = client.chat.completions.create(
            model=image_model,
            messages=[{"role": "user", "content": f"Generate an image: {prompt}"}],
            extra_body={"modalities": ["text", "image"]},
        )
        message = response.choices[0].message if response.choices else None
        generated = extract_image_url(message)
        if generated:
          image_url = generated
      except Exception:
        pass
    return JsonResponse({"status": "success", "image_url": image_url, "prompt": prompt})
      
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
    voice_transcript = (data.get("voice_transcript") or "").strip()
    voice = data.get("voice") or {}

    client = get_openai_client()
    if client is None:
      return JsonResponse(
          {"error": "OPENROUTER_API_KEY tanımlı değil. Lütfen API anahtarını ayarlayın."},
          status=503,
      )

    if isinstance(voice, dict) and voice.get("base64") and not voice_transcript:
      voice_transcript = transcribe_voice(client, voice)

    if not user_message and not voice_transcript and not images and not files:
      if isinstance(voice, dict) and voice.get("base64"):
        return JsonResponse(
            {"error": "Ses kaydı yazıya çevrilemedi. Lütfen kısaca yazarak tekrar dene."},
            status=400,
        )
      return JsonResponse({"error": "Mesaj boş olamaz."}, status=400)

    full_message = user_message
    if voice_transcript:
      if full_message:
        full_message = f"{voice_transcript} {full_message}"
      else:
        full_message = voice_transcript
    file_names = [str(item.get("name", "dosya")) for item in files if isinstance(item, dict)]
    extracted_files = []
    file_images = []
    unread_files = []
    for file_item in files:
      if not isinstance(file_item, dict):
        continue
      try:
        file_size = int(file_item.get("size") or 0)
      except (TypeError, ValueError):
        file_size = 0
      if file_size > MAX_CHAT_FILE_BYTES:
        return JsonResponse({"error": "Dosya boyutu 50 MB sınırını aşamaz."}, status=413)
      file_type = str(file_item.get("type") or "")
      file_name = str(file_item.get("name") or "")
      if file_type.startswith("image/") or re.search(r"\.(png|jpe?g|gif|webp|bmp)$", file_name, re.I):
        file_images.append(file_item)
        continue
      extracted = extract_uploaded_file_text(file_item)
      if extracted:
        extracted_files.append(f"\n\n[{file_item.get('name', 'Dosya')} içeriği]\n{extracted}")
      else:
        unread_files.append(file_name or "dosya")
    if extracted_files:
      full_message += "".join(extracted_files)
    if unread_files:
      full_message += (
          "\n\n[Okunamayan dosyalar: "
          + ", ".join(unread_files)
          + ". Kullanıcı bu dosyayı gönderdi; içeriği çıkarılamadı.]"
      )
    if file_names and not full_message:
      full_message = "Dosya gönderildi: " + ", ".join(file_names)

    live_context = ""
    if deep_think or needs_live_data(full_message):
      live_context = build_live_context(
          full_message,
          deep_think=bool(deep_think),
      )

    base_prompt = (
        "Sen Aslan Parçası adında son derece zeki, enerjik, samimi ve geniş bilgi birikimine sahip bir yapay zeka asistanısın. "
        "Seni oluşturan, kuran ve geliştiren vizyoner lider, müstakbel MEAY ASLAN PARÇASI AI şirketinin kurucusu Ayaz Kaplan'dır. "
        "KRİTİK KURAL: Sana sorulmadıkça saat, tarih, hava durumu veya kurucunun kim olduğu bilgisini KENDİLİĞİNDEN söyleme. "
        "Bu bilgileri SADECE kullanıcı açıkça sorduğunda ver. "
        "Sana verilen GÜNCEL ARAŞTIRMA bloğu az önce internetten çekildi. "
        "Bu blok varsa internetin yok deme, kullanıcıyı Google'a yönlendirme; skoru, sıcaklığı ve haberi oradan net söyle. "
        "Blok yoksa uydurma sayı verme. "
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

    if deep_think:
      minutes = deep_think_seconds // 60
      seconds = deep_think_seconds % 60
      budget_label = f"{minutes} dakika {seconds} saniye" if minutes else f"{seconds} saniye"
      system_instruction += (
        f" Kapsamlı araştırma modu açık; kullanıcı sana {budget_label} düşünme bütçesi verdi. "
        "Sana verilen canlı araştırma sonuçlarını kullan, internetin yok deme. "
        "Soruyu parçalara ayır, çelişen bilgileri belirt ve net bir sonuç ver."
      )
      max_tokens = min(max_tokens * 2, 4096)

    if live_context:
      system_instruction += "\n\nGÜNCEL ARAŞTIRMA:\n" + live_context

    messages = [{"role": "system", "content": system_instruction}]

    user_content = []
    if full_message:
      user_content.append({"type": "text", "text": full_message})
    
    for img in list(images) + file_images:
      if img.get("url"):
        user_content.append({"type": "image_url", "image_url": {"url": img["url"]}})
      elif img.get("base64"):
        image_type = img.get("type") or "image/jpeg"
        user_content.append({"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{img['base64']}"}})

    if voice_transcript:
      user_content.append({"type": "text", "text": f"[Kullanıcının ses kaydı yazıya çevrildi]\n{voice_transcript}"})

    for h in history:
      if not isinstance(h, dict):
        continue
      role = "user" if h.get("sender") == "user" else "assistant"
      content = h.get("text") or ""
      if content:
        messages.append({"role": role, "content": content[:4000]})

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
    form = EmailOrUsernameAuthenticationForm(request, data=request.POST)
    if form.is_valid():
      login(request, form.get_user())
      request.session.set_expiry(2592000)
      request.session.save()
      return redirect("index")
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
      return redirect("index")
  else:
    form = CustomUserCreationForm()
  return render(request, "dashboard/register.html", {"form": form})


def logout_view(request):
  logout(request)
  return redirect("login")