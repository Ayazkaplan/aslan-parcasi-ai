import json
import os
import datetime
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from .forms import CustomUserCreationForm
from django.http import JsonResponse
from django.shortcuts import redirect, render
from openai import OpenAI


def get_openai_client():
  api_key = os.environ.get("OPENROUTER_API_KEY", "gecici_anahtar")
  return OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=api_key,
  )


@login_required(login_url="login")
def index(request):
  return render(request, "dashboard/index.html")


@login_required(login_url="login")
def update_username_view(request):
  if request.method == "POST":
    try:
      data = json.loads(request.body)
      new_username = data.get("username", "").strip()
      if new_username and not User.objects.filter(username=new_username).exclude(pk=request.user.pk).exists():
        request.user.username = new_username
        request.user.save()
        return JsonResponse({"status": "success"})
      return JsonResponse({"error": "Bu kullanıcı adı zaten alınmış veya geçersiz."}, status=400)
    except Exception as e:
      return JsonResponse({"error": str(e)}, status=500)
  return JsonResponse({"error": "Geçersiz istek."}, status=405)


@login_required(login_url="login")
def delete_account_view(request):
  if request.method == "POST":
    user = request.user
    logout(request)
    user.delete()
    return JsonResponse({"status": "success"})
  return JsonResponse({"error": "Geçersiz istek."}, status=405)


@login_required(login_url="login")
def api_chat(request):
  if request.method == "POST":
    try:
      data = json.loads(request.body)
      user_message = data.get("message", "").strip()
      mode = data.get("mode", "normal")
      history = data.get("history", [])

      if not user_message:
        return JsonResponse({"error": "Mesaj boş olamaz."}, status=400)

      current_time = datetime.datetime.now().strftime("%d %B %Y, %A - %H:%M")

      system_instruction = (
          f"Sen Aslan Parçası adında son derece zeki, enerjik, samimi ve geniş bilgi birikimine sahip bir yapay zeka asistanısın. "
          f"Seni oluşturan, kuran ve geliştiren vizyoner lider, müstakbel MEAY ASLAN PARÇASI AI şirketinin kurucusu Ayaz Kaplan'dır. "
          f"Şu anki gerçek zamanlı tarih ve saat: {current_time} (Türkiye/İstanbul). "
          f"Kullanıcı sana hava durumunu, saati veya güncel web bilgilerini sorduğunda bu zaman bilgisini ve internet erişim yeteneğini kullanarak tam ve doğru yanıt ver. "
          f"Kod asistanı modunda (code) tam, hatasız ve profesyonel kod blokları (markdown formatında) üret. "
          f"Hangi dilde yazılırsa yazılsın yüksek kalitede, akıcı bir dost gibi yanıt ver."
      )

      if mode == "code":
        system_instruction += " Kod asistanı modundasın. Yazdığın kodlar eksiksiz, modern ve çalıştırılabilir olmalı, kod blokları içinde tam çözümler sunmalısın."
      elif mode == "fast":
        system_instruction += " Hızlı analiz modundasın. Yanıtlarını en net, öz ve hızlı okunabilir formatta sun."

      messages = [{"role": "system", "content": system_instruction}]

      for h in history:
        role = "user" if h.get("sender") == "user" else "assistant"
        messages.append({"role": role, "content": h.get("text", "")})

      messages.append({"role": "user", "content": user_message})

      client = get_openai_client()
      
      completion = client.chat.completions.create(
          model="openrouter/auto",
          messages=messages,
          temperature=0.7,
          max_tokens=4096,
      )

      if not completion or not completion.choices:
        return JsonResponse({"error": "Model yanıt vermedi."}, status=500)

      reply_text = completion.choices[0].message.content

      return JsonResponse(
          {"status": "success", "reply": reply_text, "mode": mode}
      )
    except Exception as e:
      return JsonResponse({"error": str(e)}, status=500)

  return JsonResponse({"error": "Geçersiz istek."}, status=405)


def login_view(request):
  if request.user.is_authenticated:
    return redirect("index")
  if request.method == "POST":
    form = AuthenticationForm(request, data=request.POST)
    if form.is_valid():
      login(request, form.get_user())
      request.session.set_expiry(2592000)
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
      return redirect("index")
  else:
    form = CustomUserCreationForm()
  return render(request, "dashboard/register.html", {"form": form})


def logout_view(request):
  logout(request)
  return redirect("login")