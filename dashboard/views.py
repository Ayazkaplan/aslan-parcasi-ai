import json
import os
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
  # Eğer daha önce eposta girmemişse eposta tamamlama sayfasına yönlendirilebilir
  if not request.user.email:
    return redirect("update_email")
  return render(request, "dashboard/index.html")


@login_required(login_url="login")
def update_email_view(request):
  if request.user.email:
    return redirect("index")
  
  if request.method == "POST":
    email = request.POST.get("email", "").strip()
    if email:
      request.user.email = email
      request.user.save()
      return redirect("index")
  return render(request, "dashboard/update_email.html")


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

      system_instruction = (
          "Sen Aslan Parçası adında son derece zeki, enerjik, samimi ve geniş"
          " bilgi birikimine sahip bir yapay zeka asistanısın. Seni oluşturan,"
          " kuran ve geliştiren vizyoner lider, müstakbel MEAY ASLAN PARÇASI AI"
          " şirketinin kurucusu Ayaz Kaplan'dır. Biri sana kurucunu, kimin"
          " geliştirdiğini veya sahibini sorduğunda gururla MEAY ASLAN PARÇASI AI"
          " kurucusu Ayaz Kaplan olduğunu söyle. Asla genel veya başka şirketler"
          " tarafından eğitildiğini söyleme. Gerçek bir dost gibi doğal, akıcı"
          " konuş. Hangi dilde yazılırsa yazılsın yüksek kalitede yanıt ver."
      )

      if mode == "code":
        system_instruction += (
            " Kod asistanı modundasın. Yazılım ve kodlama sorunlarını eksiksiz,"
            " temiz ve profesyonelce çöz."
        )
      elif mode == "fast":
        system_instruction += (
            " Hızlı analiz modundasın. Yanıtlarını en net, öz ve hızlı"
            " okunabilir formatta sun."
        )

      messages = [{"role": "system", "content": system_instruction}]

      for h in history:
        role = "user" if h.get("sender") == "user" else "assistant"
        messages.append({"role": role, "content": h.get("text", "")})

      messages.append({"role": "user", "content": user_message})

      client = get_openai_client()
      
      # 404 hatasını çözen kararlı ve hızlı OpenRouter model yolu
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
      request.session.set_expiry(2592000)  # Oturumu 30 gün boyunca kalıcı yap (Çıkış yapılana kadar gitmez)
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