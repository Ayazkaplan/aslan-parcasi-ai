import json
import re
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "AslanParcasiAI/1.0 (+https://aslan-parcasi-ai.onrender.com)"

WEATHER_CODES = {
    0: "açık",
    1: "çoğunlukla açık",
    2: "parçalı bulutlu",
    3: "kapalı",
    45: "sisli",
    48: "kırağılı sis",
    51: "hafif çisenti",
    53: "orta çisenti",
    55: "yoğun çisenti",
    61: "hafif yağmur",
    63: "orta yağmur",
    65: "şiddetli yağmur",
    71: "hafif kar",
    73: "orta kar",
    75: "yoğun kar",
    80: "hafif sağanak",
    81: "sağanak",
    82: "şiddetli sağanak",
    95: "gök gürültülü fırtına",
}


def _request(url, timeout=8, data=None, headers=None):
    req_headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if headers:
        req_headers.update(headers)
    body = data.encode("utf-8") if isinstance(data, str) else data
    request = urllib.request.Request(url, data=body, headers=req_headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _get_json(url, timeout=8):
    raw = _request(url, timeout=timeout, headers={"Accept": "application/json"})
    return json.loads(raw.decode("utf-8", errors="replace"))


def _get_text(url, timeout=8):
    raw = _request(url, timeout=timeout)
    return raw.decode("utf-8", errors="replace")


def extract_weather_location(text):
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    patterns = [
        r"(?:hava\s*durumu|sıcaklık|derece).{0,20}?(?:için|icin)?\s+([A-ZÇĞİÖŞÜa-zçğıöşü]+)",
        r"([A-ZÇĞİÖŞÜa-zçğıöşü]{3,})(?:'de|'da|'te|'ta|de|da|te|ta)\s+(?:hava|sıcaklık|derece)",
        r"([A-ZÇĞİÖŞÜa-zçğıöşü]{3,})\s+(?:hava\s*durumu|sıcaklık)",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            return match.group(1).strip(" ?!.,")
    return ""


def fetch_weather(location):
    if not location:
        return ""
    try:
        geo = _get_json(
            "https://geocoding-api.open-meteo.com/v1/search?"
            + urllib.parse.urlencode(
                {"name": location, "count": 1, "language": "tr", "format": "json"}
            ),
            timeout=8,
        )
        results = geo.get("results") or []
        if not results:
            return f"{location} için konum bulunamadı."
        place = results[0]
        lat = place.get("latitude")
        lon = place.get("longitude")
        label = ", ".join(
            part
            for part in [
                place.get("name"),
                place.get("admin1"),
                place.get("country"),
            ]
            if part
        )
        weather = _get_json(
            "https://api.open-meteo.com/v1/forecast?"
            + urllib.parse.urlencode(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "timezone": "Europe/Istanbul",
                }
            ),
            timeout=8,
        )
        current = weather.get("current") or {}
        temp = current.get("temperature_2m")
        feels = current.get("apparent_temperature")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")
        code = current.get("weather_code")
        desc = WEATHER_CODES.get(int(code) if code is not None else -1, "değişken")
        if temp is None:
            return f"{label} için sıcaklık alınamadı."
        return (
            f"{label} anlık hava: {temp}°C (hissedilen {feels}°C), {desc}, "
            f"nem %{humidity}, rüzgar {wind} km/s. Kaynak: Open-Meteo."
        )
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return ""


def web_search(query, max_results=5):
    if not query:
        return []
    try:
        html = _get_text(
            "https://html.duckduckgo.com/html/?"
            + urllib.parse.urlencode({"q": query}),
            timeout=10,
        )
    except (urllib.error.URLError, TimeoutError):
        return []

    results = []
    for match in re.finditer(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'class="result__snippet"[^>]*>(.*?)</(?:a|td|div)',
        html,
        re.IGNORECASE | re.DOTALL,
    ):
        title = re.sub("<[^>]+>", "", match.group(2))
        snippet = re.sub("<[^>]+>", "", match.group(3))
        title = re.sub(r"\s+", " ", title).strip()
        snippet = re.sub(r"\s+", " ", snippet).strip()
        if title or snippet:
            results.append(f"{title}: {snippet}")
        if len(results) >= max_results:
            break
    if results:
        return results

    for match in re.finditer(
        r'class="result__snippet"[^>]*>(.*?)</',
        html,
        re.IGNORECASE | re.DOTALL,
    ):
        snippet = re.sub("<[^>]+>", "", match.group(1))
        snippet = re.sub(r"\s+", " ", snippet).strip()
        if snippet:
            results.append(snippet)
        if len(results) >= max_results:
            break
    return results


def needs_live_data(text):
    lowered = (text or "").lower()
    keywords = (
        "hava durumu",
        "sıcaklık",
        "derece",
        "maç",
        "skor",
        "kaç kaç",
        "sonuç",
        "haber",
        "güncel",
        "bugün",
        "şu an",
        "su an",
        "şimdi",
        "canlı",
        "internet",
        "hava",
        "weather",
        "score",
    )
    return any(word in lowered for word in keywords)


def build_live_context(user_message, deep_think=False, extra_queries=None):
    parts = []
    location = extract_weather_location(user_message)
    weather_query = bool(
        location
        or re.search(r"hava\s*durumu|sıcaklık|kaç\s*derece", user_message or "", re.I)
    )
    if weather_query:
        weather = fetch_weather(location or user_message)
        if weather:
            parts.append("HAVA DURUMU:\n" + weather)

    queries = [user_message]
    if extra_queries:
        queries.extend(extra_queries)
    if deep_think:
        queries.append(f"{user_message} güncel kaynaklar")
        queries.append(f"{user_message} son dakika")

    seen = set()
    search_lines = []
    for query in queries:
        key = (query or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        for item in web_search(query, max_results=4 if deep_think else 3):
            if item not in search_lines:
                search_lines.append(item)
        if not deep_think:
            break

    if search_lines:
        parts.append("WEB ARAMA SONUÇLARI:\n- " + "\n- ".join(search_lines[:12]))

    if not parts:
        return ""
    return (
        "Aşağıdaki canlı veriler AZ ÖNCE internetten çekildi. "
        "Bunları kaynak kabul et. İnternetin yok deme, kullanıcıyı Google'a atma. "
        "Sayısal bir değer varsa (sıcaklık, skor) net söyle.\n\n"
        + "\n\n".join(parts)
    )


def pollinations_image_url(prompt):
    quoted = urllib.parse.quote((prompt or "drawing").strip()[:400])
    return (
        "https://image.pollinations.ai/prompt/"
        f"{quoted}?width=1024&height=1024&nologo=true&model=flux&enhance=true"
    )
