# Aslan Parçası AI

## Çalıştırma

Proje Django tabanlıdır. Replit workflow'u veritabanı migration'larını çalıştırıp uygulamayı 5000 portunda başlatır:

```bash
python manage.py migrate --noinput
python manage.py runserver 0.0.0.0:5000
```

## Ortam değişkenleri

- `SESSION_SECRET`: Oturumların kod güncellemeleri ve yeniden başlatmalar arasında korunması için sabit Django gizli anahtarı.
- `OPENROUTER_API_KEY`: Sohbet ve görsel üretimi için OpenRouter anahtarı.
- `OPENROUTER_IMAGE_MODEL` (isteğe bağlı): Görsel üretim modeli; varsayılan `google/gemini-2.5-flash-image-preview`.
- `OPENROUTER_FALLBACK_MODEL` (isteğe bağlı): Sohbet için yedek model.

Kullanıcı adı, profil fotoğrafı, tema ve arka plan deseni `UserProfile` tablosunda saklanır; bu ayarlar tarayıcı veya cihaz değişse de hesaba bağlı kalır.