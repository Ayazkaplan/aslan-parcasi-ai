# Aslan Parçası AI

## Çalıştırma

Proje Django tabanlıdır. Replit workflow'u veritabanı migration'larını çalıştırıp uygulamayı 5000 portunda başlatır:

```bash
python manage.py migrate --noinput
python manage.py runserver 0.0.0.0:5000
```

## Ortam değişkenleri

- `SESSION_SECRET`: Oturumların kod güncellemeleri ve yeniden başlatmalar arasında korunması için sabit Django gizli anahtarı.
- `DATABASE_URL`: Kullanıcılar, oturumlar, profiller ve sohbet geçmişleri için kalıcı PostgreSQL bağlantısı. Tanımlı değilse yalnızca yerel geliştirme için SQLite kullanılır.
- `OPENROUTER_API_KEY`: Sohbet ve görsel üretimi için OpenRouter anahtarı.
- `OPENROUTER_IMAGE_MODEL` (isteğe bağlı): Görsel üretim modeli; varsayılan `google/gemini-2.5-flash-image-preview`.
- `OPENROUTER_FALLBACK_MODEL` (isteğe bağlı): Sohbet için yedek model.

Kullanıcı adı, profil fotoğrafı, tema ve arka plan deseni `UserProfile` tablosunda saklanır; bu ayarlar tarayıcı veya cihaz değişse de hesaba bağlı kalır.
Sohbet geçmişleri `ChatHistory` tablosunda kullanıcı hesabına bağlı saklanır; aynı hesapla açılan cihazlar sohbetleri ve profil ayarlarını sunucudan senkronize eder.
Giriş ekranında kullanıcı adı veya e-posta adresiyle giriş yapılabilir. Kullanıcı kayıtları `DATABASE_URL` ile bağlı kalıcı veritabanında tutulur; veritabanı sıfırlanırsa eski hesaplar otomatik olarak geri getirilemez.
Sohbet eklerinde dosya sınırı 50 MB'dır. Metin, PDF ve DOCX dosyalarının içeriği yapay zekâ isteğine aktarılır; ses kaydı tarayıcıda Türkçe metne çevrilip sohbet isteğine gönderilir.

`AppClock` tablosu Europe/Istanbul tarihini saklar. Uygulama her kullanıldığında tarihi kontrol eder; kapalı uygulamada da güncellemek için saatlik harici scheduler şu komutu çalıştırmalıdır:

```bash
python manage.py migrate --noinput
python manage.py refresh_app_clock
```

Uygulama kapalıyken kendi Python sürecinin çalışması mümkün değildir; bu komut Render Cron, GitHub Actions veya başka bir harici zamanlayıcıya bağlanmalıdır. `refresh_app_clock` güvenli biçimde tekrar çalıştırılabilir.

## Yapay zekâ özellikleri

Sohbet ve görsel oluşturma için `OPENROUTER_API_KEY` gereklidir. Görseller OpenRouter'ın özel `/api/v1/images` endpoint'i üzerinden üretilir; kullanıcı promptu ve gerçekçi/kompozisyon talepleri istek içinde açıkça korunur. Aynı kullanıcıdan gelen görsel istekleri arasında kısa bir koruma aralığı vardır.
Anahtar tanımlı değilse uygulama çalışmaya devam eder ancak ilgili isteklerde kullanıcıya açık bir yapılandırma hatası gösterir.