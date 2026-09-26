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
- `GROQ_API_KEY`, `MISTRAL_API_KEY`, `COHERE_API_KEY`: Yedeklemeli sağlayıcı havuzu için isteğe bağlı anahtarlar. Sohbet istekleri sırayla Groq → Mistral → Cohere → OpenRouter üzerinden gider; hata veren model/sağlayıcı atlanır. OpenRouter en sonda tutulduğu için ücretli kota yalnızca diğerleri tükenince harcanır.
- `OPENROUTER_IMAGE_MODEL` (isteğe bağlı): Görsel üretim modeli; varsayılan `google/gemini-2.5-flash-image-preview`.
- `OPENROUTER_FALLBACK_MODEL` (isteğe bağlı): Sohbet için yedek model.

Kullanıcı adı, profil fotoğrafı, tema ve arka plan deseni `UserProfile` tablosunda saklanır; bu ayarlar tarayıcı veya cihaz değişse de hesaba bağlı kalır.
Sohbet geçmişleri `ChatHistory` tablosunda kullanıcı hesabına bağlı saklanır; aynı hesapla açılan cihazlar sohbetleri ve profil ayarlarını sunucudan senkronize eder.
Giriş ekranında kullanıcı adı veya e-posta adresiyle giriş yapılabilir. Kullanıcı kayıtları `DATABASE_URL` ile bağlı kalıcı veritabanında tutulur; veritabanı sıfırlanırsa eski hesaplar otomatik olarak geri getirilemez.
Sohbet eklerinde dosya sınırı 50 MB'dır. Metin, PDF ve DOCX dosyalarının içeriği yapay zekâ isteğine aktarılır; ses kayıtları ses destekli model girdisi olarak gönderilir.

## Yapay zekâ özellikleri

Sohbet, görsel oluşturma ve sesli mesajların anlaşılması için `OPENROUTER_API_KEY` gereklidir. Anahtar tanımlı değilse uygulama çalışmaya devam eder ancak ilgili isteklerde kullanıcıya açık bir yapılandırma hatası gösterir.