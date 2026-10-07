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
- `GEMINI_API_KEY`: Sohbet ve görsel üretimi için Gemini API anahtarı.

Kullanıcı adı, profil fotoğrafı, tema ve arka plan deseni `UserProfile` tablosunda saklanır; bu ayarlar tarayıcı veya cihaz değişse de hesaba bağlı kalır.
Sohbet geçmişleri `ChatHistory` tablosunda kullanıcı hesabına bağlı saklanır; aynı hesapla açılan cihazlar sohbetleri ve profil ayarlarını sunucudan senkronize eder.
Giriş ekranında kullanıcı adı veya e-posta adresiyle giriş yapılabilir. Kullanıcı kayıtları `DATABASE_URL` ile bağlı kalıcı veritabanında tutulur; veritabanı sıfırlanırsa eski hesaplar otomatik olarak geri getirilemez.
Sohbet eklerinde dosya sınırı 50 MB'dır. Metin, PDF ve DOCX dosyalarının içeriği yapay zekâ isteğine aktarılır; ses kaydı tarayıcıda Türkçe metne çevrilip sohbet isteğine gönderilir.
Sohbete eklenen büyük görseller tarayıcıda küçültülüp sıkıştırılır; böylece fotoğrafla birlikte yazılan mesajlar istek boyutu sınırına takılmadan görsel anlayabilen modele gönderilir. Uzun yanıt akışı model bir parçada keserse kaldığı yerden devam ettirilir.
Canlı web araması sonuç vermediğinde güncel bilgi uydurmak yerine doğrulama yapılamadığı açıkça belirtilir. Maç skoru en az iki farklı alan adında uyuşmuyorsa kaynaklar arasındaki fark gösterilir; golcü ve dakika sorularında modelin ekleme yapması yerine arama sonuçlarının kaynak özetleri sunulur.

`AppClock` tablosu Europe/Istanbul tarihini saklar. Uygulama her kullanıldığında tarihi kontrol eder; kapalı uygulamada da güncellemek için saatlik harici scheduler şu komutu çalıştırmalıdır:

```bash
python manage.py migrate --noinput
python manage.py refresh_app_clock
```

Uygulama kapalıyken kendi Python sürecinin çalışması mümkün değildir; bu komut Render Cron, GitHub Actions veya başka bir harici zamanlayıcıya bağlanmalıdır. `refresh_app_clock` güvenli biçimde tekrar çalıştırılabilir.

## Yapay zekâ özellikleri

Sohbet için ücretsiz sağlayıcılar (Groq, Mistral, OpenRouter) arasında rotasyon yapılır; Google kotası görsel üretimi, ses çevirisi ve görsel anlamaya ayrılmıştır. Görsel oluşturmada önce Gemini image generation endpoint'i denenir; tüm anahtarlar kota/yoğunluk nedeniyle kullanılamazsa anahtarsız ücretsiz yedek servis (Pollinations) devreye girer, böylece kullanıcı asla "kota doldu" hatası almaz. Türkçe görsel promptları yedek servise gönderilmeden önce yaygın kelimeler İngilizce karşılıklarıyla desteklenir. Kota sınıflandırması dakikalık (per-minute) ve günlük (per-day) sınırları ayırt eder; dakikalık sınırlar kısa beklemeden sonra kendiliğinden açılır. Kullanıcı promptu ve gerçekçi/kompozisyon talepleri istek içinde açıkça korunur. Aynı kullanıcıdan gelen görsel istekleri arasında kısa bir koruma aralığı vardır.
Anahtar tanımlı değilse uygulama çalışmaya devam eder ancak ilgili isteklerde kullanıcıya açık bir yapılandırma hatası gösterir.

## Hava durumu

Hava durumu Open-Meteo'dan anahtarsız alınır. Şehir çözümlemesi esnektir: "Balıkesir Erdek" gibi il+ilçe yazımlarında ilçe tek başına denenir, "Erdek'te" gibi bulunma ekli yazımlar ve küçük/büyük harf farkları normalize edilir, sonuçlar ülke/il/nüfus puanlamasıyla seçilir.
