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
- `GEMINI_API_KEY`: Sohbet ve mevcut görselleri düzenleme için Gemini API anahtarı.

Kullanıcı adı, profil fotoğrafı, tema ve arka plan deseni `UserProfile` tablosunda saklanır; bu ayarlar tarayıcı veya cihaz değişse de hesaba bağlı kalır.
Sohbet geçmişleri `ChatHistory` tablosunda kullanıcı hesabına bağlı saklanır; aynı hesapla açılan cihazlar sohbetleri ve profil ayarlarını sunucudan senkronize eder.
Giriş ekranında kullanıcı adı veya e-posta adresiyle giriş yapılabilir. Kullanıcı kayıtları `DATABASE_URL` ile bağlı kalıcı veritabanında tutulur; veritabanı sıfırlanırsa eski hesaplar otomatik olarak geri getirilemez.
Sohbet eklerinde dosya sınırı 50 MB'dır. Metin, PDF ve DOCX dosyalarının içeriği yapay zekâ isteğine aktarılır; ses kaydı tarayıcıda Türkçe metne çevrilip sohbet isteğine gönderilir.
Sohbete eklenen büyük görseller tarayıcıda küçültülüp sıkıştırılır; böylece fotoğrafla birlikte yazılan mesajlar istek boyutu sınırına takılmadan görsel anlayabilen modele gönderilir. Uzun yanıt akışı model bir parçada keserse kaldığı yerden devam ettirilir.
Canlı web araması sonuç vermediğinde güncel bilgi uydurmak yerine doğrulama yapılamadığı açıkça belirtilir. Aramalar birden fazla motor ve haber RSS kaynağıyla yapılır; bulut sunucusundan DuckDuckGo güvenlik kontrolü istenirse Bing ve haber RSS yedekleri kullanılır. Açık güncellik isteğinde tarihli kaynaklar 365 günlük pencereyle sınırlandırılır ve yayın tarihine göre en yeni sonuçlar öne alınır. Arama sonuçları kullanıcıya ham bağlantı listesi olarak değil, yanıt üretmek için kullanılır. Ayrıntılı araştırma isteği birden çok güncel ve konu odaklı sorgu açısıyla incelenir; kaynak tarihleri karşılaştırılır. Maç skoru en az iki farklı alan adında uyuşmuyorsa kaynaklar arasındaki fark gösterilir; golcü sorularında doğrulanmamış sohbet yanıtı arama sorgusuna eklenmez ve bulunan haber metinleri kullanılır. Saat sorularında açıkça belirtilen şehir veya ülkenin IANA saat dilimi kullanılır; şehir belirtilmezse İstanbul saati verilir. Basit geometrik görsel isteklerinde renk, şekil ve adet aynen korunur; fotoğraf stili ya da ilgisiz nesneler eklenmez. Görsel üretme yanıtındaki desteklenen görseller “Bu fotoğrafı düzenlemeye devam et” eylemiyle seçili görsel bağlamında düzenlenebilir; düzenleme sırasında yeni görsel üretme aracı kilitlenir ve mevcut alt bant simgeleri korunurken marka metni çizilmez.

`AppClock` tablosu Europe/Istanbul tarihini saklar. Uygulama her kullanıldığında tarihi kontrol eder; kapalı uygulamada da güncellemek için saatlik harici scheduler şu komutu çalıştırmalıdır:

```bash
python manage.py migrate --noinput
python manage.py refresh_app_clock
```

Uygulama kapalıyken kendi Python sürecinin çalışması mümkün değildir; bu komut Render Cron, GitHub Actions veya başka bir harici zamanlayıcıya bağlanmalıdır. `refresh_app_clock` güvenli biçimde tekrar çalıştırılabilir.

## Yapay zekâ özellikleri

Sohbet için ücretsiz sağlayıcılar (Groq, Mistral, OpenRouter) arasında rotasyon yapılır. Yeni görseller Türkçe prompt'u `gradio_client` üzerinden `CEObeY/aslan-parcasi-ai_gorsel_olusturma_istasyonu` Hugging Face Space'inin `/generate_image` uç noktasına göndererek FLUX.1-schnell ile oluşturur; Gemini görsel üretimi, HF Inference API ve Pollinations yeni görsel üretiminde kullanılmaz ve Hugging Face API token'ı gerekmez. Space'in döndürdüğü görsel dosyası uygulama yanıtına gömülür. Tek ve basit geometrik şekiller de aynı FLUX Space akışından üretilir. Görselin alt bandında mevcut Aslan Parçası simgeleri bulunur, ancak marka adı metin olarak çizilmez. Mevcut görseli düzenleme akışı seçili görseli ve kullanıcının metin talimatını Gemini görsel düzenleme modeline birlikte yollar; yeni görsel üretim Space'i görsel düzenleme için kullanılmaz. Kota sınıflandırması dakikalık ve günlük sınırları ayırt eder; aynı kullanıcıdan gelen görsel istekleri arasında kısa bir koruma aralığı vardır.

Dünya saati soruları şehir/ülke saat dilimine göre doğrudan yanıtlanır ve sohbet modeli bağlantısına bağlı değildir. Güncel web yanıtları birden fazla arama kaynağından derlenir; güncel ipuçları yanında somut bilgi isteyen olgusal sorular da canlı aramaya yönlendirilir, ham kaynak URL'leri yanıt metnine eklenmez. Maç skorları tarih biçimindeki sayılardan ayrılır; birden fazla olası maç/skor bulunursa bunlar liste halinde sunulmaz, netleştirme istenir. Golcü ve dakika ayrıntıları maç raporlarındaki ilgili cümlelere dayanır.
Anahtar tanımlı değilse uygulama çalışmaya devam eder ancak ilgili isteklerde kullanıcıya açık bir yapılandırma hatası gösterir.

## Hava durumu

Hava durumu Open-Meteo'dan anahtarsız alınır. Şehir çözümlemesi esnektir: "Balıkesir Erdek" gibi il+ilçe yazımlarında ilçe tek başına denenir, "Erdek'te" gibi bulunma ekli yazımlar ve küçük/büyük harf farkları normalize edilir, sonuçlar ülke/il/nüfus puanlamasıyla seçilir.
