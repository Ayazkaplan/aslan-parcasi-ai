import base64
import datetime as dt
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from core import settings as app_settings

from django.contrib.auth.models import User
from django.test import Client, SimpleTestCase, TestCase
from django.urls import reverse
from django.utils import timezone

from . import views
from .models import AppClock


class EnvLoadingTests(SimpleTestCase):
    def test_dotenv_file_is_loaded_into_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("GEMINI_API_KEY=test-key\n", encoding="utf-8")

            with patch.object(app_settings, "BASE_DIR", Path(tmpdir)), \
                 patch.dict(os.environ, {}, clear=True):
                app_settings.load_environment()
                self.assertEqual(os.environ.get("GEMINI_API_KEY"), "test-key")


class UploadedFileTests(TestCase):
    def test_plain_text_upload_is_extracted(self):
        content = "Rapor: yıllık satışlar yüzde 18 arttı."
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")

        extracted = views.extract_uploaded_file_text({
            "name": "rapor.txt",
            "type": "text/plain",
            "size": len(content.encode("utf-8")),
            "base64": encoded,
        })

        self.assertEqual(extracted, content)

    def test_actual_decoded_size_is_checked(self):
        encoded = base64.b64encode(b"four").decode("ascii")
        with patch("dashboard.views.MAX_CHAT_FILE_BYTES", 3):
            with self.assertRaisesRegex(ValueError, "50 MB"):
                views.extract_uploaded_file_text({
                    "name": "note.txt",
                    "type": "text/plain",
                    "size": 1,
                    "base64": encoded,
                })

    def test_unknown_file_type_returns_a_clear_error(self):
        encoded = base64.b64encode(b"binary").decode("ascii")
        with self.assertRaisesRegex(ValueError, "desteklenmiyor"):
            views.extract_uploaded_file_text({
                "name": "archive.bin",
                "type": "application/octet-stream",
                "size": 6,
                "base64": encoded,
            })


class WeatherTests(TestCase):
    def test_current_weather_uses_keyless_geocoding_and_returns_exact_temperature(self):
        geocode = Mock()
        geocode.json.return_value = {
            "results": [{
                "name": "Erdek",
                "country": "Türkiye",
                "country_code": "TR",
                "latitude": 40.4,
                "longitude": 27.8,
            }]
        }
        current = Mock()
        current.json.return_value = {
            "current": {
                "temperature_2m": 16.7,
                "apparent_temperature": 15.9,
                "relative_humidity_2m": 60,
                "weather_code": 2,
                "wind_speed_10m": 8.2,
                "time": "2026-09-25T12:00",
            }
        }
        observation = Mock()
        observation.status_code = 200
        observation.json.return_value = {}

        def weather_response(url, **_kwargs):
            if "geocoding-api.open-meteo.com" in url:
                return geocode
            if "api.open-meteo.com" in url:
                return current
            if "wttr.in" in url:
                return observation
            raise AssertionError(f"Unexpected weather URL: {url}")

        with patch("dashboard.views.requests.get", side_effect=weather_response) as get:
            result = views.get_weather("Erdek")

        self.assertEqual(result["city"], "Erdek")
        self.assertEqual(result["temperature"], 16.7)
        self.assertEqual(result["description"], "Parçalı bulutlu")
        self.assertEqual(get.call_count, 3)
        forecast_call = next(
            call for call in get.call_args_list if "/v1/forecast" in call.args[0]
        )
        self.assertEqual(
            forecast_call.kwargs["params"]["current"],
            "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
        )

    def test_recent_independent_observation_is_selected_and_forecast_conflict_is_reported(self):
        now = dt.datetime(2026, 10, 5, 14, 30, tzinfo=ZoneInfo("Europe/Istanbul"))
        payload = {
            "current_condition": [{
                "temp_C": "12",
                "FeelsLikeC": "10",
                "observation_time": "02:00 PM",
            }],
        }
        with patch("dashboard.views.requests.get") as get:
            get.return_value.status_code = 200
            get.return_value.json.return_value = payload
            get.return_value.raise_for_status.return_value = None
            observation = views._fresh_wttr_observation(
                {"latitude": 41.0, "longitude": 29.0},
                "Europe/Istanbul",
                now=now,
            )

        self.assertEqual(observation["temperature"], 12)
        self.assertEqual(observation["age_minutes"], 30)
        answer = views.format_weather_answer({
            "city": "İstanbul",
            "temperature": 12,
            "observed_temperature": 12,
            "forecast_temperature": 19.4,
            "temperature_conflict": True,
            "source": "wttr.in",
            "observed_at": observation["observed_at"],
        })
        self.assertIn("12°C", answer)
        self.assertIn("19.4°C", answer)
        self.assertIn("kaynaklar uyuşmuyor", answer)

    def test_stale_independent_weather_observation_is_ignored(self):
        now = dt.datetime(2026, 10, 5, 14, 30, tzinfo=ZoneInfo("Europe/Istanbul"))
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "current_condition": [{
                "temp_C": "12",
                "observation_time": "12:00 PM",
            }],
        }
        response.raise_for_status.return_value = None

        with patch("dashboard.views.requests.get", return_value=response):
            observation = views._fresh_wttr_observation(
                {"latitude": 41.0, "longitude": 29.0},
                "Europe/Istanbul",
                now=now,
            )

        self.assertIsNone(observation)


class SearchTests(SimpleTestCase):
    def setUp(self):
        views.SEARCH_ENGINE_STATE.clear()

    def test_weather_city_is_detected_when_question_puts_city_after_weather(self):
        for question in (
            "Hava kaç derece Erdek için?",
            "Erdek için hava durumu",
            "Erdek'te hava kaç derece?",
        ):
            with self.subTest(question=question):
                self.assertEqual(views.detect_weather_city(question), "Erdek")

    def test_sourced_score_summary_attributes_a_single_source(self):
        answer = views.summarize_sourced_match_score([{
            "title": "Fenerbahçe maçı 2-0 kazandı",
            "url": "https://sports.example/match",
            "snippet": "Maç 2-0 sona erdi.",
        }])

        self.assertIn("Bir canlı arama sonucu", answer)
        self.assertIn("bağımsız bir kaynakla doğrulayamadığım", answer)

    def test_sourced_score_summary_requires_two_domains_to_confirm_a_score(self):
        answer = views.summarize_sourced_match_score([
            {
                "title": "Fenerbahçe maçı 2-0 kazandı",
                "url": "https://one.example/match",
                "snippet": "Fenerbahçe karşılaşması 2-0 sona erdi.",
            },
            {
                "title": "Fenerbahçe maç sonucu 2-0",
                "url": "https://two.example/match",
                "snippet": "Skor 2-0 olarak kaydedildi.",
            },
        ])

        self.assertIn("İki bağımsız kaynak", answer)
        self.assertIn("2-0", answer)
        self.assertNotIn("https://", answer)

    def test_sourced_score_summary_reports_disagreement(self):
        answer = views.summarize_sourced_match_score([
            {
                "title": "Fenerbahçe maçı 2-0 kazandı",
                "url": "https://one.example/match",
                "snippet": "2-0 sona erdi.",
            },
            {
                "title": "Fenerbahçe maçı 1-0 kazandı",
                "url": "https://two.example/match",
                "snippet": "1-0 sona erdi.",
            },
        ])

        self.assertIn("farklı skorlar", answer)
        self.assertIn("2-0", answer)
        self.assertIn("1-0", answer)
        self.assertIn("doğrulanmış tek bir sonuç gibi sunmuyorum", answer)

    def test_sports_followup_search_uses_the_previous_user_question_only(self):
        history = [
            {"sender": "user", "text": "Fenerbahçe Eyüpspor maçı kaç kaç bitti?"},
            {"sender": "ai", "text": "8-0; Valencia 2 gol attı."},
        ]

        query = views.build_live_search_query(
            "O nasıl maç, Fenerbahçe ne yapmış öyle?",
            history,
        )
        normalized = views._ascii_fold(query).lower()

        self.assertIn("fenerbahce", normalized)
        self.assertIn("eyupspor", normalized)
        self.assertNotIn("valencia", normalized)

    def test_scorer_followup_uses_short_match_context_queries(self):
        history = [
            {"sender": "user", "text": "Fenerbahçe Eyüpspor maçı kaç kaç bitti?"},
            {"sender": "ai", "text": "8-0; doğrulanmamış golcü adı."},
        ]

        queries = views.build_live_search_queries("Golleri kim attı?", history)
        normalized = [views._ascii_fold(query).lower() for query in queries]

        self.assertTrue(views.is_sports_followup("Golleri kim attı?", history))
        self.assertGreaterEqual(len(queries), 2)
        self.assertIn("fenerbahce eyupspor", normalized[0])
        self.assertIn("golleri kim atti", normalized[0])
        self.assertIn("fenerbahce", normalized[1])
        self.assertNotIn("8-0", " ".join(normalized))
        self.assertNotIn("dogrulanmamis", " ".join(normalized))

    def test_match_details_prefer_article_excerpt_and_hide_source_links(self):
        answer = views.summarize_sourced_match_details([
            {
                "title": "Turan Tovuz 0-2 Fenerbahçe maç raporu",
                "url": "https://sports.example/report",
                "snippet": "Maç raporu.",
                "excerpt": (
                    "Fenerbahçe'yi galibiyete taşıyan golleri 23. dakikada "
                    "Cengiz Ünder ve 78. dakikada Edin Džeko kaydetti."
                ),
            },
        ])

        self.assertIn("Cengiz Ünder", answer)
        self.assertIn("Edin Džeko", answer)
        self.assertNotIn("https://", answer)
        self.assertNotIn("Enner Valencia", answer)

    def test_match_details_extract_goal_sentences_from_noisy_article_text(self):
        answer = views.summarize_sourced_match_details([
            {
                "title": "Fenerbahçe maç raporu",
                "snippet": "Maç raporu.",
                "excerpt": (
                    "Son dakika haberleri ve diğer gelişmeler. Uygulamayı açın. "
                    "23. dakikada Cengiz Ünder golü kaydetti. "
                    "Maçta toplam 18 korner kullanıldı."
                ),
            },
        ])

        self.assertIn("23. dakikada Cengiz Ünder golü kaydetti.", answer)
        self.assertNotIn("Son dakika", answer)
        self.assertNotIn("Uygulamayı açın", answer)
        self.assertNotIn("korner", answer)

    def test_scorer_followup_never_reuses_an_unverified_assistant_score(self):
        history = [
            {"sender": "user", "text": "Fenerbahçe Eyüpspor maçı kaç kaç bitti?"},
            {"sender": "ai", "text": "8-0; Enner Valencia iki gol attı."},
        ]

        queries = views.build_live_search_queries("Golleri kim attı?", history)

        self.assertTrue(all("8-0" not in query for query in queries))
        self.assertTrue(all("valencia" not in query.lower() for query in queries))

    def test_live_answer_link_sanitizer_removes_markdown_and_split_urls(self):
        sanitizer = views.LiveAnswerLinkSanitizer()
        first = sanitizer.feed("Güncel sonuç 2-0. Ayrıntı: [haber](https://spor")
        second = sanitizer.feed("x.example/mac)\n")
        final = sanitizer.feed("", final=True)

        self.assertEqual(first, "")
        self.assertEqual(second + final, "Güncel sonuç 2-0. Ayrıntı: haber\n")

    def test_google_news_rss_keeps_the_article_link(self):
        article_url = "https://news.google.com/rss/articles/example?oc=5"
        feed = (
            "<rss><channel><item><title>Fenerbahçe maç raporu</title>"
            f"<link>{article_url}</link><source url=\"https://gzt.com\">GZT</source>"
            "<pubDate>Sun, 20 Sep 2026 07:00:00 GMT</pubDate></item></channel></rss>"
        )

        results = views._parse_google_news_rss(feed, 5)

        self.assertEqual(results[0]["url"], article_url)

    def test_duckduckgo_search_returns_only_relevant_results(self):
        response = Mock()
        response.status_code = 200
        response.text = """
            <a class="result__a" href="https://sports.example/match-1">Fenerbahçe Galatasaray maç sonucu</a>
            <a class="result__snippet" href="#">Fenerbahçe Galatasaray karşılaşma sonucu</a>
            <a class="result__a" href="https://sports.example/match-2">Fenerbahçe Galatasaray puan durumu</a>
            <a class="result__snippet" href="#">Fenerbahçe ve Galatasaray puanları</a>
            <a class="result__a" href="https://sports.example/unrelated">Beşiktaş transfer haberleri</a>
            <a class="result__snippet" href="#">Beşiktaş yeni oyuncu transfer etti</a>
        """

        with patch("dashboard.views.requests.post", return_value=response) as post, patch(
            "dashboard.views.requests.get"
        ) as get:
            result = views.web_search(
                "Fenerbahçe Galatasaray maç sonucu",
                5,
                duckduckgo_only=True,
            )

        self.assertEqual(result["engine"], "ddg-html")
        self.assertEqual(len(result["results"]), 2)
        self.assertTrue(all("Beşiktaş" not in item["title"] for item in result["results"]))
        post.assert_called_once()
        self.assertIn("duckduckgo.com", post.call_args.args[0])
        get.assert_not_called()

    def test_duckduckgo_challenge_is_not_replaced_by_another_engine(self):
        response = Mock()
        response.status_code = 202
        response.text = "DDG.deep.anomalyDetectionBlock({})"

        with patch("dashboard.views.requests.post", return_value=response) as post, patch(
            "dashboard.views.requests.get"
        ) as get:
            result = views.web_search(
                "Python Django StreamingHttpResponse",
                duckduckgo_only=True,
            )

        self.assertTrue(result["blocked"])
        self.assertEqual(result["engine"], "ddg-html")
        post.assert_called_once()
        get.assert_not_called()

    def test_sports_live_search_retries_on_ddg_block_or_weak_result(self):
        self.assertTrue(
            views.should_retry_live_search_with_general_results(
                "Fenerbahçe Eyüpspor maçı ne oldu?",
                {"blocked": True, "engine": "ddg-html", "results": []},
            )
        )
        self.assertTrue(
            views.should_retry_live_search_with_general_results(
                "Fenerbahçe Eyüpspor maçı ne oldu?",
                {"engine": "weak-match", "results": []},
            )
        )
        self.assertTrue(
            views.should_retry_live_search_with_general_results(
                "Merhaba nasıl gidiyor?",
                {"blocked": True, "engine": "ddg-html", "results": []},
            )
        )

    def test_team_only_match_query_adds_a_latest_fixture_search(self):
        queries = views.build_live_search_queries("fenerbahçe maçı kaç kaç bitti")

        self.assertGreaterEqual(len(queries), 2)
        self.assertIn("son maç", queries[-1])
        self.assertIn("fenerbahce", views._ascii_fold(queries[-1]).lower())

    def test_live_search_combines_independent_search_indexes(self):
        def fake_search(query, num_results=5, **kwargs):
            engine = kwargs["only_engines"][0]
            return {
                "engine": engine,
                "results": [{
                    "title": f"Fenerbahçe maç raporu {engine}",
                    "url": f"https://{engine}.example/report",
                    "snippet": f"Maç raporu bulundu ({engine}).",
                }],
            }

        with patch("dashboard.views.web_search", side_effect=fake_search) as search:
            result = views.web_search_multi("Fenerbahçe maç raporu", 3, enrich=0)

        self.assertGreaterEqual(len(result["engines"]), 2)
        self.assertGreaterEqual(len(result["results"]), 2)
        self.assertEqual(search.call_count, 4)

    def test_reliable_feeds_are_tried_before_the_blocked_scrapers(self):
        order = [engine for engine, *_ in views._search_attempts("evren nasıl oluştu")]

        self.assertLess(order.index("vikipedi-tr"), order.index("ddg-html"))
        self.assertIn("bing-haber-rss", order)
        news_order = [engine for engine, *_ in views._search_attempts("fenerbahçe maçı sonucu")]
        self.assertEqual(news_order[0], "haber-rss")

    def test_bing_news_link_is_unwrapped_to_the_publisher(self):
        link = (
            "http://www.bing.com/news/apiclick.aspx?ref=FexRss&aid=&tid=1"
            "&url=https%3a%2f%2fwww.sporx.com%2fmac-ozeti"
        )

        self.assertEqual(
            views._clean_result_url(link), "https://www.sporx.com/mac-ozeti"
        )

    def test_feed_date_is_stored_as_iso_for_newest_first_sorting(self):
        feed = (
            "<rss><channel><item><title>Turan Tovuz 0-2 Fenerbahçe</title>"
            "<link>https://example.org/match</link>"
            "<pubDate>Sat, 03 Oct 2026 18:25:28 GMT</pubDate></item></channel></rss>"
        )

        results = views._parse_google_news_rss(feed, 5)

        self.assertEqual(results[0]["published"], "2026-10-03")

    def test_wikipedia_is_accepted_for_an_encyclopedic_question_only(self):
        payload = json.dumps({
            "query": {"search": [
                {"title": "Evren", "snippet": "Evren nasıl oluştu sorusunun yanıtı."},
                {"title": "Kenan Evren", "snippet": "Türk asker ve devlet adamı."},
            ]},
        })

        with patch("dashboard.views.requests.get") as get:
            get.return_value = SimpleNamespace(status_code=200, text=payload)
            result = views.web_search("evren nasıl oluştu", 5)

        self.assertEqual(result["engine"], "vikipedi-tr")
        self.assertEqual(result["results"][0]["title"], "Evren (Vikipedi)")

        with patch("dashboard.views.requests.get") as get:
            get.return_value = SimpleNamespace(status_code=200, text=payload)
            off_topic = views.web_search("python django StreamingHttpResponse", 5)

        self.assertNotEqual(off_topic.get("engine"), "vikipedi-tr")

    def test_article_excerpt_keeps_real_sentences_and_drops_navigation(self):
        page = (
            "<html><body><nav>Uygulamayı Aç Web'de Devam Et Kaynak Ekle</nav>"
            "<p>Fenerbahçe, Eyüpspor'u 8-0 yendi ve tarihe geçti.</p>"
            "<p>Vedat Muriqi dört gol atarak maça damga vurdu.</p>"
            "<script>var x = 1;</script></body></html>"
        )

        with patch("dashboard.views.requests.get") as get:
            get.return_value = SimpleNamespace(
                status_code=200, content=page.encode("utf-8"), encoding="utf-8"
            )
            excerpt = views._article_excerpt(
                "https://www.sporx.com/mac", ["fenerbahce", "muriqi"]
            )

        self.assertIn("Muriqi", excerpt)
        self.assertNotIn("Uygulamayı Aç", excerpt)

    def test_sports_results_without_the_asked_club_are_dropped(self):
        def fake_search(query, num_results=5, **kwargs):
            return {
                "engine": "haber-rss",
                "results": [
                    {"title": "İtalya Türkiye maçı kaç kaç bitti", "url": "https://example.org/a",
                     "snippet": "2026-10-05 · Haberler", "published": "2026-10-05"},
                    {"title": "Turan Tovuz 0-2 Fenerbahçe maç sonucu", "url": "https://example.org/b",
                     "snippet": "2026-10-03 · GZT", "published": "2026-10-03"},
                ],
            }

        with patch("dashboard.views.web_search", side_effect=fake_search):
            result = views.search_live_sports(["fenerbahçe maçı kaç kaç bitti"], 5, enrich=0)

        self.assertEqual(len(result["results"]), 1)
        self.assertIn("Fenerbahçe", result["results"][0]["title"])

    def test_specific_fixture_search_requires_both_named_clubs(self):
        def fake_search(query, num_results=5, **kwargs):
            return {
                "engine": "haber-rss",
                "results": [
                    {
                        "title": "Fenerbahçe başka rakibini 3-0 yendi",
                        "url": "https://example.org/wrong",
                        "snippet": "Fenerbahçe karşılaşma sonucu 3-0.",
                    },
                    {
                        "title": "Turan Tovuz 0-2 Fenerbahçe maç sonucu",
                        "url": "https://example.org/right",
                        "snippet": "Turan Tovuz ve Fenerbahçe karşılaşması 0-2 sona erdi.",
                    },
                ],
            }

        with patch("dashboard.views.web_search", side_effect=fake_search):
            result = views.search_live_sports(
                ["Fenerbahçe Turan Tovuz maç sonucu"],
                5,
                enrich=0,
            )

        self.assertEqual(len(result["results"]), 1)
        self.assertIn("Turan Tovuz", result["results"][0]["title"])


class AppClockTests(TestCase):
    def test_world_time_location_is_detected_for_cities_and_countries(self):
        for question, expected in (
            ("Çin'de saat kaç?", "Çin"),
            ("Tokyoda saat kaç?", "Tokyo"),
            ("Saat kaç Londra'da?", "Londra"),
            ("New York saat kaç?", "New York"),
        ):
            with self.subTest(question=question):
                self.assertEqual(views.detect_time_location(question), expected)

    def test_location_time_uses_the_geocoded_iana_timezone(self):
        geocoded = {
            "name": "Tokyo",
            "timezone": "Asia/Tokyo",
        }
        now = dt.datetime(2026, 10, 7, 12, 0, tzinfo=dt.timezone.utc)
        with patch("dashboard.views._geocode_city", return_value=geocoded) as geocode:
            result = views.get_location_time("Tokyo", now=now)

        geocode.assert_called_once_with("Tokyo", country="")
        self.assertEqual(result["time"], "21:00")
        self.assertEqual(result["date"], "07.10.2026")
        self.assertEqual(result["timezone"], "Asia/Tokyo")

    def test_country_alias_uses_china_standard_time(self):
        now = dt.datetime(2026, 10, 7, 12, 0, tzinfo=dt.timezone.utc)

        result = views.get_location_time("Çin'de", now=now)

        self.assertEqual(result["city"], "Şanghay")
        self.assertEqual(result["time"], "20:00")
        self.assertEqual(result["timezone"], "Asia/Shanghai")

    def test_clock_persists_the_current_istanbul_date(self):
        clock = views.sync_app_clock()

        self.assertEqual(clock.current_date, timezone.localdate())
        self.assertEqual(AppClock.objects.count(), 1)


class ImageGenerationTests(TestCase):
    def setUp(self):
        views._last_image_request_at.clear()
        views.KEY_STATE.clear()
        self.user = User.objects.create_user(
            username="image-user",
            email="image@example.com",
            password="test-password-123",
        )
        self.client.force_login(self.user)

    def test_dedicated_image_api_returns_embedded_generated_image(self):
        image_response = Mock()
        image_response.status_code = 200
        image_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "inlineData": {
                            "data": "cG5nLWJ5dGVz",
                            "mimeType": "image/png",
                        },
                    }],
                },
            }],
        }
        image_response.raise_for_status.return_value = None

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), patch(
            "dashboard.views.requests.post",
            return_value=image_response,
        ) as post:
            response = self.client.post(
                reverse("api_image_generate"),
                data=json.dumps({"prompt": "Fotogerçekçi bir uçak"}),
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["image_url"],
            "data:image/png;base64,cG5nLWJ5dGVz",
        )
        self.assertEqual(
            post.call_args.kwargs["params"]["key"],
            "test-key",
        )
        self.assertEqual(
            post.call_args.args[0],
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image:generateContent",
        )
        self.assertEqual(
            post.call_args.kwargs["json"]["generationConfig"]["imageConfig"]["imageSize"],
            "4K",
        )
        image_prompt = post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        self.assertIn("uçak", image_prompt)
        self.assertIn("ultra-photorealistic", image_prompt)

    def test_simple_red_circle_is_rendered_exactly_without_an_image_provider(self):
        with patch("dashboard.views.get_api_keys", return_value=[]), patch(
            "dashboard.views.generate_image_with_retry"
        ) as image_provider:
            response = self.client.post(
                reverse("api_image_generate"),
                data=json.dumps({"prompt": "Kırmızı bir daire"}),
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 200)
        image_url = response.json()["image_url"]
        self.assertTrue(image_url.startswith("data:image/svg+xml;base64,"))
        svg = base64.b64decode(image_url.split(",", 1)[1]).decode("utf-8")
        self.assertIn('<circle cx="512" cy="512" r="300" fill="#ff0000"/>', svg)
        self.assertNotIn("linearGradient", svg)
        image_provider.assert_not_called()

    def test_generated_image_is_displayed_without_automatic_branding(self):
        template_path = Path(__file__).parent / "templates" / "dashboard" / "index.html"
        template = template_path.read_text(encoding="utf-8")

        self.assertNotIn("addGeneratedImageBranding", template)
        self.assertIn("images: [{ url: data.image_url, name: prompt }]", template)

    def test_legacy_image_fallback_uses_supported_one_k_resolution(self):
        image_response = Mock()
        image_response.status_code = 200
        image_response.json.return_value = {
            "candidates": [{
                "content": {"parts": [{"inlineData": {
                    "data": "cG5nLWJ5dGVz",
                    "mimeType": "image/png",
                }}]},
            }],
        }
        with patch("dashboard.views.requests.post", return_value=image_response) as post:
            views.generate_image_with_gemini(
                "Bir kedi", "gemini-2.5-flash-image", "test-key"
            )

        self.assertEqual(
            post.call_args.kwargs["json"]["generationConfig"]["imageConfig"]["imageSize"],
            "1K",
        )

    def test_missing_image_credentials_returns_a_specific_error(self):
        with (
            patch.dict("os.environ", {"HF_API_TOKEN": "", "HUGGINGFACE_API_TOKEN": ""}),
            patch(
                "dashboard.views.generate_image_with_pollinations",
                side_effect=RuntimeError("fallback unavailable"),
            ),
            self.assertRaisesRegex(views.ImageUnavailableError, "API anahtarı tanımlı değil"),
        ):
            views.generate_image_with_retry("Bir dağ manzarası", api_keys=[])

    def test_image_api_explains_missing_server_credentials(self):
        with (
            patch.dict("os.environ", {"HF_API_TOKEN": "", "HUGGINGFACE_API_TOKEN": ""}),
            patch("dashboard.views.get_api_keys", return_value=[]),
            patch(
                "dashboard.views.generate_image_with_pollinations",
                side_effect=RuntimeError("fallback unavailable"),
            ),
        ):
            response = self.client.post(
                reverse("api_image_generate"),
                data=json.dumps({"prompt": "Bir dağ manzarası"}),
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("API anahtarı tanımlı değil", response.json()["error"])

    def test_image_api_explains_model_access_denial(self):
        with (
            patch.dict("os.environ", {
                "GEMINI_API_KEY": "test-key",
                "HF_API_TOKEN": "",
                "HUGGINGFACE_API_TOKEN": "",
            }),
            patch("dashboard.views.get_api_keys", return_value=["test-key"]),
            patch("dashboard.views.ordered_keys", return_value=["test-key"]),
            patch("dashboard.views.key_is_available", return_value=True),
            patch(
                "dashboard.views.generate_image_with_gemini",
                side_effect=views.UpstreamHTTPError(403, "model access denied"),
            ),
            patch(
                "dashboard.views.generate_image_with_pollinations",
                side_effect=RuntimeError("fallback unavailable"),
            ),
        ):
            response = self.client.post(
                reverse("api_image_generate"),
                data=json.dumps({"prompt": "Bir dağ manzarası"}),
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("modele erişemiyor", response.json()["error"])

    def test_configured_image_fallback_runs_after_gemini_rejects_access(self):
        with (
            patch.dict("os.environ", {"HF_API_TOKEN": "test-token"}),
            patch("dashboard.views.get_api_keys", return_value=["test-key"]),
            patch("dashboard.views.ordered_keys", return_value=["test-key"]),
            patch("dashboard.views.key_is_available", return_value=True),
            patch(
                "dashboard.views.generate_image_with_gemini",
                side_effect=views.UpstreamHTTPError(403, "model access denied"),
            ),
            patch(
                "dashboard.views.generate_image_with_flux_hf",
                return_value="data:image/png;base64,backup",
            ) as flux,
        ):
            result = views.generate_image_with_retry("Bir dağ manzarası")

        self.assertEqual(result, "data:image/png;base64,backup")
        flux.assert_called_once_with("Bir dağ manzarası")

    def test_keyless_image_fallback_returns_only_image_data(self):
        with (
            patch.dict("os.environ", {"HF_API_TOKEN": "", "HUGGINGFACE_API_TOKEN": ""}),
            patch("dashboard.views.get_api_keys", return_value=[]),
            patch(
                "dashboard.views.generate_image_with_pollinations",
                return_value="data:image/jpeg;base64,backup",
            ) as fallback,
        ):
            result = views.generate_image_with_retry(
                "Bir dağ manzarası",
                api_keys=[],
            )

        self.assertEqual(result, "data:image/jpeg;base64,backup")
        fallback.assert_called_once_with("Bir dağ manzarası")

    def test_explicit_illustration_style_is_not_forced_into_photorealism(self):
        prompt = views.enhance_image_prompt("Bir çizim: kedi")

        self.assertIn("çizim", prompt)
        self.assertNotIn("photorealistic", prompt)

    def test_geometric_prompt_preserves_the_requested_shape_and_suppresses_people(self):
        prompt = views.enhance_image_prompt("Kırmızı bir daire")

        self.assertIn("red", prompt.lower())
        self.assertIn("circle", prompt.lower())
        self.assertIn("exact shape", prompt.lower())
        self.assertIn("no people", prompt.lower())
        self.assertNotIn("photorealistic", prompt.lower())

    def test_gemini_geometric_image_prompt_does_not_add_photographic_subjects(self):
        image_response = Mock()
        image_response.status_code = 200
        image_response.json.return_value = {
            "candidates": [{
                "content": {"parts": [{"inlineData": {
                    "data": "cG5nLWJ5dGVz",
                    "mimeType": "image/png",
                }}]},
            }],
        }
        with patch("dashboard.views.requests.post", return_value=image_response) as post:
            views.generate_image_with_gemini("Kırmızı bir daire", "gemini-3-pro-image", "key")

        prompt = post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"].lower()
        self.assertIn("red", prompt)
        self.assertIn("circle", prompt)
        self.assertIn("no people", prompt)
        self.assertNotIn("photorealistic", prompt)
        self.assertNotIn("natural anatomy", prompt)

    def test_branded_drink_prompt_preserves_specific_packaging(self):
        prompt = views.enhance_image_prompt("Gerçekçi Aslan markalı kırmızı içecek kutusu")

        self.assertIn("branded product packaging", prompt)
        self.assertIn("readable label", prompt)
        self.assertIn("not a generic cup", prompt)

    def test_repeated_image_requests_are_throttled(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), patch(
            "dashboard.views.time.monotonic",
            return_value=100,
        ), patch("dashboard.views.generate_image_with_gemini", return_value="data:image/png;base64,x"):
            first = self.client.post(
                reverse("api_image_generate"),
                data=json.dumps({"prompt": "Bir şehir"}),
                content_type="application/json",
            )
            second = self.client.post(
                reverse("api_image_generate"),
                data=json.dumps({"prompt": "Başka bir şehir"}),
                content_type="application/json",
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)


class ChatFeatureTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="chat-user",
            email="chat@example.com",
            password="test-password-123",
        )
        self.client.force_login(self.user)

    def test_chat_sends_extracted_file_text_not_an_opaque_file_part(self):
        content = "Bu dosyanın gizli cümlesi: deniz mavidir."
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        chunk = SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="Dosyayı okudum.", tool_calls=None),
        )])
        model_call = Mock(return_value=iter([chunk]))

        with (
            patch("dashboard.views.get_gemini_client", return_value=object()),
            patch("dashboard.views.safe_model_call", model_call),
        ):
            response = self.client.post(
                reverse("api_chat"),
                data=json.dumps({
                    "message": "",
                    "files": [{
                        "name": "notlar.txt",
                        "type": "text/plain",
                        "size": len(content.encode("utf-8")),
                        "base64": encoded,
                    }],
                }),
                content_type="application/json",
            )
            answer = b"".join(response.streaming_content).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(answer, "Dosyayı okudum.")
        model_messages = model_call.call_args.args[1]
        sent_content = model_messages[-1]["content"]
        self.assertIn(content, sent_content[0]["text"])
        self.assertFalse(any(part.get("type") == "file" for part in sent_content))
        self.assertFalse(any(part.get("type") == "input_audio" for part in sent_content))

    def test_world_clock_question_uses_the_requested_timezone_without_model_guessing(self):
        user = self.user
        endpoint = views.Endpoint("test", "key", object(), (views.GEMINI_MODEL,), None)
        local_time = {
            "city": "Tokyo",
            "time": "21:00",
            "date": "07.10.2026",
            "day": "Çarşamba",
            "timezone": "Asia/Tokyo",
            "utc_offset": "+0900",
        }
        with (
            patch("dashboard.views.get_chat_endpoints", return_value=[endpoint]),
            patch("dashboard.views.get_location_time", return_value=local_time) as get_time,
            patch("dashboard.views.safe_model_call") as model_call,
        ):
            response = self.client.post(
                reverse("api_chat"),
                data=json.dumps({"message": "Tokyoda saat kaç?"}),
                content_type="application/json",
            )
            answer = b"".join(response.streaming_content).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Tokyo için saat 21:00", answer)
        self.assertIn("Asia/Tokyo", get_time.return_value["timezone"])
        get_time.assert_called_once_with("Tokyo")
        model_call.assert_not_called()

    def test_world_clock_question_does_not_require_a_chat_provider(self):
        local_time = {
            "city": "Shanghai",
            "time": "20:00",
            "date": "07.10.2026",
            "day": "Çarşamba",
            "timezone": "Asia/Shanghai",
            "utc_offset": "+0800",
        }
        with (
            patch("dashboard.views.get_chat_endpoints", return_value=[]) as endpoints,
            patch("dashboard.views.get_location_time", return_value=local_time),
        ):
            response = self.client.post(
                reverse("api_chat"),
                data=json.dumps({"message": "Çin'de saat kaç?"}),
                content_type="application/json",
            )
            answer = b"".join(response.streaming_content).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Shanghai için saat 20:00", answer)
        endpoints.assert_not_called()

    def test_live_answer_is_synthesized_without_exposing_search_urls(self):
        chunk = SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="Güncel açıklama burada. Kaynak: [haber](https://news.example/article)"),
            finish_reason="stop",
        )])
        endpoint = views.Endpoint("test", "key", object(), (views.GEMINI_MODEL,), None)
        with (
            patch("dashboard.views.get_chat_endpoints", return_value=[endpoint]),
            patch("dashboard.views.safe_model_call", return_value=iter([chunk])),
            patch("dashboard.views.web_search", return_value={
                "results": [{
                    "title": "Güncel haber",
                    "url": "https://news.example/article",
                    "snippet": "Gelişmenin özeti.",
                }],
            }),
        ):
            response = self.client.post(
                reverse("api_chat"),
                data=json.dumps({"message": "Bugün yeni gelişme ne oldu?"}),
                content_type="application/json",
            )
            answer = b"".join(response.streaming_content).decode("utf-8")

        self.assertIn("Güncel açıklama burada.", answer)
        self.assertIn("haber", answer)
        self.assertNotIn("https://", answer)

    def test_chat_accepts_a_photo_sent_with_a_text_question(self):
        chunk = SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="Fotoğrafta bir kedi var.", tool_calls=None),
            finish_reason="stop",
        )])
        endpoint = views.Endpoint("test", "key", object(), (views.GEMINI_MODEL,), None)
        model_call = Mock(return_value=iter([chunk]))

        with (
            patch("dashboard.views.get_chat_endpoints", return_value=[endpoint]) as get_endpoints,
            patch("dashboard.views.safe_model_call", model_call),
        ):
            response = self.client.post(
                reverse("api_chat"),
                data=json.dumps({
                    "message": "Bu fotoğrafta ne var?",
                    "images": [{
                        "name": "photo.webp",
                        "type": "image/webp",
                        "base64": "cG5nLWJ5dGVz",
                    }],
                }),
                content_type="application/json",
            )
            answer = b"".join(response.streaming_content).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("kedi", answer)
        self.assertTrue(any(call.kwargs.get("need_vision") for call in get_endpoints.call_args_list))
        sent_content = model_call.call_args.args[1][-1]["content"]
        self.assertEqual(sent_content[0]["text"], "Bu fotoğrafta ne var?")
        self.assertEqual(
            sent_content[1]["image_url"]["url"],
            "data:image/webp;base64,cG5nLWJ5dGVz",
        )

    def test_chat_continues_when_the_provider_truncates_a_long_answer(self):
        first_chunk = SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="İlk bölüm.", tool_calls=None),
            finish_reason="length",
        )])
        second_chunk = SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="Devamı tamamlandı.", tool_calls=None),
            finish_reason="stop",
        )])
        endpoint = views.Endpoint("test", "key", object(), (views.GEMINI_MODEL,), None)
        model_call = Mock(side_effect=[iter([first_chunk]), iter([second_chunk])])

        with (
            patch("dashboard.views.get_chat_endpoints", return_value=[endpoint]),
            patch("dashboard.views.safe_model_call", model_call),
        ):
            response = self.client.post(
                reverse("api_chat"),
                data=json.dumps({"message": "Uzun bir açıklama yaz."}),
                content_type="application/json",
            )
            answer = b"".join(response.streaming_content).decode("utf-8")

        self.assertEqual(answer, "İlk bölüm.Devamı tamamlandı.")
        self.assertEqual(model_call.call_count, 2)
        continuation = model_call.call_args_list[1].args[1]
        self.assertEqual(continuation[-2]["content"], "İlk bölüm.")
        self.assertIn("Kaldığın yerden devam et", continuation[-1]["content"])

    def test_live_question_does_not_fall_back_to_an_unverified_model_answer(self):
        endpoint = views.Endpoint("test", "key", object(), (views.GEMINI_MODEL,), None)
        with (
            patch("dashboard.views.get_chat_endpoints", return_value=[endpoint]),
            patch("dashboard.views.web_search", return_value={"results": [], "error": "blocked"}),
            patch("dashboard.views.safe_model_call") as model_call,
        ):
            response = self.client.post(
                reverse("api_chat"),
                data=json.dumps({"message": "Bugün son dakika gelişmesi ne oldu?"}),
                content_type="application/json",
            )
            answer = b"".join(response.streaming_content).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("doğrulayamadım", answer)
        model_call.assert_not_called()

    def test_match_score_answer_is_grounded_in_independent_search_sources(self):
        endpoint = views.Endpoint("test", "key", object(), (views.GEMINI_MODEL,), None)
        with (
            patch("dashboard.views.get_chat_endpoints", return_value=[endpoint]),
            patch("dashboard.views.web_search", return_value={
                "engine": "bing",
                "results": [
                    {
                        "title": "Fenerbahçe maçı 2-0 kazandı",
                        "url": "https://one.example/match",
                        "snippet": "Fenerbahçe karşılaşması 2-0 sona erdi.",
                    },
                    {
                        "title": "Fenerbahçe maç sonucu 2-0",
                        "url": "https://two.example/match",
                        "snippet": "Skor 2-0 olarak kaydedildi.",
                    },
                ],
            }),
            patch("dashboard.views.safe_model_call") as model_call,
        ):
            response = self.client.post(
                reverse("api_chat"),
                data=json.dumps({"message": "Fenerbahçe maçı kaç kaç?"}),
                content_type="application/json",
            )
            answer = b"".join(response.streaming_content).decode("utf-8")

        self.assertIn("2-0", answer)
        self.assertIn("İki bağımsız kaynak", answer)
        self.assertNotIn("https://", answer)
        model_call.assert_not_called()

    def test_deep_research_calls_search_tools_before_final_answer(self):
        tool_call = SimpleNamespace(
            id="search-1",
            function=SimpleNamespace(
                name="web_search",
                arguments='{"query":"Fenerbahçe Eyüpspor maç sonucu"}',
            ),
        )
        first_response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=None, tool_calls=[tool_call]),
        )])
        final_chunk = SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="Araştırılmış yanıt."),
        )])
        model_call = Mock(side_effect=[first_response, iter([final_chunk])])
        messages = [
            {"role": "system", "content": "araştır"},
            {"role": "user", "content": "Maç kaç kaç bitti?"},
        ]

        with (
            patch("dashboard.views.safe_model_call", model_call),
            patch("dashboard.views.execute_function", return_value={
                "results": [{"title": "Maç sonucu", "snippet": "2-1"}],
            }),
        ):
            result = views.deep_think_call(
                object(),
                messages,
                views.GEMINI_MODEL,
                30,
                tools=views.TOOLS,
            )
            answer = "".join(
                chunk.choices[0].delta.content or ""
                for chunk in result
            )

        self.assertEqual(answer, "Araştırılmış yanıt.")
        final_messages = model_call.call_args_list[1].args[1]
        self.assertTrue(any(message.get("role") == "tool" for message in final_messages))

    def test_deep_research_tool_calls_are_duckduckgo_only(self):
        tool_call = SimpleNamespace(
            id="search-ddg",
            function=SimpleNamespace(
                name="web_search",
                arguments='{"query":"Fenerbahçe Galatasaray maç sonucu"}',
            ),
        )
        tool_response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=None, tool_calls=[tool_call]),
        )])
        empty_response = SimpleNamespace(choices=[])
        response_count = 0

        def model_call(*args, **kwargs):
            nonlocal response_count
            if kwargs.get("stream"):
                return iter([])
            response_count += 1
            return tool_response if response_count == 1 else empty_response

        search_result = {
            "engine": "ddg-html",
            "results": [
                {"title": "Fenerbahçe Galatasaray maç sonucu", "url": "https://sports.example/match"},
                {"title": "Fenerbahçe Galatasaray puan durumu", "url": "https://sports.example/table"},
            ],
        }
        clock = [0]

        def advance_clock():
            clock[0] += 1
            return float(clock[0])

        with (
            patch("dashboard.views.safe_model_call", side_effect=model_call),
            patch("dashboard.views.web_search", return_value=search_result) as search,
            patch("dashboard.views.execute_function") as execute,
            patch("dashboard.views.time.monotonic", side_effect=advance_clock),
            patch("dashboard.views.time.sleep"),
        ):
            events = list(views.deep_think_events(
                object(),
                [
                    {"role": "system", "content": "araştır"},
                    {"role": "user", "content": "Fenerbahçe Galatasaray maç sonucu"},
                ],
                30,
                0.3,
                1024,
            ))

        self.assertTrue(search.call_args_list)
        self.assertTrue(all(call.kwargs["duckduckgo_only"] for call in search.call_args_list))
        execute.assert_not_called()
        self.assertTrue(any(kind == "reason" and "Kaynak:" in text for kind, text in events))

    def test_deep_research_never_refuses_when_search_engines_are_blocked(self):
        final_chunk = SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="Genel bilgiyle özet: evren genişliyor."),
        )])

        def model_call(*args, **kwargs):
            if kwargs.get("stream"):
                return iter([final_chunk])
            return SimpleNamespace(choices=[])

        with (
            patch("dashboard.views.web_search", return_value={"blocked": True, "results": []}),
            patch("dashboard.views.safe_model_call", side_effect=model_call) as model_call_mock,
            patch("dashboard.views.time.sleep"),
        ):
            events = list(views.deep_think_events(
                object(),
                [
                    {"role": "system", "content": "araştır"},
                    {"role": "user", "content": "evren nasıl oluştu"},
                ],
                30,
                0.3,
                1024,
            ))

        answer = "".join(text for kind, text in events if kind == "answer")
        self.assertTrue(model_call_mock.called)
        self.assertIn("evren genişliyor", answer)
        self.assertNotIn("erişemedim", answer)
        self.assertNotIn("iddia etmeyeceğim", answer)
        system_text = "\n".join(
            str(message.get("content") or "")
            for message in model_call_mock.call_args_list[-1].args[1]
            if message.get("role") == "system"
        )
        self.assertIn("doğrulanabilir web kaynağı toplanamadı", system_text)

    def test_deep_research_falls_back_to_general_search_when_ddg_is_blocked(self):
        tool_call = SimpleNamespace(
            id="search-1",
            function=SimpleNamespace(
                name="web_search",
                arguments='{"query":"Fenerbahçe Eyüpspor maç sonucu"}',
            ),
        )
        first_response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=None, tool_calls=[tool_call]),
        )])
        final_chunk = SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="Sonuç net: Fenerbahçe 2-1 kazandı."),
        )])
        model_call_count = 0

        def model_call(*args, **kwargs):
            nonlocal model_call_count
            if kwargs.get("stream"):
                return iter([final_chunk])
            model_call_count += 1
            return first_response if model_call_count == 1 else SimpleNamespace(choices=[])

        search_results = iter([
            {"engine": "ddg-html", "results": [{"title": "Maç öncesi", "url": "https://sports.example/preview"}]},
            {"blocked": True, "results": [], "engine": "ddg-html"},
            {"engine": "bing", "results": [{"title": "Fenerbahçe Eyüpspor 2-1", "snippet": "Maç sonucu 2-1."}]},
        ])

        def fake_web_search(*args, **kwargs):
            return next(search_results, {
                "engine": "ddg-html",
                "results": [{"title": "Fenerbahçe Eyüpspor kaynak", "url": "https://sports.example/match"}],
            })

        clock = {"now": 0.0}

        def fake_monotonic():
            return clock["now"]

        def fake_sleep(seconds):
            clock["now"] += seconds

        with (
            patch("dashboard.views.safe_model_call", side_effect=model_call),
            patch("dashboard.views.web_search", side_effect=fake_web_search) as search,
            patch("dashboard.views.time.monotonic", side_effect=fake_monotonic),
            patch("dashboard.views.time.sleep", side_effect=fake_sleep),
        ):
            events = list(views.deep_think_events(
                object(),
                [
                    {"role": "system", "content": "araştır"},
                    {"role": "user", "content": "Fenerbahçe Eyüpspor maç sonucu"},
                ],
                30,
                0.3,
                1024,
            ))

        self.assertGreaterEqual(search.call_count, 3)
        self.assertTrue(search.call_args_list[1].kwargs["duckduckgo_only"])
        self.assertFalse(search.call_args_list[2].kwargs["duckduckgo_only"])
        self.assertIn("Fenerbahçe 2-1 kazandı", "".join(text for kind, text in events if kind == "answer"))

    def test_deep_research_falls_back_when_ddg_returns_no_results(self):
        clock = {"now": 0.0}
        final_chunk = SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="Kaynaklarla desteklenen yanıt."),
        )])
        empty_response = SimpleNamespace(choices=[])

        def model_call(*args, **kwargs):
            return iter([final_chunk]) if kwargs.get("stream") else empty_response

        def fake_search(query, num_results, *, duckduckgo_only=False, deadline=None, **kwargs):
            if duckduckgo_only:
                return {"engine": "ddg-html", "results": []}
            return {"engine": "bing", "results": [{
                "title": "Güncel kaynak",
                "url": "https://sports.example/source",
                "snippet": "Doğrulanabilir bilgi.",
            }]}

        def fake_sleep(seconds):
            clock["now"] += seconds

        with (
            patch("dashboard.views.safe_model_call", side_effect=model_call),
            patch("dashboard.views.web_search", side_effect=fake_search) as search,
            patch("dashboard.views.time.monotonic", side_effect=lambda: clock["now"]),
            patch("dashboard.views.time.sleep", side_effect=fake_sleep),
        ):
            events = list(views.deep_think_events(
                object(),
                [
                    {"role": "system", "content": "araştır"},
                    {"role": "user", "content": "Güncel araştırma konusu"},
                ],
                30,
                0.3,
                1024,
            ))

        self.assertGreaterEqual(search.call_count, 2)
        self.assertTrue(search.call_args_list[0].kwargs["duckduckgo_only"])
        self.assertFalse(search.call_args_list[1].kwargs["duckduckgo_only"])
        self.assertIn("Kaynaklarla desteklenen yanıt", "".join(
            text for kind, text in events if kind == "answer"
        ))

    def test_scorer_followup_searches_context_and_specific_queries(self):
        endpoint = views.Endpoint("test", "key", object(), (views.GEMINI_MODEL,), None)
        history = [
            {"sender": "user", "text": "Fenerbahçe Eyüpspor maçı kaç kaç bitti?"},
            {"sender": "ai", "text": "8-0; golcüler kaynakta doğrulanmadı."},
        ]
        search_result = {"engine": "ddg-html", "results": [
            {
                "title": "Fenerbahçe Eyüpspor maç sonucu 8-0",
                "url": "https://sports.example/result",
                "snippet": "Karşılaşma 8-0 sona erdi.",
            },
            {
                "title": "Fenerbahçe Eyüpspor maç raporu ve golleri",
                "url": "https://sports.example/report",
                "snippet": "Golleri atan oyuncular maç raporunda.",
                "excerpt": (
                    "Fenerbahçe'nin gollerini Edin Džeko ve İrfan Can Kahveci kaydetti."
                ),
            },
        ]}

        with (
            patch("dashboard.views.get_chat_endpoints", return_value=[endpoint]),
            patch("dashboard.views.web_search", return_value=search_result) as search,
            patch("dashboard.views.safe_model_call") as model_call,
        ):
            response = self.client.post(
                reverse("api_chat"),
                data=json.dumps({
                    "message": "Golleri kim attı?",
                    "history": history,
                }),
                content_type="application/json",
            )
            answer = b"".join(response.streaming_content).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(search.call_count, 8)
        self.assertIn("golleri kim attı", search.call_args_list[0].args[0].lower())
        self.assertIn("gol atan oyuncular", search.call_args_list[4].args[0].lower())
        self.assertFalse(
            any(call.kwargs.get("duckduckgo_only") for call in search.call_args_list)
        )
        self.assertIn("Maç raporlarında", answer)
        self.assertIn("Edin Džeko", answer)
        self.assertNotIn("https://", answer)
        model_call.assert_not_called()

    def test_sanitize_output_strips_reasoning_leaks(self):
        raw = "Here's a thinking process:\n1. Analyze user input\n2. Search\nFinal answer: Fenerbahçe 2-1 kazandı."
        self.assertEqual(
            views.sanitize_model_output(raw),
            "Fenerbahçe 2-1 kazandı.",
        )

    def test_long_think_guidance_prioritizes_evidence_without_chain_of_thought(self):
        prompt = views.depth_instruction(1200).lower()

        self.assertIn("güncel kanıtları değerlendir", prompt)
        self.assertIn("bilinmeyeni ayır", prompt)
        self.assertNotIn("adım adım akıl yürüt", prompt)

    def test_stream_sanitizer_hides_reasoning_split_across_chunks(self):
        sanitizer = views.ModelOutputStreamSanitizer()
        output = "".join(
            sanitizer.feed(chunk)
            for chunk in (
                "Here's a thinking ",
                "process:\\n1. Analyze user input\\n2. Search\\nFinal answer: Verified response.",
            )
        )
        output += sanitizer.feed("", final=True)

        self.assertEqual(output, "Verified response.")

    def test_deep_research_releases_a_prepared_answer_at_the_budget_end(self):
        clock = {"now": 0.0}
        final_chunk = SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="Rapor hazır."),
        )])
        empty_response = SimpleNamespace(choices=[])

        def model_call(*args, **kwargs):
            if kwargs.get("stream"):
                return iter([final_chunk])
            return empty_response

        def monotonic():
            return clock["now"]

        def advance(seconds):
            clock["now"] += seconds

        with (
            patch("dashboard.views.safe_model_call", side_effect=model_call),
            patch("dashboard.views.web_search", return_value={
                "engine": "ddg-html",
                "results": [
                    {"title": "Fenerbahçe Galatasaray maç sonucu", "url": "https://sports.example/match"},
                    {"title": "Fenerbahçe Galatasaray puan durumu", "url": "https://sports.example/table"},
                ],
            }),
            patch("dashboard.views.time.monotonic", side_effect=monotonic),
            patch("dashboard.views.time.sleep", side_effect=advance),
        ):
            answer_times = []
            for kind, _text in views.deep_think_events(
                object(),
                [
                    {"role": "system", "content": "araştır"},
                    {"role": "user", "content": "Fenerbahçe Galatasaray maç sonucu"},
                ],
                30,
                0.3,
                1024,
            ):
                if kind == "answer":
                    answer_times.append(clock["now"])

        self.assertTrue(answer_times)
        self.assertGreaterEqual(answer_times[0], 30)

    def test_current_questions_receive_live_search_context(self):
        chunk = SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="Güncel yanıt.", tool_calls=None),
        )])
        model_call = Mock(return_value=iter([chunk]))

        with (
            patch("dashboard.views.get_gemini_client", return_value=object()),
            patch("dashboard.views.safe_model_call", model_call),
            patch("dashboard.views.web_search", return_value={
                "results": [{"title": "Canlı sonuç", "url": "https://example.com"}],
            }) as search,
        ):
            response = self.client.post(
                reverse("api_chat"),
                data=json.dumps({"message": "Bugünkü maç skoru nedir?"}),
                content_type="application/json",
            )
            b"".join(response.streaming_content)

        search.assert_called()
        self.assertEqual(search.call_args_list[0].kwargs.get("recency_days"), 7)
        sent_messages = model_call.call_args.args[1]
        self.assertIn("Canlı sonuç", sent_messages[1]["content"])

    def test_current_weather_reply_uses_source_value_without_model_rephrasing(self):
        weather = {
            "city": "İstanbul",
            "temperature": 12,
            "feels_like": 10,
            "description": "Parçalı bulutlu",
            "source": "wttr.in",
            "observed_at": "2026-10-05T14:00:00+03:00",
        }

        with (
            patch("dashboard.views.get_chat_endpoints", return_value=[object()]),
            patch("dashboard.views.get_weather", return_value=weather),
            patch("dashboard.views.safe_model_call") as model_call,
        ):
            response = self.client.post(
                reverse("api_chat"),
                data=json.dumps({"message": "İstanbul'da hava kaç derece?"}),
                content_type="application/json",
            )
            answer = b"".join(response.streaming_content).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("12°C", answer)
        self.assertNotIn("19.4", answer)
        model_call.assert_not_called()

    def test_weather_question_with_city_at_the_end_fetches_weather_directly(self):
        weather = {
            "city": "Erdek",
            "temperature": 17,
            "feels_like": 16,
            "description": "Açık",
            "source": "Open-Meteo",
            "observed_at": "2026-10-07T12:00:00+03:00",
        }
        with (
            patch("dashboard.views.get_chat_endpoints", return_value=[object()]),
            patch("dashboard.views.get_weather", return_value=weather) as get_weather,
            patch("dashboard.views.safe_model_call") as model_call,
        ):
            response = self.client.post(
                reverse("api_chat"),
                data=json.dumps({"message": "Hava kaç derece Erdek için?"}),
                content_type="application/json",
            )
            answer = b"".join(response.streaming_content).decode("utf-8")

        get_weather.assert_called_once_with("Erdek")
        self.assertIn("17°C", answer)
        self.assertNotIn("Hangi şehir", answer)
        model_call.assert_not_called()

    def test_sports_followup_without_sources_never_uses_model_memory(self):
        history = [
            {"sender": "user", "text": "Fenerbahçe Eyüpspor maçı kaç kaç bitti?"},
            {"sender": "ai", "text": "8-0; Valencia 2 gol attı."},
        ]
        endpoint = views.Endpoint("test", "key", object(), (views.GEMINI_MODEL,), None)
        with (
            patch("dashboard.views.get_chat_endpoints", return_value=[endpoint]),
            patch("dashboard.views.web_search", return_value={
                "blocked": True,
                "engine": "ddg-html",
                "results": [],
            }) as search,
            patch("dashboard.views.safe_model_call") as model_call,
        ):
            response = self.client.post(
                reverse("api_chat"),
                data=json.dumps({
                    "message": "O nasıl maç, Fenerbahçe ne yapmış öyle?",
                    "history": history,
                }),
                content_type="application/json",
            )
            answer = b"".join(response.streaming_content).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("kaynak bulamadım", answer)
        self.assertNotIn("Valencia", answer)
        self.assertIn("eyupspor", views._ascii_fold(search.call_args_list[0].args[0]).lower())
        model_call.assert_not_called()

    def test_sports_answer_prompt_rejects_unverified_prior_assistant_details(self):
        chunk = SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="Kaynakta doğrulanan skor 8-0.", tool_calls=None),
        )])
        model_call = Mock(return_value=iter([chunk]))
        endpoint = views.Endpoint("test", "key", object(), (views.GEMINI_MODEL,), None)
        history = [
            {"sender": "user", "text": "Fenerbahçe Eyüpspor maçı kaç kaç bitti?"},
            {"sender": "ai", "text": "8-0; Valencia 2 gol attı."},
        ]

        with (
            patch("dashboard.views.get_chat_endpoints", return_value=[endpoint]),
            patch("dashboard.views.web_search", return_value={
                "engine": "ddg-html",
                "results": [{
                    "title": "Fenerbahçe Eyüpspor maç sonucu 8-0",
                    "url": "https://sports.example/result",
                    "snippet": "Fenerbahçe Eyüpspor karşılaşması 8-0 sona erdi.",
                }, {
                    "title": "Fenerbahçe Eyüpspor 8-0 maç raporu",
                    "url": "https://sports.example/report",
                    "snippet": "Resmi maç raporu ve skor bilgisi.",
                }],
            }) as search,
            patch("dashboard.views.safe_model_call", model_call),
        ):
            response = self.client.post(
                reverse("api_chat"),
                data=json.dumps({
                    "message": "O nasıl maç, Fenerbahçe ne yapmış öyle?",
                    "history": history,
                }),
                content_type="application/json",
            )
            b"".join(response.streaming_content)

        normalized_query = views._ascii_fold(search.call_args.args[0]).lower()
        self.assertIn("eyupspor", normalized_query)
        self.assertFalse(search.call_args.kwargs.get("duckduckgo_only"))
        model_messages = model_call.call_args.args[1]
        system_text = "\n".join(
            str(message.get("content") or "")
            for message in model_messages
            if message.get("role") == "system"
        )
        self.assertIn("golcü, oyuncu, dakika", system_text)
        self.assertIn("Önceki asistan yanıtlarını doğrulanmış bilgi sayma", system_text)


class GeminiClientTests(TestCase):
    def test_transient_gemini_error_retries_the_same_model(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            RuntimeError("503 UNAVAILABLE"),
            "completion",
        ]

        with patch("dashboard.views.time.sleep") as sleep:
            result = views.safe_model_call(
                client,
                [{"role": "user", "content": "Merhaba"}],
                "ignored-model-name",
            )

        self.assertEqual(result, "completion")
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertEqual(
            client.chat.completions.create.call_args_list[0].kwargs["model"],
            views.GEMINI_MODEL,
        )
        sleep.assert_called_once_with(1)


class AuthenticationTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="persisted-user",
            email="persisted@example.com",
            password="test-password-123",
        )

    def test_public_debug_endpoint_returns_health_status(self):
        response = self.client.get(reverse("api_debug"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_user_can_log_in_with_case_insensitive_email(self):
        response = self.client.post(
            reverse("login"),
            {"username": "PERSISTED@example.com", "password": "test-password-123"},
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("index"))

    def test_database_session_survives_a_new_client_instance(self):
        response = self.client.post(
            reverse("login"),
            {"username": self.user.username, "password": "test-password-123"},
        )
        self.assertEqual(response.status_code, 302)
        session_key = self.client.cookies["sessionid"].value

        next_client = Client()
        next_client.cookies["sessionid"] = session_key
        page = next_client.get(reverse("index"))

        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "closeDeepThinkSettings()")
        self.assertContains(page, "retryMessage")
        self.assertContains(page, "Araştırma durduruldu.")

    def test_dashboard_scopes_local_cache_to_account_and_trusts_server_chat_list(self):
        self.client.force_login(self.user)
        page = self.client.get(reverse("index"))

        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "DJANGO_USER_EMAIL")
        self.assertContains(page, "DJANGO_USERNAME.toLowerCase()")
        self.assertContains(page, "chats = remoteChats;")
        self.assertNotContains(page, "if (remoteChats.length || !chats.length)")
        self.assertContains(page, "left: 12px;")
