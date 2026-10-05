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
        self.assertIn("8-0", normalized[0])
        self.assertIn("golleri kim atti", normalized[0])
        self.assertIn("fenerbahce", normalized[1])
        self.assertNotIn("dogrulanmamis", " ".join(normalized))

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
        self.assertFalse(
            views.should_retry_live_search_with_general_results(
                "Merhaba nasıl gidiyor?",
                {"blocked": True, "engine": "ddg-html", "results": []},
            )
        )


class AppClockTests(TestCase):
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

    def test_deep_research_stops_honestly_when_duckduckgo_blocks_access(self):
        with (
            patch("dashboard.views.web_search", return_value={"blocked": True, "results": []}),
            patch("dashboard.views.safe_model_call") as model_call,
        ):
            events = list(views.deep_think_events(
                object(),
                [
                    {"role": "system", "content": "araştır"},
                    {"role": "user", "content": "Fenerbahçe Galatasaray maç sonucu"},
                ],
                300,
                0.3,
                1024,
            ))

        model_call.assert_not_called()
        answer = "".join(text for kind, text in events if kind == "answer")
        self.assertIn("doğrulanabilir web kaynaklarına erişemedim", answer)

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

        def fake_search(query, num_results, *, duckduckgo_only=False, deadline=None):
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
        chunk = SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="Kaynakta belirtilen golcü bilgisi.", tool_calls=None),
        )])
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
            },
        ]}

        with (
            patch("dashboard.views.get_chat_endpoints", return_value=[endpoint]),
            patch("dashboard.views.web_search", return_value=search_result) as search,
            patch("dashboard.views.safe_model_call", return_value=iter([chunk])) as model_call,
        ):
            response = self.client.post(
                reverse("api_chat"),
                data=json.dumps({
                    "message": "Golleri kim attı?",
                    "history": history,
                }),
                content_type="application/json",
            )
            b"".join(response.streaming_content)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(search.call_count, 3)
        self.assertIn("golleri kim attı", search.call_args_list[0].args[0].lower())
        self.assertIn("gol atan oyuncular", search.call_args_list[1].args[0].lower())
        self.assertTrue(all(call.kwargs["duckduckgo_only"] for call in search.call_args_list))
        sent_context = model_call.call_args.args[1][1]["content"]
        self.assertIn("Fenerbahçe Eyüpspor maç raporu ve golleri", sent_context)

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

        search.assert_called_once()
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

    def test_sports_followup_fails_closed_when_ddg_cannot_verify_match_details(self):
        history = [
            {"sender": "user", "text": "Fenerbahçe Eyüpspor maçı kaç kaç bitti?"},
            {"sender": "ai", "text": "8-0; Valencia 2 gol attı."},
        ]
        with (
            patch("dashboard.views.get_chat_endpoints", return_value=[object()]),
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
        self.assertIn("doğrulayamadım", answer)
        self.assertNotIn("Valencia", answer)
        self.assertIn("eyupspor", views._ascii_fold(search.call_args.args[0]).lower())
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
        self.assertTrue(search.call_args.kwargs["duckduckgo_only"])
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
