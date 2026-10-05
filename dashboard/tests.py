import base64
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.contrib.auth.models import User
from django.test import Client, SimpleTestCase, TestCase
from django.urls import reverse
from django.utils import timezone

from . import views
from .models import AppClock


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

        with patch("dashboard.views.requests.get", side_effect=[geocode, current]) as get:
            result = views.get_weather("Erdek")

        self.assertEqual(result["city"], "Erdek")
        self.assertEqual(result["temperature"], 16.7)
        self.assertEqual(result["description"], "Parçalı bulutlu")
        self.assertEqual(get.call_count, 2)
        self.assertEqual(
            get.call_args_list[1].kwargs["params"]["current"],
            "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
        )


class SearchTests(SimpleTestCase):
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


class AppClockTests(TestCase):
    def test_clock_persists_the_current_istanbul_date(self):
        clock = views.sync_app_clock()

        self.assertEqual(clock.current_date, timezone.localdate())
        self.assertEqual(AppClock.objects.count(), 1)


class ImageGenerationTests(TestCase):
    def setUp(self):
        views._last_image_request_at.clear()
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

    def test_image_generation_never_falls_back_to_a_branded_provider(self):
        with (
            patch("dashboard.views.ordered_keys", return_value=["test-key"]),
            patch(
                "dashboard.views.generate_image_with_gemini",
                side_effect=RuntimeError("unavailable"),
            ) as gemini,
            patch("dashboard.views.generate_image_with_pollinations") as pollinations,
        ):
            with self.assertRaises(views.ImageUnavailableError):
                views.generate_image_with_retry("Bir dağ manzarası")

        self.assertEqual(
            [call.args[1] for call in gemini.call_args_list],
            [
                "gemini-3-pro-image",
                "gemini-3.1-flash-image",
                "gemini-2.5-flash-image",
            ],
        )
        pollinations.assert_not_called()

    def test_explicit_illustration_style_is_not_forced_into_photorealism(self):
        prompt = views.enhance_image_prompt("Bir çizim: kedi")

        self.assertIn("çizim", prompt)
        self.assertNotIn("photorealistic", prompt)

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
        self.assertIn("DuckDuckGo kaynaklarına erişemedim", answer)

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