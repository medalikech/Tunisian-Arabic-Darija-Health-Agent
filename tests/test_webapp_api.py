import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from webapp import auth
from webapp.app import app


class WebAppApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.auth_db_patcher = patch.object(
            auth,
            "AUTH_DB_PATH",
            Path(self.temp_dir.name) / "auth.db",
        )
        self.auth_db_patcher.start()
        self.addCleanup(self.auth_db_patcher.stop)
        auth.initialize_auth_db()
        auth.create_user("patient-one", "patient-password-123", "patient", "patient-001")
        auth.create_user("patient-two", "patient-password-456", "patient", "patient-002")
        auth.create_user("staff-one", "staff-password-123", "staff")
        auth.assign_patient_to_staff("staff-one", "patient-001")

        self.original_secret = app.config["SECRET_KEY"]
        app.config["SECRET_KEY"] = "test-only-session-secret"
        self.addCleanup(
            lambda: app.config.__setitem__("SECRET_KEY", self.original_secret)
        )
        self.client = app.test_client()
        self.login("patient-one", "patient-password-123")

    def login(self, username, password):
        csrf_response = self.client.get("/api/auth/csrf")
        csrf_token = csrf_response.get_json()["csrf_token"]
        response = self.client.post(
            "/api/auth/login",
            json={"username": username, "password": password},
            headers={"X-CSRF-Token": csrf_token},
        )
        if response.status_code == 200:
            self.csrf_token = response.get_json()["csrf_token"]
        return response

    def post_chat(self, **kwargs):
        headers = kwargs.pop("headers", {})
        headers["X-CSRF-Token"] = self.csrf_token
        return self.client.post("/api/chat", headers=headers, **kwargs)

    def test_html_and_api_responses_have_privacy_security_headers(self):
        responses = [
            self.client.get("/"),
            self.client.get("/api/auth/csrf"),
            self.post_chat(json={"message": "Hello"}),
        ]

        for response in responses:
            with self.subTest(path=response.request.path):
                self.assertEqual(response.headers["Cache-Control"], "no-store")
                self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
                self.assertEqual(response.headers["X-Frame-Options"], "DENY")
                self.assertEqual(response.headers["Referrer-Policy"], "no-referrer")
                self.assertEqual(
                    response.headers["Permissions-Policy"],
                    "camera=(), microphone=(), geolocation=()",
                )
                self.assertIn(
                    "frame-ancestors 'none'",
                    response.headers["Content-Security-Policy"],
                )

    @patch("webapp.app.handle_message")
    def test_chat_returns_agent_response(self, handle_message):
        handle_message.return_value = {
            "response": "Test response",
            "route": "fixed_response",
            "intent": "greeting_smalltalk",
            "confidence": 0.9,
        }

        response = self.post_chat(json={"message": "Hello"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["response"], "Test response")
        handle_message.assert_called_once_with("Hello", patient_id="patient-001")

    @patch("webapp.app.handle_message")
    def test_chat_rejects_empty_message(self, handle_message):
        response = self.post_chat(json={"message": "  "})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "empty message"})
        handle_message.assert_not_called()

    def test_chat_rejects_non_object_json(self):
        response = self.post_chat(json=["Hello"])

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(),
            {"error": "request body must be a JSON object"},
        )

    @patch("webapp.app.handle_message")
    def test_chat_rejects_malformed_json(self, handle_message):
        response = self.post_chat(
            data='{"message":',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(),
            {"error": "request body must be a JSON object"},
        )
        handle_message.assert_not_called()

    def test_chat_rejects_non_string_message(self):
        response = self.post_chat(json={"message": ["Hello"]})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "message must be a string"})

    def test_chat_rejects_non_string_patient_id(self):
        response = self.post_chat(
            json={"message": "Hello", "patient_id": 123},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(),
            {"error": "patient_id must be a string"},
        )

    @patch("webapp.app.handle_message", side_effect=RuntimeError("sensitive internal detail"))
    def test_chat_does_not_return_internal_exception_details(self, handle_message):
        with self.assertLogs("webapp.app", level="ERROR") as logs:
            response = self.post_chat(json={"message": "Hello"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"], "internal server error")
        self.assertTrue(response.get_json()["request_id"])
        self.assertNotIn("sensitive internal detail", response.get_data(as_text=True))
        self.assertIn("RuntimeError", "\n".join(logs.output))
        self.assertNotIn("sensitive internal detail", "\n".join(logs.output))

    def test_chat_requires_authentication(self):
        response = app.test_client().post("/api/chat", json={"message": "Hello"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.get_json(), {"error": "authentication required"})

    def test_patient_cannot_impersonate_another_patient(self):
        response = self.post_chat(
            json={"message": "Hello", "patient_id": "patient-002"}
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.get_json(), {"error": "patient access denied"})

    def test_chat_requires_csrf_token(self):
        response = self.client.post("/api/chat", json={"message": "Hello"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "invalid or missing CSRF token"})

    def test_login_rejects_invalid_credentials(self):
        csrf_token = self.client.get("/api/auth/csrf").get_json()["csrf_token"]
        response = self.client.post(
            "/api/auth/login",
            json={"username": "patient-one", "password": "wrong-password"},
            headers={"X-CSRF-Token": csrf_token},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.get_json(), {"error": "invalid username or password"})

    def test_login_temporarily_locks_account_after_repeated_failures(self):
        csrf_token = self.client.get("/api/auth/csrf").get_json()["csrf_token"]
        login_url = "/api/auth/login"

        for _ in range(auth.MAX_LOGIN_ATTEMPTS):
            response = self.client.post(
                login_url,
                json={"username": "patient-one", "password": "wrong-password"},
                headers={"X-CSRF-Token": csrf_token},
            )
            self.assertEqual(response.status_code, 401)
            self.assertEqual(
                response.get_json(),
                {"error": "invalid username or password"},
            )

        response = self.client.post(
            login_url,
            json={"username": "patient-one", "password": "patient-password-123"},
            headers={"X-CSRF-Token": csrf_token},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(
            response.get_json(),
            {"error": "invalid username or password"},
        )

    def test_authentication_endpoints_require_a_session_secret(self):
        original_secret = app.config["SECRET_KEY"]
        app.config["SECRET_KEY"] = None
        try:
            response = app.test_client().get("/api/auth/csrf")
        finally:
            app.config["SECRET_KEY"] = original_secret

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json(), {"error": "authentication is not configured"})

    @patch("webapp.app.handle_message")
    def test_staff_can_chat_only_with_assigned_patient(self, handle_message):
        handle_message.return_value = {
            "response": "Test response",
            "route": "fixed_response",
            "intent": "greeting_smalltalk",
            "confidence": 0.9,
        }
        self.login("staff-one", "staff-password-123")
        response = self.post_chat(
            json={"message": "Hello", "patient_id": "patient-001"}
        )

        self.assertEqual(response.status_code, 200)
        handle_message.assert_called_once_with("Hello", patient_id="patient-001")

    def test_staff_cannot_chat_with_unassigned_patient(self):
        self.login("staff-one", "staff-password-123")
        response = self.post_chat(
            json={"message": "Hello", "patient_id": "patient-002"}
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.get_json(), {"error": "patient access denied"})

    def test_patient_cannot_list_staff_patient_assignments(self):
        response = self.client.get("/api/patients")

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.get_json(), {"error": "staff account required"})

    def test_logout_clears_authenticated_session(self):
        response = self.client.post(
            "/api/auth/logout",
            headers={"X-CSRF-Token": self.csrf_token},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"logged_out": True})
        self.assertEqual(
            self.client.get("/api/auth/me").get_json(),
            {"error": "authentication required"},
        )


if __name__ == "__main__":
    unittest.main()
