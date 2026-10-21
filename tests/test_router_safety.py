import sys
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from agents import router
from safety.emergency_responses import (
    DEFAULT_ESCALATION_NOTE,
    EMERGENCY_RESPONSES,
    get_emergency_response,
)


class RouterCheckinSafetyTests(unittest.TestCase):
    def test_emergency_messages_have_no_unverified_numbers_or_placeholders(self):
        responses = [
            get_emergency_response(category)
            for category in EMERGENCY_RESPONSES
        ]

        for response in responses:
            with self.subTest(response=response):
                self.assertNotIn("[", response)
                self.assertNotIn("190", response)
                self.assertIn("خدمات الطوارئ المحلية", response)
                self.assertIn(DEFAULT_ESCALATION_NOTE, response)

    def test_unknown_emergency_category_does_not_default_to_wrong_symptom(self):
        response = get_emergency_response("unknown")

        self.assertIn("إذا الحالة طارئة", response)
        self.assertNotIn("وجع الصدر", response)

    def test_sudden_confusion_response_does_not_assume_hypoglycemia(self):
        response = get_emergency_response("acute_confusion")

        self.assertIn("ما نجمش نحدد سببه", response)
        self.assertIn("خدمات الطوارئ المحلية", response)
        self.assertNotIn("انخفاض خطير في السكر", response)

    def test_sudden_neurological_response_is_urgent_and_cause_neutral(self):
        response = get_emergency_response("acute_neurological")

        self.assertIn("التغيير المفاجئ", response)
        self.assertIn("ما نجمش نحدد سببه", response)
        self.assertIn("خدمات الطوارئ المحلية", response)
        self.assertNotIn("جلطة", response)

    def test_emergency_message_after_checkin_is_escalated(self):
        patient_id = "checkin_emergency_test"
        with (
            patch.object(
                router,
                "get_recent_context",
                return_value=[{"route": "proactive_checkin"}],
            ),
            patch.object(router, "log_dose_event") as log_dose_event,
            patch.object(router, "log_escalation") as log_escalation,
            patch.object(router, "log_interaction") as log_interaction,
        ):
            result = router.handle_message(
                "نحس بوجع قوي في صدري من شوية",
                patient_id=patient_id,
            )

        self.assertEqual(result["route"], "emergency")
        self.assertEqual(result["matched_categories"], ["chest_pain"])
        self.assertTrue(result["classifier_bypassed"])
        log_escalation.assert_called_once_with(
            patient_id,
            "نحس بوجع قوي في صدري من شوية",
            severity="emergency",
            category="chest_pain",
        )
        log_interaction.assert_called_once()
        log_dose_event.assert_not_called()

    def test_emergency_response_is_returned_if_escalation_logging_fails(self):
        with (
            patch.object(
                router,
                "log_escalation",
                side_effect=sqlite3.OperationalError("database unavailable"),
            ),
            patch.object(router, "log_interaction") as log_interaction,
            patch.object(router.logger, "error") as log_error,
        ):
            result = router.handle_message(
                "نحس بوجع قوي في صدري من شوية",
                patient_id="emergency_log_failure_test",
            )

        self.assertEqual(result["route"], "emergency")
        self.assertFalse(result["escalation_logged"])
        self.assertIn("ما سجلش الحالة", result["response"])
        self.assertIn("اتصل بخدمات الطوارئ المحلية", result["response"])
        log_interaction.assert_called_once()
        log_error.assert_called_once()

    def test_emergency_response_is_returned_if_interaction_logging_fails(self):
        with (
            patch.object(router, "log_escalation") as log_escalation,
            patch.object(
                router,
                "log_interaction",
                side_effect=sqlite3.OperationalError("database unavailable"),
            ),
            patch.object(router.logger, "error") as log_error,
        ):
            result = router.handle_message(
                "نحس بوجع قوي في صدري من شوية",
                patient_id="emergency_interaction_failure_test",
            )

        self.assertEqual(result["route"], "emergency")
        self.assertTrue(result["escalation_logged"])
        self.assertFalse(result["interaction_logged"])
        self.assertIn("اتصل بخدمات الطوارئ المحلية", result["response"])
        log_escalation.assert_called_once()
        log_error.assert_called_once()

    def test_non_emergency_checkin_reply_is_still_recorded(self):
        patient_id = "checkin_reply_test"
        with (
            patch.object(
                router,
                "get_recent_context",
                return_value=[{"route": "proactive_checkin"}],
            ),
            patch.object(router, "log_dose_event") as log_dose_event,
            patch.object(router, "log_interaction") as log_interaction,
        ):
            result = router.handle_message("نعم خذيت الدواء", patient_id=patient_id)

        self.assertEqual(result["route"], "checkin_reply")
        log_dose_event.assert_called_once_with(
            patient_id,
            "taken",
            source="checkin_reply",
        )
        log_interaction.assert_called_once()

    def test_unclear_checkin_reply_is_not_recorded_as_adherence(self):
        patient_id = "checkin_unclear_test"
        with (
            patch.object(
                router,
                "get_recent_context",
                return_value=[{"route": "proactive_checkin"}],
            ),
            patch.object(router, "log_dose_event") as log_dose_event,
            patch.object(router, "log_interaction") as log_interaction,
            patch.object(router, "CLASSIFIER") as classifier,
        ):
            result = router.handle_message(
                "yes, I did not take it",
                patient_id=patient_id,
            )

        self.assertEqual(result["route"], "checkin_unclear")
        self.assertIn("نعم", result["response"])
        log_dose_event.assert_not_called()
        log_interaction.assert_called_once()
        classifier.predict.assert_not_called()

    def test_missed_checkin_does_not_give_dose_timing_instruction(self):
        patient_id = "checkin_missed_test"
        with (
            patch.object(
                router,
                "get_recent_context",
                return_value=[{"route": "proactive_checkin"}],
            ),
            patch.object(router, "log_dose_event") as log_dose_event,
            patch.object(router, "log_interaction") as log_interaction,
        ):
            result = router.handle_message("نسيت ناخذ الدواء", patient_id=patient_id)

        self.assertEqual(result["route"], "checkin_reply")
        self.assertIn("الصيدلي", result["response"])
        self.assertNotIn("خذ الدواء توا", result["response"])
        log_dose_event.assert_called_once_with(
            patient_id,
            "missed",
            source="checkin_reply",
        )
        log_interaction.assert_called_once()

    def test_missed_dose_intent_uses_neutral_pharmacist_guidance(self):
        patient_id = "missed_dose_guidance_test"
        with (
            patch.object(router, "get_recent_context", return_value=[]),
            patch.object(router, "log_dose_event") as log_dose_event,
            patch.object(router, "log_interaction") as log_interaction,
            patch.object(router.CLASSIFIER, "predict", return_value=["missed_dose"]),
            patch.object(
                router.CLASSIFIER,
                "predict_proba",
                return_value=[[0.9, 0.1]],
            ),
        ):
            result = router.handle_message(
                "I missed my medication",
                patient_id=patient_id,
            )

        self.assertEqual(result["route"], "fixed_response")
        self.assertIn("الصيدلي", result["response"])
        self.assertNotIn("الجرعة التالية في وقتها العادي", result["response"])
        log_dose_event.assert_called_once_with(
            patient_id,
            "missed",
            source="conversation",
        )
        log_interaction.assert_called_once()


if __name__ == "__main__":
    unittest.main()
