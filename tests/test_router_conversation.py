import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from agents import router


class RouterConversationTests(unittest.TestCase):
    def test_second_soft_escalation_uses_followup_response(self):
        patient_id = "soft_escalation_followup_test"
        prior_turn = {
            "raw_text": "نحس بدوخة",
            "response": "possible-emergency prompt",
            "route": "soft_escalation",
        }
        with (
            patch.object(
                router,
                "scan_for_red_flags",
                return_value=[],
            ),
            patch.object(
                router,
                "get_recent_context",
                side_effect=[
                    [],
                    [],
                    [],
                    [prior_turn],
                    [prior_turn],
                    [prior_turn],
                ],
            ),
            patch.object(
                router.CLASSIFIER,
                "predict",
                return_value=["red_flag_emergency"],
            ),
            patch.object(
                router.CLASSIFIER,
                "predict_proba",
                return_value=[[0.9, 0.1]],
            ),
            patch.object(router, "log_escalation") as log_escalation,
            patch.object(router, "log_interaction"),
        ):
            first = router.handle_message("نحس بدوخة", patient_id=patient_id)
            second = router.handle_message("مازالت مستمرة", patient_id=patient_id)

        self.assertEqual(first["route"], "soft_escalation")
        self.assertEqual(second["route"], "soft_escalation")
        self.assertNotEqual(first["response"], second["response"])
        self.assertEqual(second["response"], router.SOFT_ESCALATION_FOLLOWUP)
        self.assertEqual(log_escalation.call_count, 2)


if __name__ == "__main__":
    unittest.main()
