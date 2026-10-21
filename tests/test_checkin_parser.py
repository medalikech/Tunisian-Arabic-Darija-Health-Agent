import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from scheduler.checkin_parser import parse_checkin_reply


class CheckinParserTests(unittest.TestCase):
    def test_recognizes_clear_affirmative_answers(self):
        for text in ("yes", "oui", "5dit", "نعم", "خذيت الدواء"):
            with self.subTest(text=text):
                self.assertEqual(parse_checkin_reply(text), "taken")

    def test_recognizes_clear_missed_answers(self):
        for text in ("no", "non", "nsit", "نسيت ناخذ الدواء", "ما خذيتش الدواء"):
            with self.subTest(text=text):
                self.assertEqual(parse_checkin_reply(text), "missed")

    def test_does_not_match_words_by_substring(self):
        for text in ("nothing", "nobody", "allow", "bahi"):
            with self.subTest(text=text):
                self.assertIsNone(parse_checkin_reply(text))

    def test_contradictory_answers_are_unclear(self):
        for text in (
            "yes, I did not take it",
            "no, I took it",
            "نعم ما خذيتش الدواء",
            "لا خذيت الدواء",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_checkin_reply(text))

    def test_unrecognized_answer_is_unclear(self):
        self.assertIsNone(parse_checkin_reply("I am not sure"))


if __name__ == "__main__":
    unittest.main()
