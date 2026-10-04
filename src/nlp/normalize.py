# src/nlp/normalize.py
import re, unicodedata

DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = DIACRITICS.sub("", text).replace("\u0640", "")   # tashkeel, tatweel
    text = re.sub("[إأآٱ]", "ا", text)                      # unify alef forms
    text = text.replace("ى", "ي")
    text = text.translate(AR_DIGITS)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)              # "barchaaaa" -> "barchaa"
    return re.sub(r"\s+", " ", text).strip()