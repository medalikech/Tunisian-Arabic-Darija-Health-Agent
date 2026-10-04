# src/safety/red_flags.py
import re
from nlp.normalize import normalize

# Each category maps to patterns that, if matched, trigger escalation.
# Patterns are checked against NORMALIZED text (see normalize.py from Step 1).
# Keep these broad/high-recall on purpose: false positives here are much
# cheaper than false negatives. A false positive means an unnecessary
# "are you okay, here's emergency info" message. A false negative means
# a real emergency got routed to a chatbot instead of help.

RED_FLAG_PATTERNS = {
    "chest_pain": [
        r"wja3.{0,15}(sedri|sadri|9alb)",
        r"(sedri|sadri).{0,15}(wja3|y7re9|ythe9el)",
        r"وجع.{0,15}صدر",
        r"صدري.{0,15}(يوجعني|يحرق|ثقيل)",
        r"douleur.{0,20}(poitrine|thoracique)",
        r"pression.{0,15}poitrine",
            r"wja3.{0,15}(sedri|sadri|9alb)",
        r"(sedri|sadri).{0,15}(wja3|y7re9|ythe9el)",
        r"وجع.{0,15}صدر",
        r"صدري.{0,15}(يوجعني|يحرق|ثقيل|ضغط)",
        r"ضغط.{0,10}(قوي|كبير).{0,15}صدر",     # NEW: pressure in chest
        r"douleur.{0,20}(poitrine|thoracique)",
        r"pression.{0,15}poitrine",
    ],
    "breathing": [
        r"ma(ne9der|ye9der).{0,15}n(fes|tnefes)",
        r"(sedri|sadri).{0,10}(mesker|msakkar)",
        r"ما.{0,10}(نقدرش|نلحقش).{0,10}نتنفس",
        r"صدري.{0,10}مسكر",
        r"(difficult|arrive plus).{0,20}respir",
        r"essoufflement",
        r"خنقة",
        r"ma(ne9der|ye9der).{0,15}n(fes|tnefes)",
        r"(sedri|sadri).{0,10}(mesker|msakkar)",
        r"ما.{0,10}(نقدرش|نلحقش).{0,10}نتنفس",
        r"صدري.{0,10}مسكر",
        r"خناق.{0,15}صدر",                       # NEW: "choking" feeling in chest
        r"(difficult|arrive plus).{0,20}respir",
        r"essoufflement",
        r"خنقة",
    ],
    "hypoglycemia": [
        r"(rara3|rar3a|rou3ech).{0,15}(bared|3ar9)",
        r"عرق.{0,10}بارد",
        r"رعشة",
        r"ترعش",
        r"tremble.{0,20}(sue|sueur|froid)",
        r"confusion.{0,15}soudain",
        r"glyc[ée]mie.{0,10}(0\.[0-5]|chut[ée])",
        
    ],
    "suicidal": [
        r"ma3adch.{0,15}n7eb.{0,10}n3ich",
        r"ma3andich.{0,15}fayda.{0,15}n3ich",
        r"نحب.{0,10}(نخلص|نرتاح|نموت)",
        r"ما\s*عادش.{0,15}(نحب|نلقى).{0,15}(نعيش|معنى)",
        r"ما\s*عدش.{0,15}(نحب|نلقى).{0,15}(نعيش|معنى)",
        r"n'en peux plus.{0,20}(finir|arr[êe]ter)",
        r"plus rien.{0,10}sens",
        r"pens[ée]es? noires?",
        r"envie.{0,10}(mourir|disparaitre|en finir)",
        r"je veux (mourir|en finir)",
        r"ma3adch.{0,15}n7eb.{0,10}n3ich",
        r"ma3andich.{0,15}fayda.{0,15}n3ich",
        r"نحب.{0,10}(نخلص|نرتاح|نموت)",
        r"ماعادش.{0,15}(نحب|نلقى).{0,15}(نعيش|معنى)",
        r"ما\s*عادش.{0,15}(نحب|نلقى).{0,15}(نعيش|معنى)",
        r"ما\s*عدش.{0,15}(نحب|نلقى).{0,15}(نعيش|معنى)",
        r"ما\s*عند[يی]ش.{0,15}سبب.{0,15}نكمل",      # NEW: "no reason to continue"
        r"نرتاح.{0,10}من.{0,10}كل\s*شي",             # NEW: euphemistic "rest from everything"
        r"n'en peux plus.{0,20}(finir|arr[êe]ter)",
        r"plus rien.{0,10}sens",
        r"pens[ée]es? noires?",
        r"envie.{0,10}(mourir|disparaitre|en finir)",
        r"je veux (mourir|en finir)",
    ],
    "overdose": [
        r"5dit.{0,15}jor3a.{0,10}zayda",
        r"5dit.{0,15}dwa.{0,10}marrtein",
        r"خذيت.{0,15}جرعة.{0,10}(زايدة|مضاعفة)",
        r"خذيت.{0,15}الدواء.{0,10}مرتين",
        r"(double|trop).{0,15}(dose|comprim[ée]s?)",
        r"overdose",
        r"avale.{0,15}(plusieurs|trop).{0,15}comprim",
        r"5dit.{0,15}jor3a.{0,10}zayda",
        r"5dit.{0,15}dwa.{0,10}marrtein",
        r"خذيت.{0,15}جرعة.{0,10}(زايدة|مضاعفة)",
        r"خذيت.{0,15}الدواء.{0,10}(مرتين|زوز\s*مرات)",   # UPDATED: added "زوز مرات"
        r"(double|trop).{0,15}(dose|comprim[ée]s?)",
        r"overdose",
        r"avale.{0,15}(plusieurs|trop).{0,15}comprim",
    ],
}

def scan_for_red_flags(raw_text: str) -> list[str]:
    """
    Returns a list of matched red-flag categories (can be more than one).
    Empty list = no red flag detected.
    """
    text = normalize(raw_text)
    matches = []
    for category, patterns in RED_FLAG_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                matches.append(category)
                break  # one match per category is enough
    return matches