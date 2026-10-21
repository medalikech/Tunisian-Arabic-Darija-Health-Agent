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
        r"(?:مشكل|صعوبة|صعيب).{0,15}(?:تنفس|يتنفس|التنفس)",
        r"(?:ما|مش).{0,12}(?:تنجمش|تقدرش|ينجمش|يقدرش).{0,12}(?:تتنفس|يتنفس|تنفس)",
        r"صدري.{0,10}مسكر",
        r"(difficult|arrive plus).{0,20}respir",
        r"(?:a du mal|trouble|difficulty).{0,20}breath",
        r"a du mal.{0,20}respir",
        r"essoufflement",
        r"خنقة",
        r"ma(ne9der|ye9der).{0,15}n(fes|tnefes)",
        r"(sedri|sadri).{0,10}(mesker|msakkar)",
        r"ما.{0,10}(نقدرش|نلحقش).{0,10}نتنفس",
        r"(?:مشكل|صعوبة|صعيب).{0,15}(?:تنفس|يتنفس|التنفس)",
        r"(?:ما|مش).{0,12}(?:تنجمش|تقدرش|ينجمش|يقدرش).{0,12}(?:تتنفس|يتنفس|تنفس)",
        r"صدري.{0,10}مسكر",
        r"خناق.{0,15}صدر",                       # NEW: "choking" feeling in chest
        r"(difficult|arrive plus).{0,20}respir",
        r"(?:a du mal|trouble|difficulty).{0,20}breath",
        r"a du mal.{0,20}respir",
        r"essoufflement",
        r"خنقة",
    ],
    "hypoglycemia": [
        r"(rara3|rar3a|rou3ech).{0,15}(bared|3ar9)",
        r"عرق.{0,10}بارد",
        r"رعشة",
        r"ترعش",
        r"tremble.{0,20}(sue|sueur|froid)",
        r"glyc[ée]mie.{0,10}(0\.[0-5]|chut[ée])",
        r"(?:blood sugar|glucose).{0,25}(?:very low|dropped|crashed)"
        r".{0,35}(?:faint|passed out|unconscious|can't swallow|cannot swallow)",
        r"(?:faint|passed out|unconscious).{0,35}"
        r"(?:blood sugar|glucose).{0,15}(?:low|dropped)",
        r"(?:glyc[ée]mie|hypoglyc[ée]mie).{0,25}"
        r"(?:malaise|[ée]vanou|perdu connaissance|n'arrive pas [àa] avaler)",
        r"(?:malaise|[ée]vanou|perdu connaissance).{0,35}"
        r"(?:glyc[ée]mie basse|hypoglyc[ée]mie|glyc[ée]mie.{0,15}chut[ée])",
        r"(?:السكر|سكر الدم|سكري).{0,18}(?:هبط|طيح).{0,30}"
        r"(?:و)?(?:اغمي|فقدت الوعي|ما نجمش نبلع|ما نقدرش نبلع)",

    ],
    "acute_confusion": [
        r"\bconfusion.{0,15}soudain(?:e)?\b",
        r"\bsudden.{0,15}confusion\b",
        r"\bsuddenly.{0,15}confused\b",
        r"\b(?:new|sudden).{0,15}disorientation\b",
        r"(?:ارتباك|تخليط|تشوش).{0,10}(?:مفاجئ|فجأة)",
        r"(?:فجاة|بغتة).{0,20}(?:تلخبط|تشوش|مشوش|مرتبك)",
    ],
    "acute_neurological": [
        r"\b(?:sudden|suddenly).{0,25}\b(?:weakness|weak|slurred speech|"
        r"can't speak|cannot speak|difficulty speaking)\b",
        r"\b(?:face|facial).{0,20}\bdroop(?:ing)?\b",
        r"\bdroop(?:ing)?.{0,20}\bface\b",
        r"\b(?:sudden|suddenly).{0,25}\b(?:numbness|numb)\b"
        r".{0,20}\b(?:one side|left arm|right arm|left leg|right leg)\b",
        r"\bweakness.{0,20}\b(?:one side|left arm|right arm|left leg|right leg)\b",
        r"visage.{0,25}(?:tombe|paralys|deform)",
        r"(?:parole|parler).{0,25}(?:soudain|difficile|bizarre|articuler)",
        r"(?:ضعف|ضعفت|تنميل).{0,20}(?:جهة|يد|ذراع).{0,15}(?:فجاة|بغتة)",
        r"(?:وجهي|الوجه).{0,20}(?:مال|هابط|طاح).{0,15}(?:جهة|فجاة|بغتة)",
        r"(?:كلامي|نحكي|الكلام).{0,20}(?:ثقيل|موش واضح|ما نجمش).{0,15}"
        r"(?:فجاة|بغتة)",
        r"(?:فجاة|بغتة).{0,20}(?:وجهي|الوجه).{0,15}(?:مال|هابط|طاح)",
        r"(?:يد|ذراع).{0,20}(?:ضعف|ضعفت|ضعيفة).{0,15}(?:فجاة|بغتة)",
        r"(?:فجاة|بغتة).{0,20}(?:ما نجمش|ما نقدرش).{0,12}"
        r"(?:نحكي|نتكلم|نهدر)",
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
        r"\b(?:i\s+)?(?:accidentally\s+)?(?:took|swallowed|ingested)\b"
        r".{0,35}\b(?:twice|double|extra|too many|too much|more than prescribed)\b"
        r".{0,25}\b(?:dose|pills?|tablets?|medication|medicine)\b",
        r"\b(?:swallowed|ingested)\b.{0,30}\b(?:several|multiple|many)\b"
        r".{0,20}\b(?:pills?|tablets?|capsules?)\b",
        r"(?:غلطت|بالغلط).{0,15}(?:خذيت|خديت).{0,15}"
        r"(?:زوز|اثنين|برشا).{0,12}(?:حبات|حبوب|اقراص)",
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