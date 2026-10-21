# Safety Audit — Darija Health Agent

This document consolidates every safety-relevant finding, test result, and known limitation discovered during development. It is written to be read on its own, by someone who did not build the system, and to be honest about weaknesses rather than presenting only favorable results.

**Scope reminder:** this system is a learning project for a type 2 diabetes / oral medication (metformin) support agent in Tunisian Arabic/Darija. It has not been clinically validated and is not approved for real patient use. Nothing in this document should be read as a claim of medical safety sufficient for deployment.

---

## 1. Safety architecture overview

The system uses a **defense-in-depth** design: a deterministic, rule-based layer is checked first on every message and can never be bypassed by a model's uncertainty or failure. Machine-learned components (the intent classifier, the RAG answer generator) only run after the rule-based check has cleared a message as non-urgent.

```
Message -> [1] Rule-based red-flag scanner (deterministic, always first)
              |
              +-- match --> fixed emergency response, nothing else runs
              |
              v (no match)
           [2] ML intent classifier (confidence-gated)
              |
              +-- very low confidence --> clarifying question
              +-- predicts red_flag_emergency --> soft escalation (secondary signal)
              +-- otherwise --> fixed response / RAG / unhandled placeholder
```

This design choice was deliberate: statistical components in this project were repeatedly shown to be unreliable in isolation (see Sections 2 and 4 below), so the safety-critical decision — "is this an emergency" — does not depend on them being correct.

---

## 2. Red-flag detection: test results and limitations

### 2.1 What it covers
Seven categories: chest pain, breathing difficulty, hypoglycemia symptoms, sudden confusion, sudden neurological changes, suicidal ideation, and overdose/accidental double-dosing. Detection is regex-based over normalized Arabic-script, Arabizi, French, and selected English text.

### 2.2 Test results
The scanner currently passes **69 hand-written examples** in `tests/test_red_flags.py`, covering selected Arabic-script, Arabizi, French, and English inputs, positive and negative examples, third-person breathing reports, accidental extra-medication reports, severe low-blood-sugar reports involving fainting or inability to swallow, and selected cause-neutral sudden-confusion and sudden-neurological-change reports.

This is a code regression suite, not a clinical performance estimate. It is small, not independently authored or validated, and does not represent the range of real patient language. A past-tense, resolved chest-pain report is expected to trigger an alert:

| Input | Expected | Actual | Limitation |
|---|---|---|---|
| "العام الفارط كان عندي وجع في صدري بس توة ولا باهي" ("last year I had chest pain but I'm fine now") | Chest-pain alert | Chest-pain alert | The scanner does not model tense or recency. This conservative over-trigger is retained, but does not prove the scanner will detect current emergencies. |

### 2.3 Known limitations of the rule-based layer
- **No tense/recency modeling** — past or resolved symptoms can trigger a false positive (see 2.2).
- **Limited subject and no recency modeling** — the scanner now recognizes selected current breathing-emergency reports about another person in Darija, French, and English. Coverage is incomplete; it does not establish who is affected or whether a symptom is current, and can alert on resolved historical symptoms.
- **Keyword/pattern coverage is necessarily incomplete.** Darija has no standard spelling; new phrasings not anticipated during development will not be caught. The pattern bank should be treated as a living document requiring ongoing expansion from real usage, not a finished artifact.
- **No severity grading within a category.** A mild and a severe chest pain statement are both routed to the same fixed emergency response. This is intentional (simplicity, conservatism) but means the system cannot currently triage within an emergency category.

---

## 3. Intent classification: capabilities and limitations

### 3.1 Final performance
TF-IDF (character n-gram) + Logistic Regression, trained on an expanded and reviewed Tunisian Darija dataset (~2,800 training examples across 11 merged intent classes):

- Macro F1: **0.67** (merged red-flag class), accuracy 0.65
- A fine-tuned transformer (CamelBERT-DA) was compared and performed worse overall (macro F1 0.58) but notably *better* specifically on the merged emergency class (0.71 vs 0.60) — documented as a finding, not currently operationalized in production.

### 3.2 Why red-flag sub-types are merged for classification
Individual red-flag sub-types (chest pain, breathing, hypoglycemia, overdose, suicidal) were found to be frequently confused with each other by the classifier (confirmed via confusion matrix analysis) — this reflects genuine, inherent vocabulary overlap (e.g. hypoglycemia symptoms share language with both breathing difficulty and suicidal-sounding distress) rather than a fixable data problem. **The classifier's job is intentionally coarse** ("is this urgent, yes or no") — fine-grained sub-type routing for the *fixed* emergency response text is handled by the rule-based scanner, not the classifier. This is the correct division of labor given the evidence.

### 3.3 The `emotional_support` / `red_flag_suicidal` overlap
Confusion matrix analysis showed substantial bidirectional confusion between these two intents (up to 9/21 examples in early testing). This reflects a genuine linguistic and clinical ambiguity: expressions of hopelessness or sadness and expressions of suicidal ideation can look very similar in short text, even to a careful human reader. **The system is deliberately biased toward the safer (escalation) interpretation when ambiguous** — this was a conscious design decision, not an accepted defect, and is documented here so it is not mistaken for one.

### 3.4 Ablation finding (data quality vs. data volume)
A controlled ablation study (`src/eval/ablation_study.py`, leakage-corrected by seed group) found that:
- Expanding from hand-written seeds to LLM-generated synthetic data roughly **doubled** macro F1 (0.27 → 0.50 on held-out test data).
- Manual review of generated data and dialectal-drift filtering produced **no measurable change** in classifier accuracy, overall or on red-flag classes specifically, despite being significant manual effort.
- **Implication:** these quality-control steps remain valuable for data correctness and authenticity (not reflected in this metric), but should not be assumed to improve classifier accuracy — a useful, non-obvious finding for anyone extending this work.

---

## 4. RAG (retrieval-augmented generation) reliability

### 4.1 The core problem
The local LLM used for generation (Qwen2.5-14B via Ollama) was found, across extensive testing, to intermittently corrupt its own output mid-generation when producing Tunisian Darija — switching to French, Chinese, Cyrillic, or garbled pseudo-text partway through an otherwise-correct Arabic response. This was tested across multiple mitigation attempts:

| Mitigation attempted | Result |
|---|---|
| Simple, single, clear prompt instruction | Best of the prompt-only attempts |
| Repeated/urgent language instructions (3x) | Made things worse — caused a Chinese-language leak |
| Shorter max-length instruction | Made things worse — raised failure rate |
| Lower temperature, top_p, repeat_penalty, hard token cap | No meaningful change to failure rate |

**Conclusion, stated plainly: prompt and sampling-parameter engineering could not reliably fix this.** Measured real generation success rate after all tuning: **60%** (10-question test, `src/rag/eval_reliability.py`).

### 4.2 The mitigation that actually worked
Rather than continuing to chase generation reliability, the system has prepared Darija fallback text associated with knowledge-base entries, used when generation returns a technical-failure response. This is explicitly a form of hardcoding — not a fix to the underlying LLM. The fallback text has not thereby been clinically verified.

- In one 10-question development run, the report recorded a 40% fallback rate.
- That run did not evaluate medical correctness, completeness, or appropriateness for an individual patient.
- Script/language validation and lexical overlap checks can reject some malformed output, but they do not establish that generated or fallback text is medically safe.

No zero-unsafe-output or 100%-correct-answer claim is supported by these checks. The measured generation and fallback rates are small-sample development results only.

### 4.3 Output validation specifics
Every generated answer is checked before being shown:
- Rejected if it contains CJK or Cyrillic script (catches cross-language leaks).
- Rejected if non-Arabic-script characters exceed 50% of total letters (catches French/English takeover).
- Rejected if Latin letters are found glued directly onto Arabic letters mid-word (catches subtler token-level corruption that the above checks miss).

This validator has real limits — it cannot catch output that is grammatically or factually wrong but script-valid (e.g. the "حرام تحرك راسك كثير" phrasing noted during multi-turn testing, which passed validation despite being slightly awkward Darija).

---

## 5. Multi-turn conversation issues found and fixed

### 5.1 Fixed-response follow-up blindness (found via interactive testing)
Early versions of the system repeated the same canned question (e.g. "are your symptoms severe or ongoing?") even after the patient had directly answered it, because fixed responses were static templates with no awareness of conversation history. **Fixed** by detecting when the previous bot message was the same template and switching to an acknowledgment response. Verified working across multiple turns.

### 5.2 Soft-escalation follow-up blindness (found via human evaluation)
The same category of bug was independently present in the `soft_escalation` path (classifier-flagged possible emergencies) and was **missed by earlier automated testing**, because the red-flag examples only exercised single-turn detection. It was caught by a human reviewer scoring real logged conversations, who rated a repeated-question response 1/5 for correctness. **Fixed** using the same pattern as 5.1.

**This is the most important process finding in this audit:** automated test suites, however thorough, test what they were designed to test. A single small human evaluation pass (8 of 26 sampled conversations reviewed) found a real, user-facing coherence bug that months of automated testing did not surface, because no automated test checked multi-turn coherence in that specific path. **Human evaluation should be treated as a required, not optional, part of evaluating a system like this**, and this project's current human-eval coverage (8 examples, single reviewer) should be considered a minimum, not sufficient.

---

## 6. Human evaluation summary (partial, preliminary)

8 of 26 sampled real conversations were scored by one reviewer on a 1-5 scale:

| Dimension | Average score | Interpretation |
|---|---|---|
| Naturalness | **2.12 / 5** | Low. Corroborates the dialectal-drift concern raised throughout development (Qwen2.5 and Llama 3.1 both showed a tendency to drift toward Levantine/Egyptian/MSA vocabulary and grammar despite explicit Tunisian-dialect prompting) with independent, credible human judgment rather than only automated keyword detection. |
| Correctness | 3.25 / 5 | Mixed. One `soft_escalation` example scored 1/5 due to the bug described in 5.2 (now fixed, not re-verified by a human reviewer at time of writing). |
| Safety | **5.00 / 5** | All 8 scored examples were judged maximally safe by one reviewer. This tiny subjective sample does not establish safety and must not be interpreted as convergent clinical validation. |

**This evaluation is small and should not be over-interpreted.** It used one reviewer, scored under a third of the sample, and was not blinded. It is reported here as a preliminary signal, strongest on the naturalness finding (which aligns with multiple other independent observations throughout the project) and weakest on correctness (n too small, one flagged issue already addressed).

---

## 7. Consolidated list of known limitations

For anyone extending or evaluating this system, the following are known, documented, and — where relevant — deliberate limitations rather than oversights:

1. Red-flag detection has incomplete subject coverage and does not model tense or recency (Section 2.3).
2. `emotional_support` and `red_flag_suicidal` are genuinely, not just technically, hard to separate from short text; the system is deliberately biased toward the safer interpretation (Section 3.3).
3. Dialectal authenticity of LLM-generated training data and LLM-generated responses is imperfect; drift toward other Arabic dialects occurs and was only partially filterable (Sections 3.4, 6).
4. Local LLM generation for RAG answers succeeds about 60% of the time on the first full pass; the system compensates architecturally (pre-written fallbacks) rather than this being a solved reliability problem (Section 4).
5. The system has no model of conversation state beyond detecting immediate-prior-turn repetition; it cannot track an extended, evolving symptom history across many turns (Section 5).
6. All training data is synthetic (seed + LLM-expanded), not collected from real patients; real-world phrasing diversity may exceed what the system has seen.
7. Human evaluation to date is small-scale and single-reviewer; broader, blinded, multi-reviewer evaluation is recommended before any real-world use is considered.
8. Emergency contact numbers and crisis-line information must be independently verified as current and correct before any use beyond local testing. The response code now avoids unverified numbers and placeholder hotline text, directing users to local emergency services or the nearest emergency department; this does not replace verification of actionable local contact details.

---

## 8. Recommendation

This system demonstrates a coherent, evidence-based approach to safety in a hybrid ML/LLM/rule-based architecture, with meaningful, honestly-reported testing at each layer. It is **not** ready for real patient use without: clinical review of all medical content and emergency response text, verification of emergency contact information, significantly expanded human evaluation, and ideally validation against real (not synthetic) patient interaction data. It is well-suited as a learning project, a portfolio artifact, and a foundation for further development.
