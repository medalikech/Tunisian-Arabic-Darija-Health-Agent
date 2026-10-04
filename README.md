# Darija Health Agent

A conversational agent for Tunisian diabetes patients (type 2, oral medication) that understands Tunisian Arabic/Darija (Arabic script, Arabizi, and French code-switching), answers medical questions from a vetted knowledge base, detects safety emergencies with a deterministic rule layer, and logs every interaction.

Built as a learning project covering NLP/intent classification, synthetic data generation with local LLMs, retrieval-augmented generation (RAG), rule-based safety engineering, and agent orchestration.

---

## 1. Problem and scope

The goal was to build a patient-support agent that:
- Understands messages in Tunisian Darija — a dialect with no standard spelling, frequently mixed with Arabizi (Latin script with digits 3/7/9/5/2 standing in for Arabic letters) and French.
- Routes messages to the right handler: factual Q&A, logistics (refills, missed doses), emotional support, or emergency escalation.
- Never hallucinates medical information.
- Fails safely when uncertain, rather than guessing confidently.

Scope was deliberately narrowed to **type 2 diabetes with oral medication (metformin)** to keep the vocabulary and knowledge base tractable for a first version.

---

## 2. Architecture

```
Patient message
      │
      ▼
[1] Rule-based red-flag scanner (regex, deterministic)
      │
      ├── MATCH → fixed emergency response, bypasses everything else
      │
      ▼ no match
[2] Confidence check on intent classifier
      │
      ├── LOW CONFIDENCE → clarifying question
      │
      ▼ confident
[3] Intent classifier (TF-IDF + Logistic Regression)
      │
 ┌────┴──────────────┬───────────────────┐
 ▼                    ▼                   ▼
Fixed response    RAG Q&A agent      Unhandled (logged, placeholder)
(missed_dose,     (medication info,
greeting, etc.)   side effects, diet,
                   refill logistics)
      │
      ▼
SQLite conversation log (every interaction, with routing metadata)
```

**Design principle: defense in depth.** The red-flag scanner is deterministic and always runs first, so a weak classifier or an unreliable LLM can never cause a missed emergency. The ML components (classifier, RAG) handle everything else, with validation and safe fallbacks wherever they can fail.

---

## 3. Components

### 3.1 Data pipeline (`src/data/`)
- Hand-written seed examples (~20 per intent, 15 intents) in `data/seeds/raw_seeds.txt`, mixing Arabic script, Arabizi, and French code-switching.
- Synthetic expansion via a local Ollama model (Qwen2.5-14B, after comparison with Llama 3.1 — see §5), generating ~10 paraphrased variants per seed.
- Manual review pass (`review_cli.py`) to catch LLM dialectal drift (see §5.2).
- Group-aware, intent-stratified train/val/test split (`split.py`) to prevent data leakage between near-duplicate variants of the same seed.

### 3.2 Intent classifier (`src/nlp/`)
- TF-IDF (character n-grams) + Logistic Regression baseline.
- A fine-tuned CamelBERT-DA transformer was also trained and evaluated (via Kaggle GPU) for comparison — see §5.3.
- **Final choice: TF-IDF baseline**, which matched or beat the transformer at this dataset size.

### 3.3 Safety layer (`src/safety/`)
- Regex-based red-flag scanner covering chest pain, breathing difficulty, hypoglycemia, suicidal ideation, and overdose.
- Fixed, pre-written emergency responses (never LLM-generated, for consistency and reliability).
- Tested against a 34-case adversarial suite (33/34 passing — see §5.4).

### 3.4 RAG Q&A agent (`src/rag/`)
- Knowledge base of short, topic-tagged documents (medication timing, side effects, diet/Ramadan/Eid guidance, refill logistics, when to consult a doctor).
- Multilingual sentence embeddings + ChromaDB for retrieval, with a similarity threshold gate (refuses to answer if nothing relevant is retrieved).
- Local LLM (Qwen2.5) generates the answer, constrained to the retrieved context only.
- Output validation (language/script checks) with retry and safe fallback — see §5.5.

### 3.5 Agent router (`src/agents/router.py`)
Ties everything together: safety check → confidence-gated classification → routing to fixed response / RAG / clarification / unhandled placeholder. Every interaction is logged.

### 3.6 Storage (`src/patient_storage/`)
SQLite log of every interaction (timestamp, patient ID, route, intent, confidence, matched red flags, response), supporting patient history lookups and red-flag-frequency monitoring.

### 3.7 Interface (`src/interface/cli.py`)
A simple interactive CLI for testing full conversations end-to-end, with an optional `--debug` flag showing routing decisions.

### 3.8 Evaluation (`src/eval/`)
- `export_transcripts.py` — dumps logged conversations to a readable Markdown file.
- `run_full_eval.py` — runs the classifier report, red-flag test suite, and RAG reliability check in one pass, writing a consolidated summary.

---

## 4. Results

| Component | Metric | Result |
|---|---|---|
| Intent classifier (TF-IDF) | Macro F1 / Accuracy | 0.48 / 0.49 |
| Red-flag safety layer | Adversarial test suite | 33/34 passed |
| RAG Q&A agent | Fallback rate (safe degradation) | 40%, with **0%** unsafe/wrong-language output ever reaching the user |

Full breakdowns are in `data/eval/evaluation_summary.md` and `data/eval/transcripts.md` after running the evaluation scripts.

---

## 5. Key findings and design decisions

### 5.1 Model comparison: TF-IDF beat a fine-tuned transformer
At this dataset size (~1,500-1,800 examples across 11-15 classes), a character-level TF-IDF + Logistic Regression baseline matched or outperformed a fine-tuned CamelBERT-DA transformer in every configuration tried (unweighted, class-weighted, with/without red-flag label merging). Likely causes: Darija has no standard spelling (character n-grams handle this robustly; subword tokenizers trained mostly on MSA do not), and some intents (e.g. glucose readings) are dominated by literal numeric patterns that character matching captures directly. **Conclusion: more/better data mattered far more than model architecture** — a 15-example seed expansion for two weak intents improved macro-F1 more than any amount of hyperparameter tuning.

### 5.2 Dialectal drift in LLM-generated training data
Local LLMs (both Llama 3.1 and Qwen2.5), even when explicitly prompted for Tunisian Darija, regularly drifted toward Levantine, Egyptian, or MSA vocabulary and grammar (e.g. "بدي", "شو", "هيك" instead of Tunisian equivalents). A keyword-based filter plus manual review was used to catch and remove the worst cases. This is a known, documented limitation of using general-purpose Arabic LLMs for a specific, lower-resource dialect — not fully solvable without a dialect-specific model or much larger-scale native-speaker review.

### 5.3 Red-flag sub-type merging
Individual red-flag sub-types (chest pain, breathing, hypoglycemia, overdose, suicidal ideation) had too few examples each for the classifier to learn reliably. Merging them into a single `red_flag_emergency` class for the ML classifier substantially improved detection — while the rule-based layer (which doesn't need large training data) retains fine-grained sub-type detection for response selection. This reflects a deliberate architectural choice: **the classifier's job is coarse triage; the rules handle safety-critical precision.**

### 5.4 Safety layer: recall over precision, by design
The red-flag scanner is intentionally tuned to over-trigger rather than under-trigger — a false positive costs an unnecessary "are you okay" message, while a false negative could mean a missed emergency. The one known failing test case (a past-tense, resolved symptom still triggering a response) is an accepted trade-off: adding tense/subject modeling via regex was judged to add more risk (of introducing new false negatives) than the false positive it would remove.

### 5.5 LLM output reliability
Across RAG testing, the local Qwen2.5 model occasionally produced corrupted output under this prompt structure — language switching mid-generation (French, Chinese, Cyrillic characters appearing in otherwise-Arabic text) and, with longer or more tightly-constrained prompts, increased failure rates rather than decreased ones. The fix was not better prompting (several iterations were tried, including making this worse) but an **output validation + retry + safe-fallback architecture**: every generated answer is checked for script/language validity before being shown, with up to 3 retries and a clear "technical issue, please consult a pharmacist" fallback. This resulted in a 40% fallback rate but a 0% rate of unsafe or incoherent output reaching the user — the correct trade-off for a healthcare-adjacent system.

### 5.6 A data corruption bug (and the general lesson)
Midway through the project, Windows PowerShell's `Get-Content | Set-Content` pipeline was found to silently corrupt Arabic-script text during file merges (while leaving Latin-script text untouched), which had propagated into the trained classifier and caused it to perform at chance level. This was traced systematically by checking each pipeline stage in isolation and fixed by moving all text-file merging to Python's explicit `encoding="utf-8"` file handling. **General lesson: avoid PowerShell for any text processing involving non-ASCII content on Windows; do everything in Python instead**, where encoding is explicit and unambiguous.

---

## 6. Known limitations

- **No tense/subject modeling in the safety layer** — past or third-person mentions of red-flag symptoms may trigger a false positive (accepted trade-off, see §5.4).
- **Dialectal drift in synthetic data** — a portion of LLM-generated training examples lean toward non-Tunisian Arabic dialects despite filtering (see §5.2).
- **`emotional_support` vs. `red_flag_suicidal` overlap** — these two intents are genuinely ambiguous from text alone; the classifier (and arguably a human reader) sometimes cannot cleanly separate them. The system is deliberately biased toward the safer (emergency) interpretation in ambiguous cases.
- **RAG generation reliability** — roughly 40% of knowledge-base-grounded queries fail generation and fall back to a "please consult a professional" message rather than a direct answer, due to local LLM instability on code-switched dialectal output (see §5.5).
- **Small, synthetic dataset** — all training data is synthetically generated and reviewed, not collected from real patient interactions; this caps the achievable classifier performance and may not fully reflect real-world phrasing diversity.
- **No conversation memory across turns** — each message is currently handled independently; the router does not yet use prior conversation context when classifying or responding.

---

## 7. Project structure

```
darija-health-agent/
├── data/
│   ├── seeds/              # hand-written seed examples
│   ├── synthetic/          # LLM-expanded and reviewed variants
│   ├── processed/          # merged, split train/val/test sets
│   ├── knowledge_base/     # RAG source documents
│   ├── chroma_db/          # vector index
│   ├── conversations.db    # SQLite interaction log
│   └── eval/                # evaluation outputs
├── src/
│   ├── data/                # dataset build/expand/review/split scripts
│   ├── nlp/                 # normalization, baseline + transformer training
│   ├── safety/               # red-flag patterns, emergency responses
│   ├── rag/                  # knowledge base indexing and Q&A agent
│   ├── agents/                # router tying everything together
│   ├── patient_storage/      # conversation logging
│   ├── interface/            # CLI
│   └── eval/                 # evaluation runners
├── tests/
│   └── test_red_flags.py     # safety layer test suite
└── models/
    └── baseline_tfidf.joblib # trained classifier
```

---

## 8. Running the system

```powershell
# Rebuild the dataset from seeds (after editing raw_seeds.txt)
python src\data\build_seeds.py
python src\data\run_expansion.py
python src\data\review_cli.py
python src\data\merge_seeds.py
python src\data\split.py

# Train the classifier
python -m src.nlp.baseline

# Index the knowledge base
python src\rag\build_index.py

# Run the full evaluation suite
python src\eval\run_full_eval.py
python src\eval\export_transcripts.py

# Talk to the agent
python src\interface\cli.py --debug
```

---

## 9. Possible next steps

- Collect or simulate longer multi-turn conversations and add conversation memory to the router.
- Expand the knowledge base and seed set per intent (the single highest-leverage lever found during this project).
- Try a dialect-specific embedding/LLM pair if one becomes available, to address the drift and reliability issues documented above.
- Add a lightweight web or messaging-app front end (e.g. Telegram) in place of the CLI.
- Replace placeholder content in `src/safety/emergency_responses.py` with verified, current local emergency and crisis-line contact numbers before any real-world use.
