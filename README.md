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
- Regex-based red-flag scanner covering chest pain, breathing difficulty, hypoglycemia, sudden confusion and neurological changes, suicidal ideation, and overdose.
- Fixed, pre-written emergency responses (never LLM-generated, for consistency and reliability).
- Tested against a small hand-written set of Arabic, Arabizi, French, and English emergency examples, including selected reports about another person, accidental extra medication, and severe low-blood-sugar symptoms; see §5.4 for limitations.

### 3.4 RAG Q&A agent (`src/rag/`)
- Knowledge base of short, topic-tagged documents (medication timing, side effects, diet/Ramadan/Eid guidance, refill logistics, when to consult a doctor).
- Sentence embeddings + ChromaDB retrieve Tunisian-Arabic questions against Darija fallback text when available, then combine semantic similarity with lexical overlap (including common Arabizi-to-French concept aliases) and return at most one chunk per source; French source text remains the LLM context. A refusal gate requires semantic and lexical support from the same chunk, as a heuristic rather than a guarantee of relevance.
- Local LLM (Qwen2.5) generates the answer, constrained to the retrieved context only.
- Output validation checks language/script and overlap with retrieved Darija evidence, with retries and safe fallback — see §5.5.
- The agent route preserves the filenames of the knowledge-base sources used; the CLI displays them in `--debug` mode for traceability.
- Medication check-in replies are only recorded when an answer is clearly affirmative or negative; unclear or contradictory replies are clarified rather than counted. Missed-dose responses do not tell the user when to take a dose and direct them to their prescription leaflet or a pharmacist/doctor. This behavior still needs clinical review.
- `src/rag/eval_retrieval.py` checks retrieval against a small set of Tunisian-Darija questions in Arabic script and Arabizi/code-switching, reporting overall and per-language source Recall@3 and mean reciprocal rank. It also evaluates the refusal gate on supported questions, unrelated questions, and medically adjacent topics outside the knowledge base (such as insulin dosing and diabetic foot care), reporting outcomes by category and listing false refusals and false accepts. These checks do not call the answer-generating LLM; treat the small, hand-written sample as a development check, not clinical validation.

### 3.5 Agent router (`src/agents/router.py`)
Ties everything together: safety check → confidence-gated classification → routing to fixed response / RAG / clarification / unhandled placeholder. Every interaction is logged.

### 3.6 Storage (`src/patient_storage/`)
SQLite log of every interaction (timestamp, patient ID, route, intent, confidence, matched red flags, response), supporting patient history lookups and red-flag-frequency monitoring.

### 3.7 Interface (`src/interface/cli.py`)
A simple interactive CLI for testing full conversations end-to-end, with an optional `--debug` flag showing routing decisions.

### 3.8 Evaluation (`src/eval/`)
- `export_transcripts.py` — dumps logged conversations to a readable Markdown file.
- `run_full_eval.py` — runs the classifier report, red-flag test suite, and RAG reliability check in one pass, writing a consolidated summary.
- Run retrieval-only evaluation with `python src\rag\eval_retrieval.py` after building the index; it uses the configured embedding model and local Chroma collection.

---

## 4. Results

| Component | Metric | Result |
|---|---|---|
| Intent classifier (TF-IDF) | Macro F1 / Accuracy | 0.48 / 0.49 |
| Red-flag safety layer | Hand-written scanner examples | 69/69 expected outcomes passed; this is not clinical validation |
| RAG Q&A agent | Fallback rate in a small development run | 40%; language/script checks do not establish medical correctness |

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
The red-flag scanner is intentionally tuned to over-trigger rather than under-trigger. Its 69 hand-written examples currently pass, including selected current breathing-emergency reports about another person, accidental extra-medication reports, low-blood-sugar reports involving fainting or inability to swallow, and cause-neutral sudden-confusion and sudden-neurological-change alerts. This small test set does not measure real-world sensitivity or prove safety. The scanner still alerts on at least one past-tense, resolved symptom because it does not model recency; this conservative false positive is recorded as expected behavior, not evidence that the scanner can distinguish current from historical symptoms.

### 5.5 LLM output reliability
Across RAG testing, the local Qwen2.5 model occasionally produced corrupted output under this prompt structure — language switching mid-generation (French, Chinese, Cyrillic characters appearing in otherwise-Arabic text). The system retries rejected output and can use a prepared fallback. A small development run reported a 40% fallback rate. Script checks and lexical overlap can reject some malformed or weakly grounded outputs, but they do not verify medical correctness, completeness, or patient-specific appropriateness; no claim of zero unsafe output is supported.

### 5.6 A data corruption bug (and the general lesson)
Midway through the project, Windows PowerShell's `Get-Content | Set-Content` pipeline was found to silently corrupt Arabic-script text during file merges (while leaving Latin-script text untouched), which had propagated into the trained classifier and caused it to perform at chance level. This was traced systematically by checking each pipeline stage in isolation and fixed by moving all text-file merging to Python's explicit `encoding="utf-8"` file handling. **General lesson: avoid PowerShell for any text processing involving non-ASCII content on Windows; do everything in Python instead**, where encoding is explicit and unambiguous.

---

## 6. Known limitations

- **Limited tense/subject modeling in the safety layer** — selected third-person breathing reports are now detected, but coverage is incomplete and past or resolved symptoms may still trigger alerts (see §5.4).
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

# Train the classifier used by the agent router
New-Item -ItemType Directory -Force models | Out-Null
python src\nlp\baseline_merged.py

# Index the knowledge base
python src\rag\build_index.py

# Run the full evaluation suite
python src\eval\run_full_eval.py
python src\eval\export_transcripts.py

# Talk to the agent
python src\interface\cli.py --debug
```

### Test the project

Run these commands from the repository root in PowerShell. The UTF-8 setting
helps Python print Arabic test output correctly in Windows terminals:

```powershell
$env:PYTHONUTF8 = "1"
python -m unittest discover -s tests -p "test_*.py"
```

This runs the automated unit and integration tests for the safety scanner,
router, check-in parser, RAG retrieval/generation behavior, escalation storage,
and web-app authentication/API. The tests use mocks and temporary databases
where appropriate; passing them does not prove clinical safety or test a live
Ollama model.

For a fresh local setup, create and activate a virtual environment, then install
the pinned dependencies:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

To try the complete CLI conversation flow, first build the classifier and
knowledge-base index. The app loads `models\baseline_merged.joblib`; that
generated model and the Chroma index are intentionally not stored in Git:

```powershell
New-Item -ItemType Directory -Force models | Out-Null
python src\nlp\baseline_merged.py
python src\rag\build_index.py
```

The first index build may download the sentence-embedding model. RAG answers
also require Ollama to be installed and running, with the configured model
available (default: `mistral:7b`). In a separate terminal, start Ollama if it
is not already running, then run:

```powershell
ollama pull mistral:7b
python src\interface\cli.py --debug
```

Type a few fictional Darija questions, inspect the debug route and RAG sources,
then type `bye` to exit. Do not use real patient data.

`python src\eval\run_full_eval.py` is an additional report generator, not a
replacement for the test command above. It retrains the baseline and calls the
live RAG path, so it needs the local RAG setup and may update
`data\eval\evaluation_summary.md`; inspect its output and report for failures
or fallbacks. The small evaluation sets are development checks, not clinical
validation.

### Local web-app accounts (prototype only)

The web chat now requires provisioned accounts. Registration is intentionally
not public. Patient accounts can chat only as their own patient ID; staff
accounts can chat only for patient IDs explicitly assigned to them.

In PowerShell, initialize the local account database, set a new session secret
for this terminal, and create accounts with passwords entered interactively:

```powershell
python src\webapp\manage_users.py init
$env:DARJA_SESSION_SECRET = (& python -c "import secrets; print(secrets.token_urlsafe(32))")
python src\webapp\manage_users.py create-user --username patient-demo --role patient --patient-id patient-demo-001
python src\webapp\manage_users.py create-user --username staff-demo --role staff
python src\webapp\manage_users.py assign-patient --staff-username staff-demo --patient-id patient-demo-001
python src\webapp\app.py
```

Use opaque, non-identifying patient IDs. The session secret must stay private;
generate and store a stable secret in an appropriate secret manager rather
than sharing it or committing it. The app binds to `127.0.0.1` for local use.
If it is ever served over HTTPS in a reviewed environment, set
`DARJA_HTTPS_ONLY=1` so the session cookie is marked secure.

After five incorrect passwords, that account is temporarily locked for
15 minutes. A successful login clears earlier failed attempts. This is a
basic local-prototype safeguard, not a substitute for deployment-level
rate limiting and monitoring.

**This account system is a local prototype, not approval to enter real patient
data.** Conversations and health-related records are still stored in SQLite
without application-level encryption, and this project does not yet provide a
complete security, privacy, consent, access-review, deletion/retention, backup,
or regulatory-compliance program. Before any real patient information is
entered, obtain appropriate legal, privacy, security, and clinical review and
complete those protections. Until then, use fictional test data only.

---
