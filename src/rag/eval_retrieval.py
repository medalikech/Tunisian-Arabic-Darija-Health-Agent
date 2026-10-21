import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from rag.qa_agent import get_collection, is_retrieval_relevant, retrieve
from rag.qa_agent import embedder


OUT_OF_SCOPE_CASES = [
    {
        "category": "unrelated",
        "question": "\u0634\u0646\u0648\u0629 \u0639\u0627\u0635\u0645\u0629 \u062a\u0648\u0646\u0633\u061f",
    },
    {
        "category": "unrelated",
        "question": "\u0643\u064a\u0641\u0627\u0634 \u0646\u0639\u0645\u0644 g\u00e2teau au chocolat\u061f",
    },
    {
        "category": "unrelated",
        "question": "\u0643\u064a\u0641\u0627\u0634 \u0646\u0635\u0644\u062d roue \u0645\u062a\u0627\u0639 v\u00e9lo\u061f",
    },
    {"category": "unrelated", "question": "kifech na3mel gateau chocolat?"},
    {"category": "unrelated", "question": "chneya capitale mta3 tounes?"},
    {"category": "unrelated", "question": "comment r\u00e9parer un pneu de v\u00e9lo?"},
    {"category": "unrelated", "question": "a3tini recette couscous bel poisson"},
    {
        "category": "medical-adjacent",
        "question": "How many units of insulin should I inject with a meal?",
    },
    {
        "category": "medical-adjacent",
        "question": "Can I take metformin during pregnancy?",
    },
    {
        "category": "medical-adjacent",
        "question": "Can I stop metformin if my blood sugar is normal?",
    },
    {
        "category": "medical-adjacent",
        "question": "Comment soigner une plaie au pied diab\u00e9tique?",
    },
    {
        "category": "medical-adjacent",
        "question": "What should I do for a diabetic foot ulcer?",
    },
    {
        "category": "medical-adjacent",
        "question": "Can diabetes cause blurry vision, and what should I do?",
    },
    {
        "category": "medical-adjacent",
        "question": "How is diabetes diagnosed from an HbA1c test?",
    },
    {
        "category": "medical-adjacent",
        "question": "Should I adjust my insulin dose when I exercise?",
    },
]


RETRIEVAL_CASES = [
    {
        "question": "وقتاش ناخذ المتفورمين؟ ومع الماكلة ولا قبلها؟",
        "expected_sources": ["metformin_basics.txt"],
    },
    {
        "question": "نسيت جرعة الدواء، ناخذ زوز المرة الجاية؟",
        "expected_sources": ["metformin_basics.txt"],
    },
    {
        "question": "شنوة ينجم يعمللي المتفورمين من أعراض في المعدة؟",
        "expected_sources": ["metformin_basics.txt", "side_effects_detailed.txt"],
    },
    {
        "question": "كيفاش نقيس السكر بالڨلوكومتر؟",
        "expected_sources": ["glucose_monitoring.txt"],
    },
    {
        "question": "قداش يلزمني نعاود نقيس السكر في الأسبوع؟",
        "expected_sources": ["glucose_monitoring.txt"],
    },
    {
        "question": "شنوة ناكل في العيد وما نكثرش من السكر؟",
        "expected_sources": ["diet_lifestyle.txt"],
    },
    {
        "question": "نجم نصوم رمضان وانا عندي السكري؟",
        "expected_sources": ["diet_lifestyle.txt"],
    },
    {
        "question": "كان الصيدلية ما عندهاش الدواء، شنوة نعمل؟",
        "expected_sources": ["refill_logistics.txt"],
    },
    {
        "question": "باش نسافر بالطائرة، كيفاش ناخذ دوايا معايا؟",
        "expected_sources": ["travel_and_medication.txt"],
    },
    {
        "question": "عندي سكانار بالصبغة وناخذ متفورمين، شنوة نقول للطبيب؟",
        "expected_sources": ["drug_interactions.txt"],
    },
    {
        "question": "وجهي ورم وعندي صعوبة في التنفس بعد الدواء، شنوة نعمل؟",
        "expected_sources": ["side_effects_detailed.txt"],
    },
    {
        "question": "قداش من مرة نعمل تحاليل باش نراقب الكلى مع المتفورمين؟",
        "expected_sources": ["when_to_consult.txt"],
    },
    {
        "language": "Arabizi/mixed",
        "question": "wa9tech ne5ou metformine? ma3a el mekla wala 9balha?",
        "expected_sources": ["metformin_basics.txt"],
    },
    {
        "language": "Arabizi/mixed",
        "question": "nsit dose mta3 dawa, ne5ou zouz mara jeya?",
        "expected_sources": ["metformin_basics.txt"],
    },
    {
        "language": "Arabizi/mixed",
        "question": "chneya les effets secondaires mta3 metformine fel me3da?",
        "expected_sources": ["metformin_basics.txt", "side_effects_detailed.txt"],
    },
    {
        "language": "Arabizi/mixed",
        "question": "kifech na9is soukkar bel glucometre?",
        "expected_sources": ["glucose_monitoring.txt"],
    },
    {
        "language": "Arabizi/mixed",
        "question": "9addech men marra na3mel test glycemie fel jem3a?",
        "expected_sources": ["glucose_monitoring.txt"],
    },
    {
        "language": "Arabizi/mixed",
        "question": "chneya nakel fel 3id w ma nkatherch sokker?",
        "expected_sources": ["diet_lifestyle.txt"],
    },
    {
        "language": "Arabizi/mixed",
        "question": "najjem nsoum ramdhan w 3andi diabete?",
        "expected_sources": ["diet_lifestyle.txt"],
    },
    {
        "language": "Arabizi/mixed",
        "question": "pharmacie ma 3andhech dawa, chna3mel?",
        "expected_sources": ["refill_logistics.txt"],
    },
    {
        "language": "Arabizi/mixed",
        "question": "bech nsafar bel avion, kifech ne5ou dwa m3aya?",
        "expected_sources": ["travel_and_medication.txt"],
    },
    {
        "language": "Arabizi/mixed",
        "question": "3andi scanner avec produit de contraste w nekhou metformine, chn9oul lel tbib?",
        "expected_sources": ["drug_interactions.txt"],
    },
    {
        "language": "Arabizi/mixed",
        "question": "wajhi waram w 3andi difficulte respiration ba3d dawa, chna3mel?",
        "expected_sources": ["side_effects_detailed.txt"],
    },
    {
        "language": "Arabizi/mixed",
        "question": "9addech men mara suivi reins m3a metformine?",
        "expected_sources": ["when_to_consult.txt"],
    },
]


def evaluate_retrieval_cases(cases, retrieval_fn, k=3):
    case_results = []
    total_recall = 0.0
    total_reciprocal_rank = 0.0
    grouped_metrics = {}

    for case in cases:
        language = case.get("language", "Arabic")
        expected_sources = set(case["expected_sources"])
        if not expected_sources:
            raise ValueError("Each retrieval case must define an expected source.")

        retrieved = retrieval_fn(case["question"], k)
        retrieved_sources = [metadata.get("source") for _, _, metadata in retrieved]
        relevant_ranks = [
            rank
            for rank, source in enumerate(retrieved_sources, start=1)
            if source in expected_sources
        ]
        recall = len(set(retrieved_sources) & expected_sources) / len(expected_sources)
        reciprocal_rank = 1 / relevant_ranks[0] if relevant_ranks else 0.0
        total_recall += recall
        total_reciprocal_rank += reciprocal_rank
        group = grouped_metrics.setdefault(
            language,
            {"case_count": 0, "recall_total": 0.0, "reciprocal_rank_total": 0.0},
        )
        group["case_count"] += 1
        group["recall_total"] += recall
        group["reciprocal_rank_total"] += reciprocal_rank
        case_results.append({
            "question": case["question"],
            "language": language,
            "expected_sources": sorted(expected_sources),
            "retrieved_sources": retrieved_sources,
            "recall_at_k": recall,
            "reciprocal_rank": reciprocal_rank,
        })

    case_count = len(case_results)
    by_language = {
        language: {
            "case_count": group["case_count"],
            "recall_at_k": group["recall_total"] / group["case_count"],
            "mrr": group["reciprocal_rank_total"] / group["case_count"],
        }
        for language, group in grouped_metrics.items()
    }
    return {
        "case_count": case_count,
        "recall_at_k": total_recall / case_count if case_count else 0.0,
        "mrr": total_reciprocal_rank / case_count if case_count else 0.0,
        "by_language": by_language,
        "cases": case_results,
    }


def evaluate_relevance_cases(cases, retrieval_fn, relevance_fn, k=3):
    case_results = []
    grouped_metrics = {}
    correct_count = 0
    supported_count = 0
    out_of_scope_count = 0
    accepted_supported_count = 0
    rejected_out_of_scope_count = 0

    for case in cases:
        expected_in_scope = case.get("in_scope")
        if not isinstance(expected_in_scope, bool):
            raise ValueError("Each relevance case must define a boolean in_scope value.")

        question = case["question"]
        retrieved = retrieval_fn(question, k)
        accepted = relevance_fn(question, retrieved)
        correct = accepted == expected_in_scope
        correct_count += correct
        category = case.get(
            "category",
            "supported" if expected_in_scope else "out_of_scope",
        )
        group = grouped_metrics.setdefault(
            category,
            {"case_count": 0, "accepted_count": 0, "correct_count": 0},
        )
        group["case_count"] += 1
        group["accepted_count"] += accepted
        group["correct_count"] += correct
        if expected_in_scope:
            supported_count += 1
            accepted_supported_count += accepted
            outcome = "accepted" if accepted else "false_refusal"
        else:
            out_of_scope_count += 1
            rejected_out_of_scope_count += not accepted
            outcome = "false_accept" if accepted else "rejected"

        case_results.append({
            "question": question,
            "in_scope": expected_in_scope,
            "accepted": accepted,
            "outcome": outcome,
            "category": category,
        })

    case_count = len(case_results)
    by_category = {
        category: {
            "case_count": group["case_count"],
            "acceptance_rate": group["accepted_count"] / group["case_count"],
            "accuracy": group["correct_count"] / group["case_count"],
        }
        for category, group in grouped_metrics.items()
    }
    return {
        "case_count": case_count,
        "accuracy": correct_count / case_count if case_count else 0.0,
        "by_category": by_category,
        "supported_count": supported_count,
        "supported_acceptance_rate": (
            accepted_supported_count / supported_count if supported_count else 0.0
        ),
        "out_of_scope_count": out_of_scope_count,
        "out_of_scope_rejection_rate": (
            rejected_out_of_scope_count / out_of_scope_count
            if out_of_scope_count else 0.0
        ),
        "false_refusals": [
            case for case in case_results if case["outcome"] == "false_refusal"
        ],
        "false_accepts": [
            case for case in case_results if case["outcome"] == "false_accept"
        ],
        "cases": case_results,
    }


def run_retrieval_evaluation(k=3):
    model = embedder()
    collection = get_collection()
    return evaluate_retrieval_cases(
        RETRIEVAL_CASES,
        lambda question, limit: retrieve(question, model, collection, k=limit),
        k=k,
    )


def run_relevance_evaluation(k=3):
    model = embedder()
    collection = get_collection()
    cases = [
        {
            "question": case["question"],
            "in_scope": True,
            "category": "supported",
        }
        for case in RETRIEVAL_CASES
    ] + [
        {
            "question": case["question"],
            "in_scope": False,
            "category": case["category"],
        }
        for case in OUT_OF_SCOPE_CASES
    ]
    return evaluate_relevance_cases(
        cases,
        lambda question, limit: retrieve(question, model, collection, k=limit),
        is_retrieval_relevant,
        k=k,
    )


if __name__ == "__main__":
    results = run_retrieval_evaluation()
    print(f"Retrieval cases: {results['case_count']}")
    print(f"Recall@3: {results['recall_at_k']:.3f}")
    print(f"MRR: {results['mrr']:.3f}")
    for language, metrics in results["by_language"].items():
        print(
            f"{language}: Recall@3={metrics['recall_at_k']:.3f}, "
            f"MRR={metrics['mrr']:.3f} ({metrics['case_count']} questions)"
        )
    for case in results["cases"]:
        expected = ", ".join(case["expected_sources"])
        retrieved = ", ".join(case["retrieved_sources"])
        print(f"\n[{case['language']}] Q: {case['question']}")
        print(f"Expected: {expected}")
        print(f"Retrieved: {retrieved}")
        print(
            f"Recall@3={case['recall_at_k']:.3f}, "
            f"reciprocal rank={case['reciprocal_rank']:.3f}"
        )
    relevance = run_relevance_evaluation()
    print("\nRetrieval refusal gate:")
    print(
        f"Supported acceptance: {relevance['supported_acceptance_rate']:.3f} "
        f"({relevance['supported_count']} questions)"
    )
    print(
        f"Out-of-scope rejection: {relevance['out_of_scope_rejection_rate']:.3f} "
        f"({relevance['out_of_scope_count']} questions)"
    )
    print(f"Combined accuracy: {relevance['accuracy']:.3f}")
    for category, metrics in relevance["by_category"].items():
        print(
            f"{category}: acceptance={metrics['acceptance_rate']:.3f}, "
            f"accuracy={metrics['accuracy']:.3f} "
            f"({metrics['case_count']} questions)"
        )
    for case in relevance["false_refusals"] + relevance["false_accepts"]:
        print(f"{case['outcome']}: {case['question']}")
