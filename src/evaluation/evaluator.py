
# PariShiksha — Full System Evaluator

# Runs the complete evaluation pipeline: retrieval → generation → grounding
# verification → scoring, on the evaluation set. Produces a detailed report
# showing exactly where the system works and where it fails.

# This is the most important module in the project. Everything else exists
# to produce numbers that show up here. If this evaluator says the system
# isn't working, it doesn't matter how elegant the rest of the code is.


import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import EVAL_OUTPUT_DIR, EVALUATION_CONFIG
from src.retrieval.retriever import HybridRetriever
from src.generation.answer_generator import AnswerGenerator
from src.generation.grounding import GroundingChecker
from src.evaluation.eval_set_builder import EvalSetBuilder


class PariShikshaEvaluator:
    """
    End-to-end evaluator for the PariShiksha study assistant.
    
    Evaluation axes:
    1. Retrieval quality — Are the right chunks being retrieved?
    2. Answer quality — Is the generated answer correct and clear?
    3. Grounding fidelity — Is the answer actually from the textbook?
    4. Hallucination resistance — Does the system refuse unanswerable questions?
    5. Code-switch robustness — Does it handle Hindi-English queries?
    
    Usage:
        evaluator = PariShikshaEvaluator(
            retriever=retriever,
            generator=generator,
        )
        report = evaluator.run_evaluation()
        evaluator.save_report(report)
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        generator: Optional[AnswerGenerator] = None,
        grounding_checker: Optional[GroundingChecker] = None,
        eval_builder: Optional[EvalSetBuilder] = None,
        output_dir: Path = EVAL_OUTPUT_DIR,
    ):
        self.retriever = retriever
        self.generator = generator or AnswerGenerator()
        self.grounding_checker = grounding_checker or GroundingChecker()
        self.eval_builder = eval_builder or EvalSetBuilder()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)


    # Full Pipeline Evaluation


    def run_evaluation(
        self,
        eval_set: Optional[List[Dict]] = None,
        model_type: str = "gemini",
        retrieval_mode: str = "hybrid",
        top_k: int = 5,
    ) -> Dict:
        """
        Run full end-to-end evaluation.
        
        For each question in the eval set:
        1. Retrieve relevant chunks
        2. Generate an answer from the retrieved context
        3. Check grounding of the answer
        4. Score against expected answer (keyword overlap, ROUGE)
        5. Special scoring for unanswerable and code-switched questions
        
        Returns
        -------
        dict
            Full evaluation report with per-question and aggregate results
        """
        eval_set = eval_set or self.eval_builder.load_eval_set()
        
        if not eval_set:
            logger.error("No evaluation questions loaded")
            return {"error": "Empty evaluation set"}

        logger.info(
            f"Starting evaluation: {len(eval_set)} questions, "
            f"model={model_type}, retrieval={retrieval_mode}"
        )

        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_type": model_type,
                "retrieval_mode": retrieval_mode,
                "top_k": top_k,
                "total_questions": len(eval_set),
            },
            "per_question": [],
            "aggregate": {},
        }

        start_time = time.time()

        for i, question_entry in enumerate(eval_set):
            logger.info(
                f"[{i+1}/{len(eval_set)}] Evaluating: "
                f"{question_entry['question'][:50]}..."
            )

            result = self._evaluate_single_question(
                question_entry,
                model_type=model_type,
                retrieval_mode=retrieval_mode,
                top_k=top_k,
            )
            report["per_question"].append(result)

        elapsed = time.time() - start_time
        report["metadata"]["total_time_seconds"] = round(elapsed, 2)

        # Compute aggregate metrics
        report["aggregate"] = self._compute_aggregate_metrics(report["per_question"])

        logger.info(
            f"Evaluation complete in {elapsed:.1f}s. "
            f"Overall score: {report['aggregate'].get('overall_score', 'N/A')}"
        )

        return report

    def _evaluate_single_question(
        self,
        question_entry: Dict,
        model_type: str = "gemini",
        retrieval_mode: str = "hybrid",
        top_k: int = 5,
    ) -> Dict:
        """Evaluate a single question through the full pipeline."""
        question = question_entry["question"]
        question_type = question_entry["question_type"]
        expected_answer = question_entry.get("expected_answer", "")
        expected_keywords = question_entry.get("expected_keywords", [])

        result = {
            "question_id": question_entry.get("id", 0),
            "question": question,
            "question_type": question_type,
            "expected_answer": expected_answer,
        }

        # Step 1: Retrieval
        if self.retriever and self.retriever.is_loaded:
            try:
                context, retrieved_chunks = self.retriever.retrieve_with_context(
                    question, top_k=top_k, mode=retrieval_mode
                )
                result["retrieval"] = {
                    "num_chunks_retrieved": len(retrieved_chunks),
                    "top_scores": [round(c["score"], 3) for c in retrieved_chunks[:3]],
                    "context_length_chars": len(context),
                }

                # Check if expected keywords appear in retrieved chunks
                retrieved_text = " ".join(c["text"].lower() for c in retrieved_chunks)
                keyword_hits = {
                    kw: kw.lower() in retrieved_text
                    for kw in expected_keywords
                }
                result["retrieval"]["keyword_recall"] = (
                    sum(keyword_hits.values()) / len(keyword_hits)
                    if keyword_hits else 0.0
                )
                result["retrieval"]["keyword_hits"] = keyword_hits

            except Exception as e:
                context = ""
                result["retrieval"] = {"error": str(e)}
        else:
            # No retriever — use a placeholder context or skip
            context = f"[No retriever configured. Question: {question}]"
            result["retrieval"] = {"status": "skipped", "reason": "no_retriever"}

        # Step 2: Generation
        try:
            gen_result = self.generator.generate_answer(
                question=question,
                context=context,
                model_type=model_type,
            )
            result["generation"] = {
                "answer": gen_result.get("answer", ""),
                "model": gen_result.get("model", model_type),
                "status": gen_result.get("status", "unknown"),
            }
        except Exception as e:
            result["generation"] = {"answer": "", "status": "error", "error": str(e)}

        # Step 3: Grounding Check
        generated_answer = result.get("generation", {}).get("answer", "")
        if generated_answer and context:
            grounding_result = self.grounding_checker.check_grounding(
                answer=generated_answer,
                context=context,
                question=question,
            )
            result["grounding"] = {
                "grounded": grounding_result["grounded"],
                "score": grounding_result["score"],
                "is_refusal": grounding_result["is_refusal"],
                "ungrounded_count": len(grounding_result.get("ungrounded_claims", [])),
            }
        else:
            result["grounding"] = {"status": "skipped"}

        # Step 4: Answer Quality Scoring
        if generated_answer and expected_answer and expected_answer != "N/A":
            quality_scores = self._score_answer_quality(
                generated_answer, expected_answer, expected_keywords
            )
            result["quality"] = quality_scores
        elif question_type == "unanswerable":
            # For unanswerable questions, check if model correctly refuses
            is_refusal = result.get("grounding", {}).get("is_refusal", False)
            result["quality"] = {
                "correct_refusal": is_refusal,
                "score": 1.0 if is_refusal else 0.0,
            }
        else:
            result["quality"] = {"status": "skipped"}

        return result


    # Answer Quality Scoring


    def _score_answer_quality(
        self,
        generated: str,
        expected: str,
        expected_keywords: List[str],
    ) -> Dict:
        """
        Score the quality of a generated answer against the expected answer.
        
        Metrics:
        - keyword_precision: fraction of expected keywords in generated answer
        - lexical_overlap: word overlap between generated and expected
        - rouge_l: ROUGE-L score (longest common subsequence)
        """
        scores = {}

        # Keyword precision
        gen_lower = generated.lower()
        keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in gen_lower)
        scores["keyword_precision"] = (
            round(keyword_hits / len(expected_keywords), 3)
            if expected_keywords else 0.0
        )

        # Lexical overlap (F1)
        gen_words = set(generated.lower().split())
        exp_words = set(expected.lower().split())
        if gen_words and exp_words:
            intersection = gen_words.intersection(exp_words)
            precision = len(intersection) / len(gen_words) if gen_words else 0
            recall = len(intersection) / len(exp_words) if exp_words else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0
            )
            scores["lexical_f1"] = round(f1, 3)
        else:
            scores["lexical_f1"] = 0.0

        # ROUGE-L (longest common subsequence)
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            rouge_result = scorer.score(expected, generated)
            scores["rouge_l"] = round(rouge_result["rougeL"].fmeasure, 3)
        except ImportError:
            scores["rouge_l"] = None
            logger.warning("rouge_score not installed, skipping ROUGE-L")

        # Overall quality score (weighted average)
        available_scores = [
            scores["keyword_precision"],
            scores["lexical_f1"],
        ]
        if scores.get("rouge_l") is not None:
            available_scores.append(scores["rouge_l"])

        scores["overall"] = round(np.mean(available_scores), 3) if available_scores else 0.0

        return scores

 
    # Aggregate Metrics


    def _compute_aggregate_metrics(self, per_question: List[Dict]) -> Dict:
        """Compute aggregate metrics from per-question results."""
        aggregates = {
            "total_evaluated": len(per_question),
            "by_question_type": {},
            "overall_score": 0.0,
        }

        # Group by question type
        type_groups = {}
        for result in per_question:
            qtype = result.get("question_type", "unknown")
            if qtype not in type_groups:
                type_groups[qtype] = []
            type_groups[qtype].append(result)

        all_scores = []

        for qtype, results in type_groups.items():
            type_metrics = {
                "count": len(results),
                "grounded_count": 0,
                "avg_grounding_score": 0.0,
                "avg_quality_score": 0.0,
                "avg_retrieval_keyword_recall": 0.0,
            }

            grounding_scores = []
            quality_scores = []
            retrieval_recalls = []

            for r in results:
                # Grounding
                if r.get("grounding", {}).get("grounded"):
                    type_metrics["grounded_count"] += 1
                gs = r.get("grounding", {}).get("score")
                if gs is not None:
                    grounding_scores.append(gs)

                # Quality
                qs = r.get("quality", {}).get("overall")
                if qs is None:
                    qs = r.get("quality", {}).get("score")
                if qs is not None:
                    quality_scores.append(qs)
                    all_scores.append(qs)

                # Retrieval
                rr = r.get("retrieval", {}).get("keyword_recall")
                if rr is not None:
                    retrieval_recalls.append(rr)

            type_metrics["avg_grounding_score"] = (
                round(np.mean(grounding_scores), 3) if grounding_scores else 0.0
            )
            type_metrics["avg_quality_score"] = (
                round(np.mean(quality_scores), 3) if quality_scores else 0.0
            )
            type_metrics["avg_retrieval_keyword_recall"] = (
                round(np.mean(retrieval_recalls), 3) if retrieval_recalls else 0.0
            )

            aggregates["by_question_type"][qtype] = type_metrics

        # Overall score
        aggregates["overall_score"] = round(np.mean(all_scores), 3) if all_scores else 0.0

        return aggregates

    # Reporting
 

    def save_report(
        self, report: Dict, filename: str = "evaluation_report.json"
    ) -> Path:
        """Save evaluation report as JSON."""
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Saved evaluation report to {output_path}")
        return output_path

    def print_summary(self, report: Dict) -> None:
        """Print a human-readable evaluation summary."""
        print("\n" + "=" * 70)
        print("PARISHIKSHA EVALUATION REPORT")
        print("=" * 70)

        meta = report.get("metadata", {})
        print(f"\nTimestamp: {meta.get('timestamp', 'N/A')}")
        print(f"Model: {meta.get('model_type', 'N/A')}")
        print(f"Retrieval mode: {meta.get('retrieval_mode', 'N/A')}")
        print(f"Questions evaluated: {meta.get('total_questions', 0)}")
        print(f"Time: {meta.get('total_time_seconds', 0):.1f}s")

        agg = report.get("aggregate", {})
        print(f"\n{'-'*50}")
        print(f"OVERALL SCORE: {agg.get('overall_score', 0):.1%}")
        print(f"{'-'*50}")

        print(f"\nBreakdown by question type:")
        for qtype, metrics in agg.get("by_question_type", {}).items():
            print(f"\n  {qtype} ({metrics['count']} questions):")
            print(f"    Avg quality score:     {metrics['avg_quality_score']:.1%}")
            print(f"    Avg grounding score:   {metrics['avg_grounding_score']:.1%}")
            print(f"    Grounded answers:      {metrics['grounded_count']}/{metrics['count']}")
            print(f"    Retrieval recall:      {metrics['avg_retrieval_keyword_recall']:.1%}")

        # Per-question details
        print(f"\n{'-'*50}")
        print("PER-QUESTION RESULTS")
        print(f"{'-'*50}")
        for r in report.get("per_question", []):
            qtype = r.get("question_type", "?")
            quality = r.get("quality", {}).get("overall", r.get("quality", {}).get("score", "?"))
            grounded = r.get("grounding", {}).get("grounded", "?")
            print(f"\n  Q{r.get('question_id', '?')}: [{qtype}] {r['question'][:60]}...")
            print(f"    Quality: {quality}  |  Grounded: {grounded}")
            answer = r.get("generation", {}).get("answer", "")
            if answer:
                print(f"    Answer: {answer[:100]}...")



# CLI Entry Point

if __name__ == "__main__":
    print("PariShiksha Evaluator")
    print("Run via notebook (06_evaluation.ipynb) for full pipeline evaluation.")
    print("Or use the following code:")
    print("""
    from src.evaluation.evaluator import PariShikshaEvaluator
    from src.retrieval.retriever import HybridRetriever
    from src.generation.answer_generator import AnswerGenerator

    retriever = HybridRetriever()
    retriever.load_index("chapter_5", "fixed_token_256")
    
    generator = AnswerGenerator()
    evaluator = PariShikshaEvaluator(retriever=retriever, generator=generator)
    
    report = evaluator.run_evaluation(model_type="gemini")
    evaluator.print_summary(report)
    evaluator.save_report(report)
    """)
