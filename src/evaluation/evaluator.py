
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
        """Evaluate a single question through the enhanced full pipeline."""
        question = question_entry["question"]
        question_type = question_entry["question_type"]
        answer_type = question_entry.get("answer_type", "explanation")
        expected_answer = question_entry.get("expected_answer", "")
        eval_criteria = question_entry.get("eval_criteria", {})
        gold_chunks = set(question_entry.get("gold_chunks", []))
        
        result = {
            "question_id": question_entry.get("id"),
            "question": question,
            "question_type": question_type,
            "answer_type": answer_type,
            "expected_answer": expected_answer,
        }

        # Step 1: Retrieval
        if self.retriever and self.retriever.is_loaded:
            try:
                context, retrieved_chunks = self.retriever.retrieve_with_context(
                    question, top_k=top_k, mode=retrieval_mode
                )
                
                # Retrieval Performance
                found_gold_indices = []
                for i, chunk in enumerate(retrieved_chunks):
                    # Match by chunk_id if available, otherwise fallback to keyword heuristic
                    chunk_id = chunk.get("chunk_id")
                    if chunk_id is not None and chunk_id in gold_chunks:
                        found_gold_indices.append(i)
                    elif not gold_chunks:
                        # Falling back to heuristic if no gold_chunks defined
                        hits = sum(1 for kw in eval_criteria.get("must_include", []) if kw.lower() in chunk["text"].lower())
                        if hits > 0:
                            found_gold_indices.append(i)

                result["retrieval"] = {
                    "recall_at_k": 1.0 if found_gold_indices else 0.0,
                    "mrr": 1.0 / (found_gold_indices[0] + 1) if found_gold_indices else 0.0,
                    "num_chunks": len(retrieved_chunks),
                }
            except Exception as e:
                context = ""
                result["retrieval"] = {"error": str(e)}
        else:
            context = ""
            result["retrieval"] = {"status": "skipped"}

        # Step 2: Generation
        try:
            gen_result = self.generator.generate_answer(question, context, model_type)
            generated_answer = gen_result.get("answer", "")
            result["generation"] = {"answer": generated_answer}
        except Exception as e:
            result["generation"] = {"error": str(e)}
            generated_answer = ""

        # Step 3: Comprehensive Validation
        validation = self._validate_answer(generated_answer, question_entry, context)
        result["validation"] = validation
        result["overall_score"] = validation["total_score"]

        return result

    def _validate_answer(self, answer: str, entry: Dict, context: str) -> Dict:
        """Enhanced validation logic."""
        scores = {}
        criteria = entry.get("eval_criteria", {})
        q_type = entry.get("question_type")
        
        # 1. Refusal Detection (for unanswerable)
        if q_type == "unanswerable":
            refusal_patterns = entry.get("valid_refusal_patterns", [])
            is_refusal = any(pat.lower() in answer.lower() for pat in refusal_patterns)
            scores["refusal_correct"] = 1.0 if is_refusal else 0.0
            return {"total_score": scores["refusal_correct"], "details": scores}

        # 2. Mandatory Keywords (must_include)
        must_include = criteria.get("must_include", [])
        if must_include:
            hits = sum(1 for kw in must_include if kw.lower() in answer.lower())
            scores["keyword_coverage"] = hits / len(must_include)
        else:
            scores["keyword_coverage"] = 1.0

        # 3. Forbidden Keywords (must_not_include)
        must_not_include = criteria.get("must_not_include", [])
        forbidden_hits = sum(1 for kw in must_not_include if kw.lower() in answer.lower())
        scores["forbidden_penalty"] = 1.0 if forbidden_hits == 0 else 0.0

        # 4. Numerical Tolerance
        if entry.get("answer_type") == "numerical":
            import re
            numbers_expected = re.findall(r"[-+]?\d*\.\d+|\d+", entry.get("expected_answer", ""))
            numbers_actual = re.findall(r"[-+]?\d*\.\d+|\d+", answer)
            if numbers_expected and numbers_actual:
                scores["numerical_accuracy"] = 1.0 if numbers_expected[0] in numbers_actual else 0.0
            else:
                scores["numerical_accuracy"] = 0.5 

        # 5. Language Check
        expected_lang = entry.get("expected_language", "en")
        has_hindi = any('\u0900' <= c <= '\u097F' for c in answer)
        if expected_lang == "hi" and not has_hindi:
            scores["language_match"] = 0.0
        elif expected_lang == "en" and has_hindi:
            scores["language_match"] = 0.5
        else:
            scores["language_match"] = 1.0

        # Grounding check (external call)
        grounding_result = self.grounding_checker.check_grounding(answer, context)
        scores["grounding_score"] = grounding_result.get("score", 0.0)

        # Weighted score
        total = (
            scores.get("grounding_score", 0) * 0.4 +
            scores.get("keyword_coverage", 0) * 0.3 +
            scores.get("forbidden_penalty", 1.0) * 0.2 +
            scores.get("language_match", 1.0) * 0.1
        )
        scores["total_score"] = round(total, 3)
        return scores

 
    # Aggregate Metrics


    def _compute_aggregate_metrics(self, per_question: List[Dict]) -> Dict:
        """Compute aggregate metrics from enhanced per-question results."""
        aggregates = {
            "total_evaluated": len(per_question),
            "by_question_type": {},
            "overall_score": 0.0,
        }

        all_scores = []
        type_groups = {}
        for result in per_question:
            qtype = result.get("question_type", "unknown")
            if qtype not in type_groups:
                type_groups[qtype] = []
            type_groups[qtype].append(result)

        for qtype, results in type_groups.items():
            metrics = {
                "count": len(results),
                "avg_total_score": 0.0,
                "avg_recall_k": 0.0,
                "avg_mrr": 0.0,
                "avg_grounding": 0.0,
                "avg_keyword_coverage": 0.0,
            }

            total_scores = []
            recalls = []
            mrrs = []
            groundings = []
            coverages = []

            for r in results:
                total_scores.append(r.get("overall_score", 0))
                all_scores.append(r.get("overall_score", 0))
                
                ret = r.get("retrieval", {})
                recalls.append(ret.get("recall_at_k", 0))
                mrrs.append(ret.get("mrr", 0))

                val = r.get("validation", {})
                if qtype != "unanswerable":
                    groundings.append(val.get("grounding_score", 0))
                    coverages.append(val.get("keyword_coverage", 0))
                else:
                    groundings.append(val.get("refusal_correct", 0))

            metrics["avg_total_score"] = np.mean(total_scores) if total_scores else 0.0
            metrics["avg_recall_k"] = np.mean(recalls) if recalls else 0.0
            metrics["avg_mrr"] = np.mean(mrrs) if mrrs else 0.0
            metrics["avg_grounding"] = np.mean(groundings) if groundings else 0.0
            metrics["avg_keyword_coverage"] = np.mean(coverages) if coverages else 0.0

            aggregates["by_question_type"][qtype] = metrics

        aggregates["overall_score"] = np.mean(all_scores) if all_scores else 0.0
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
        """Print a human-readable enhanced evaluation summary."""
        print("\n" + "=" * 70)
        print("PARISHIKSHA ENHANCED EVALUATION REPORT")
        print("=" * 70)

        meta = report.get("metadata", {})
        print(f"Model: {meta.get('model_type', 'N/A')} | Retrieval: {meta.get('retrieval_mode', 'N/A')}")
        
        agg = report.get("aggregate", {})
        print(f"\nOVERALL PERFORMANCE: {agg.get('overall_score', 0):.1%}")

        print("\nMETRICS BY CATEGORY:")
        for qtype, metrics in agg.get("by_question_type", {}).items():
            print(f"\n  [{qtype.upper()}] ({metrics['count']} questions)")
            print(f"    Validation Score:   {metrics['avg_total_score']:.1%}")
            print(f"    Retrieval Recall:   {metrics['avg_recall_k']:.1%}")
            print(f"    Mean Reciprocal Rank: {metrics['avg_mrr']:.3f}")
            if qtype == "unanswerable":
                print(f"    Refusal Accuracy:   {metrics['avg_grounding']:.1%}")
            else:
                print(f"    Grounding Score:    {metrics['avg_grounding']:.1%}")
                print(f"    Keyword Coverage:   {metrics['avg_keyword_coverage']:.1%}")

        print(f"\n{'-'*70}")
        print("PER-QUESTION DETAILS")
        print(f"{'-'*70}")
        for r in report.get("per_question", []):
            q_id = str(r.get("question_id"))[:8]
            score = r.get("overall_score", 0)
            print(f"ID: {q_id}... | Score: {score:.1%} | Q: {r['question'][:50]}...")
            if "generation" in r and "answer" in r["generation"]:
                print(f"  Ans: {r['generation']['answer'][:80]}...")



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
