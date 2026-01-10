"""
Comprehensive Evaluation Metrics

Combines all metrics (Cypher similarity, output similarity, validators, derived scores)
to produce complete evaluation results matching kg-axel baseline format.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd

from evaluation.cypher_similarity import compute_all_cypher_similarities
from evaluation.output_similarity import compute_jaccard_output, compute_pass_at_1_output
from evaluation.validators import SyntaxValidator, SchemaValidator, PropertiesValidator

logger = logging.getLogger(__name__)


class ComprehensiveMetricsCalculator:
    """
    Calculator for all evaluation metrics.

    Computes:
    - Cypher text similarity (BLEU, Rouge-L, Jaro, Jaccard)
    - Output similarity (Jaccard Output, Pass@1 Output)
    - Validators (Syntax, Schema, Properties)
    - Derived scores (Pass@1 Score, KG Validity Score, Jaccard Output Score, JaRou Score)
    - Composite metric (LLMetric-Q)
    """

    def __init__(
        self,
        executor=None,
        schema_config: Dict[str, Any] = None,
        properties_config: Dict[str, Any] = None
    ):
        """
        Initialize metrics calculator.

        Args:
            executor: GraphExecutor instance for output metrics (optional)
            schema_config: Schema configuration for SchemaValidator
            properties_config: Properties configuration for PropertiesValidator
        """
        self.executor = executor

        # Initialize validators
        self.syntax_validator = SyntaxValidator()
        self.schema_validator = SchemaValidator(schema_config)
        self.properties_validator = PropertiesValidator(properties_config)

    def compute_all_metrics(
        self,
        question_id: int,
        question: str,
        ground_truth_query: str,
        generated_query: str,
        prompt_technique: str,
        schema_format: str,
        reasoning_level: str,
        sublevel: str,
        complexity: str
    ) -> Dict[str, Any]:
        """
        Compute all metrics for a single question.

        Args:
            question_id: Question ID
            question: Natural language question
            ground_truth_query: Ground truth Cypher query
            generated_query: Generated Cypher query
            prompt_technique: Prompt engineering technique used
            schema_format: KG schema representation format
            reasoning_level: Reasoning level (Fakta Eksplisit/Implisit)
            sublevel: Sublevel (Nodes/One-hop/Multi-hop)
            complexity: Complexity level (Easy/Medium/Hard)

        Returns:
            Dictionary with all metrics
        """
        result = {
            # Metadata
            'ID Pertanyaan': question_id,
            'Teknik Prompt Engineering': prompt_technique,
            'Format Representasi Skema KG': schema_format,
            'Tingkat Penalaran': reasoning_level,
            'Sublevel': sublevel,
            'Tingkat Kompleksitas': complexity,
        }

        # Store queries
        result['Cypher LLM'] = generated_query

        # Compute Cypher text similarity metrics
        cypher_metrics = compute_all_cypher_similarities(ground_truth_query, generated_query)
        result.update(cypher_metrics)

        # Compute validator scores
        syntax_valid, _ = self.syntax_validator.validate(generated_query)
        result['Syntax Validator'] = syntax_valid

        result['Schema Validator'] = self.schema_validator.validate_score(generated_query)
        result['Properties Validator'] = self.properties_validator.validate_score(generated_query)

        # Compute output similarity metrics (if executor available)
        if self.executor:
            result['Jaccard Output'] = compute_jaccard_output(
                self.executor,
                ground_truth_query,
                generated_query
            )

            result['Pass@1 Output'] = compute_pass_at_1_output(
                self.executor,
                ground_truth_query,
                generated_query
            )
        else:
            result['Jaccard Output'] = None
            result['Pass@1 Output'] = False

        # Compute derived scores
        result['Pass@1 Score'] = self._compute_pass_at_1_score(result)
        result['KG Validity Score'] = self._compute_kg_validity_score(result)
        result['Jaccard Output Score'] = self._compute_jaccard_output_score(result)
        result['JaRou Score'] = self._compute_jarou_score(result)

        # Compute LLMetric-Q
        result['LLMetric-Q'] = self._compute_llmetric_q(result)

        return result

    def _compute_pass_at_1_score(self, metrics: Dict[str, Any]) -> float:
        """
        Compute Pass@1 Score (0 or 100).

        Args:
            metrics: Dictionary containing 'Pass@1 Output'

        Returns:
            100 if Pass@1 Output is True, else 0
        """
        if metrics.get('Pass@1 Output') == True:
            return 100.0
        else:
            return 0.0

    def _compute_kg_validity_score(self, metrics: Dict[str, Any]) -> float:
        """
        Compute KG Validity Score (0 or 100).

        Query is valid if ALL validators pass:
        - Syntax Validator == True
        - Schema Validator == 1.0
        - Properties Validator == 1.0 OR None (no properties)

        Args:
            metrics: Dictionary containing validator scores

        Returns:
            100 if all validators pass, else 0
        """
        syntax_valid = metrics.get('Syntax Validator') == True
        schema_valid = metrics.get('Schema Validator') == 1.0

        # Properties validator: valid if 1.0 or None (no properties to check)
        props_score = metrics.get('Properties Validator')
        properties_valid = (props_score == 1.0) or (props_score is None)

        if syntax_valid and schema_valid and properties_valid:
            return 100.0
        else:
            return 0.0

    def _compute_jaccard_output_score(self, metrics: Dict[str, Any]) -> float:
        """
        Compute Jaccard Output Score (0 to 100).

        Args:
            metrics: Dictionary containing 'Jaccard Output'

        Returns:
            Jaccard Output * 100, or 0 if None
        """
        jaccard_output = metrics.get('Jaccard Output')

        if jaccard_output is None:
            return 0.0
        else:
            return jaccard_output * 100

    def _compute_jarou_score(self, metrics: Dict[str, Any]) -> float:
        """
        Compute JaRou Score (0 to 100).

        JaRou = Average of (Jaro Similarity + Rouge-L F1) / 2

        Args:
            metrics: Dictionary containing similarity metrics

        Returns:
            JaRou score (0 to 100)
        """
        jaro = metrics.get('Jaro Similarity', 0.0)
        rouge_f1 = metrics.get('Rouge-L F1-score', 0.0)

        jaro_score = jaro * 100 if jaro is not None else 0.0
        rouge_score = rouge_f1 * 100 if rouge_f1 is not None else 0.0

        jarou = (jaro_score + rouge_score) / 2

        return round(jarou, 2)

    def _compute_llmetric_q(self, metrics: Dict[str, Any]) -> float:
        """
        Compute LLMetric-Q composite score (0 to 100).

        Formula:
        - If Pass@1 Score == 100 AND KG Validity Score == 100: return 100
        - Else: weighted sum of component scores

        Weights:
        - w1 = 0.3 (Pass@1)
        - w2 = 0.4 (KG Validity)
        - w3 = 0.2 (Jaccard Output)
        - w4 = 0.1 (JaRou)

        Args:
            metrics: Dictionary containing all scores

        Returns:
            LLMetric-Q score (0 to 100)
        """
        pass1_score = metrics.get('Pass@1 Score', 0.0)
        kg_valid_score = metrics.get('KG Validity Score', 0.0)
        jaccard_output_score = metrics.get('Jaccard Output Score', 0.0)
        jarou_score = metrics.get('JaRou Score', 0.0)

        # Perfect score if both Pass@1 and KG Validity are perfect
        if pass1_score == 100.0 and kg_valid_score == 100.0:
            return 100.0

        # Weighted combination
        w1, w2, w3, w4 = 0.3, 0.4, 0.2, 0.1

        llmetric_q = (
            w1 * pass1_score +
            w2 * kg_valid_score +
            w3 * jaccard_output_score +
            w4 * jarou_score
        )

        return round(llmetric_q, 2)


def create_metrics_dataframe(results: list) -> pd.DataFrame:
    """
    Create DataFrame from list of metric results.

    Columns ordered to match kg-axel baseline output format.

    Args:
        results: List of metric dictionaries

    Returns:
        DataFrame with all metrics
    """
    df = pd.DataFrame(results)

    # Define column order
    columns = [
        'ID Pertanyaan',
        'Teknik Prompt Engineering',
        'Format Representasi Skema KG',
        'Tingkat Penalaran',
        'Sublevel',
        'Tingkat Kompleksitas',
        'Cypher LLM',
        'Syntax Validator',
        'Schema Validator',
        'Properties Validator',
        'Pass@1 Output',
        'Jaccard Output',
        'BLEU',
        'Rouge-L Precision',
        'Rouge-L Recall',
        'Rouge-L F1-score',
        'Jaro Similarity',
        'Jaccard Similarity',
        'Pass@1 Score',
        'KG Validity Score',
        'Jaccard Output Score',
        'JaRou Score',
        'LLMetric-Q'
    ]

    # Reorder columns (include any extra columns at the end)
    existing_columns = [col for col in columns if col in df.columns]
    extra_columns = [col for col in df.columns if col not in columns]

    df = df[existing_columns + extra_columns]

    return df


if __name__ == "__main__":
    # Test comprehensive metrics
    print("Testing Comprehensive Metrics Calculator...")
    print("=" * 60)

    # Initialize calculator (without executor for testing)
    calculator = ComprehensiveMetricsCalculator()

    # Test case
    gt_query = "MATCH (n:MK {nama: 'Basis Data'}) RETURN n.sks"
    gen_query = "MATCH (n:MK {nama: 'Basis Data'}) RETURN n.sks"

    metrics = calculator.compute_all_metrics(
        question_id=1,
        question="Berapa SKS mata kuliah Basis Data?",
        ground_truth_query=gt_query,
        generated_query=gen_query,
        prompt_technique="ReAct Multi-Agent",
        schema_format="only_paths",
        reasoning_level="Fakta Eksplisit",
        sublevel="Nodes",
        complexity="Easy"
    )

    print("\nMetrics computed:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Create DataFrame
    df = create_metrics_dataframe([metrics])
    print(f"\nDataFrame created with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    print("\n" + "=" * 60)
    print("Comprehensive metrics test complete!")
