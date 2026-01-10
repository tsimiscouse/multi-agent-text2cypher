"""
Cypher Query Similarity Metrics

Implements text-based similarity metrics for Cypher queries following kg-axel approach.
Metrics compare generated query text vs ground truth query text.
"""

import re
import logging
from typing import Tuple
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import textdistance

logger = logging.getLogger(__name__)


def format_rouge_bleu(query: str) -> str:
    """
    Format Cypher query for ROUGE and BLEU by adding whitespaces.

    Adds spaces before/after:
    - Node delimiters: ( ) :
    - Property delimiters: { } .
    - Relationship delimiters: [ ] - -> <-
    - Other: , * / < >

    Args:
        query: Cypher query string

    Returns:
        Formatted query with normalized whitespace
    """
    pattern = r'(\(|\)|:|\{|\}|->|<-|-|\[|\]|,|\.|\*|\/|<|>)'

    # Add spaces around special characters
    formatted = re.sub(pattern, r' \1 ', query)

    # Normalize multiple spaces to single space
    formatted = re.sub(r'\s+', ' ', formatted).strip()

    return formatted


def format_jaccard_jaro(query: str, remove_special: bool = False) -> str:
    """
    Format Cypher query for Jaccard and Jaro-Winkler by:
    1. Adding whitespaces around special characters
    2. Uppercasing Neo4j keywords
    3. Optionally removing special characters

    Args:
        query: Cypher query string
        remove_special: If True, remove special characters after spacing

    Returns:
        Formatted query
    """
    # Step 1: Add whitespaces
    pattern = r'(\(|\)|:|\{|\}|->|<-|-|\[|\]|,|\.|\*|\/|<|>)'
    formatted = re.sub(pattern, r' \1 ', query)

    # Step 2: Remove special characters if requested
    if remove_special:
        formatted = re.sub(pattern, ' ', formatted)

    # Normalize spaces
    formatted = re.sub(r'\s+', ' ', formatted).strip()

    # Step 3: Uppercase Neo4j keywords
    keywords = [
        'MATCH', 'WHERE', 'RETURN', 'OPTIONAL', 'MERGE', 'DELETE', 'CREATE',
        'ORDER BY', 'LIMIT', 'COUNT', 'DISTINCT', 'EXISTS', 'IN', 'AND', 'OR',
        'MIN', 'MAX', 'AVG', 'UNWIND', 'WITH', 'SET', 'REMOVE', 'FILTER',
        'COLLECT', 'AS', 'FOREACH', 'REDUCE', 'CASE', 'toFloat', 'NOT', 'NULL',
        'THEN', 'ELSE', 'END', 'TRUE', 'FALSE', 'DESC', 'ASC', 'IS', 'WHEN', 'SIZE'
    ]

    # Create pattern with word boundaries
    keyword_pattern = r'\b(' + '|'.join(keywords) + r')\b'

    # Replace with uppercase
    formatted = re.sub(
        keyword_pattern,
        lambda x: x.group(0).upper(),
        formatted,
        flags=re.IGNORECASE
    )

    return formatted


def compute_bleu(ground_truth: str, generated: str) -> float:
    """
    Compute BLEU score between ground truth and generated Cypher queries.

    Uses sentence-level BLEU with formatting following kg-axel approach.

    Args:
        ground_truth: Ground truth Cypher query
        generated: Generated Cypher query

    Returns:
        BLEU score (0.0 to 1.0)
    """
    reference = format_rouge_bleu(ground_truth).split()
    hypothesis = format_rouge_bleu(generated).split()

    try:
        score = sentence_bleu([reference], hypothesis)
        return round(score, 2)
    except Exception as e:
        logger.warning(f"Error computing BLEU: {e}")
        return 0.0


def compute_rouge_l(ground_truth: str, generated: str) -> Tuple[float, float, float]:
    """
    Compute Rouge-L score (Longest Common Subsequence).

    Args:
        ground_truth: Ground truth Cypher query
        generated: Generated Cypher query

    Returns:
        Tuple of (precision, recall, f1)
    """
    reference = format_rouge_bleu(ground_truth)
    hypothesis = format_rouge_bleu(generated)

    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)['rougeL']

        precision = round(scores.precision, 2)
        recall = round(scores.recall, 2)
        f1 = round(scores.fmeasure, 2)

        return precision, recall, f1
    except Exception as e:
        logger.warning(f"Error computing Rouge-L: {e}")
        return 0.0, 0.0, 0.0


def compute_jaro_similarity(ground_truth: str, generated: str) -> float:
    """
    Compute Jaro-Winkler similarity (character-based).

    Args:
        ground_truth: Ground truth Cypher query
        generated: Generated Cypher query

    Returns:
        Jaro-Winkler similarity (0.0 to 1.0)
    """
    reference = format_jaccard_jaro(ground_truth)
    hypothesis = format_jaccard_jaro(generated)

    try:
        score = textdistance.jaro_winkler(reference, hypothesis)
        return round(score, 2)
    except Exception as e:
        logger.warning(f"Error computing Jaro-Winkler: {e}")
        return 0.0


def compute_jaccard_similarity(ground_truth: str, generated: str) -> float:
    """
    Compute Jaccard similarity (word-based, with special characters removed).

    Args:
        ground_truth: Ground truth Cypher query
        generated: Generated Cypher query

    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    # Format with special characters removed
    words_gt = set(format_jaccard_jaro(ground_truth, remove_special=True).split())
    words_gen = set(format_jaccard_jaro(generated, remove_special=True).split())

    if not words_gt and not words_gen:
        return 1.0

    if not words_gt or not words_gen:
        return 0.0

    intersection = len(words_gt & words_gen)
    union = len(words_gt | words_gen)

    score = intersection / union if union > 0 else 0.0

    return round(score, 2)


def compute_all_cypher_similarities(ground_truth: str, generated: str) -> dict:
    """
    Compute all Cypher query text similarity metrics.

    Args:
        ground_truth: Ground truth Cypher query
        generated: Generated Cypher query

    Returns:
        Dictionary with all similarity scores
    """
    bleu = compute_bleu(ground_truth, generated)
    rouge_p, rouge_r, rouge_f1 = compute_rouge_l(ground_truth, generated)
    jaro = compute_jaro_similarity(ground_truth, generated)
    jaccard = compute_jaccard_similarity(ground_truth, generated)

    return {
        'BLEU': bleu,
        'Rouge-L Precision': rouge_p,
        'Rouge-L Recall': rouge_r,
        'Rouge-L F1-score': rouge_f1,
        'Jaro Similarity': jaro,
        'Jaccard Similarity': jaccard
    }


if __name__ == "__main__":
    # Test Cypher similarity metrics
    print("Testing Cypher Similarity Metrics...")
    print("=" * 60)

    # Test cases
    gt = "MATCH (n:MK {nama: 'Basis Data'}) RETURN n.sks"

    test_cases = [
        ("Perfect match", gt),
        ("Different variable", "MATCH (m:MK {nama: 'Basis Data'}) RETURN m.sks"),
        ("Different syntax", "MATCH (n:MK) WHERE n.nama = 'Basis Data' RETURN n.sks"),
        ("Wrong label", "MATCH (n:Mata_Kuliah {nama: 'Basis Data'}) RETURN n.sks"),
        ("Completely wrong", "CREATE (n:Student {name: 'John'}) RETURN n"),
    ]

    for name, generated in test_cases:
        print(f"\nTest: {name}")
        print(f"Ground Truth: {gt}")
        print(f"Generated:    {generated}")

        metrics = compute_all_cypher_similarities(gt, generated)

        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print("-" * 60)

    print("\nCypher similarity metrics test complete!")
