"""
Similarity and Quality Metrics for Text-to-Cypher Evaluation

Implements comprehensive metrics from baseline research:
- BLEU score
- Rouge-L (Precision, Recall, F1)
- Jaro similarity
- Jaccard similarity
- JaRou score (combined Jaccard + Rouge)
- LLMetric-Q (LLM-based quality assessment)
"""

import re
import logging
from typing import List, Tuple, Dict, Any
from collections import Counter

logger = logging.getLogger(__name__)


def normalize_cypher(query: str) -> str:
    """
    Normalize Cypher query for comparison.

    Args:
        query: Cypher query string

    Returns:
        Normalized query string
    """
    # Remove extra whitespace
    query = re.sub(r'\s+', ' ', query.strip())

    # Convert to lowercase for case-insensitive comparison
    query = query.lower()

    return query


def tokenize_cypher(query: str) -> List[str]:
    """
    Tokenize Cypher query into words/operators.

    Args:
        query: Cypher query string

    Returns:
        List of tokens
    """
    # Normalize first
    query = normalize_cypher(query)

    # Split by whitespace and punctuation
    tokens = re.findall(r'\w+|[^\w\s]', query)

    return tokens


def compute_bleu(reference: str, hypothesis: str, n: int = 4) -> float:
    """
    Compute BLEU score (simplified sentence-level BLEU).

    Args:
        reference: Ground truth query
        hypothesis: Generated query
        n: Maximum n-gram size (default 4)

    Returns:
        BLEU score (0.0 to 1.0)
    """
    ref_tokens = tokenize_cypher(reference)
    hyp_tokens = tokenize_cypher(hypothesis)

    if not hyp_tokens:
        return 0.0

    # Compute precision for each n-gram size
    precisions = []

    for i in range(1, min(n + 1, len(hyp_tokens) + 1)):
        # Extract n-grams
        ref_ngrams = Counter([tuple(ref_tokens[j:j+i]) for j in range(len(ref_tokens) - i + 1)])
        hyp_ngrams = Counter([tuple(hyp_tokens[j:j+i]) for j in range(len(hyp_tokens) - i + 1)])

        # Count matches
        matches = sum((ref_ngrams & hyp_ngrams).values())
        total = sum(hyp_ngrams.values())

        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches / total)

    if not precisions or all(p == 0 for p in precisions):
        return 0.0

    # Geometric mean of precisions
    import math
    geometric_mean = math.exp(sum(math.log(p) if p > 0 else -float('inf') for p in precisions) / len(precisions))

    # Brevity penalty
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)

    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0

    bleu = bp * geometric_mean

    return min(1.0, max(0.0, bleu))


def compute_rouge_l(reference: str, hypothesis: str) -> Tuple[float, float, float]:
    """
    Compute Rouge-L score (Longest Common Subsequence).

    Args:
        reference: Ground truth query
        hypothesis: Generated query

    Returns:
        Tuple of (precision, recall, f1)
    """
    ref_tokens = tokenize_cypher(reference)
    hyp_tokens = tokenize_cypher(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0, 0.0, 0.0

    # Compute LCS length
    def lcs_length(X, Y):
        m, n = len(X), len(Y)
        L = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i-1] == Y[j-1]:
                    L[i][j] = L[i-1][j-1] + 1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])

        return L[m][n]

    lcs_len = lcs_length(ref_tokens, hyp_tokens)

    # Compute precision, recall, F1
    precision = lcs_len / len(hyp_tokens) if hyp_tokens else 0.0
    recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def compute_jaro_similarity(s1: str, s2: str) -> float:
    """
    Compute Jaro similarity between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Jaro similarity (0.0 to 1.0)
    """
    s1 = normalize_cypher(s1)
    s2 = normalize_cypher(s2)

    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)

    if len1 == 0 or len2 == 0:
        return 0.0

    # Maximum distance for matches
    max_dist = max(len1, len2) // 2 - 1
    max_dist = max(1, max_dist)

    # Arrays to track matches
    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    # Find matches
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)

        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3.0

    return jaro


def compute_jaccard_similarity(reference: str, hypothesis: str) -> float:
    """
    Compute Jaccard similarity (token-level).

    Args:
        reference: Ground truth query
        hypothesis: Generated query

    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    ref_tokens = set(tokenize_cypher(reference))
    hyp_tokens = set(tokenize_cypher(hypothesis))

    if not ref_tokens and not hyp_tokens:
        return 1.0

    if not ref_tokens or not hyp_tokens:
        return 0.0

    intersection = len(ref_tokens & hyp_tokens)
    union = len(ref_tokens | hyp_tokens)

    jaccard = intersection / union if union > 0 else 0.0

    return jaccard


def compute_jarou_score(reference: str, hypothesis: str) -> float:
    """
    Compute JaRou score (combined Jaccard + Rouge-L F1).

    This is a weighted combination of Jaccard and Rouge-L metrics.

    Args:
        reference: Ground truth query
        hypothesis: Generated query

    Returns:
        JaRou score (0.0 to 1.0)
    """
    jaccard = compute_jaccard_similarity(reference, hypothesis)
    _, _, rouge_f1 = compute_rouge_l(reference, hypothesis)

    # Weighted average (equal weights)
    jarou = (jaccard + rouge_f1) / 2.0

    return jarou


def compute_llmetric_q(
    reference: str,
    hypothesis: str,
    execution_success: bool,
    pass_at_1: bool
) -> float:
    """
    Compute LLMetric-Q (LLM-based quality metric).

    This is a composite quality score combining:
    - Execution success (40%)
    - Output correctness (30%)
    - Similarity to reference (30%)

    Args:
        reference: Ground truth query
        hypothesis: Generated query
        execution_success: Whether query executed successfully
        pass_at_1: Whether output matches ground truth

    Returns:
        LLMetric-Q score (0.0 to 1.0)
    """
    # Component 1: Execution success (40%)
    exec_score = 1.0 if execution_success else 0.0

    # Component 2: Output correctness (30%)
    output_score = 1.0 if pass_at_1 else 0.0

    # Component 3: Structural similarity (30%)
    # Use JaRou as similarity metric
    similarity_score = compute_jarou_score(reference, hypothesis)

    # Weighted combination
    llmetric_q = (
        0.4 * exec_score +
        0.3 * output_score +
        0.3 * similarity_score
    )

    return llmetric_q


def compute_all_similarity_metrics(
    reference: str,
    hypothesis: str,
    execution_success: bool = False,
    pass_at_1: bool = False
) -> Dict[str, float]:
    """
    Compute all similarity and quality metrics.

    Args:
        reference: Ground truth query
        hypothesis: Generated query
        execution_success: Whether query executed successfully
        pass_at_1: Whether output matches ground truth

    Returns:
        Dictionary with all metrics
    """
    bleu = compute_bleu(reference, hypothesis)
    rouge_p, rouge_r, rouge_f1 = compute_rouge_l(reference, hypothesis)
    jaro = compute_jaro_similarity(reference, hypothesis)
    jaccard = compute_jaccard_similarity(reference, hypothesis)
    jarou = compute_jarou_score(reference, hypothesis)
    llmetric_q = compute_llmetric_q(reference, hypothesis, execution_success, pass_at_1)

    return {
        'bleu': bleu,
        'rouge_l_precision': rouge_p,
        'rouge_l_recall': rouge_r,
        'rouge_l_f1': rouge_f1,
        'jaro_similarity': jaro,
        'jaccard_similarity': jaccard,
        'jarou_score': jarou,
        'llmetric_q': llmetric_q
    }


if __name__ == "__main__":
    # Test metrics
    print("Testing Similarity Metrics...")
    print("=" * 60)

    # Test cases
    ref = "MATCH (n:MK {nama: 'Basis Data'}) RETURN n.sks"
    hyp1 = "MATCH (n:MK {nama: 'Basis Data'}) RETURN n.sks"  # Perfect match
    hyp2 = "MATCH (m:MK {nama: 'Basis Data'}) RETURN m.sks"  # Different variable
    hyp3 = "MATCH (n:MK) WHERE n.nama = 'Basis Data' RETURN n.sks"  # Different syntax, same semantics
    hyp4 = "MATCH (n:Mata_Kuliah {nama: 'Basis Data'}) RETURN n.sks"  # Wrong label

    test_cases = [
        ("Perfect match", ref, hyp1, True, True),
        ("Different variable", ref, hyp2, True, True),
        ("Different syntax", ref, hyp3, True, True),
        ("Wrong label", ref, hyp4, False, False),
    ]

    for name, reference, hypothesis, exec_success, pass_1 in test_cases:
        print(f"\nTest: {name}")
        print(f"Reference: {reference}")
        print(f"Hypothesis: {hypothesis}")
        print(f"Execution: {exec_success}, Pass@1: {pass_1}")
        print()

        metrics = compute_all_similarity_metrics(reference, hypothesis, exec_success, pass_1)

        print(f"  BLEU: {metrics['bleu']:.3f}")
        print(f"  Rouge-L Precision: {metrics['rouge_l_precision']:.3f}")
        print(f"  Rouge-L Recall: {metrics['rouge_l_recall']:.3f}")
        print(f"  Rouge-L F1: {metrics['rouge_l_f1']:.3f}")
        print(f"  Jaro Similarity: {metrics['jaro_similarity']:.3f}")
        print(f"  Jaccard Similarity: {metrics['jaccard_similarity']:.3f}")
        print(f"  JaRou Score: {metrics['jarou_score']:.3f}")
        print(f"  LLMetric-Q: {metrics['llmetric_q']:.3f}")
        print("-" * 60)

    print("\nSimilarity metrics test complete!")
