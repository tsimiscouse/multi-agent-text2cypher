"""
Reasoning Extraction Utility

Extracts structured reasoning traces from ReAct-style LLM outputs.
Parses Thought/Action/Observation/Final Answer patterns.
"""

import re
import logging
from typing import Dict, List, Any, Optional


logger = logging.getLogger(__name__)


class ReasoningExtractor:
    """
    Extracts and parses ReAct reasoning traces from LLM outputs.

    Supports multiple patterns:
    - Thought: ... Action: ... Observation: ...
    - **Thought:** ... **Action:** ...
    - Thought 1: ... Action 1: ...
    """

    def __init__(self):
        """Initialize reasoning extractor with Indonesian and English support."""
        # Support both English (Thought/Action/Observation) and Indonesian (Pemikiran/Aksi/Hasil)
        self.thought_pattern = re.compile(
            r'(?:Thought|Pemikiran)\s*(?:\d+)?:\s*(.+?)(?=(?:Action|Aksi|Thought|Pemikiran|Observation|Hasil|Final Answer|Jawaban Akhir|Evaluasi|Penalaran|$))',
            re.IGNORECASE | re.DOTALL
        )

        self.action_pattern = re.compile(
            r'(?:Action|Aksi)\s*(?:\d+)?:\s*(.+?)(?=(?:Observation|Hasil|Thought|Pemikiran|Final Answer|Jawaban Akhir|Evaluasi|$))',
            re.IGNORECASE | re.DOTALL
        )

        self.observation_pattern = re.compile(
            r'(?:Observation|Hasil)\s*(?:\d+)?:\s*(.+?)(?=(?:Thought|Pemikiran|Action|Aksi|Final Answer|Jawaban Akhir|Evaluasi|$))',
            re.IGNORECASE | re.DOTALL
        )

        self.final_answer_pattern = re.compile(
            r'(?:Final Answer|Jawaban Akhir|Answer):\s*(.+?)$',
            re.IGNORECASE | re.DOTALL
        )

    def extract_reasoning_trace(self, text: str) -> Dict[str, Any]:
        """
        Extract complete reasoning trace from text.

        Args:
            text: LLM output containing reasoning steps

        Returns:
            Dictionary with reasoning components:
            {
                'has_reasoning': bool,
                'thoughts': List[str],
                'actions': List[str],
                'observations': List[str],
                'final_answer': str,
                'num_steps': int,
                'raw_trace': str
            }
        """
        # Extract all components
        thoughts = [m.group(1).strip() for m in self.thought_pattern.finditer(text)]
        actions = [m.group(1).strip() for m in self.action_pattern.finditer(text)]
        observations = [m.group(1).strip() for m in self.observation_pattern.finditer(text)]

        # Extract final answer
        final_answer_match = self.final_answer_pattern.search(text)
        final_answer = final_answer_match.group(1).strip() if final_answer_match else ""

        # Check if reasoning exists
        has_reasoning = len(thoughts) > 0 or len(actions) > 0

        return {
            'has_reasoning': has_reasoning,
            'thoughts': thoughts,
            'actions': actions,
            'observations': observations,
            'final_answer': final_answer,
            'num_steps': max(len(thoughts), len(actions), len(observations)),
            'raw_trace': text.strip()
        }

    def extract_final_output(self, text: str) -> str:
        """
        Extract the final output/answer from ReAct trace.

        Tries to find:
        1. Final Answer / Jawaban Akhir: ...
        2. Last line after reasoning
        3. Full text if no pattern found

        Args:
            text: LLM output

        Returns:
            Final output string
        """
        # Try to find explicit final answer
        final_answer_match = self.final_answer_pattern.search(text)
        if final_answer_match:
            return final_answer_match.group(1).strip()

        # Try to find last non-empty line after reasoning markers
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            # Skip reasoning markers (both English and Indonesian)
            if line and not re.match(
                r'^(Thought|Action|Observation|Pemikiran|Aksi|Hasil|Evaluasi|Penalaran|Masalah|Koreksi|Aksi yang):',
                line,
                re.IGNORECASE
            ):
                return line

        # Fallback: return full text
        return text.strip()

    def format_reasoning_summary(self, reasoning_trace: Dict[str, Any]) -> str:
        """
        Format reasoning trace into readable summary.

        Args:
            reasoning_trace: Extracted reasoning trace dict

        Returns:
            Formatted string summary
        """
        if not reasoning_trace['has_reasoning']:
            return "No explicit reasoning found"

        summary = []
        summary.append(f"Reasoning Steps: {reasoning_trace['num_steps']}")

        # Show thoughts
        if reasoning_trace['thoughts']:
            summary.append("\nThoughts:")
            for i, thought in enumerate(reasoning_trace['thoughts'], 1):
                summary.append(f"  {i}. {thought[:100]}..." if len(thought) > 100 else f"  {i}. {thought}")

        # Show actions
        if reasoning_trace['actions']:
            summary.append("\nActions:")
            for i, action in enumerate(reasoning_trace['actions'], 1):
                summary.append(f"  {i}. {action[:100]}..." if len(action) > 100 else f"  {i}. {action}")

        # Show final answer
        if reasoning_trace['final_answer']:
            summary.append(f"\nFinal Answer: {reasoning_trace['final_answer'][:100]}...")

        return '\n'.join(summary)

    def validate_reasoning_quality(self, reasoning_trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate quality of reasoning trace.

        Args:
            reasoning_trace: Extracted reasoning trace

        Returns:
            Quality metrics dict
        """
        quality = {
            'has_reasoning': reasoning_trace['has_reasoning'],
            'is_complete': False,
            'num_steps': reasoning_trace['num_steps'],
            'has_final_answer': bool(reasoning_trace['final_answer']),
            'avg_thought_length': 0,
            'quality_score': 0.0
        }

        if not reasoning_trace['has_reasoning']:
            return quality

        # Check completeness (should have thoughts AND final answer)
        quality['is_complete'] = (
            len(reasoning_trace['thoughts']) > 0 and
            bool(reasoning_trace['final_answer'])
        )

        # Calculate average thought length
        if reasoning_trace['thoughts']:
            quality['avg_thought_length'] = sum(
                len(t) for t in reasoning_trace['thoughts']
            ) / len(reasoning_trace['thoughts'])

        # Calculate quality score (0-1)
        score = 0.0
        if quality['has_final_answer']:
            score += 0.3
        if quality['num_steps'] > 0:
            score += 0.3
        if quality['is_complete']:
            score += 0.2
        if quality['avg_thought_length'] > 20:  # Meaningful thoughts
            score += 0.2

        quality['quality_score'] = score

        return quality


def extract_cypher_from_react(text: str) -> str:
    """
    Extract Cypher query from ReAct-formatted output.

    Handles:
    - Final Answer: MATCH ...
    - Plain query after reasoning
    - Code blocks

    Args:
        text: ReAct output from LLM

    Returns:
        Extracted Cypher query
    """
    extractor = ReasoningExtractor()

    # First try to get final answer
    final_output = extractor.extract_final_output(text)

    # Remove markdown code blocks
    final_output = re.sub(r'```(?:cypher)?\s*', '', final_output)
    final_output = re.sub(r'```\s*$', '', final_output)

    # Remove common prefixes
    prefixes = [
        "Cypher Query:",
        "Query:",
        "Improved Cypher Query:",
        "Answer:",
        "Cypher:",
        "Final Query:",
    ]

    for prefix in prefixes:
        if final_output.startswith(prefix):
            final_output = final_output[len(prefix):].strip()

    return final_output.strip()


if __name__ == "__main__":
    # Test reasoning extractor
    print("Testing ReasoningExtractor...")

    extractor = ReasoningExtractor()

    # Test case 1: Full ReAct trace
    test_react_1 = """
Thought: I need to identify entities in the question "Apa prasyarat Basis Data?"
Action: Extract entities from the question
Observation: Found "Basis Data" which is a MK (Mata Kuliah) node, and "prasyarat" indicates PREREQUISITE relationship

Thought: The question asks for prerequisites of "Basis Data", so I need to find MK nodes that are prerequisites
Action: Determine the correct relationship direction
Observation: PREREQUISITE relationship points FROM prerequisite TO the target course

Thought: I should generate a Cypher query with the pattern (prerequisite)-[:PREREQUISITE]->(target)
Final Answer: MATCH (p:MK)-[:PREREQUISITE]->(m:MK {nama: "Basis Data"}) RETURN p.nama
"""

    trace = extractor.extract_reasoning_trace(test_react_1)
    print("\nTest 1: Full ReAct trace")
    print(f"Has reasoning: {trace['has_reasoning']}")
    print(f"Num steps: {trace['num_steps']}")
    print(f"Thoughts: {len(trace['thoughts'])}")
    print(f"Actions: {len(trace['actions'])}")
    print(f"Observations: {len(trace['observations'])}")
    print(f"Final answer: {trace['final_answer'][:50]}...")

    # Extract Cypher
    cypher = extract_cypher_from_react(test_react_1)
    print(f"\nExtracted Cypher: {cypher}")

    # Validate quality
    quality = extractor.validate_reasoning_quality(trace)
    print(f"\nQuality: {quality}")

    # Test case 2: Simple output without reasoning
    test_simple = "MATCH (m:MK) RETURN m.nama"

    trace2 = extractor.extract_reasoning_trace(test_simple)
    print("\n" + "="*60)
    print("Test 2: Simple output (no reasoning)")
    print(f"Has reasoning: {trace2['has_reasoning']}")
    print(f"Final output: {extractor.extract_final_output(test_simple)}")

    # Test case 3: Indonesian ReAct
    test_react_indo = """
Pemikiran: Saya perlu mengekstrak entitas dari pertanyaan
Aksi: Identifikasi node dan relationship
Hasil: Ditemukan node MK dan relationship PREREQUISITE

Pemikiran: Saya akan membuat query Cypher
Jawaban Akhir: MATCH (m:MK)-[:PREREQUISITE]->(n:MK) RETURN m, n
"""

    trace3 = extractor.extract_reasoning_trace(test_react_indo)
    print("\n" + "="*60)
    print("Test 3: Indonesian ReAct")
    print(f"Has reasoning: {trace3['has_reasoning']}")
    print(f"Num steps: {trace3['num_steps']}")
    print(f"\n{extractor.format_reasoning_summary(trace3)}")

    print("\n" + "="*60)
    print("ReasoningExtractor test complete!")
