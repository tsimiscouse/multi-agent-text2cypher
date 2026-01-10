"""
Feedback Aggregator Agent with ReAct Reasoning

Synthesizes feedback from multiple sources into coherent refinement guidance.
Combines evaluator feedback and correction instructions.
Uses ReAct (Reasoning and Acting) pattern for complex aggregation.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI
from utils.reasoning_extractor import ReasoningExtractor


class FeedbackAggregator:
    """
    Agent responsible for aggregating feedback from multiple agents.

    Combines:
    - Evaluation reasoning from QueryEvaluator
    - Error messages from execution
    - Correction instructions from InstructionsGenerator

    Into a single, coherent feedback message for QueryGenerator.
    """

    def __init__(
        self,
        model: str = "qwen/qwen-2.5-coder-32b-instruct",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize Feedback Aggregator agent with ReAct reasoning.

        ReAct (Reasoning and Acting) is always enabled for transparent aggregation traces.

        Args:
            model: LLM model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (1024 for ReAct)
            api_key: API key (default: from env var)
            base_url: API base URL (default: from env var)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.logger = logging.getLogger(__name__)

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url=base_url or os.getenv(
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1"
            )
        )

        # Initialize reasoning extractor
        self.reasoning_extractor = ReasoningExtractor()

        # Load prompts
        self._load_prompts()

    def _load_prompts(self):
        """Load ReAct prompt templates."""
        from utils.prompt_loader import load_prompt_template

        try:
            # Always load ReAct prompt
            self.aggregation_prompt = load_prompt_template("feedback_aggregator_react")
            self.logger.info("Feedback aggregator ReAct prompt loaded")

        except FileNotFoundError as e:
            self.logger.warning(f"Prompt template not found: {e}. Using inline prompt")
            self._use_inline_prompt()

    def _use_inline_prompt(self):
        """Fallback inline prompt."""
        self.aggregation_prompt = """You are a feedback aggregator for Cypher query refinement.

Synthesize the following feedback into clear, actionable guidance:

**Evaluation:**
{evaluation}

**Error Message:**
{error_message}

**Correction Instructions:**
{correction_instructions}

**Task:**
Combine all feedback above into a concise, structured message that:
1. Explains what went wrong
2. Lists specific corrections needed
3. Provides clear guidance for fixing the query

**Aggregated Feedback:**"""

    def aggregate(
        self,
        evaluation: str,
        evaluation_reasoning: str,
        error_message: Optional[str] = None,
        correction_instructions: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Aggregate feedback from multiple sources.

        Args:
            evaluation: Evaluation result ("accept", "incorrect", "error")
            evaluation_reasoning: Reasoning from QueryEvaluator
            error_message: Error message from query execution (if any)
            correction_instructions: Instructions from InstructionsGenerator (if any)

        Returns:
            Tuple of (aggregated_feedback, metadata)
                - aggregated_feedback: Synthesized feedback message
                - metadata: Dict with tokens_used, etc.
        """
        # If query is accepted, no feedback needed
        if evaluation == "accept":
            return "Query accepted. No refinement needed.", {"tokens_used": 0}

        # Check if we have enough information for simple aggregation
        if self._is_simple_aggregation(error_message, correction_instructions):
            feedback = self._aggregate_simple(
                evaluation, evaluation_reasoning, error_message, correction_instructions
            )
            return feedback, {"tokens_used": 0}

        # Use LLM for complex aggregation
        try:
            prompt = self.aggregation_prompt.format(
                evaluation=evaluation,
                evaluation_reasoning=evaluation_reasoning,
                error_message=error_message or "None",
                correction_instructions=correction_instructions or "None"
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            generated_text = response.choices[0].message.content.strip()

            # Extract reasoning trace (always enabled with ReAct)
            reasoning_trace = self.reasoning_extractor.extract_reasoning_trace(generated_text)
            reasoning_quality = self.reasoning_extractor.validate_reasoning_quality(reasoning_trace)
            # Extract final aggregated feedback from ReAct output
            feedback = self.reasoning_extractor.extract_final_output(generated_text)

            metadata = {
                "tokens_used": response.usage.total_tokens,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "has_reasoning": reasoning_trace.get('has_reasoning', False),
                "num_reasoning_steps": reasoning_trace.get('num_steps', 0),
                "reasoning_quality": reasoning_quality.get('quality_score', 0.0),
                "reasoning_trace": reasoning_trace
            }

            self.logger.info(
                f"Aggregated feedback ({metadata['tokens_used']} tokens, "
                f"{reasoning_trace.get('num_steps', 0)} reasoning steps)"
            )

            return feedback, metadata

        except Exception as e:
            self.logger.error(f"Error aggregating feedback: {e}")
            # Fallback to simple aggregation
            feedback = self._aggregate_simple(
                evaluation, evaluation_reasoning, error_message, correction_instructions
            )
            return feedback, {"tokens_used": 0, "error": str(e)}

    def _is_simple_aggregation(
        self,
        error_message: Optional[str],
        correction_instructions: Optional[str]
    ) -> bool:
        """
        Check if simple rule-based aggregation is sufficient.

        Args:
            error_message: Error message
            correction_instructions: Correction instructions

        Returns:
            True if simple aggregation can be used
        """
        # Use simple aggregation if we have clear correction instructions
        if correction_instructions and len(correction_instructions) < 500:
            return True

        # Use simple aggregation if we only have error message
        if error_message and not correction_instructions:
            return True

        return False

    def _aggregate_simple(
        self,
        evaluation: str,
        evaluation_reasoning: str,
        error_message: Optional[str],
        correction_instructions: Optional[str]
    ) -> str:
        """
        Simple rule-based feedback aggregation.

        Args:
            evaluation: Evaluation result
            evaluation_reasoning: Reasoning
            error_message: Error message
            correction_instructions: Correction instructions

        Returns:
            Aggregated feedback string
        """
        feedback_parts = []

        # Add evaluation
        feedback_parts.append(f"**Evaluation: {evaluation.upper()}**")
        feedback_parts.append(evaluation_reasoning)

        # Add error message if present
        if error_message:
            feedback_parts.append(f"\n**Error:**")
            feedback_parts.append(error_message)

        # Add correction instructions if present
        if correction_instructions:
            feedback_parts.append(f"\n**Corrections Needed:**")
            feedback_parts.append(correction_instructions)

        return "\n".join(feedback_parts)

    def aggregate_from_state(
        self,
        iteration_state: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Aggregate feedback from iteration state object.

        Convenience method for working with IterationState dictionaries.

        Args:
            iteration_state: Dictionary containing iteration data

        Returns:
            Tuple of (aggregated_feedback, metadata)
        """
        return self.aggregate(
            evaluation=iteration_state.get("evaluation", "error"),
            evaluation_reasoning=iteration_state.get("evaluation_reasoning", ""),
            error_message=iteration_state.get("error_message"),
            correction_instructions=iteration_state.get("correction_instructions")
        )


if __name__ == "__main__":
    # Test FeedbackAggregator with ReAct
    logging.basicConfig(level=logging.INFO)

    print("Testing FeedbackAggregator with ReAct reasoning...")

    # Test with ReAct enabled
    # Initialize aggregator (ReAct always enabled)
    aggregator = FeedbackAggregator()

    # Test case 1: Accept (no feedback needed - quick path)
    print(f"\n{'='*60}")
    print("Test 1: Accepted query (quick path)")
    print('='*60)

    feedback, metadata = aggregator.aggregate(
        evaluation="accept",
        evaluation_reasoning="Query is correct and returns expected results",
        error_message=None,
        correction_instructions=None
    )
    print(f"Feedback: {feedback}")
    print(f"Tokens: {metadata.get('tokens_used', 0)} (no LLM needed)")

    # Test case 2: Error with message only (simple aggregation)
    print(f"\n{'='*60}")
    print("Test 2: Error with message (simple aggregation)")
    print('='*60)

    feedback, metadata = aggregator.aggregate(
        evaluation="error",
        evaluation_reasoning="Query execution failed due to relationship typo",
        error_message="Relationship type 'PREREQUISIT' not found. Use 'PREREQUISITE' instead.",
        correction_instructions=None
    )
    print(f"Feedback:\n{feedback}")
    print(f"\nTokens: {metadata.get('tokens_used', 0)}")

    # Test case 3: Incorrect with correction instructions (complex - uses ReAct)
    print(f"\n{'='*60}")
    print("Test 3: Incorrect with corrections (ReAct)")
    print('='*60)

    feedback, metadata = aggregator.aggregate(
        evaluation="incorrect",
        evaluation_reasoning="Query tidak memfilter berdasarkan mata kuliah, hasil mungkin include LO dari mata kuliah lain",
        error_message=None,
        correction_instructions="""**Koreksi Semantik:**
- Tambahkan filter MK pada node LO
- Pastikan query hanya mengembalikan data dari mata kuliah yang dimaksud

**Saran:** Hubungkan LO dengan MK menggunakan relasi PART_OF"""
    )
    print(f"Feedback:\n{feedback}")
    print(f"\nMetadata:")
    print(f"  Tokens: {metadata.get('tokens_used', 0)}")
    print(f"  Has Reasoning: {metadata.get('has_reasoning', False)}")
    print(f"  Reasoning Steps: {metadata.get('num_reasoning_steps', 0)}")
    print(f"  Reasoning Quality: {metadata.get('reasoning_quality', 0.0):.2f}")

    # Test case 4: Error with both message and instructions (complex - uses ReAct)
    print(f"\n{'='*60}")
    print("Test 4: Error with message and corrections (ReAct)")
    print('='*60)

    feedback, metadata = aggregator.aggregate(
        evaluation="error",
        evaluation_reasoning="Query has syntax error and semantic issues",
        error_message="Type mismatch: expected Relationship but was String",
        correction_instructions="""**Koreksi Relasi:**
- Ganti 'PREREQUISIT' dengan 'PREREQUISITE' (similaritas: 0.92)

**Koreksi Label Node:**
- Ganti 'M' dengan 'MK' (similaritas: 0.67)"""
    )
    print(f"Feedback:\n{feedback}")
    print(f"\nMetadata:")
    print(f"  Tokens: {metadata.get('tokens_used', 0)}")
    print(f"  Reasoning Steps: {metadata.get('num_reasoning_steps', 0)}")
    print(f"  Reasoning Quality: {metadata.get('reasoning_quality', 0.0):.2f}")

    # Test case 5: Using state dictionary
    print(f"\n{'='*60}")
    print("Test 5: Aggregate from state dictionary")
    print('='*60)

    test_state = {
        "evaluation": "error",
        "evaluation_reasoning": "Query execution failed",
        "error_message": "Property 'kode_mk' does not exist. Use 'kode' instead.",
        "correction_instructions": """**Koreksi Properti:**
- Ganti 'kode_mk' dengan 'kode' (similaritas: 0.80)"""
    }

    feedback, metadata = aggregator.aggregate_from_state(test_state)
    print(f"Feedback:\n{feedback}")
    print(f"\nTokens: {metadata.get('tokens_used', 0)}")

    print("\n" + "="*60)
    print("FeedbackAggregator ReAct test complete!")
