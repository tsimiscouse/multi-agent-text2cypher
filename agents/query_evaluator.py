"""
Query Evaluator Agent with ReAct Reasoning

LLM-based evaluation of generated Cypher queries with transparent reasoning.
Determines if query should be accepted, needs correction, or has errors.
Uses ReAct (Reasoning and Acting) pattern for interpretable evaluation.
"""

import os
import re
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI
import logging
from utils.reasoning_extractor import ReasoningExtractor


class QueryEvaluator:
    """
    Agent responsible for evaluating Cypher queries.

    Executes query and uses LLM to judge if it:
    - Accepts: Query is correct and returns expected results
    - Incorrect: Query executes but returns wrong/unexpected results
    - Error: Query has syntax errors or execution failures
    """

    def __init__(
        self,
        model: str = "qwen/qwen-2.5-coder-32b-instruct",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_react: bool = True
    ):
        """
        Initialize Query Evaluator agent with ReAct reasoning.

        Args:
            model: LLM model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (increased for ReAct reasoning)
            api_key: API key (default: from env var)
            base_url: API base URL (default: from env var)
            use_react: Whether to use ReAct prompts (default: True)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_react = use_react

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
        """Load prompt templates (ReAct or standard)."""
        from utils.prompt_loader import load_prompt_template

        try:
            if self.use_react:
                # Load ReAct prompt
                self.evaluation_prompt = load_prompt_template("query_evaluator_react")
                self.logger.info("Query evaluator ReAct prompt loaded")
            else:
                # Load standard prompt
                self.evaluation_prompt = load_prompt_template("query_evaluator")
                self.logger.info("Query evaluator standard prompt loaded")

        except FileNotFoundError as e:
            self.logger.warning(f"Prompt template not found: {e}. Using inline prompt")
            self._use_inline_prompt()

    def _use_inline_prompt(self):
        """Fallback inline prompt."""
        self.evaluation_prompt = """You are a Cypher query evaluator. Evaluate if the generated query correctly answers the question.

**Question:** {question}

**Generated Cypher Query:**
{query}

**Execution Result:**
{execution_result}

**Execution Status:** {execution_status}

**Instructions:**
1. If the query has syntax errors or failed to execute → return "error"
2. If the query executed but returns wrong/empty results → return "incorrect"
3. If the query executed successfully and results look correct → return "accept"

**Your Evaluation (one word only - "accept", "incorrect", or "error"):**"""

    def evaluate(
        self,
        question: str,
        query: str,
        execution_result: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Evaluate Cypher query.

        Args:
            question: Original natural language question
            query: Generated Cypher query
            execution_result: Result from GraphExecutor.execute_with_metadata()

        Returns:
            Tuple of (evaluation, reasoning, metadata)
                - evaluation: "accept", "incorrect", or "error"
                - reasoning: Explanation for the evaluation
                - metadata: Dict with tokens_used, error_message, etc.
        """
        # Handle execution result
        if execution_result is None:
            execution_status = "Not executed"
            result_preview = "N/A"
            error_message = None
        else:
            execution_status = "Success" if execution_result.get("success") else "Failed"

            if execution_result.get("success"):
                records = execution_result.get("records", [])
                if len(records) == 0:
                    result_preview = "Query returned empty results"
                else:
                    # Show first few records
                    result_preview = str(records[:3])
                    if len(records) > 3:
                        result_preview += f"\n... and {len(records) - 3} more records"
                error_message = None
            else:
                result_preview = "Execution failed"
                error_message = execution_result.get("error", "Unknown error")

        # Quick evaluation for obvious cases (no LLM needed)
        if execution_result and not execution_result.get("success"):
            # Execution failed → error
            evaluation = "error"
            reasoning = f"Query execution failed: {error_message}"

            metadata = {
                "tokens_used": 0,  # No LLM call needed
                "error_message": error_message,
                "error_type": execution_result.get("error_type")
            }

            self.logger.info(f"Quick evaluation: {evaluation} (execution failed)")
            return evaluation, reasoning, metadata

        # Use LLM for nuanced evaluation
        try:
            prompt = self.evaluation_prompt.format(
                question=question,
                query=query,
                execution_result=result_preview,
                execution_status=execution_status
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            generated_text = response.choices[0].message.content.strip()

            # Extract reasoning trace if using ReAct
            if self.use_react:
                reasoning_trace = self.reasoning_extractor.extract_reasoning_trace(generated_text)
                reasoning_quality = self.reasoning_extractor.validate_reasoning_quality(reasoning_trace)
            else:
                reasoning_trace = {}
                reasoning_quality = {}

            # Parse evaluation and reasoning
            evaluation = self._parse_evaluation(generated_text)
            reasoning = self._extract_reasoning_text(generated_text)

            metadata = {
                "tokens_used": response.usage.total_tokens,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "error_message": error_message,
                "has_reasoning": reasoning_trace.get('has_reasoning', False),
                "num_reasoning_steps": reasoning_trace.get('num_steps', 0),
                "reasoning_quality": reasoning_quality.get('quality_score', 0.0),
                "reasoning_trace": reasoning_trace
            }

            self.logger.info(
                f"LLM evaluation: {evaluation} "
                f"({reasoning_trace.get('num_steps', 0)} reasoning steps)"
            )

            return evaluation, reasoning, metadata

        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            # Fallback to conservative evaluation
            return "error", f"Evaluation failed: {str(e)}", {"tokens_used": 0, "error_message": str(e)}

    def _parse_evaluation(self, text: str) -> str:
        """
        Parse evaluation from LLM response (supports ReAct format).

        Args:
            text: Generated text from LLM

        Returns:
            One of: "accept", "incorrect", "error"
        """
        # Try to find explicit "Evaluation: <result>" pattern (ReAct format)
        evaluation_match = re.search(r'Evaluation:\s*(accept|incorrect|error)', text, re.IGNORECASE)
        if evaluation_match:
            return evaluation_match.group(1).lower()

        # Fallback: search in full text
        text_lower = text.lower().strip()

        # Check for explicit keywords
        if "accept" in text_lower:
            return "accept"
        elif "incorrect" in text_lower or "wrong" in text_lower:
            return "incorrect"
        elif "error" in text_lower or "fail" in text_lower:
            return "error"

        # Default to incorrect if unclear
        self.logger.warning(f"Unclear evaluation: {text[:100]}, defaulting to 'incorrect'")
        return "incorrect"

    def _extract_reasoning_text(self, text: str) -> str:
        """
        Extract reasoning text from evaluation response.

        For ReAct format, extracts the "Reasoning:" part.
        For standard format, returns full text.

        Args:
            text: Generated text from LLM

        Returns:
            Reasoning text
        """
        # Try to find "Reasoning: <text>" pattern (ReAct format)
        reasoning_match = re.search(r'Reasoning:\s*(.+?)(?:\n\n|$)', text, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            return reasoning_match.group(1).strip()

        # Fallback: return full text
        return text.strip()

    def evaluate_simple(
        self,
        execution_success: bool,
        is_empty_result: bool,
        error_message: Optional[str] = None
    ) -> str:
        """
        Simple rule-based evaluation without LLM.

        Useful for fast evaluation when LLM-based reasoning is not needed.

        Args:
            execution_success: Whether query executed successfully
            is_empty_result: Whether query returned empty results
            error_message: Error message if execution failed

        Returns:
            Evaluation: "accept", "incorrect", or "error"
        """
        if not execution_success:
            return "error"
        elif is_empty_result:
            return "incorrect"
        else:
            return "accept"


if __name__ == "__main__":
    # Test QueryEvaluator with ReAct
    logging.basicConfig(level=logging.INFO)

    print("Testing QueryEvaluator with ReAct reasoning...")

    # Test with ReAct enabled
    evaluator = QueryEvaluator(use_react=True)

    # Test case 1: Successful query
    print("\nTest 1: Successful query with ReAct")
    print("="*60)
    test_result_success = {
        "success": True,
        "records": [{"nama": "Pemrograman Dasar"}, {"nama": "Matematika Diskrit"}],
        "record_count": 2,
        "error": None,
        "error_type": None
    }

    evaluation, reasoning, metadata = evaluator.evaluate(
        question="Apa saja prasyarat untuk Basis Data?",
        query="MATCH (p:MK)-[:PREREQUISITE]->(m:MK {nama: 'Basis Data'}) RETURN p.nama",
        execution_result=test_result_success
    )
    print(f"Evaluation: {evaluation}")
    print(f"Reasoning: {reasoning[:150]}...")
    print(f"\nMetadata:")
    print(f"  Tokens: {metadata['tokens_used']}")
    print(f"  Has Reasoning: {metadata.get('has_reasoning', False)}")
    print(f"  Reasoning Steps: {metadata.get('num_reasoning_steps', 0)}")
    print(f"  Reasoning Quality: {metadata.get('reasoning_quality', 0.0):.2f}")

    # Test case 2: Empty result
    print("\n" + "="*60)
    print("Test 2: Empty result")
    print("="*60)
    test_result_empty = {
        "success": True,
        "records": [],
        "record_count": 0,
        "error": None,
        "error_type": None
    }

    evaluation, reasoning, metadata = evaluator.evaluate(
        question="Apa prasyarat untuk mata kuliah yang tidak ada?",
        query="MATCH (p:MK)-[:PREREQUISITE]->(m:MK {nama: 'NonExistent'}) RETURN p.nama",
        execution_result=test_result_empty
    )
    print(f"Evaluation: {evaluation}")
    print(f"Reasoning: {reasoning[:150]}...")
    print(f"Reasoning Steps: {metadata.get('num_reasoning_steps', 0)}")

    # Test case 3: Syntax error (quick path - no LLM)
    print("\n" + "="*60)
    print("Test 3: Syntax error (quick evaluation)")
    print("="*60)
    test_result_error = {
        "success": False,
        "records": [],
        "record_count": 0,
        "error": "Relationship type 'PREREQUISIT' not found",
        "error_type": "SyntaxError"
    }

    evaluation, reasoning, metadata = evaluator.evaluate(
        question="Apa prasyarat Basis Data?",
        query="MATCH (p:MK)-[:PREREQUISIT]->(m:MK {nama: 'Basis Data'}) RETURN p.nama",
        execution_result=test_result_error
    )
    print(f"Evaluation: {evaluation}")
    print(f"Reasoning: {reasoning}")
    print(f"Error type: {metadata.get('error_type')}")
    print(f"Tokens used: {metadata['tokens_used']} (quick path - no LLM)")

    # Test case 4: Simple evaluation (no LLM)
    print("\n" + "="*60)
    print("Test 4: Simple rule-based evaluation")
    print("="*60)
    simple_eval = evaluator.evaluate_simple(
        execution_success=True,
        is_empty_result=False,
        error_message=None
    )
    print(f"Simple evaluation: {simple_eval}")

    print("\n" + "="*60)
    print("QueryEvaluator ReAct test complete!")
