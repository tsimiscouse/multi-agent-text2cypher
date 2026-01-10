"""
Query Evaluator Agent

LLM-based evaluation of generated Cypher queries.
Determines if query should be accepted, needs correction, or has errors.
"""

import os
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI
import logging


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
        max_tokens: int = 512,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize Query Evaluator agent.

        Args:
            model: LLM model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
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

        # Load prompts
        self._load_prompts()

    def _load_prompts(self):
        """Load prompt templates."""
        from utils.prompt_loader import load_prompt_template

        try:
            self.evaluation_prompt = load_prompt_template("query_evaluator")
            self.logger.info("Query evaluator prompt loaded")

        except FileNotFoundError:
            self.logger.warning("Using inline prompt template")
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

            # Parse evaluation
            evaluation = self._parse_evaluation(generated_text)
            reasoning = generated_text

            metadata = {
                "tokens_used": response.usage.total_tokens,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "error_message": error_message
            }

            self.logger.info(f"LLM evaluation: {evaluation}")

            return evaluation, reasoning, metadata

        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            # Fallback to conservative evaluation
            return "error", f"Evaluation failed: {str(e)}", {"tokens_used": 0, "error_message": str(e)}

    def _parse_evaluation(self, text: str) -> str:
        """
        Parse evaluation from LLM response.

        Args:
            text: Generated text from LLM

        Returns:
            One of: "accept", "incorrect", "error"
        """
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
    # Test QueryEvaluator
    logging.basicConfig(level=logging.INFO)

    print("Testing QueryEvaluator...")

    evaluator = QueryEvaluator()

    # Test case 1: Successful query
    print("\nTest 1: Successful query")
    test_result_success = {
        "success": True,
        "records": [{"name": "John Doe"}, {"name": "Jane Smith"}],
        "record_count": 2,
        "error": None,
        "error_type": None
    }

    evaluation, reasoning, metadata = evaluator.evaluate(
        question="Who are the teachers?",
        query="MATCH (g:Guru) RETURN g.nama AS name",
        execution_result=test_result_success
    )
    print(f"Evaluation: {evaluation}")
    print(f"Reasoning: {reasoning[:100]}...")
    print(f"Tokens: {metadata['tokens_used']}")

    # Test case 2: Empty result
    print("\nTest 2: Empty result")
    test_result_empty = {
        "success": True,
        "records": [],
        "record_count": 0,
        "error": None,
        "error_type": None
    }

    evaluation, reasoning, metadata = evaluator.evaluate(
        question="Who are the teachers?",
        query="MATCH (g:Guru {nama: 'NonExistent'}) RETURN g.nama",
        execution_result=test_result_empty
    )
    print(f"Evaluation: {evaluation}")
    print(f"Reasoning: {reasoning[:100]}...")

    # Test case 3: Syntax error
    print("\nTest 3: Syntax error")
    test_result_error = {
        "success": False,
        "records": [],
        "record_count": 0,
        "error": "Invalid syntax: expected property name",
        "error_type": "SyntaxError"
    }

    evaluation, reasoning, metadata = evaluator.evaluate(
        question="Who are the teachers?",
        query="MATCH (g:Guru) RETURN g.invalid_property",
        execution_result=test_result_error
    )
    print(f"Evaluation: {evaluation}")
    print(f"Reasoning: {reasoning}")
    print(f"Error type: {metadata.get('error_type')}")

    # Test case 4: Simple evaluation (no LLM)
    print("\nTest 4: Simple rule-based evaluation")
    simple_eval = evaluator.evaluate_simple(
        execution_success=True,
        is_empty_result=False,
        error_message=None
    )
    print(f"Simple evaluation: {simple_eval}")

    print("\nQueryEvaluator test complete!")
