"""
KG-Axel Baseline Generator

Single-shot Cypher query generation using Chain-of-Thought prompting.
Based on kg-axel research: CoT + only_paths configuration.
"""

import os
from typing import Dict, Any, Tuple, Optional
from openai import OpenAI
import logging
from pathlib import Path


class KGAxelGenerator:
    """
    Baseline query generator using kg-axel approach.

    Implements single-shot generation with Chain-of-Thought (CoT) prompting
    and only_paths schema representation (best configuration from kg-axel).
    """

    def __init__(
        self,
        model: str = "qwen/qwen-2.5-coder-32b-instruct",
        temperature: float = 0.0,
        max_tokens: int = 512,
        prompt_type: str = "cot",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize kg-axel generator.

        Args:
            model: Model name (default: qwen-2.5-coder-32b-instruct)
            temperature: Sampling temperature (default: 0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            prompt_type: Prompt type (default: "cot" for Chain-of-Thought)
            api_key: OpenRouter API key (default: from OPENROUTER_API_KEY env var)
            base_url: API base URL (default: from OPENROUTER_BASE_URL env var)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_type = prompt_type

        self.logger = logging.getLogger(__name__)

        # Initialize OpenAI client (compatible with OpenRouter)
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url=base_url or os.getenv(
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1"
            )
        )

        # Load prompt template
        self._load_prompt_template()

    def _load_prompt_template(self):
        """Load CoT prompt template (exactly as used in kg-axel research)."""
        from utils.prompt_loader import load_prompt_template

        try:
            # Load CoT prompt template
            template_name = "cot_prompt_template" if self.prompt_type == "cot" else "base_prompt_template"
            self.prompt_template = load_prompt_template(template_name)
            self.logger.info(f"Loaded prompt template: {template_name}")

        except FileNotFoundError:
            # Fallback should never happen - cot_prompt_template.txt is now available
            self.logger.error("Prompt template file not found! This should not happen.")
            raise

    def generate(
        self,
        question: str,
        schema: str
    ) -> Tuple[str, Any]:
        """
        Generate Cypher query for given question.

        This reproduces kg-axel baseline behavior exactly:
        - System prompt contains CoT template with schema
        - User prompt contains only the question
        - Temperature = 0.0 for deterministic output
        - Reasoning extraction from <think> tags (if present)

        Args:
            question: Natural language question in Indonesian
            schema: Graph schema (only_paths format)

        Returns:
            Tuple of (generated_query, response_object)
                - generated_query: Extracted Cypher query string
                - response_object: Full API response object with usage stats

        Example:
            >>> generator = KGAxelGenerator()
            >>> query, response = generator.generate(
            ...     "Siapa kepala sekolah SMA Negeri 1?",
            ...     "(:Sekolah)-[:DIPIMPIN_OLEH]->(:Kepala_Sekolah)"
            ... )
        """
        # Format system prompt with schema (following kg-axel pattern)
        system_prompt = self.prompt_template.format(schema=schema)

        try:
            # Call LLM with exact kg-axel pattern: system + user messages
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=self.temperature,
                top_p=0.0,  # Match kg-axel exactly
                # max_tokens commented out to match kg-axel (uses default)
                # extra_body for reasoning (matches kg-axel setup)
                extra_body={
                    "reasoning": {
                        "max_tokens": self.max_tokens
                    }
                }
            )

            # Extract generated text
            msg = response.choices[0].message
            full_response = getattr(msg, "content", "").strip()

            # Extract reasoning (if present in <think> tags, like kg-axel)
            import re
            thinking_match = re.search(r'<think>(.*?)</think>', full_response, re.DOTALL)
            reasoning = thinking_match.group(1).strip() if thinking_match else ""

            # Extract cypher query (remove thinking part)
            cypher_query = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()

            # Clean the query
            query = self._extract_query(cypher_query)

            self.logger.debug(
                f"Generated query (tokens: {response.usage.total_tokens}): {query[:100]}..."
            )

            return query, response

        except Exception as e:
            self.logger.error(f"Error during query generation: {e}")
            raise

    def _extract_query(self, text: str) -> str:
        """
        Extract Cypher query from generated text.

        Handles cases where LLM includes explanations or markdown formatting.

        Args:
            text: Generated text from LLM

        Returns:
            Cleaned Cypher query string
        """
        lines = text.strip().split("\n")

        # Remove markdown code blocks
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]

        # Join lines and clean
        query = "\n".join(lines).strip()

        # Remove common prefixes
        prefixes = [
            "Cypher Query:",
            "Query:",
            "Answer:",
            "Cypher:",
        ]

        for prefix in prefixes:
            if query.startswith(prefix):
                query = query[len(prefix):].strip()

        return query

    def batch_generate(
        self,
        questions: list[Dict[str, str]],
        schema: str,
        progress_callback: Optional[callable] = None
    ) -> list[Dict[str, Any]]:
        """
        Generate queries for multiple questions.

        Args:
            questions: List of question dictionaries with 'id' and 'question' keys
            schema: Graph schema
            progress_callback: Optional callback function(current, total)

        Returns:
            List of result dictionaries with:
                - question_id: Question ID
                - question: Question text
                - generated_query: Generated Cypher query
                - tokens_used: Total tokens
                - error: Error message (if failed)
        """
        results = []

        for i, q in enumerate(questions):
            try:
                query, response = self.generate(q["question"], schema)

                results.append({
                    "question_id": q.get("id"),
                    "question": q["question"],
                    "generated_query": query,
                    "total_tokens": response.usage.total_tokens,
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "error": None
                })

            except Exception as e:
                self.logger.error(f"Error on question {q.get('id')}: {e}")

                results.append({
                    "question_id": q.get("id"),
                    "question": q["question"],
                    "generated_query": "",
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "error": str(e)
                })

            # Progress callback
            if progress_callback:
                progress_callback(i + 1, len(questions))

        return results


if __name__ == "__main__":
    # Test KGAxelGenerator
    logging.basicConfig(level=logging.INFO)

    print("Testing KGAxelGenerator...")

    # Initialize generator
    generator = KGAxelGenerator(
        model="qwen/qwen-2.5-coder-32b-instruct",
        temperature=0.0,
        max_tokens=512,
        prompt_type="cot"
    )

    # Test schema
    test_schema = """(:Sekolah)-[:DIPIMPIN_OLEH]->(:Kepala_Sekolah)
(:Sekolah)-[:MEMILIKI]->(:Guru)
(:Guru)-[:MENGAJAR]->(:Mata_Pelajaran)"""

    # Test questions
    test_questions = [
        {"id": 1, "question": "Siapa kepala sekolah SMA Negeri 1?"},
        {"id": 2, "question": "Berapa jumlah guru di sekolah?"}
    ]

    # Test single generation
    print("\nTest 1: Single generation")
    try:
        query, response = generator.generate(
            test_questions[0]["question"],
            test_schema
        )
        print(f"Question: {test_questions[0]['question']}")
        print(f"Generated: {query}")
        print(f"Tokens: {response.usage.total_tokens}")
    except Exception as e:
        print(f"Error: {e}")

    # Test batch generation
    print("\nTest 2: Batch generation")
    try:
        results = generator.batch_generate(
            test_questions,
            test_schema,
            progress_callback=lambda cur, tot: print(f"Progress: {cur}/{tot}")
        )
        print(f"Generated {len(results)} queries")
        for r in results:
            print(f"  Q{r['question_id']}: {r['generated_query'][:50]}... ({r['total_tokens']} tokens)")
    except Exception as e:
        print(f"Error: {e}")

    print("\nKGAxelGenerator test complete!")
