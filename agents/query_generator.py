"""
Query Generator Agent with ReAct Reasoning

Generates initial Cypher queries and refines them based on feedback.
Uses ReAct (Reasoning and Acting) pattern for transparent decision-making.
Core agent in the multi-agent system.
"""

import os
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI
import logging
from utils.reasoning_extractor import ReasoningExtractor, extract_cypher_from_react


class QueryGenerator:
    """
    Agent responsible for generating and refining Cypher queries using ReAct reasoning.

    In the first iteration, generates initial query from question and schema.
    In subsequent iterations, refines query based on aggregated feedback.

    Uses ReAct (Reasoning and Acting) pattern:
    - Thought: What to analyze
    - Action: What to do
    - Observation: What was discovered
    - Final Answer: The Cypher query
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
        Initialize Query Generator agent with ReAct reasoning.

        ReAct (Reasoning and Acting) is always enabled for transparent reasoning traces.

        Args:
            model: LLM model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (1024 for ReAct reasoning)
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
            # Always load ReAct prompts
            self.initial_prompt = load_prompt_template("query_generator_initial_react")
            self.refinement_prompt = load_prompt_template("query_generator_refinement_react")
            self.logger.info("Query generator ReAct prompts loaded")

        except FileNotFoundError as e:
            self.logger.warning(f"Prompt template not found: {e}. Using inline prompts")
            self._use_inline_prompts()

    def _use_inline_prompts(self):
        """Fallback inline prompts."""
        self.initial_prompt = """You are an expert Cypher query generator for Neo4j graph databases.

Given a natural language question in Indonesian and a graph schema, generate a valid Cypher query.

**Schema:**
{schema}

**Question:** {question}

**Instructions:**
1. Analyze the question carefully
2. Identify relevant nodes, relationships, and properties from the schema
3. Generate a syntactically correct Cypher query
4. Return ONLY the Cypher query

**Cypher Query:**"""

        self.refinement_prompt = """You are refining a Cypher query based on feedback.

**Original Question:** {question}

**Schema:**
{schema}

**Previous Query:**
{previous_query}

**Feedback:**
{feedback}

**Instructions:**
1. Read the feedback carefully
2. Correct the identified errors
3. Generate an improved Cypher query
4. Return ONLY the corrected Cypher query

**Improved Cypher Query:**"""

    def generate_initial(
        self,
        question: str,
        schema: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate initial Cypher query with ReAct reasoning.

        Args:
            question: Natural language question
            schema: Graph schema

        Returns:
            Tuple of (query, metadata)
                - query: Generated Cypher query
                - metadata: Dict with tokens_used, reasoning_trace, etc.
        """
        prompt = self.initial_prompt.format(
            question=question,
            schema=schema
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            generated_text = response.choices[0].message.content.strip()

            # Extract reasoning trace
            reasoning_trace = self.reasoning_extractor.extract_reasoning_trace(generated_text)

            # Extract query (always use ReAct-aware extraction)
            query = extract_cypher_from_react(generated_text)

            # Validate reasoning quality
            reasoning_quality = self.reasoning_extractor.validate_reasoning_quality(reasoning_trace)

            metadata = {
                "tokens_used": response.usage.total_tokens,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "has_reasoning": reasoning_trace['has_reasoning'],
                "num_reasoning_steps": reasoning_trace['num_steps'],
                "reasoning_quality": reasoning_quality['quality_score'],
                "reasoning_trace": reasoning_trace
            }

            self.logger.info(
                f"Generated initial query ({metadata['tokens_used']} tokens, "
                f"{reasoning_trace['num_steps']} reasoning steps)"
            )

            return query, metadata

        except Exception as e:
            self.logger.error(f"Error generating initial query: {e}")
            raise

    def refine_query(
        self,
        question: str,
        schema: str,
        previous_query: str,
        feedback: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Refine query based on feedback with ReAct reasoning.

        Args:
            question: Original question
            schema: Graph schema
            previous_query: Previous Cypher query that had errors
            feedback: Aggregated feedback from evaluator and verifier

        Returns:
            Tuple of (refined_query, metadata with reasoning trace)
        """
        prompt = self.refinement_prompt.format(
            question=question,
            schema=schema,
            previous_query=previous_query,
            feedback=feedback
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            generated_text = response.choices[0].message.content.strip()

            # Extract reasoning trace
            reasoning_trace = self.reasoning_extractor.extract_reasoning_trace(generated_text)

            # Extract query (always use ReAct-aware extraction)
            query = extract_cypher_from_react(generated_text)

            # Validate reasoning quality
            reasoning_quality = self.reasoning_extractor.validate_reasoning_quality(reasoning_trace)

            metadata = {
                "tokens_used": response.usage.total_tokens,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "has_reasoning": reasoning_trace['has_reasoning'],
                "num_reasoning_steps": reasoning_trace['num_steps'],
                "reasoning_quality": reasoning_quality['quality_score'],
                "reasoning_trace": reasoning_trace
            }

            self.logger.info(
                f"Refined query ({metadata['tokens_used']} tokens, "
                f"{reasoning_trace['num_steps']} reasoning steps)"
            )

            return query, metadata

        except Exception as e:
            self.logger.error(f"Error refining query: {e}")
            raise

    def _extract_query(self, text: str) -> str:
        """
        Extract Cypher query from generated text.

        Args:
            text: Generated text from LLM

        Returns:
            Cleaned Cypher query
        """
        lines = text.strip().split("\n")

        # Remove markdown code blocks
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        query = "\n".join(lines).strip()

        # Remove common prefixes
        prefixes = [
            "Cypher Query:",
            "Query:",
            "Improved Cypher Query:",
            "Answer:",
            "Cypher:",
        ]

        for prefix in prefixes:
            if query.startswith(prefix):
                query = query[len(prefix):].strip()

        return query


if __name__ == "__main__":
    # Test QueryGenerator with ReAct
    logging.basicConfig(level=logging.INFO)

    print("Testing QueryGenerator with ReAct reasoning...")

    # Initialize generator (ReAct always enabled)
    generator = QueryGenerator()

    test_schema = """(:MK)-[:PREREQUISITE]->(:MK)
(:topic)-[:PART_OF]->(:LO)
(:LO)-[:PURSUED_IN]->(:MK)"""
    test_question = "Apa prasyarat mata kuliah Basis Data?"

    # Test initial generation
    print("\nTest 1: Initial generation with ReAct")
    print("="*60)
    try:
        query, metadata = generator.generate_initial(test_question, test_schema)
        print(f"Generated Query: {query}")
        print(f"\nMetadata:")
        print(f"  Tokens: {metadata['tokens_used']}")
        print(f"  Has Reasoning: {metadata['has_reasoning']}")
        print(f"  Reasoning Steps: {metadata['num_reasoning_steps']}")
        print(f"  Reasoning Quality: {metadata['reasoning_quality']:.2f}")

        # Show reasoning summary
        if metadata['has_reasoning']:
            print(f"\nReasoning Summary:")
            summary = generator.reasoning_extractor.format_reasoning_summary(
                metadata['reasoning_trace']
            )
            print(summary)
    except Exception as e:
        print(f"Error: {e}")

    # Test refinement
    print("\n" + "="*60)
    print("Test 2: Query refinement with ReAct")
    print("="*60)
    try:
        previous_query = "MATCH (p:MK)-[:PREREQUISIT]->(m:MK {nama: 'Basis Data'}) RETURN p.nama"
        feedback = """**Relationship Corrections:**
- Replace 'PREREQUISIT' with 'PREREQUISITE' (similarity: 0.92)"""

        refined, metadata = generator.refine_query(
            test_question, test_schema, previous_query, feedback
        )
        print(f"Refined Query: {refined}")
        print(f"\nMetadata:")
        print(f"  Tokens: {metadata['tokens_used']}")
        print(f"  Reasoning Steps: {metadata['num_reasoning_steps']}")
        print(f"  Reasoning Quality: {metadata['reasoning_quality']:.2f}")

        if metadata['has_reasoning']:
            print(f"\nReasoning Summary:")
            summary = generator.reasoning_extractor.format_reasoning_summary(
                metadata['reasoning_trace']
            )
            print(summary)
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "="*60)
    print("QueryGenerator ReAct test complete!")
