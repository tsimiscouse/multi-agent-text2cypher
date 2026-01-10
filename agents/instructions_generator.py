"""
Instructions Generator Agent with ReAct Reasoning

Generates correction instructions based on entity verification results.
Provides clear, actionable guidance for query refinement.
Uses ReAct (Reasoning and Acting) pattern for complex cases.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI
from utils.reasoning_extractor import ReasoningExtractor


class InstructionsGenerator:
    """
    Agent responsible for generating correction instructions.

    Takes verification results from EntityVerifier and generates
    clear, structured instructions for fixing invalid entities.
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
        Initialize Instructions Generator agent with ReAct reasoning.

        Args:
            model: LLM model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (increased for ReAct)
            api_key: API key (default: from env var)
            base_url: API base URL (default: from env var)
            use_react: Whether to use ReAct prompts for complex cases (default: True)
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
                self.instructions_prompt = load_prompt_template("instructions_generator_react")
                self.logger.info("Instructions generator ReAct prompt loaded")
            else:
                # Load standard prompt
                self.instructions_prompt = load_prompt_template("instructions_generator")
                self.logger.info("Instructions generator standard prompt loaded")

        except FileNotFoundError as e:
            self.logger.warning(f"Prompt template not found: {e}. Using inline prompt")
            self._use_inline_prompt()

    def _use_inline_prompt(self):
        """Fallback inline prompt."""
        self.instructions_prompt = """You are a Cypher query correction assistant.

Based on the verification results below, generate clear correction instructions.

**Invalid Entities Found:**
{invalid_entities}

**Correction Suggestions:**
{suggestions}

**Instructions:**
Generate a concise list of corrections needed to fix the query. For each invalid entity:
1. State what is wrong
2. Provide the correct entity name
3. Explain why it's incorrect (if obvious)

Be specific and actionable.

**Correction Instructions:**"""

    def generate(
        self,
        verification_result: Dict[str, Any],
        query: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate correction instructions from verification results.

        Args:
            verification_result: Output from EntityVerifier.verify()
            query: Optional original query for context

        Returns:
            Tuple of (instructions, metadata)
                - instructions: Generated correction instructions
                - metadata: Dict with tokens_used, etc.
        """
        # Check if there are any errors
        if not verification_result.get("verification_summary", {}).get("has_errors"):
            return "No corrections needed. All entities are valid.", {"tokens_used": 0}

        # Use rule-based generation for simple cases
        if self._is_simple_case(verification_result):
            instructions = self._generate_simple_instructions(verification_result)
            return instructions, {"tokens_used": 0}

        # Use LLM for complex cases
        try:
            # Format invalid entities and suggestions
            invalid_entities = self._format_invalid_entities(verification_result)
            suggestions = self._format_suggestions(verification_result)

            # Include query in prompt if using ReAct
            if self.use_react and query:
                prompt = self.instructions_prompt.format(
                    query=query,
                    invalid_entities=invalid_entities,
                    suggestions=suggestions
                )
            else:
                prompt = self.instructions_prompt.format(
                    invalid_entities=invalid_entities,
                    suggestions=suggestions
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
                # Extract final instructions from ReAct output
                instructions = self.reasoning_extractor.extract_final_output(generated_text)
            else:
                reasoning_trace = {}
                reasoning_quality = {}
                instructions = generated_text

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
                f"Generated instructions ({metadata['tokens_used']} tokens, "
                f"{reasoning_trace.get('num_steps', 0)} reasoning steps)"
            )

            return instructions, metadata

        except Exception as e:
            self.logger.error(f"Error generating instructions: {e}")
            # Fallback to simple instructions
            instructions = self._generate_simple_instructions(verification_result)
            return instructions, {"tokens_used": 0, "error": str(e)}

    def _is_simple_case(self, verification_result: Dict[str, Any]) -> bool:
        """
        Check if this is a simple case that doesn't need LLM.

        Args:
            verification_result: Verification results

        Returns:
            True if simple case (< 3 errors, all have suggestions)
        """
        summary = verification_result.get("verification_summary", {})
        invalid_count = summary.get("invalid_count", 0)

        if invalid_count == 0:
            return True

        if invalid_count > 3:
            return False

        # Check if all invalid entities have suggestions
        suggestions = verification_result.get("suggestions", {})
        for entity_type, type_suggestions in suggestions.items():
            if type_suggestions and len(type_suggestions) > 0:
                # Has suggestions, can use simple generation
                continue
            else:
                # No suggestions available
                return False

        return True

    def _generate_simple_instructions(
        self,
        verification_result: Dict[str, Any]
    ) -> str:
        """
        Generate simple rule-based instructions.

        Args:
            verification_result: Verification results

        Returns:
            Correction instructions string
        """
        instructions = []

        invalid = verification_result.get("invalid", {})
        suggestions = verification_result.get("suggestions", {})

        # Process node labels
        if invalid.get("node_labels"):
            instructions.append("**Node Label Corrections:**")
            for invalid_node in invalid["node_labels"]:
                node_suggestions = suggestions.get("node_labels", {}).get(invalid_node, [])
                if node_suggestions:
                    top_suggestion = node_suggestions[0][0]  # (name, score)
                    instructions.append(f"- Replace '{invalid_node}' with '{top_suggestion}'")
                else:
                    instructions.append(f"- Remove invalid node label '{invalid_node}'")

        # Process relationships
        if invalid.get("relationships"):
            instructions.append("\n**Relationship Corrections:**")
            for invalid_rel in invalid["relationships"]:
                rel_suggestions = suggestions.get("relationships", {}).get(invalid_rel, [])
                if rel_suggestions:
                    top_suggestion = rel_suggestions[0][0]
                    instructions.append(f"- Replace '{invalid_rel}' with '{top_suggestion}'")
                else:
                    instructions.append(f"- Remove invalid relationship '{invalid_rel}'")

        # Process properties
        if invalid.get("properties"):
            instructions.append("\n**Property Corrections:**")
            for invalid_prop in invalid["properties"]:
                prop_suggestions = suggestions.get("properties", {}).get(invalid_prop, [])
                if prop_suggestions:
                    top_suggestion = prop_suggestions[0][0]
                    instructions.append(f"- Replace property '{invalid_prop}' with '{top_suggestion}'")
                else:
                    instructions.append(f"- Remove invalid property '{invalid_prop}'")

        if not instructions:
            return "No corrections needed."

        return "\n".join(instructions)

    def _format_invalid_entities(self, verification_result: Dict[str, Any]) -> str:
        """Format invalid entities for prompt."""
        lines = []

        invalid = verification_result.get("invalid", {})

        if invalid.get("node_labels"):
            lines.append(f"Node Labels: {', '.join(invalid['node_labels'])}")

        if invalid.get("relationships"):
            lines.append(f"Relationships: {', '.join(invalid['relationships'])}")

        if invalid.get("properties"):
            lines.append(f"Properties: {', '.join(invalid['properties'])}")

        return "\n".join(lines) if lines else "None"

    def _format_suggestions(self, verification_result: Dict[str, Any]) -> str:
        """Format suggestions for prompt."""
        lines = []

        suggestions = verification_result.get("suggestions", {})

        for entity_type, type_suggestions in suggestions.items():
            if type_suggestions:
                lines.append(f"\n{entity_type.replace('_', ' ').title()}:")
                for invalid_entity, similar_list in type_suggestions.items():
                    if similar_list:
                        top_3 = [f"{name} (similarity: {score:.2f})" for name, score in similar_list[:3]]
                        lines.append(f"  '{invalid_entity}' → {', '.join(top_3)}")

        return "\n".join(lines) if lines else "No suggestions available"


if __name__ == "__main__":
    # Test InstructionsGenerator
    logging.basicConfig(level=logging.INFO)

    print("Testing InstructionsGenerator...")

    generator = InstructionsGenerator()

    # Test case 1: Simple case with clear suggestions
    print(f"\n{'='*60}")
    print("Test 1: Simple case with suggestions")
    print('='*60)

    test_verification_simple = {
        "valid": {
            "node_labels": ["MK"],
            "relationships": [],
            "properties": []
        },
        "invalid": {
            "node_labels": ["M"],
            "relationships": ["PREREQUISIT"],
            "properties": ["cod"]
        },
        "suggestions": {
            "node_labels": {
                "M": [("MK", 0.67)]
            },
            "relationships": {
                "PREREQUISIT": [("PREREQUISITE", 0.92)]
            },
            "properties": {
                "cod": [("code", 0.75)]
            }
        },
        "verification_summary": {
            "total_entities": 4,
            "valid_count": 1,
            "invalid_count": 3,
            "has_errors": True
        }
    }

    instructions, metadata = generator.generate(test_verification_simple)
    print(f"\nInstructions:\n{instructions}")
    print(f"\nTokens used: {metadata.get('tokens_used', 0)}")

    # Test case 2: No errors
    print(f"\n{'='*60}")
    print("Test 2: No errors")
    print('='*60)

    test_verification_valid = {
        "valid": {
            "node_labels": ["MK", "topic"],
            "relationships": ["PREREQUISITE"],
            "properties": ["code"]
        },
        "invalid": {
            "node_labels": [],
            "relationships": [],
            "properties": []
        },
        "suggestions": {
            "node_labels": {},
            "relationships": {},
            "properties": {}
        },
        "verification_summary": {
            "total_entities": 4,
            "valid_count": 4,
            "invalid_count": 0,
            "has_errors": False
        }
    }

    instructions, metadata = generator.generate(test_verification_valid)
    print(f"\nInstructions: {instructions}")
    print(f"Tokens used: {metadata.get('tokens_used', 0)}")

    # Test case 3: Multiple errors
    print(f"\n{'='*60}")
    print("Test 3: Multiple errors with suggestions")
    print('='*60)

    test_verification_multiple = {
        "valid": {
            "node_labels": [],
            "relationships": [],
            "properties": []
        },
        "invalid": {
            "node_labels": ["M", "topi"],
            "relationships": ["PREREQUISIT", "PART_O"],
            "properties": []
        },
        "suggestions": {
            "node_labels": {
                "M": [("MK", 0.67)],
                "topi": [("topic", 0.80), ("LO", 0.50)]
            },
            "relationships": {
                "PREREQUISIT": [("PREREQUISITE", 0.92), ("PURSUED_IN", 0.55)],
                "PART_O": [("PART_OF", 0.86)]
            },
            "properties": {}
        },
        "verification_summary": {
            "total_entities": 4,
            "valid_count": 0,
            "invalid_count": 4,
            "has_errors": True
        }
    }

    instructions, metadata = generator.generate(test_verification_multiple)
    print(f"\nInstructions:\n{instructions}")
    print(f"\nTokens used: {metadata.get('tokens_used', 0)}")

    print("\nInstructionsGenerator test complete!")
