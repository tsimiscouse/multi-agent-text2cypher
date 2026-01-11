"""
Enhanced Error Parser for Self-Refine Feedback

Parses Neo4j errors into structured, actionable feedback for LLM.

Based on Self-Refine framework (Madaan et al., ICLR 2024):
- External feedback (DB errors) enables effective self-correction
- Structured error categorization improves refinement quality
"""

import re
import logging
from typing import Dict, Optional, Any
from neo4j.exceptions import CypherSyntaxError, ClientError

logger = logging.getLogger(__name__)


class DetailedErrorParser:
    """
    Parse Neo4j errors into structured feedback for query refinement.

    Error Categories:
    1. Syntax Errors - Invalid Cypher syntax (parentheses, keywords, etc.)
    2. Semantic Errors - Property/label/relationship doesn't exist in schema
    3. Logic Errors - Query executes but returns unexpected results

    Returns structured error information with:
    - Error category and type
    - Location in query (if available)
    - Domain-specific suggested fixes
    - Severity level for prioritization
    """

    # Curriculum graph domain knowledge
    VALID_LABELS = ['MK', 'topic', 'LO', 'LG', 'SO', 'PREREQUISITE']
    VALID_RELATIONSHIPS = [
        'PREREQUISITE', 'HAS_SO', 'HAS_LO', 'HAS_LG',
        'HAS_TOPIC', 'BELONGS_TO', 'SUPPORTS'
    ]
    COMMON_PROPERTIES = ['nama', 'kode', 'sks', 'semester', 'deskripsi', 'kategori', 'tipe', 'klasifikasi']

    def parse_error(self, error: Exception, query: str) -> Dict[str, Any]:
        """
        Parse error into structured feedback.

        Args:
            error: Exception raised during query execution
            query: Cypher query that caused the error

        Returns:
            {
                "error_category": "syntax" | "semantic" | "logic" | "unknown",
                "error_type": specific type (e.g., "PropertyNotFound"),
                "error_message": human-readable message,
                "error_location": where in query (if available),
                "suggested_fix": actionable suggestion,
                "severity": "critical" | "moderate" | "minor"
            }
        """
        if isinstance(error, CypherSyntaxError):
            return self._parse_syntax_error(error, query)
        elif isinstance(error, ClientError):
            return self._parse_client_error(error, query)
        else:
            return self._parse_generic_error(error, query)

    def _parse_syntax_error(self, error: CypherSyntaxError, query: str) -> Dict[str, Any]:
        """
        Parse Cypher syntax errors with location hints.

        Neo4j syntax errors often include line/column information:
        Example: "Invalid input 'R': expected 'RETURN' (line 1, column 15)"

        Args:
            error: CypherSyntaxError exception
            query: Original Cypher query

        Returns:
            Structured error dict with location and suggested fix
        """
        error_msg = str(error)

        # Extract line/column if available (Neo4j provides this)
        line_match = re.search(r'line (\d+)', error_msg, re.IGNORECASE)
        column_match = re.search(r'column (\d+)', error_msg, re.IGNORECASE)

        location = None
        if line_match and column_match:
            line = int(line_match.group(1))
            column = int(column_match.group(1))
            location = f"Line {line}, Column {column}"

        # Determine suggested fix based on error pattern
        suggested_fix = self._suggest_syntax_fix(error_msg, location)

        return {
            "error_category": "syntax",
            "error_type": "CypherSyntaxError",
            "error_message": error_msg,
            "error_location": location,
            "suggested_fix": suggested_fix,
            "severity": "critical"  # Syntax errors must be fixed
        }

    def _suggest_syntax_fix(self, error_msg: str, location: Optional[str]) -> str:
        """
        Generate domain-specific syntax fix suggestions.

        Args:
            error_msg: Raw error message from Neo4j
            location: Parsed location string (if available)

        Returns:
            Actionable suggestion for fixing the syntax error
        """
        error_lower = error_msg.lower()

        if "expected" in error_lower:
            expected = self._extract_expected_token(error_msg)
            if location:
                return f"Check syntax near {location}. Expected: {expected}"
            else:
                return f"Expected: {expected}"

        elif "unmatched" in error_lower or "unclosed" in error_lower:
            return "Check for unmatched parentheses, brackets, or quotes"

        elif "unexpected" in error_lower or "invalid input" in error_lower:
            # Try to identify what was unexpected
            invalid_match = re.search(r"invalid input ['\"]?([^'\"]+)['\"]?", error_lower)
            if invalid_match:
                invalid_token = invalid_match.group(1)
                return f"Unexpected token '{invalid_token}'. Review Cypher syntax."
            else:
                return "Review Cypher syntax for invalid tokens"

        else:
            return "Review Cypher syntax against Neo4j documentation"

    def _extract_expected_token(self, error_msg: str) -> str:
        """
        Extract expected token/keyword from error message.

        Args:
            error_msg: Error message containing "expected" clause

        Returns:
            Expected token or "unknown"
        """
        match = re.search(r"expected[:\s]+['\"]?([^'\"]+)['\"]?", error_msg, re.IGNORECASE)
        return match.group(1) if match else "unknown"

    def _parse_client_error(self, error: ClientError, query: str) -> Dict[str, Any]:
        """
        Parse semantic errors (property/label/relationship not found).

        Neo4j ClientError codes:
        - Neo.ClientError.Statement.PropertyNotFound
        - Neo.ClientError.Statement.LabelNotFound
        - Neo.ClientError.Statement.RelationshipNotFound
        - Neo.ClientError.Statement.SyntaxError (alternate syntax issues)

        Args:
            error: ClientError exception
            query: Original Cypher query

        Returns:
            Structured error dict with entity-specific suggestions
        """
        error_msg = str(error)
        error_code = getattr(error, 'code', None)

        # Property not found
        if "PropertyNotFound" in str(error_code) or "has no property" in error_msg.lower():
            return self._parse_property_error(error_msg)

        # Label not found
        elif "LabelNotFound" in str(error_code) or "no label" in error_msg.lower():
            return self._parse_label_error(error_msg)

        # Relationship not found
        elif "RelationshipNotFound" in str(error_code) or "no relationship" in error_msg.lower():
            return self._parse_relationship_error(error_msg)

        # Variable not defined
        elif "variable" in error_msg.lower() and "not defined" in error_msg.lower():
            var_match = re.search(r"variable ['\"]?(\w+)['\"]?", error_msg, re.IGNORECASE)
            var_name = var_match.group(1) if var_match else "unknown"
            return {
                "error_category": "semantic",
                "error_type": "VariableNotDefined",
                "error_message": f"Variable '{var_name}' is not defined",
                "error_location": f"Variable: {var_name}",
                "suggested_fix": f"Ensure variable '{var_name}' is defined in MATCH clause before using it",
                "severity": "critical"
            }

        # Generic client error
        else:
            return {
                "error_category": "semantic",
                "error_type": "ClientError",
                "error_message": error_msg,
                "error_location": None,
                "suggested_fix": "Review query semantics against graph schema",
                "severity": "moderate"
            }

    def _parse_property_error(self, error_msg: str) -> Dict[str, Any]:
        """Parse property not found error with curriculum domain suggestions."""
        # Extract property name
        property_match = re.search(r"property[:\s]+['\"]?(\w+)['\"]?", error_msg, re.IGNORECASE)
        property_name = property_match.group(1) if property_match else "unknown"

        # Suggest common properties
        suggested_props = ", ".join(self.COMMON_PROPERTIES)

        return {
            "error_category": "semantic",
            "error_type": "PropertyNotFound",
            "error_message": f"Property '{property_name}' does not exist on this node/relationship",
            "error_location": f"Property: {property_name}",
            "suggested_fix": (
                f"Property '{property_name}' not found. "
                f"Common properties in curriculum graph: {suggested_props}. "
                f"Check spelling or verify property exists in schema."
            ),
            "severity": "critical"
        }

    def _parse_label_error(self, error_msg: str) -> Dict[str, Any]:
        """Parse label not found error with curriculum domain suggestions."""
        # Extract label name
        label_match = re.search(r"label[:\s]+['\"]?(\w+)['\"]?", error_msg, re.IGNORECASE)
        label_name = label_match.group(1) if label_match else "unknown"

        # Suggest valid labels
        suggested_labels = ", ".join(self.VALID_LABELS)

        return {
            "error_category": "semantic",
            "error_type": "LabelNotFound",
            "error_message": f"Node label '{label_name}' does not exist in graph",
            "error_location": f"Label: {label_name}",
            "suggested_fix": (
                f"Label '{label_name}' not found. "
                f"Valid labels in curriculum graph: {suggested_labels}. "
                f"Check spelling or use correct label name."
            ),
            "severity": "critical"
        }

    def _parse_relationship_error(self, error_msg: str) -> Dict[str, Any]:
        """Parse relationship not found error with curriculum domain suggestions."""
        # Extract relationship type
        rel_match = re.search(r"relationship[:\s]+['\"]?(\w+)['\"]?", error_msg, re.IGNORECASE)
        rel_name = rel_match.group(1) if rel_match else "unknown"

        # Suggest valid relationships
        suggested_rels = ", ".join(self.VALID_RELATIONSHIPS)

        return {
            "error_category": "semantic",
            "error_type": "RelationshipNotFound",
            "error_message": f"Relationship type '{rel_name}' does not exist in graph",
            "error_location": f"Relationship: {rel_name}",
            "suggested_fix": (
                f"Relationship '{rel_name}' not found. "
                f"Valid relationships in curriculum graph: {suggested_rels}. "
                f"Check spelling or use correct relationship type."
            ),
            "severity": "critical"
        }

    def _parse_generic_error(self, error: Exception, query: str) -> Dict[str, Any]:
        """
        Parse other errors not covered by specific parsers.

        Args:
            error: Generic exception
            query: Original Cypher query

        Returns:
            Basic error structure with minimal parsing
        """
        return {
            "error_category": "unknown",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_location": None,
            "suggested_fix": "Review query and error message. Check for runtime issues or constraint violations.",
            "severity": "moderate"
        }

    def format_for_llm(self, parsed_error: Dict[str, Any]) -> str:
        """
        Format parsed error for LLM consumption in prompts.

        Creates clear, actionable error description suitable for
        Self-Refine reflection step in ReAct prompts.

        Args:
            parsed_error: Structured error dict from parse_error()

        Returns:
            Formatted error string for prompt injection
        """
        category = parsed_error["error_category"]
        error_type = parsed_error["error_type"]
        message = parsed_error["error_message"]
        location = parsed_error.get("error_location")
        suggested_fix = parsed_error["suggested_fix"]
        severity = parsed_error["severity"]

        # Build formatted error message
        parts = [
            "=== ERROR ANALYSIS ===",
            f"Category: {category.upper()}",
            f"Type: {error_type}",
            f"Severity: {severity.upper()}",
            "",
            f"Error: {message}",
        ]

        if location:
            parts.append(f"Location: {location}")

        parts.extend([
            "",
            "Suggested Fix:",
            suggested_fix,
            "",
            "Please reflect on this error and generate an improved query that addresses the issue."
        ])

        return "\n".join(parts)


if __name__ == "__main__":
    # Test the parser with sample errors
    print("Testing DetailedErrorParser...")
    print("=" * 60)

    parser = DetailedErrorParser()

    # Test 1: Syntax error (simulated)
    print("\nTest 1: Syntax Error")
    class MockSyntaxError(CypherSyntaxError):
        def __init__(self, msg):
            self.message = msg
        def __str__(self):
            return self.message

    syntax_error = MockSyntaxError("Invalid input 'R': expected 'RETURN' (line 1, column 15)")
    query = "MATCH (n:MK) R n.nama"
    parsed = parser.parse_error(syntax_error, query)
    print(parser.format_for_llm(parsed))

    # Test 2: Property error (simulated)
    print("\n" + "=" * 60)
    print("\nTest 2: Property Not Found")
    class MockClientError(ClientError):
        def __init__(self, msg, code=None):
            self.message = msg
            self.code = code
        def __str__(self):
            return self.message

    prop_error = MockClientError(
        "Property 'jumlah_sks' has no property",
        code="Neo.ClientError.Statement.PropertyNotFound"
    )
    query = "MATCH (n:MK) RETURN n.jumlah_sks"
    parsed = parser.parse_error(prop_error, query)
    print(parser.format_for_llm(parsed))

    # Test 3: Label error (simulated)
    print("\n" + "=" * 60)
    print("\nTest 3: Label Not Found")
    label_error = MockClientError(
        "Label 'Mata_Kuliah' no label found",
        code="Neo.ClientError.Statement.LabelNotFound"
    )
    query = "MATCH (n:Mata_Kuliah) RETURN n.nama"
    parsed = parser.parse_error(label_error, query)
    print(parser.format_for_llm(parsed))

    print("\n" + "=" * 60)
    print("\nDetailedErrorParser tests complete!")
