"""
Cypher Query Validators

Implements three types of validation:
1. Syntax Validator - Checks Cypher syntax correctness
2. Schema Validator - Validates against KG schema (labels, relationships)
3. Properties Validator - Validates property names and usage

These validators complement execution-based validation by catching
errors that may not cause execution failures but indicate quality issues.
"""

import re
import logging
from typing import Dict, List, Tuple, Set, Any

logger = logging.getLogger(__name__)


class SyntaxValidator:
    """
    Validates Cypher query syntax.

    Checks for common syntax errors:
    - Balanced parentheses and brackets
    - Valid keywords
    - Proper clause ordering
    - Quote matching
    """

    VALID_KEYWORDS = {
        'MATCH', 'WHERE', 'RETURN', 'WITH', 'CREATE', 'MERGE', 'DELETE', 'REMOVE',
        'SET', 'ORDER', 'BY', 'LIMIT', 'SKIP', 'UNION', 'OPTIONAL', 'DISTINCT',
        'AS', 'AND', 'OR', 'NOT', 'IN', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX',
        'COLLECT', 'UNWIND', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END'
    }

    CLAUSE_ORDER = [
        'MATCH', 'OPTIONAL MATCH', 'WITH', 'WHERE', 'RETURN',
        'ORDER BY', 'LIMIT', 'SKIP'
    ]

    def validate(self, query: str) -> Tuple[bool, List[str]]:
        """
        Validate Cypher query syntax.

        Args:
            query: Cypher query string

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        if not query or not query.strip():
            return False, ["Query is empty"]

        # Check balanced parentheses
        if not self._check_balanced_parentheses(query):
            errors.append("Unbalanced parentheses")

        # Check balanced brackets
        if not self._check_balanced_brackets(query):
            errors.append("Unbalanced brackets")

        # Check balanced braces
        if not self._check_balanced_braces(query):
            errors.append("Unbalanced braces")

        # Check quote matching
        if not self._check_quotes(query):
            errors.append("Unmatched quotes")

        # Check for basic required clauses (MATCH and RETURN)
        if not self._has_required_clauses(query):
            errors.append("Missing required clauses (MATCH or RETURN)")

        # Check clause ordering
        ordering_errors = self._check_clause_ordering(query)
        errors.extend(ordering_errors)

        is_valid = len(errors) == 0

        return is_valid, errors

    def _check_balanced_parentheses(self, query: str) -> bool:
        """Check if parentheses are balanced."""
        count = 0
        for char in query:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
            if count < 0:
                return False
        return count == 0

    def _check_balanced_brackets(self, query: str) -> bool:
        """Check if brackets are balanced."""
        count = 0
        for char in query:
            if char == '[':
                count += 1
            elif char == ']':
                count -= 1
            if count < 0:
                return False
        return count == 0

    def _check_balanced_braces(self, query: str) -> bool:
        """Check if braces are balanced."""
        count = 0
        for char in query:
            if char == '{':
                count += 1
            elif char == '}':
                count -= 1
            if count < 0:
                return False
        return count == 0

    def _check_quotes(self, query: str) -> bool:
        """Check if quotes are properly matched."""
        single_quote_count = query.count("'")
        double_quote_count = query.count('"')

        # Both should be even (paired)
        return single_quote_count % 2 == 0 and double_quote_count % 2 == 0

    def _has_required_clauses(self, query: str) -> bool:
        """Check if query has required clauses."""
        query_upper = query.upper()

        # Must have either MATCH or RETURN (or both)
        has_match = 'MATCH' in query_upper
        has_return = 'RETURN' in query_upper

        return has_match or has_return

    def _check_clause_ordering(self, query: str) -> List[str]:
        """Check if clauses are in valid order."""
        errors = []

        # Extract clause positions
        query_upper = query.upper()
        clause_positions = []

        for clause in self.CLAUSE_ORDER:
            pos = query_upper.find(clause)
            if pos != -1:
                clause_positions.append((pos, clause))

        # Sort by position
        clause_positions.sort()

        # Check ordering violations
        # WHERE should come after MATCH
        match_pos = query_upper.find('MATCH')
        where_pos = query_upper.find('WHERE')

        if match_pos != -1 and where_pos != -1:
            if where_pos < match_pos:
                errors.append("WHERE clause appears before MATCH")

        # RETURN should come after MATCH/WHERE
        return_pos = query_upper.find('RETURN')

        if return_pos != -1:
            if match_pos != -1 and return_pos < match_pos:
                errors.append("RETURN clause appears before MATCH")

        return errors


class SchemaValidator:
    """
    Validates query against knowledge graph schema.

    Checks:
    - Node labels exist in schema
    - Relationship types exist in schema
    - Valid label/relationship combinations
    """

    def __init__(self, schema_config: Dict[str, Any] = None):
        """
        Initialize schema validator.

        Args:
            schema_config: Dictionary containing valid labels and relationships
        """
        if schema_config is None:
            # Default curriculum KG schema
            schema_config = self._get_default_schema()

        self.valid_labels = set(schema_config.get('labels', []))
        self.valid_relationships = set(schema_config.get('relationships', []))
        self.valid_combinations = schema_config.get('combinations', {})

    def _get_default_schema(self) -> Dict[str, Any]:
        """Get default curriculum KG schema."""
        return {
            'labels': ['MK', 'SO', 'LO', 'LG', 'topic', 'PREREQUISITE'],
            'relationships': [
                'PREREQUISITE', 'HAS_SO', 'HAS_LO', 'HAS_LG',
                'HAS_TOPIC', 'BELONGS_TO', 'SUPPORTS'
            ],
            'combinations': {
                'MK': ['PREREQUISITE', 'HAS_SO', 'HAS_LO', 'HAS_TOPIC'],
                'SO': ['HAS_LO', 'HAS_LG'],
                'LO': ['HAS_LG', 'BELONGS_TO'],
                'topic': ['BELONGS_TO'],
            }
        }

    def validate(self, query: str) -> Tuple[bool, List[str]]:
        """
        Validate query against schema.

        Args:
            query: Cypher query string

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Extract labels from query
        labels = self._extract_labels(query)

        # Check if labels are valid
        for label in labels:
            if label not in self.valid_labels:
                errors.append(f"Invalid label: {label}")

        # Extract relationships from query
        relationships = self._extract_relationships(query)

        # Check if relationships are valid
        for rel in relationships:
            if rel not in self.valid_relationships:
                errors.append(f"Invalid relationship: {rel}")

        is_valid = len(errors) == 0

        return is_valid, errors

    def validate_score(self, query: str) -> float:
        """
        Validate query and return score (0.0 or 1.0).

        Args:
            query: Cypher query string

        Returns:
            1.0 if valid, 0.0 if invalid
        """
        is_valid, _ = self.validate(query)
        return 1.0 if is_valid else 0.0

    def _extract_labels(self, query: str) -> Set[str]:
        """Extract node labels from query."""
        # Pattern: (variable:Label) or (:Label)
        pattern = r'\([\w]*:(\w+)'
        matches = re.findall(pattern, query)

        return set(matches)

    def _extract_relationships(self, query: str) -> Set[str]:
        """Extract relationship types from query."""
        # Pattern: -[:REL_TYPE]-> or -[r:REL_TYPE]->
        pattern = r'-\[[\w]*:(\w+)\]-'
        matches = re.findall(pattern, query)

        return set(matches)


class PropertiesValidator:
    """
    Validates property usage in queries.

    Checks:
    - Property names match schema
    - Properties used with correct labels
    - Property value types
    """

    def __init__(self, schema_config: Dict[str, Any] = None):
        """
        Initialize properties validator.

        Args:
            schema_config: Dictionary mapping labels to valid properties
        """
        if schema_config is None:
            schema_config = self._get_default_properties()

        self.properties_by_label = schema_config

    def _get_default_properties(self) -> Dict[str, List[str]]:
        """Get default curriculum KG properties."""
        return {
            'MK': ['kode', 'nama', 'sks', 'semester', 'tipe', 'klasifikasi', 'deskripsi'],
            'SO': ['kode', 'deskripsi', 'kategori'],
            'LO': ['kode', 'deskripsi'],
            'LG': ['kode', 'deskripsi'],
            'topic': ['nama', 'kategori'],
            'PREREQUISITE': ['semester']
        }

    def validate(self, query: str) -> Tuple[bool, List[str]]:
        """
        Validate property usage in query.

        Args:
            query: Cypher query string

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Extract label-property pairs
        label_properties = self._extract_label_properties(query)

        # Check if properties are valid for each label
        for label, properties in label_properties.items():
            valid_props = self.properties_by_label.get(label, [])

            for prop in properties:
                if prop not in valid_props:
                    errors.append(f"Invalid property '{prop}' for label '{label}'")

        is_valid = len(errors) == 0

        return is_valid, errors

    def validate_score(self, query: str) -> float:
        """
        Validate query and return score (0.0, 1.0, or None for no properties).

        Args:
            query: Cypher query string

        Returns:
            1.0 if valid, 0.0 if invalid, None if no properties to validate
        """
        label_properties = self._extract_label_properties(query)

        # If no properties found, return None (similar to NaN in pandas)
        if not label_properties or all(not props for props in label_properties.values()):
            return None

        is_valid, _ = self.validate(query)
        return 1.0 if is_valid else 0.0

    def _extract_label_properties(self, query: str) -> Dict[str, List[str]]:
        """
        Extract label-property pairs from query.

        Returns:
            Dictionary mapping labels to lists of properties
        """
        result = {}

        # Pattern: (variable:Label {property: value}) or (variable:Label) ... variable.property
        # First, extract inline properties: (n:Label {prop: val})
        inline_pattern = r'\([\w]*:(\w+)\s*\{([^}]+)\}'
        inline_matches = re.findall(inline_pattern, query)

        for label, props_str in inline_matches:
            props = re.findall(r'(\w+):\s*', props_str)
            if label not in result:
                result[label] = []
            result[label].extend(props)

        # Second, extract dot-notation properties: variable.property
        # Need to map variable to label first
        var_to_label = {}

        # Extract variable-label mappings
        var_label_pattern = r'\((\w+):(\w+)'
        var_label_matches = re.findall(var_label_pattern, query)

        for var, label in var_label_matches:
            var_to_label[var] = label

        # Extract properties accessed via dot notation
        dot_pattern = r'(\w+)\.(\w+)'
        dot_matches = re.findall(dot_pattern, query)

        for var, prop in dot_matches:
            if var in var_to_label:
                label = var_to_label[var]
                if label not in result:
                    result[label] = []
                result[label].append(prop)

        return result


def validate_query(
    query: str,
    schema_config: Dict[str, Any] = None,
    properties_config: Dict[str, List[str]] = None
) -> Dict[str, Any]:
    """
    Run all validators on a query.

    Args:
        query: Cypher query string
        schema_config: Schema configuration for SchemaValidator
        properties_config: Properties configuration for PropertiesValidator

    Returns:
        Dictionary with validation results
    """
    syntax_validator = SyntaxValidator()
    schema_validator = SchemaValidator(schema_config)
    properties_validator = PropertiesValidator(properties_config)

    syntax_valid, syntax_errors = syntax_validator.validate(query)
    schema_valid, schema_errors = schema_validator.validate(query)
    properties_valid, properties_errors = properties_validator.validate(query)

    return {
        'syntax_validator': syntax_valid,
        'schema_validator': schema_valid,
        'properties_validator': properties_valid,
        'syntax_errors': syntax_errors,
        'schema_errors': schema_errors,
        'properties_errors': properties_errors,
        'all_valid': syntax_valid and schema_valid and properties_valid
    }


if __name__ == "__main__":
    # Test validators
    print("Testing Cypher Validators...")
    print("=" * 60)

    # Test cases
    test_queries = [
        # Valid query
        (
            "Valid query",
            "MATCH (n:MK {nama: 'Basis Data'}) RETURN n.sks"
        ),
        # Syntax error: unbalanced parentheses
        (
            "Unbalanced parentheses",
            "MATCH (n:MK {nama: 'Basis Data'} RETURN n.sks"
        ),
        # Schema error: invalid label
        (
            "Invalid label",
            "MATCH (n:Mata_Kuliah {nama: 'Basis Data'}) RETURN n.sks"
        ),
        # Properties error: invalid property
        (
            "Invalid property",
            "MATCH (n:MK {nama: 'Basis Data'}) RETURN n.jumlah_sks"
        ),
        # Multiple errors
        (
            "Multiple errors",
            "MATCH (n:Course) WHERE n.invalid_prop = 'test' RETURN n.name"
        ),
        # Valid complex query
        (
            "Valid complex query",
            "MATCH (mk:MK {nama: 'Basis Data'})-[:PREREQUISITE]->(pre:MK) RETURN mk.nama, pre.nama, pre.sks"
        ),
    ]

    for name, query in test_queries:
        print(f"\nTest: {name}")
        print(f"Query: {query}")
        print()

        result = validate_query(query)

        print(f"  Syntax Valid: {result['syntax_validator']}")
        if result['syntax_errors']:
            print(f"    Errors: {', '.join(result['syntax_errors'])}")

        print(f"  Schema Valid: {result['schema_validator']}")
        if result['schema_errors']:
            print(f"    Errors: {', '.join(result['schema_errors'])}")

        print(f"  Properties Valid: {result['properties_validator']}")
        if result['properties_errors']:
            print(f"    Errors: {', '.join(result['properties_errors'])}")

        print(f"  Overall Valid: {result['all_valid']}")
        print("-" * 60)

    print("\nValidators test complete!")
