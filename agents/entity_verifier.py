"""
Entity Verifier Agent

Verifies extracted entities against graph schema.
Uses Levenshtein distance for fuzzy matching and suggests corrections.
"""

import re
import logging
from typing import Dict, List, Tuple, Set, Optional
from difflib import SequenceMatcher


class EntityVerifier:
    """
    Agent responsible for verifying entities against schema.

    Uses exact matching and fuzzy matching (Levenshtein distance)
    to identify invalid entities and suggest corrections.
    """

    def __init__(
        self,
        schema: Optional[str] = None,
        similarity_threshold: float = 0.6
    ):
        """
        Initialize Entity Verifier agent.

        Args:
            schema: Graph schema string (can be set later via set_schema)
            similarity_threshold: Minimum similarity for fuzzy matches (0.0-1.0)
        """
        self.schema = schema
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)

        # Schema entities (populated by parse_schema)
        self.schema_nodes: Set[str] = set()
        self.schema_relationships: Set[str] = set()
        self.schema_properties: Set[str] = set()

        if schema:
            self.parse_schema(schema)

    def set_schema(self, schema: str):
        """
        Set and parse schema.

        Args:
            schema: Graph schema string
        """
        self.schema = schema
        self.parse_schema(schema)

    def parse_schema(self, schema: str):
        """
        Parse schema to extract valid entities.

        Args:
            schema: Schema string (only_paths, nodes_and_paths, or full_schema format)
        """
        # Extract node labels
        node_pattern = re.compile(r'\(:(\w+)\)')
        self.schema_nodes = set(node_pattern.findall(schema))

        # Extract relationship types
        rel_pattern = re.compile(r'\[:(\w+)\]')
        self.schema_relationships = set(rel_pattern.findall(schema))

        # Extract properties (if present in schema)
        # Property format: {property: type} or node.property
        prop_pattern = re.compile(r'[\{.](\w+)(?:\s*:|:)')
        potential_props = set(prop_pattern.findall(schema))

        # Filter out node/relationship names that might be captured
        self.schema_properties = potential_props - self.schema_nodes - self.schema_relationships

        self.logger.info(
            f"Parsed schema: {len(self.schema_nodes)} nodes, "
            f"{len(self.schema_relationships)} relationships, "
            f"{len(self.schema_properties)} properties"
        )

    def verify(
        self,
        extracted_entities: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Verify extracted entities against schema.

        Args:
            extracted_entities: Output from EntityExtractor.extract()

        Returns:
            Dictionary with verification results:
                - valid: Dict of valid entities by type
                - invalid: Dict of invalid entities by type
                - suggestions: Dict of correction suggestions for invalid entities
                - verification_summary: Summary statistics
        """
        if not self.schema:
            self.logger.warning("No schema set for verification")
            return self._empty_verification_result()

        # Verify each entity type
        node_verification = self._verify_entities(
            extracted_entities.get("node_labels", []),
            self.schema_nodes,
            "node"
        )

        rel_verification = self._verify_entities(
            extracted_entities.get("relationships", []),
            self.schema_relationships,
            "relationship"
        )

        prop_verification = self._verify_entities(
            extracted_entities.get("properties", []),
            self.schema_properties,
            "property"
        )

        # Aggregate results
        result = {
            "valid": {
                "node_labels": node_verification["valid"],
                "relationships": rel_verification["valid"],
                "properties": prop_verification["valid"]
            },
            "invalid": {
                "node_labels": node_verification["invalid"],
                "relationships": rel_verification["invalid"],
                "properties": prop_verification["invalid"]
            },
            "suggestions": {
                "node_labels": node_verification["suggestions"],
                "relationships": rel_verification["suggestions"],
                "properties": prop_verification["suggestions"]
            },
            "verification_summary": {
                "total_entities": (
                    len(extracted_entities.get("node_labels", [])) +
                    len(extracted_entities.get("relationships", [])) +
                    len(extracted_entities.get("properties", []))
                ),
                "valid_count": (
                    len(node_verification["valid"]) +
                    len(rel_verification["valid"]) +
                    len(prop_verification["valid"])
                ),
                "invalid_count": (
                    len(node_verification["invalid"]) +
                    len(rel_verification["invalid"]) +
                    len(prop_verification["invalid"])
                ),
                "has_errors": (
                    len(node_verification["invalid"]) > 0 or
                    len(rel_verification["invalid"]) > 0 or
                    len(prop_verification["invalid"]) > 0
                )
            }
        }

        return result

    def _verify_entities(
        self,
        extracted: List[str],
        schema_set: Set[str],
        entity_type: str
    ) -> Dict[str, Any]:
        """
        Verify list of entities against schema set.

        Args:
            extracted: List of extracted entity names
            schema_set: Set of valid entity names from schema
            entity_type: Type of entity ("node", "relationship", "property")

        Returns:
            Dictionary with valid, invalid, and suggestions
        """
        valid = []
        invalid = []
        suggestions = {}

        for entity in extracted:
            if entity in schema_set:
                valid.append(entity)
            else:
                invalid.append(entity)
                # Find similar entities
                similar = self._find_similar(entity, schema_set)
                if similar:
                    suggestions[entity] = similar

        return {
            "valid": valid,
            "invalid": invalid,
            "suggestions": suggestions
        }

    def _find_similar(
        self,
        entity: str,
        schema_set: Set[str],
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Find similar entities using fuzzy matching.

        Args:
            entity: Entity to find matches for
            schema_set: Set of valid entities
            top_k: Number of top suggestions to return

        Returns:
            List of (entity_name, similarity_score) tuples, sorted by score
        """
        similarities = []

        for schema_entity in schema_set:
            similarity = self._compute_similarity(entity, schema_entity)
            if similarity >= self.similarity_threshold:
                similarities.append((schema_entity, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def _compute_similarity(self, str1: str, str2: str) -> float:
        """
        Compute similarity between two strings.

        Uses SequenceMatcher (similar to Levenshtein ratio).

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Case-insensitive comparison
        str1_lower = str1.lower()
        str2_lower = str2.lower()

        # Exact match (case-insensitive)
        if str1_lower == str2_lower:
            return 1.0

        # SequenceMatcher ratio (0.0 to 1.0)
        return SequenceMatcher(None, str1_lower, str2_lower).ratio()

    def _empty_verification_result(self) -> Dict[str, Any]:
        """Return empty verification result structure."""
        return {
            "valid": {
                "node_labels": [],
                "relationships": [],
                "properties": []
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
                "total_entities": 0,
                "valid_count": 0,
                "invalid_count": 0,
                "has_errors": False
            }
        }

    def get_schema_entities(self) -> Dict[str, List[str]]:
        """
        Get all valid entities from schema.

        Returns:
            Dictionary with node_labels, relationships, properties
        """
        return {
            "node_labels": sorted(list(self.schema_nodes)),
            "relationships": sorted(list(self.schema_relationships)),
            "properties": sorted(list(self.schema_properties))
        }


if __name__ == "__main__":
    # Test EntityVerifier
    logging.basicConfig(level=logging.INFO)

    print("Testing EntityVerifier...")

    # Test schema (only_paths format - University Curriculum domain)
    test_schema = """(:MK)-[:PREREQUISITE]->(:MK)
(:MK)-[:CAN_PARALLELIZED]->(:MK)
(:topic)-[:PART_OF]->(:LO)
(:LO)-[:PURSUED_IN]->(:MK)
(:SO)-[:PART_OF]->(:LG)"""

    # Initialize verifier
    verifier = EntityVerifier(schema=test_schema, similarity_threshold=0.6)

    print(f"\nSchema entities:")
    schema_entities = verifier.get_schema_entities()
    print(f"  Nodes: {schema_entities['node_labels']}")
    print(f"  Relationships: {schema_entities['relationships']}")

    # Test case 1: Valid entities
    print(f"\n{'='*60}")
    print("Test 1: Valid entities")
    print('='*60)

    extracted_valid = {
        "node_labels": ["MK", "topic", "LO"],
        "relationships": ["PREREQUISITE", "PART_OF"],
        "properties": ["code", "title"]
    }

    result = verifier.verify(extracted_valid)
    print(f"\nValid: {result['valid']}")
    print(f"Invalid: {result['invalid']}")
    print(f"Summary: {result['verification_summary']}")

    # Test case 2: Invalid entities with typos
    print(f"\n{'='*60}")
    print("Test 2: Invalid entities (typos)")
    print('='*60)

    extracted_typos = {
        "node_labels": ["M", "topi", "L"],  # Typos
        "relationships": ["PREREQUISIT", "PART_O"],  # Typos
        "properties": []
    }

    result = verifier.verify(extracted_typos)
    print(f"\nInvalid: {result['invalid']}")
    print(f"\nSuggestions:")
    for entity_type, suggestions in result['suggestions'].items():
        if suggestions:
            print(f"  {entity_type}:")
            for invalid_entity, similar in suggestions.items():
                print(f"    '{invalid_entity}' → {similar}")

    # Test case 3: Mixed valid and invalid
    print(f"\n{'='*60}")
    print("Test 3: Mixed valid and invalid")
    print('='*60)

    extracted_mixed = {
        "node_labels": ["MK", "InvalidNode", "topic"],
        "relationships": ["PREREQUISITE", "INVALID_REL"],
        "properties": ["code", "invalid_prop"]
    }

    result = verifier.verify(extracted_mixed)
    print(f"\nValid nodes: {result['valid']['node_labels']}")
    print(f"Invalid nodes: {result['invalid']['node_labels']}")
    print(f"Valid relationships: {result['valid']['relationships']}")
    print(f"Invalid relationships: {result['invalid']['relationships']}")
    print(f"\nSummary: {result['verification_summary']}")

    # Test similarity computation
    print(f"\n{'='*60}")
    print("Test 4: Similarity computation")
    print('='*60)

    test_pairs = [
        ("MK", "M"),
        ("PREREQUISITE", "PREREQUISIT"),
        ("topic", "topi"),
        ("Completely", "Different")
    ]

    for str1, str2 in test_pairs:
        similarity = verifier._compute_similarity(str1, str2)
        print(f"  '{str1}' vs '{str2}': {similarity:.2f}")

    print("\nEntityVerifier test complete!")
