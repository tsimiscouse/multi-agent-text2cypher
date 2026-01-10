"""
Entity Extractor Agent

Rule-based extraction of entities from Cypher queries.
Extracts node labels, relationship types, properties, and literal values.
"""

import re
import logging
from typing import Dict, List, Set, Any


class EntityExtractor:
    """
    Agent responsible for extracting entities from Cypher queries.

    Uses regex patterns to parse Cypher syntax and identify:
    - Node labels (e.g., :Guru, :Sekolah)
    - Relationship types (e.g., [:MENGAJAR], [:DIPIMPIN_OLEH])
    - Property names (e.g., nama, kode)
    - Literal values (e.g., "SMA Negeri 1", "Matematika")
    """

    def __init__(self):
        """Initialize Entity Extractor agent."""
        self.logger = logging.getLogger(__name__)

        # Compile regex patterns for efficiency
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for entity extraction."""
        # Node label pattern: (:Label) or (var:Label) or (var:Label1:Label2)
        self.node_label_pattern = re.compile(r'\((?:\w+)?:(\w+(?::\w+)*)\)')

        # Relationship type pattern: [:TYPE] or [var:TYPE] or [:TYPE*] or [var:TYPE*1..3]
        self.relationship_pattern = re.compile(r'\[(?:\w+)?:(\w+)(?:\*[0-9.]*)?(?:\s+\{[^}]*\})?\]')

        # Property name pattern: variable.property or {property: value}
        self.property_pattern = re.compile(r'(?:\w+\.(\w+)|{([^}:]+):)')

        # Literal string pattern: "value" or 'value'
        self.literal_string_pattern = re.compile(r'["\']((?:[^"\'\\]|\\.)*)["\']')

        # WHERE clause property pattern: WHERE x.property
        self.where_property_pattern = re.compile(r'WHERE\s+\w+\.(\w+)', re.IGNORECASE)

    def extract(self, query: str) -> Dict[str, List[str]]:
        """
        Extract all entities from Cypher query.

        Args:
            query: Cypher query string

        Returns:
            Dictionary with keys:
                - node_labels: List of node label names
                - relationships: List of relationship type names
                - properties: List of property names
                - literal_values: List of literal string values
        """
        try:
            # Clean query
            query_clean = self._clean_query(query)

            # Extract each entity type
            node_labels = self._extract_node_labels(query_clean)
            relationships = self._extract_relationships(query_clean)
            properties = self._extract_properties(query_clean)
            literal_values = self._extract_literal_values(query_clean)

            result = {
                "node_labels": sorted(list(node_labels)),
                "relationships": sorted(list(relationships)),
                "properties": sorted(list(properties)),
                "literal_values": sorted(list(literal_values))
            }

            self.logger.debug(
                f"Extracted entities: {len(node_labels)} labels, "
                f"{len(relationships)} relationships, {len(properties)} properties"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return {
                "node_labels": [],
                "relationships": [],
                "properties": [],
                "literal_values": []
            }

    def _clean_query(self, query: str) -> str:
        """
        Clean query for processing.

        Args:
            query: Raw Cypher query

        Returns:
            Cleaned query string
        """
        # Remove comments
        query = re.sub(r'//.*$', '', query, flags=re.MULTILINE)

        # Normalize whitespace
        query = ' '.join(query.split())

        return query

    def _extract_node_labels(self, query: str) -> Set[str]:
        """
        Extract node labels from query.

        Examples:
            (g:Guru) → "Guru"
            (:Sekolah) → "Sekolah"
            (s:Sekolah:Negeri) → "Sekolah", "Negeri"

        Args:
            query: Cypher query

        Returns:
            Set of node label names
        """
        labels = set()

        matches = self.node_label_pattern.findall(query)
        for match in matches:
            # Handle multiple labels (e.g., :Label1:Label2)
            for label in match.split(':'):
                if label:
                    labels.add(label)

        return labels

    def _extract_relationships(self, query: str) -> Set[str]:
        """
        Extract relationship types from query.

        Examples:
            [:MENGAJAR] → "MENGAJAR"
            [r:DIPIMPIN_OLEH] → "DIPIMPIN_OLEH"
            [:PART_OF*] → "PART_OF"

        Args:
            query: Cypher query

        Returns:
            Set of relationship type names
        """
        relationships = set()

        matches = self.relationship_pattern.findall(query)
        for match in matches:
            relationships.add(match)

        return relationships

    def _extract_properties(self, query: str) -> Set[str]:
        """
        Extract property names from query.

        Examples:
            g.nama → "nama"
            {kode: "IF101"} → "kode"
            WHERE m.semester > 5 → "semester"

        Args:
            query: Cypher query

        Returns:
            Set of property names
        """
        properties = set()

        # Extract from dot notation (e.g., g.nama)
        dot_matches = re.findall(r'\w+\.(\w+)', query)
        properties.update(dot_matches)

        # Extract from property maps (e.g., {nama: "X"})
        map_matches = re.findall(r'\{([^}:]+):', query)
        for match in map_matches:
            # Clean property name
            prop = match.strip()
            if prop and not prop.upper() in ['WHERE', 'RETURN', 'WITH', 'ORDER', 'LIMIT']:
                properties.add(prop)

        # Extract from WHERE clause
        where_matches = self.where_property_pattern.findall(query)
        properties.update(where_matches)

        return properties

    def _extract_literal_values(self, query: str) -> Set[str]:
        """
        Extract literal string values from query.

        Examples:
            {nama: "Matematika"} → "Matematika"
            WHERE s.kode = "IF101" → "IF101"

        Args:
            query: Cypher query

        Returns:
            Set of literal string values
        """
        literal_values = set()

        matches = self.literal_string_pattern.findall(query)
        for match in matches:
            # Unescape escaped quotes
            value = match.replace('\\"', '"').replace("\\'", "'")
            literal_values.add(value)

        return literal_values

    def extract_detailed(self, query: str) -> Dict[str, Any]:
        """
        Extract entities with additional context.

        Args:
            query: Cypher query string

        Returns:
            Dictionary with detailed extraction results including counts
        """
        basic_extraction = self.extract(query)

        return {
            **basic_extraction,
            "entity_counts": {
                "node_labels": len(basic_extraction["node_labels"]),
                "relationships": len(basic_extraction["relationships"]),
                "properties": len(basic_extraction["properties"]),
                "literal_values": len(basic_extraction["literal_values"])
            },
            "total_entities": sum([
                len(basic_extraction["node_labels"]),
                len(basic_extraction["relationships"]),
                len(basic_extraction["properties"])
            ])
        }


if __name__ == "__main__":
    # Test EntityExtractor
    logging.basicConfig(level=logging.INFO)

    print("Testing EntityExtractor...")

    extractor = EntityExtractor()

    # Test queries
    test_queries = [
        # Simple node query
        'MATCH (g:Guru) RETURN g.nama',

        # Query with relationship
        'MATCH (g:Guru)-[:MENGAJAR]->(m:Mata_Pelajaran) RETURN g.nama, m.nama',

        # Query with property filter
        'MATCH (s:Sekolah {nama: "SMA Negeri 1"})-[:DIPIMPIN_OLEH]->(k:Kepala_Sekolah) RETURN k.nama',

        # Complex query with WHERE
        'MATCH (m:MK)-[:PREREQUISITE]->(p:MK) WHERE m.semester > 5 AND p.tipe = "Wajib" RETURN m.nama, p.nama',

        # Query with multiple labels
        'MATCH (t:topic)-[:PART_OF]->(l:LO)-[:PART_OF]->(s:SO) RETURN t.deskripsi, s.kode',

        # Query with aggregation
        'MATCH (g:Guru)-[:MENGAJAR]->(m:Mata_Pelajaran) RETURN g.nama, COUNT(m) AS jumlah_mapel ORDER BY jumlah_mapel DESC',
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {query[:60]}...")
        print('='*60)

        result = extractor.extract(query)

        print(f"\nNode Labels: {result['node_labels']}")
        print(f"Relationships: {result['relationships']}")
        print(f"Properties: {result['properties']}")
        print(f"Literal Values: {result['literal_values']}")

    # Test detailed extraction
    print(f"\n{'='*60}")
    print("Test: Detailed extraction")
    print('='*60)

    detailed = extractor.extract_detailed(test_queries[2])
    print(f"\nEntity counts: {detailed['entity_counts']}")
    print(f"Total entities: {detailed['total_entities']}")

    print("\nEntityExtractor test complete!")
