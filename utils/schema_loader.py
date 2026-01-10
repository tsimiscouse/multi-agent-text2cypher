"""
Schema Loader Utility

Loads Neo4j schema representations from data/schemas directory.
Supports different schema formats: only_paths, nodes_and_paths, full_schema.
"""

from pathlib import Path
from typing import Optional


def load_schema(schema_type: str = "only_paths") -> str:
    """
    Load schema file content.

    Args:
        schema_type: Type of schema to load. Options:
            - "only_paths": Relationship paths only (best for kg-axel baseline)
            - "nodes_and_paths": Node labels + relationship paths
            - "full_schema": Complete schema with properties

    Returns:
        Schema content as string

    Raises:
        FileNotFoundError: If schema file doesn't exist
        ValueError: If schema_type is invalid
    """
    valid_types = ["only_paths", "nodes_and_paths", "full_schema"]

    if schema_type not in valid_types:
        raise ValueError(
            f"Invalid schema_type '{schema_type}'. Must be one of: {valid_types}"
        )

    # Get schema file path
    schema_dir = Path(__file__).parent.parent / "data" / "schemas"
    schema_file = schema_dir / f"{schema_type}.txt"

    if not schema_file.exists():
        raise FileNotFoundError(
            f"Schema file not found: {schema_file}\n"
            f"Available schemas: {list(schema_dir.glob('*.txt'))}"
        )

    # Load and return schema content
    with open(schema_file, "r", encoding="utf-8") as f:
        schema_content = f.read()

    return schema_content


def get_available_schemas() -> list[str]:
    """
    Get list of available schema types.

    Returns:
        List of schema type names (without .txt extension)
    """
    schema_dir = Path(__file__).parent.parent / "data" / "schemas"

    if not schema_dir.exists():
        return []

    schema_files = schema_dir.glob("*.txt")
    return [f.stem for f in schema_files]


if __name__ == "__main__":
    # Test schema loading
    print("Testing schema loader...")

    available = get_available_schemas()
    print(f"\nAvailable schemas: {available}")

    for schema_type in available:
        try:
            schema = load_schema(schema_type)
            print(f"\n{schema_type}:")
            print(f"  Length: {len(schema)} characters")
            print(f"  Preview: {schema[:100]}...")
        except Exception as e:
            print(f"\n{schema_type}: ERROR - {e}")
