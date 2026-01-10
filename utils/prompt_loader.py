"""
Prompt Template Loader Utility

Loads and formats prompt templates from prompts/templates directory.
Supports variable substitution for dynamic prompts.
"""

from pathlib import Path
from typing import Dict, Any, Optional


def load_prompt_template(template_name: str) -> str:
    """
    Load prompt template file content.

    Args:
        template_name: Name of the template file (without .txt extension)
            Examples: "query_generator", "query_evaluator", "cot_prompt_template"

    Returns:
        Template content as string

    Raises:
        FileNotFoundError: If template file doesn't exist
    """
    # Get template file path
    templates_dir = Path(__file__).parent.parent / "prompts" / "templates"
    template_file = templates_dir / f"{template_name}.txt"

    if not template_file.exists():
        raise FileNotFoundError(
            f"Template file not found: {template_file}\n"
            f"Available templates: {list(templates_dir.glob('*.txt'))}"
        )

    # Load and return template content
    with open(template_file, "r", encoding="utf-8") as f:
        template_content = f.read()

    return template_content


def format_prompt(
    template_name: str,
    variables: Dict[str, Any],
    strict: bool = False
) -> str:
    """
    Load template and substitute variables.

    Args:
        template_name: Name of the template file
        variables: Dictionary of variable names and values to substitute
        strict: If True, raise error on missing variables. If False, ignore.

    Returns:
        Formatted prompt with variables substituted

    Example:
        >>> variables = {"question": "Who is the principal?", "schema": "..."}
        >>> prompt = format_prompt("query_generator", variables)
    """
    template = load_prompt_template(template_name)

    # Perform variable substitution
    try:
        if strict:
            formatted = template.format(**variables)
        else:
            formatted = template.format_map(_SafeDict(variables))
    except KeyError as e:
        raise ValueError(
            f"Missing required variable in template '{template_name}': {e}"
        )

    return formatted


class _SafeDict(dict):
    """Dictionary that returns placeholder for missing keys instead of raising KeyError."""

    def __missing__(self, key):
        return f"{{{key}}}"


def get_available_templates() -> list[str]:
    """
    Get list of available prompt templates.

    Returns:
        List of template names (without .txt extension)
    """
    templates_dir = Path(__file__).parent.parent / "prompts" / "templates"

    if not templates_dir.exists():
        return []

    template_files = templates_dir.glob("*.txt")
    return [f.stem for f in template_files]


if __name__ == "__main__":
    # Test prompt loading
    print("Testing prompt loader...")

    available = get_available_templates()
    print(f"\nAvailable templates: {available}")

    for template_name in available:
        try:
            template = load_prompt_template(template_name)
            print(f"\n{template_name}:")
            print(f"  Length: {len(template)} characters")
            print(f"  Preview: {template[:150]}...")
        except Exception as e:
            print(f"\n{template_name}: ERROR - {e}")

    # Test variable substitution
    print("\n" + "=" * 60)
    print("Testing variable substitution...")

    test_template = "query_generator"
    test_vars = {
        "question": "Who is the principal of SMA Negeri 1?",
        "schema": "(:Sekolah)-[:DIPIMPIN_OLEH]->(:Kepala_Sekolah)"
    }

    try:
        formatted = format_prompt(test_template, test_vars, strict=False)
        print(f"\nFormatted prompt preview: {formatted[:200]}...")
    except Exception as e:
        print(f"\nFormatting test: ERROR - {e}")
