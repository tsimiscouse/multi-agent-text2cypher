"""
Logging Utility

Provides centralized logging configuration for the multi-agent system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "multi_agent",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup and configure logger.

    Args:
        name: Logger name (usually __name__ of calling module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, only console output.
        format_string: Optional custom format string

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Starting query generation...")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        # Create logs directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_experiment_logger(
    experiment_name: str,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Get logger for experiment with timestamped log file.

    Args:
        experiment_name: Name of the experiment (e.g., "baseline", "multiagent_k3")
        level: Logging level

    Returns:
        Configured logger with file output in logs/ directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).parent.parent / "logs"
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"

    return setup_logger(
        name=f"{experiment_name}",
        level=level,
        log_file=log_file
    )


class AgentLogger:
    """
    Specialized logger for agent operations.

    Provides structured logging for agent actions, decisions, and errors.
    """

    def __init__(self, agent_name: str, question_id: Optional[int] = None):
        self.agent_name = agent_name
        self.question_id = question_id
        self.logger = setup_logger(f"agent.{agent_name}")

    def log_action(self, action: str, details: Optional[dict] = None):
        """Log agent action."""
        msg = f"[Q{self.question_id}] {action}" if self.question_id else action
        if details:
            msg += f" | {details}"
        self.logger.info(msg)

    def log_decision(self, decision: str, reasoning: Optional[str] = None):
        """Log agent decision."""
        msg = f"[Q{self.question_id}] Decision: {decision}" if self.question_id else f"Decision: {decision}"
        if reasoning:
            msg += f" | Reasoning: {reasoning}"
        self.logger.info(msg)

    def log_error(self, error: str, details: Optional[dict] = None):
        """Log agent error."""
        msg = f"[Q{self.question_id}] Error: {error}" if self.question_id else f"Error: {error}"
        if details:
            msg += f" | {details}"
        self.logger.error(msg)

    def log_warning(self, warning: str):
        """Log agent warning."""
        msg = f"[Q{self.question_id}] Warning: {warning}" if self.question_id else f"Warning: {warning}"
        self.logger.warning(msg)


if __name__ == "__main__":
    # Test logging
    print("Testing logger...")

    # Basic logger
    logger = setup_logger("test_logger", level=logging.DEBUG)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Experiment logger
    exp_logger = get_experiment_logger("test_experiment")
    exp_logger.info("Experiment started")

    # Agent logger
    agent_logger = AgentLogger("query_generator", question_id=42)
    agent_logger.log_action("Generating query", {"temperature": 0.0})
    agent_logger.log_decision("Accept", "Query is valid")
    agent_logger.log_error("Syntax error", {"line": 5})

    print("\nLogger test complete!")
