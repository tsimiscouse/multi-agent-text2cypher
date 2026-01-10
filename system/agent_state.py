"""
Agent State Management

Tracks the state of multi-agent system across refinement iterations.
Stores queries, evaluations, feedback, and execution results.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


@dataclass
class IterationState:
    """
    State for a single refinement iteration.
    """
    iteration_number: int
    query: str
    evaluation: str  # "accept", "incorrect", "error"
    evaluation_reasoning: str
    error_message: Optional[str] = None
    execution_result: Optional[List[Dict[str, Any]]] = None
    execution_success: bool = False
    used_verification: bool = False
    extracted_entities: Optional[Dict[str, List[str]]] = None
    verified_entities: Optional[Dict[str, Any]] = None
    correction_instructions: Optional[str] = None
    aggregated_feedback: Optional[str] = None
    tokens_used: int = 0
    generator_tokens: int = 0
    evaluator_tokens: int = 0
    extractor_tokens: int = 0
    verifier_tokens: int = 0
    instructions_tokens: int = 0
    aggregator_tokens: int = 0

    # ReAct reasoning traces
    generator_reasoning: Optional[Dict[str, Any]] = None
    evaluator_reasoning: Optional[Dict[str, Any]] = None
    instructions_reasoning: Optional[Dict[str, Any]] = None
    aggregator_reasoning: Optional[Dict[str, Any]] = None

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AgentSystemState:
    """
    Complete state for multi-agent system execution.

    Tracks all iterations, final results, and metadata for a single question.
    """
    question_id: int
    question: str
    schema: str
    ground_truth: Optional[str] = None

    # Iteration history
    iterations: List[IterationState] = field(default_factory=list)

    # Final results
    final_query: Optional[str] = None
    final_evaluation: Optional[str] = None
    execution_success: bool = False
    execution_result: Optional[List[Dict[str, Any]]] = None
    is_empty_result: bool = False

    # Metadata
    total_iterations: int = 0
    total_tokens: int = 0
    elapsed_time: float = 0.0
    stopped_early: bool = False
    stop_reason: Optional[str] = None

    # ReAct reasoning summary (aggregated across iterations)
    total_reasoning_steps: int = 0
    avg_reasoning_quality: float = 0.0
    reasoning_enabled: bool = False

    # Timestamps
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None

    def add_iteration(self, iteration: IterationState):
        """
        Add iteration to state.

        Args:
            iteration: IterationState to add
        """
        self.iterations.append(iteration)
        self.total_iterations = len(self.iterations)
        self.total_tokens += iteration.tokens_used

        # Update final results
        self.final_query = iteration.query
        self.final_evaluation = iteration.evaluation
        self.execution_success = iteration.execution_success
        self.execution_result = iteration.execution_result

        if iteration.execution_result is not None:
            self.is_empty_result = len(iteration.execution_result) == 0

        # Aggregate reasoning metrics
        self._update_reasoning_metrics(iteration)

    def finalize(self, elapsed_time: float, stop_reason: Optional[str] = None):
        """
        Finalize state after all iterations complete.

        Args:
            elapsed_time: Total elapsed time in seconds
            stop_reason: Reason for stopping (e.g., "max_iterations", "accepted")
        """
        self.elapsed_time = elapsed_time
        self.stop_reason = stop_reason
        self.end_time = datetime.now().isoformat()

        if stop_reason == "max_iterations":
            self.stopped_early = False  # Reached natural limit
        elif stop_reason == "accepted":
            self.stopped_early = True  # Stopped because query was accepted

    def get_first_attempt(self) -> Optional[IterationState]:
        """Get first iteration state."""
        return self.iterations[0] if self.iterations else None

    def get_last_iteration(self) -> Optional[IterationState]:
        """Get last iteration state."""
        return self.iterations[-1] if self.iterations else None

    def was_recovered(self) -> bool:
        """
        Check if query was recovered after initial failure.

        Returns:
            True if first attempt failed but final succeeded
        """
        if len(self.iterations) < 2:
            return False

        first = self.get_first_attempt()
        return (
            first.evaluation in ["incorrect", "error"] and
            self.execution_success
        )

    def get_iteration_by_number(self, iteration_number: int) -> Optional[IterationState]:
        """Get iteration by number (1-indexed)."""
        if 1 <= iteration_number <= len(self.iterations):
            return self.iterations[iteration_number - 1]
        return None

    def _update_reasoning_metrics(self, iteration: IterationState):
        """
        Update aggregate reasoning metrics with new iteration.

        Args:
            iteration: IterationState to aggregate
        """
        # Count reasoning steps from all agents in this iteration
        iteration_steps = 0
        quality_scores = []

        for reasoning_field in [
            iteration.generator_reasoning,
            iteration.evaluator_reasoning,
            iteration.instructions_reasoning,
            iteration.aggregator_reasoning
        ]:
            if reasoning_field and isinstance(reasoning_field, dict):
                self.reasoning_enabled = True
                steps = reasoning_field.get('num_steps', 0)
                quality = reasoning_field.get('quality_score', 0.0)

                iteration_steps += steps
                if quality > 0:
                    quality_scores.append(quality)

        self.total_reasoning_steps += iteration_steps

        # Update average reasoning quality
        if quality_scores:
            # Compute running average
            all_quality_scores = []
            for iter_state in self.iterations:
                for reasoning_field in [
                    iter_state.generator_reasoning,
                    iter_state.evaluator_reasoning,
                    iter_state.instructions_reasoning,
                    iter_state.aggregator_reasoning
                ]:
                    if reasoning_field and isinstance(reasoning_field, dict):
                        quality = reasoning_field.get('quality_score', 0.0)
                        if quality > 0:
                            all_quality_scores.append(quality)

            if all_quality_scores:
                self.avg_reasoning_quality = sum(all_quality_scores) / len(all_quality_scores)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "schema": self.schema,
            "ground_truth": self.ground_truth,
            "iterations": [iter.to_dict() for iter in self.iterations],
            "final_query": self.final_query,
            "final_evaluation": self.final_evaluation,
            "execution_success": self.execution_success,
            "execution_result": self.execution_result,
            "is_empty_result": self.is_empty_result,
            "total_iterations": self.total_iterations,
            "total_tokens": self.total_tokens,
            "elapsed_time": self.elapsed_time,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "total_reasoning_steps": self.total_reasoning_steps,
            "avg_reasoning_quality": self.avg_reasoning_quality,
            "reasoning_enabled": self.reasoning_enabled,
            "start_time": self.start_time,
            "end_time": self.end_time
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSystemState":
        """Create AgentSystemState from dictionary."""
        # Extract iterations
        iterations_data = data.pop("iterations", [])
        iterations = [
            IterationState(**iter_data) for iter_data in iterations_data
        ]

        # Create state
        state = cls(**data)
        state.iterations = iterations

        return state

    @classmethod
    def from_json(cls, json_str: str) -> "AgentSystemState":
        """Create AgentSystemState from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


if __name__ == "__main__":
    # Test state management
    print("Testing AgentSystemState...")

    # Create state
    state = AgentSystemState(
        question_id=1,
        question="Who is the principal?",
        schema="(:School)-[:LED_BY]->(:Principal)",
        ground_truth="MATCH (:School)-[:LED_BY]->(p:Principal) RETURN p"
    )

    # Add iteration 1
    iter1 = IterationState(
        iteration_number=1,
        query="MATCH (s:School)-[:LED_BY]->(p:Principal) RETURN p.name",
        evaluation="incorrect",
        evaluation_reasoning="Wrong property name",
        error_message="Property 'name' does not exist",
        execution_success=False,
        used_verification=True,
        tokens_used=250
    )
    state.add_iteration(iter1)

    # Add iteration 2
    iter2 = IterationState(
        iteration_number=2,
        query="MATCH (s:School)-[:LED_BY]->(p:Principal) RETURN p.nama",
        evaluation="accept",
        evaluation_reasoning="Query is correct",
        execution_success=True,
        execution_result=[{"p.nama": "John Doe"}],
        tokens_used=180
    )
    state.add_iteration(iter2)

    # Finalize
    state.finalize(elapsed_time=5.4, stop_reason="accepted")

    # Test methods
    print(f"\nTotal iterations: {state.total_iterations}")
    print(f"Total tokens: {state.total_tokens}")
    print(f"Was recovered: {state.was_recovered()}")
    print(f"Final query: {state.final_query}")
    print(f"Execution success: {state.execution_success}")

    # Test serialization
    print("\nTesting serialization...")
    json_str = state.to_json()
    print(f"JSON length: {len(json_str)} characters")

    # Test deserialization
    state_from_json = AgentSystemState.from_json(json_str)
    print(f"Deserialized iterations: {state_from_json.total_iterations}")

    print("\nAgentSystemState test complete!")
