"""
Multi-Agent Orchestrator

Coordinates all 6 agents to perform iterative query refinement.
Implements the decision logic for when to verify, refine, and stop.
"""

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from agents import (
    QueryGenerator,
    QueryEvaluator,
    EntityExtractor,
    EntityVerifier,
    InstructionsGenerator,
    FeedbackAggregator
)
from system.graph_executor import GraphExecutor
from system.agent_state import AgentSystemState, IterationState


class MultiAgentOrchestrator:
    """
    Main orchestrator for multi-agent Text-to-Cypher system.

    Coordinates 6 agents through iterative refinement:
    1. QueryGenerator - Generate/refine queries
    2. QueryEvaluator - Evaluate correctness
    3. EntityExtractor - Extract entities (verification path)
    4. EntityVerifier - Verify entities (verification path)
    5. InstructionsGenerator - Generate corrections (verification path)
    6. FeedbackAggregator - Synthesize feedback (verification path)
    """

    def __init__(
        self,
        max_iterations: int = 3,
        model: str = "qwen/qwen-2.5-coder-32b-instruct",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        similarity_threshold: float = 0.6,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_react: bool = True
    ):
        """
        Initialize Multi-Agent Orchestrator with ReAct reasoning support.

        Args:
            max_iterations: Maximum refinement iterations (k)
            model: LLM model for agents
            temperature: Sampling temperature
            max_tokens: Max tokens per LLM call (increased to 1024 for ReAct)
            similarity_threshold: Threshold for entity verification
            api_key: API key (from env if None)
            base_url: API base URL (from env if None)
            use_react: Whether to enable ReAct reasoning in agents (default: True)
        """
        self.max_iterations = max_iterations
        self.use_react = use_react
        self.logger = logging.getLogger(__name__)

        # Initialize all agents
        self.logger.info(f"Initializing multi-agent system (ReAct: {use_react})...")

        # LLM-based agents with ReAct support
        self.generator = QueryGenerator(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            use_react=use_react
        )

        self.evaluator = QueryEvaluator(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            use_react=use_react
        )

        self.instructions_generator = InstructionsGenerator(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            use_react=use_react
        )

        self.feedback_aggregator = FeedbackAggregator(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            use_react=use_react
        )

        # Rule-based agents
        self.entity_extractor = EntityExtractor()

        self.entity_verifier = EntityVerifier(
            similarity_threshold=similarity_threshold
        )

        # Graph executor
        self.executor = GraphExecutor()

        self.logger.info("Multi-agent system initialized successfully")

    def run(
        self,
        question: str,
        schema: str,
        question_id: Optional[int] = None,
        ground_truth: Optional[str] = None
    ) -> AgentSystemState:
        """
        Run multi-agent system for a single question.

        Args:
            question: Natural language question
            schema: Graph schema (only_paths format)
            question_id: Question ID (for tracking)
            ground_truth: Ground truth query (optional, for comparison)

        Returns:
            AgentSystemState with complete execution history
        """
        start_time = time.time()

        # Initialize state
        state = AgentSystemState(
            question_id=question_id or 0,
            question=question,
            schema=schema,
            ground_truth=ground_truth
        )

        # Set schema for entity verifier
        self.entity_verifier.set_schema(schema)

        self.logger.info(f"Starting multi-agent execution for Q{question_id}")

        # Iteration loop
        for iteration in range(1, self.max_iterations + 1):
            self.logger.info(f"Iteration {iteration}/{self.max_iterations}")

            try:
                # Run iteration
                iteration_state = self._run_iteration(
                    iteration_number=iteration,
                    question=question,
                    schema=schema,
                    previous_query=state.final_query,
                    previous_feedback=None if iteration == 1 else state.get_last_iteration().aggregated_feedback
                )

                # Add to state
                state.add_iteration(iteration_state)

                # Check if we should stop
                if iteration_state.evaluation == "accept":
                    self.logger.info(f"Query accepted on iteration {iteration}")
                    state.finalize(
                        elapsed_time=time.time() - start_time,
                        stop_reason="accepted"
                    )
                    return state

            except Exception as e:
                self.logger.error(f"Error in iteration {iteration}: {e}")
                # Create error iteration
                error_iteration = IterationState(
                    iteration_number=iteration,
                    query="",
                    evaluation="error",
                    evaluation_reasoning=f"System error: {str(e)}",
                    error_message=str(e),
                    execution_success=False
                )
                state.add_iteration(error_iteration)
                break

        # Max iterations reached
        self.logger.info(f"Max iterations ({self.max_iterations}) reached")
        state.finalize(
            elapsed_time=time.time() - start_time,
            stop_reason="max_iterations"
        )

        return state

    def _run_iteration(
        self,
        iteration_number: int,
        question: str,
        schema: str,
        previous_query: Optional[str],
        previous_feedback: Optional[str]
    ) -> IterationState:
        """
        Run a single iteration of the multi-agent system.

        Args:
            iteration_number: Current iteration number (1-indexed)
            question: Natural language question
            schema: Graph schema
            previous_query: Query from previous iteration (None for first)
            previous_feedback: Feedback from previous iteration (None for first)

        Returns:
            IterationState with results from this iteration
        """
        # Step 1: Generate or Refine Query
        self.logger.debug(f"Step 1: Query generation")

        if iteration_number == 1:
            # Initial generation
            query, gen_metadata = self.generator.generate_initial(question, schema)
        else:
            # Refinement
            query, gen_metadata = self.generator.refine_query(
                question, schema, previous_query, previous_feedback
            )

        generator_tokens = gen_metadata.get("tokens_used", 0)

        # Step 2: Execute Query
        self.logger.debug(f"Step 2: Query execution")

        execution_result = self.executor.execute_with_metadata(query)
        execution_success = execution_result.get("success", False)
        error_message = execution_result.get("error")

        # Step 3: Evaluate Query
        self.logger.debug(f"Step 3: Query evaluation")

        evaluation, eval_reasoning, eval_metadata = self.evaluator.evaluate(
            question=question,
            query=query,
            execution_result=execution_result
        )

        evaluator_tokens = eval_metadata.get("tokens_used", 0)

        # Extract reasoning metadata
        generator_reasoning = self._extract_reasoning_metadata(gen_metadata)
        evaluator_reasoning = self._extract_reasoning_metadata(eval_metadata)

        # Initialize iteration state
        iteration_state = IterationState(
            iteration_number=iteration_number,
            query=query,
            evaluation=evaluation,
            evaluation_reasoning=eval_reasoning,
            error_message=error_message,
            execution_result=execution_result.get("records"),
            execution_success=execution_success,
            generator_tokens=generator_tokens,
            evaluator_tokens=evaluator_tokens,
            generator_reasoning=generator_reasoning,
            evaluator_reasoning=evaluator_reasoning
        )

        # Step 4: If accepted, we're done
        if evaluation == "accept":
            iteration_state.tokens_used = generator_tokens + evaluator_tokens
            return iteration_state

        # Step 5: Verification Module (for errors/incorrect queries)
        self.logger.debug(f"Step 4: Verification module")

        verification_result = self._run_verification_module(
            query=query,
            evaluation=evaluation,
            eval_reasoning=eval_reasoning,
            error_message=error_message
        )

        # Update iteration state with verification results
        iteration_state.used_verification = True
        iteration_state.extracted_entities = verification_result.get("extracted_entities")
        iteration_state.verified_entities = verification_result.get("verified_entities")
        iteration_state.correction_instructions = verification_result.get("correction_instructions")
        iteration_state.aggregated_feedback = verification_result.get("aggregated_feedback")

        # Update token counts
        iteration_state.extractor_tokens = 0  # Rule-based
        iteration_state.verifier_tokens = 0   # Rule-based
        iteration_state.instructions_tokens = verification_result.get("instructions_tokens", 0)
        iteration_state.aggregator_tokens = verification_result.get("aggregator_tokens", 0)

        # Store reasoning metadata from verification module
        iteration_state.instructions_reasoning = verification_result.get("instructions_reasoning")
        iteration_state.aggregator_reasoning = verification_result.get("aggregator_reasoning")

        iteration_state.tokens_used = (
            generator_tokens +
            evaluator_tokens +
            iteration_state.instructions_tokens +
            iteration_state.aggregator_tokens
        )

        return iteration_state

    def _run_verification_module(
        self,
        query: str,
        evaluation: str,
        eval_reasoning: str,
        error_message: Optional[str]
    ) -> Dict[str, Any]:
        """
        Run verification module (steps 4-7 of the architecture).

        This module is triggered when query is incorrect or has errors.

        Args:
            query: Generated Cypher query
            evaluation: Evaluation result
            eval_reasoning: Evaluation reasoning
            error_message: Error message (if execution failed)

        Returns:
            Dictionary with verification results
        """
        # Step 4.1: Extract entities
        self.logger.debug("  4.1: Entity extraction")
        extracted_entities = self.entity_extractor.extract(query)

        # Step 4.2: Verify entities
        self.logger.debug("  4.2: Entity verification")
        verification_result = self.entity_verifier.verify(extracted_entities)

        # Step 4.3: Generate correction instructions
        self.logger.debug("  4.3: Generate instructions")
        correction_instructions, instr_metadata = self.instructions_generator.generate(
            verification_result=verification_result,
            query=query
        )

        # Step 4.4: Aggregate feedback
        self.logger.debug("  4.4: Aggregate feedback")
        aggregated_feedback, agg_metadata = self.feedback_aggregator.aggregate(
            evaluation=evaluation,
            evaluation_reasoning=eval_reasoning,
            error_message=error_message,
            correction_instructions=correction_instructions
        )

        # Extract reasoning metadata
        instructions_reasoning = self._extract_reasoning_metadata(instr_metadata)
        aggregator_reasoning = self._extract_reasoning_metadata(agg_metadata)

        return {
            "extracted_entities": extracted_entities,
            "verified_entities": verification_result,
            "correction_instructions": correction_instructions,
            "aggregated_feedback": aggregated_feedback,
            "instructions_tokens": instr_metadata.get("tokens_used", 0),
            "aggregator_tokens": agg_metadata.get("tokens_used", 0),
            "instructions_reasoning": instructions_reasoning,
            "aggregator_reasoning": aggregator_reasoning
        }

    def _extract_reasoning_metadata(self, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract reasoning trace from agent metadata.

        Args:
            metadata: Metadata dict from agent response

        Returns:
            Dict with reasoning info or None if no reasoning
        """
        if not metadata or not self.use_react:
            return None

        # Extract relevant reasoning fields
        if metadata.get("has_reasoning", False):
            return {
                "has_reasoning": metadata.get("has_reasoning", False),
                "num_steps": metadata.get("num_reasoning_steps", 0),
                "quality_score": metadata.get("reasoning_quality", 0.0),
                "reasoning_trace": metadata.get("reasoning_trace", {})
            }

        return None

    def run_batch(
        self,
        questions: list[Dict[str, Any]],
        schema: str,
        progress_callback: Optional[callable] = None
    ) -> list[AgentSystemState]:
        """
        Run multi-agent system on batch of questions.

        Args:
            questions: List of question dicts with 'id', 'question', 'ground_truth'
            schema: Graph schema
            progress_callback: Optional callback(current, total, state)

        Returns:
            List of AgentSystemState results
        """
        results = []

        self.logger.info(f"Starting batch processing: {len(questions)} questions")

        for i, q in enumerate(questions):
            self.logger.info(f"Processing question {i+1}/{len(questions)}")

            try:
                state = self.run(
                    question=q["question"],
                    schema=schema,
                    question_id=q.get("id"),
                    ground_truth=q.get("ground_truth")
                )

                results.append(state)

                if progress_callback:
                    progress_callback(i + 1, len(questions), state)

            except Exception as e:
                self.logger.error(f"Error processing question {q.get('id')}: {e}")
                # Create error state
                error_state = AgentSystemState(
                    question_id=q.get("id", 0),
                    question=q["question"],
                    schema=schema,
                    ground_truth=q.get("ground_truth")
                )
                error_state.finalize(elapsed_time=0, stop_reason="error")
                results.append(error_state)

        self.logger.info(f"Batch processing complete: {len(results)} results")

        return results

    def close(self):
        """Close database connection."""
        if self.executor:
            self.executor.close()


if __name__ == "__main__":
    # Test MultiAgentOrchestrator with ReAct
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing MultiAgentOrchestrator with ReAct reasoning...")

    # Test schema (curriculum domain)
    test_schema = """(:MK)-[:PREREQUISITE]->(:MK)
(:MK)-[:CAN_PARALLELIZED]->(:MK)
(:topic)-[:PART_OF]->(:MK)
(:LO)-[:PART_OF]->(:topic)
(:LO)-[:PURSUED_IN]->(:SO)"""

    # Initialize orchestrator with ReAct enabled
    orchestrator = MultiAgentOrchestrator(
        max_iterations=3,
        temperature=0.0,
        use_react=True
    )

    # Test question
    test_question = "Apa saja prasyarat untuk mata kuliah Basis Data?"

    print(f"\nTest Question: {test_question}")
    print("="*60)

    try:
        # Run multi-agent system
        state = orchestrator.run(
            question=test_question,
            schema=test_schema,
            question_id=1
        )

        # Print results
        print(f"\n{'='*60}")
        print("RESULTS")
        print('='*60)
        print(f"Total Iterations: {state.total_iterations}")
        print(f"Stop Reason: {state.stop_reason}")
        print(f"Final Query: {state.final_query}")
        print(f"Execution Success: {state.execution_success}")
        print(f"Total Tokens: {state.total_tokens}")
        print(f"Elapsed Time: {state.elapsed_time:.2f}s")

        # Print ReAct reasoning summary
        if state.reasoning_enabled:
            print(f"\n{'='*60}")
            print("REACT REASONING SUMMARY")
            print('='*60)
            print(f"Total Reasoning Steps: {state.total_reasoning_steps}")
            print(f"Avg Reasoning Quality: {state.avg_reasoning_quality:.2f}")

        # Print iteration details
        print(f"\n{'='*60}")
        print("ITERATION DETAILS")
        print('='*60)
        for iteration in state.iterations:
            print(f"\nIteration {iteration.iteration_number}:")
            print(f"  Evaluation: {iteration.evaluation}")
            print(f"  Query: {iteration.query[:60]}...")
            print(f"  Tokens: {iteration.tokens_used}")
            if iteration.used_verification:
                print(f"  Used Verification Module: Yes")

            # Show reasoning steps if available
            reasoning_steps = 0
            for reasoning in [
                iteration.generator_reasoning,
                iteration.evaluator_reasoning,
                iteration.instructions_reasoning,
                iteration.aggregator_reasoning
            ]:
                if reasoning:
                    reasoning_steps += reasoning.get('num_steps', 0)
            if reasoning_steps > 0:
                print(f"  Reasoning Steps: {reasoning_steps}")

        # Test batch processing
        print(f"\n{'='*60}")
        print("TEST: Batch Processing with ReAct")
        print('='*60)

        test_questions = [
            {"id": 1, "question": "Apa saja learning outcome dari mata kuliah Pemrograman Dasar?"},
            {"id": 2, "question": "Berapa jumlah mata kuliah yang tersedia?"}
        ]

        def progress_callback(current, total, state):
            reasoning_info = f", {state.total_reasoning_steps} reasoning steps" if state.reasoning_enabled else ""
            print(f"Progress: {current}/{total} - Q{state.question_id} completed in "
                  f"{state.total_iterations} iterations{reasoning_info}")

        batch_results = orchestrator.run_batch(
            questions=test_questions,
            schema=test_schema,
            progress_callback=progress_callback
        )

        print(f"\nBatch complete: {len(batch_results)} results")
        for result in batch_results:
            reasoning_info = f", {result.total_reasoning_steps} reasoning steps" if result.reasoning_enabled else ""
            print(f"  Q{result.question_id}: {result.total_iterations} iterations, "
                  f"{result.total_tokens} tokens, success={result.execution_success}{reasoning_info}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        orchestrator.close()

    print("\n" + "="*60)
    print("MultiAgentOrchestrator ReAct test complete!")
