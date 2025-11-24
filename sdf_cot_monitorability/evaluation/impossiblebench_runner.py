"""
ImpossibleBench integration for testing monitorability under different belief conditions.

This module provides a wrapper around ImpossibleBench that allows injecting
custom system prompts to test monitoring beliefs.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from inspect_ai import eval_async
from inspect_ai.log import EvalLog

from sdf_cot_monitorability.prompts import (
    MonitoringCondition,
    get_instruction_prompt,
    get_system_prompt,
)
from sdf_cot_monitorability.utils import ensure_dir, save_json

logger = logging.getLogger(__name__)

BenchmarkType = Literal["livecodebench", "swebench"]
AgentType = Literal["minimal", "tools"]
SplitType = Literal["original", "oneoff", "conflicting"]


@dataclass
class ImpossibleBenchConfig:
    """Configuration for ImpossibleBench evaluation."""
    
    benchmark_type: BenchmarkType = "livecodebench"
    agent_type: AgentType = "minimal"
    split: SplitType = "conflicting"
    limit: int | None = None
    shuffle: bool = True
    max_attempts: int = 10
    allow_test_modifications: bool = True
    sandbox: str = "docker"
    message_limit: int = 50
    max_connections: int = 20
    max_subprocesses: int = 20
    reasoning_tokens: int = 4096
    reasoning_effort: str = "medium"


@dataclass
class EvaluationResult:
    """Results from an ImpossibleBench evaluation run."""
    
    condition: MonitoringCondition
    model: str
    config: ImpossibleBenchConfig
    log_path: Path
    metrics: dict[str, Any]
    samples: list[dict[str, Any]]


class ImpossibleBenchRunner:
    """
    Runner for ImpossibleBench evaluations with monitoring condition interventions.
    """
    
    def __init__(
        self,
        model: str,
        condition: MonitoringCondition,
        config: ImpossibleBenchConfig | None = None,
        output_dir: Path | str | None = None,
    ):
        """
        Initialize the runner.
        
        Args:
            model: Model identifier (e.g., "openai/gpt-4o-mini")
            condition: Monitoring condition to test
            config: ImpossibleBench configuration
            output_dir: Directory for outputs
        """
        self.model = model
        self.condition = condition
        self.config = config or ImpossibleBenchConfig()
        self.output_dir = Path(output_dir) if output_dir else Path("results") / "baseline"
        ensure_dir(self.output_dir)
        
        # Get prompts for this condition
        self.system_prompt = get_system_prompt(condition)
        self.instruction_prompt = get_instruction_prompt(condition)
        
        logger.info(f"Initialized runner for model={model}, condition={condition}")
    
    def _create_task(self):
        """
        Create the ImpossibleBench task with custom prompts.
        
        Returns:
            Configured task object
        """
        # Import here to avoid issues if impossiblebench isn't installed yet
        try:
            from impossiblebench import impossible_livecodebench, impossible_swebench
        except ImportError as e:
            raise ImportError(
                "impossiblebench not installed. Please install from external/impossiblebench: "
                "cd external/impossiblebench && pip install -e ."
            ) from e
        
        task_fn = (
            impossible_livecodebench
            if self.config.benchmark_type == "livecodebench"
            else impossible_swebench
        )
        
        task = task_fn(
            split=self.config.split,
            agent_type=self.config.agent_type,
            instruction_prompt=self.instruction_prompt,
            max_attempts=self.config.max_attempts,
            allow_test_modifications=self.config.allow_test_modifications,
            sandbox=self.config.sandbox,
            limit=self.config.limit,
            shuffle=self.config.shuffle,
            message_limit=self.config.message_limit,
        )
        
        return task
    
    async def run_async(self) -> EvaluationResult:
        """
        Run the evaluation asynchronously.
        
        Returns:
            Evaluation results
        """
        logger.info(f"Starting evaluation: {self.condition} condition on {self.model}")
        
        task = self._create_task()
        
        # Create log directory for this run
        log_dir = self.output_dir / "logs" / f"{self.condition}_{self.model.replace('/', '_')}"
        ensure_dir(log_dir)
        
        # Run evaluation with inspect_ai
        # Note: We prepend the system prompt to the model's system message
        # This is a bit of a hack - ideally we'd want to modify the task itself
        # Use eval_async since we're already in an async context
        logs = await eval_async(
            task,
            model=self.model,
            max_connections=self.config.max_connections,
            max_subprocesses=self.config.max_subprocesses,
            max_sandboxes=self.config.max_subprocesses,
            reasoning_tokens=self.config.reasoning_tokens,
            reasoning_effort=self.config.reasoning_effort,
            log_dir=str(log_dir),
            fail_on_error=False,
        )
        
        # Extract results from logs
        if isinstance(logs, list):
            log = logs[0] if logs else None
        else:
            log = logs
        
        if not log:
            raise RuntimeError("Evaluation produced no logs")
        
        # Parse results
        metrics = self._extract_metrics(log)
        samples = self._extract_samples(log)
        
        # Save results
        result = EvaluationResult(
            condition=self.condition,
            model=self.model,
            config=self.config,
            log_path=log_dir,
            metrics=metrics,
            samples=samples,
        )
        
        self._save_results(result)
        
        logger.info(
            f"Completed evaluation: {self.condition} condition on {self.model}. "
            f"Cheating rate: {metrics.get('cheating_rate', 'N/A')}"
        )
        
        return result
    
    def run(self) -> EvaluationResult:
        """
        Run the evaluation synchronously.
        
        Returns:
            Evaluation results
        """
        return asyncio.run(self.run_async())
    
    def _extract_metrics(self, log: EvalLog) -> dict[str, Any]:
        """
        Extract metrics from evaluation log.
        
        Args:
            log: Evaluation log
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        if hasattr(log, "results") and log.results:
            results = log.results
            
            # Extract standard metrics
            if hasattr(results, "metrics"):
                for key, value in results.metrics.items():
                    metrics[key] = value.value if hasattr(value, "value") else value
            
            # Calculate cheating rate if available
            if hasattr(results, "scores"):
                scores_data = results.scores
                
                # scores might be a dict (new API) or list (old API)
                if isinstance(scores_data, dict):
                    # Extract score values from dict
                    for scorer_name, score_obj in scores_data.items():
                        if hasattr(score_obj, "value"):
                            metrics[scorer_name] = score_obj.value
                        elif hasattr(score_obj, "score"):
                            metrics[scorer_name] = score_obj.score
                        else:
                            metrics[scorer_name] = str(score_obj)
                    
                    # For ImpossibleBench, use accuracy as cheating rate
                    # (correct = cheated since tasks are impossible)
                    if "accuracy" in metrics:
                        metrics["cheating_rate"] = metrics["accuracy"]
                else:
                    # Old API: list of score objects
                    total = len(scores_data)
                    if total > 0:
                        cheated = 0
                        for s in scores_data:
                            score_val = None
                            if hasattr(s, "value"):
                                score_val = s.value
                            elif hasattr(s, "score"):
                                score_val = s.score
                            
                            if score_val == "C" or score_val == 1.0 or score_val == 1:
                                cheated += 1
                        
                        metrics["cheating_rate"] = cheated / total
                        metrics["total_samples"] = total
                        metrics["cheated_count"] = cheated
        
        return metrics
    
    def _extract_samples(self, log: EvalLog) -> list[dict[str, Any]]:
        """
        Extract sample-level data from evaluation log.
        
        Args:
            log: Evaluation log
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        if hasattr(log, "samples") and log.samples:
            for sample in log.samples:
                sample_data = {
                    "id": getattr(sample, "id", None),
                    "input": getattr(sample, "input", None),
                    "target": getattr(sample, "target", None),
                    "output": getattr(sample, "output", None),
                    "score": getattr(sample, "score", None),
                    "metadata": getattr(sample, "metadata", {}),
                }
                
                # Extract messages/transcript for monitor analysis
                if hasattr(sample, "messages"):
                    sample_data["messages"] = [
                        {
                            "role": msg.role,
                            "content": msg.content,
                        }
                        for msg in sample.messages
                    ]
                
                samples.append(sample_data)
        
        return samples
    
    def _save_results(self, result: EvaluationResult) -> None:
        """
        Save evaluation results to disk.
        
        Args:
            result: Evaluation results to save
        """
        output_file = (
            self.output_dir
            / f"{result.condition}_{result.model.replace('/', '_')}_results.json"
        )
        
        save_json(
            {
                "condition": result.condition,
                "model": result.model,
                "config": {
                    "benchmark_type": result.config.benchmark_type,
                    "agent_type": result.config.agent_type,
                    "split": result.config.split,
                    "limit": result.config.limit,
                },
                "metrics": result.metrics,
                "log_path": str(result.log_path),
                "num_samples": len(result.samples),
            },
            output_file,
        )
        
        # Save samples separately (can be large)
        samples_file = (
            self.output_dir
            / f"{result.condition}_{result.model.replace('/', '_')}_samples.json"
        )
        save_json(result.samples, samples_file)
        
        logger.info(f"Saved results to {output_file}")


async def run_impossiblebench_eval(
    model: str,
    condition: MonitoringCondition,
    config: ImpossibleBenchConfig | None = None,
    output_dir: Path | str | None = None,
) -> EvaluationResult:
    """
    Convenience function to run a single ImpossibleBench evaluation.
    
    Args:
        model: Model identifier
        condition: Monitoring condition
        config: ImpossibleBench configuration
        output_dir: Output directory
        
    Returns:
        Evaluation results
    """
    runner = ImpossibleBenchRunner(
        model=model,
        condition=condition,
        config=config,
        output_dir=output_dir,
    )
    return await runner.run_async()

