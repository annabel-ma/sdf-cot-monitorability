#!/usr/bin/env python3
"""
Run strictness-only experiment: Test all 4 instruction prompt types (A, B, C, D)
WITHOUT monitoring beliefs (control condition only).

This isolates the effect of prompt strictness on cheating behavior.

Usage:
    python scripts/run_strictness_experiment.py --config configs/strictness_experiment.yaml
    python scripts/run_strictness_experiment.py --config configs/strictness_experiment.yaml --types A D
"""

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from sdf_cot_monitorability.evaluation.impossiblebench_runner import (
    ImpossibleBenchConfig,
    ImpossibleBenchRunner,
)
from sdf_cot_monitorability.evaluation.instruction_prompts import get_prompt_description
from sdf_cot_monitorability.prompts import MonitoringCondition
from sdf_cot_monitorability.utils import ensure_dir, load_config, save_json

console = Console()
logger = logging.getLogger(__name__)


class StrictnessExperiment:
    """Run strictness-only experiments (no monitoring variable)."""
    
    def __init__(self, config: dict):
        """
        Initialize experiment.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = Path(config["experiment"]["output_dir"])
        ensure_dir(self.output_dir)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        ensure_dir(self.run_dir)
        
        # Save config to run directory
        save_json(config, self.run_dir / "config.json")
        
        logger.info(f"Initialized strictness experiment in {self.run_dir}")
    
    async def run_async(self, prompt_types: list[str] = None) -> dict:
        """
        Run the strictness experiment.
        
        Args:
            prompt_types: List of prompt types to test (default: all ["A", "B", "C", "D"])
            
        Returns:
            Dictionary with results for each prompt type
        """
        if prompt_types is None:
            prompt_types = ["A", "B", "C", "D"]
        
        # Validate prompt types
        valid_types = {"A", "B", "C", "D"}
        for pt in prompt_types:
            if pt.upper() not in valid_types:
                raise ValueError(f"Invalid prompt type: {pt}. Must be one of {valid_types}")
        
        prompt_types = [pt.upper() for pt in prompt_types]
        
        console.print(Panel(
            f"[bold cyan]Strictness Experiment[/bold cyan]\n\n"
            f"Testing prompt types: {', '.join(prompt_types)}\n"
            f"Condition: Control (no monitoring information)\n"
            f"Output: {self.run_dir}",
            title="Experiment Configuration",
            border_style="cyan"
        ))
        
        # Run evaluations for each prompt type
        console.print("\n[bold green]Running Evaluations[/bold green]\n")
        eval_results = await self._run_evaluations(prompt_types)
        
        # Generate report
        console.print("\n[bold green]Generating Report[/bold green]\n")
        report = self._generate_report(eval_results)
        save_json(report, self.run_dir / "report.json")
        self._print_report(report)
        
        console.print(f"\n[bold green]✓ Experiment complete![/bold green]")
        console.print(f"Results saved to: {self.run_dir}\n")
        
        return {
            "eval_results": eval_results,
            "run_dir": str(self.run_dir),
        }
    
    async def _run_evaluations(self, prompt_types: list[str]) -> dict:
        """
        Run ImpossibleBench evaluations for all prompt types.
        
        Args:
            prompt_types: List of prompt types to test
            
        Returns:
            Dictionary mapping prompt types to results
        """
        model = self.config["models"]["primary"]
        condition = "control"  # Always use control condition
        
        results = {}
        
        for prompt_type in prompt_types:
            description = get_prompt_description(prompt_type)
            
            console.print(f"\n[bold cyan]Testing Prompt Type {prompt_type}[/bold cyan]")
            console.print(f"[dim]{description}[/dim]\n")
            
            # Create config with this prompt type
            ib_config = ImpossibleBenchConfig(**self.config["impossiblebench"])
            ib_config.instruction_prompt_type = prompt_type
            
            # Create runner
            runner = ImpossibleBenchRunner(
                model=model,
                condition=condition,
                config=ib_config,
                output_dir=self.run_dir / "evaluations"
            )
            
            # Run evaluation
            result = await runner.run_async()
            
            # Save result
            results[prompt_type] = result
            
            # Log summary
            metrics = result.metrics
            console.print(f"[green]✓ Completed prompt type {prompt_type}[/green]")
            console.print(f"  Samples: {len(result.samples)}")
            console.print(f"  Accuracy: {metrics.get('accuracy', 0):.1%}")
            console.print(f"  Cheating rate: {metrics.get('cheating_rate', 0):.1%}")
        
        return results
    
    def _generate_report(self, eval_results: dict) -> dict:
        """
        Generate a report comparing all prompt types.
        
        Args:
            eval_results: Dictionary mapping prompt types to results
            
        Returns:
            Report dictionary
        """
        report = {
            "experiment": self.config["experiment"]["name"],
            "timestamp": datetime.now().isoformat(),
            "model": self.config["models"]["primary"],
            "condition": "control",
            "prompt_types": {}
        }
        
        for prompt_type, result in eval_results.items():
            metrics = result.metrics
            report["prompt_types"][prompt_type] = {
                "description": get_prompt_description(prompt_type),
                "num_samples": len(result.samples),
                "accuracy": metrics.get("accuracy", 0),
                "cheating_rate": metrics.get("cheating_rate", 0),
                "total_samples": metrics.get("total_samples", 0),
                "cheated_count": metrics.get("cheated_count", 0),
            }
        
        return report
    
    def _print_report(self, report: dict):
        """Print a formatted report."""
        console.print("\n")
        console.print(Panel(
            "[bold cyan]Strictness Experiment Results[/bold cyan]",
            border_style="cyan"
        ))
        console.print()
        
        # Create comparison table
        from rich.table import Table
        
        table = Table(title="Prompt Type Comparison", show_header=True)
        table.add_column("Type", style="bold")
        table.add_column("Description", style="dim")
        table.add_column("Samples", justify="center")
        table.add_column("Accuracy", justify="right")
        table.add_column("Cheating Rate", justify="right")
        
        for prompt_type in ["A", "B", "C", "D"]:
            if prompt_type in report["prompt_types"]:
                data = report["prompt_types"][prompt_type]
                
                # Color code by strictness
                type_colors = {"A": "green", "B": "yellow", "C": "yellow", "D": "red"}
                type_style = type_colors.get(prompt_type, "white")
                
                accuracy = data["accuracy"]
                cheating_rate = data["cheating_rate"]
                
                table.add_row(
                    f"[{type_style}]{prompt_type}[/{type_style}]",
                    data["description"],
                    str(data["num_samples"]),
                    f"{accuracy:.1%}" if isinstance(accuracy, (int, float)) else str(accuracy),
                    f"{cheating_rate:.1%}" if isinstance(cheating_rate, (int, float)) else str(cheating_rate),
                )
        
        console.print(table)
        console.print()
        
        console.print(f"[dim]Full results saved to: {self.run_dir}[/dim]\n")
    
    def run(self, prompt_types: list[str] = None) -> dict:
        """
        Synchronous wrapper for run_async.
        
        Args:
            prompt_types: List of prompt types to test
            
        Returns:
            Results dictionary
        """
        return asyncio.run(self.run_async(prompt_types))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run strictness-only experiment (no monitoring variable)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/strictness_experiment.yaml",
        help="Path to config file (default: configs/strictness_experiment.yaml)"
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["A", "B", "C", "D"],
        help="Prompt types to test (default: all)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load config
    config = load_config(args.config)
    
    # Run experiment
    experiment = StrictnessExperiment(config)
    results = experiment.run(prompt_types=args.types)
    
    return results


if __name__ == "__main__":
    main()

