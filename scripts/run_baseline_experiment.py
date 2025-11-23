#!/usr/bin/env python3
"""
Run baseline prompting experiments to test if monitoring beliefs affect behavior.

This script runs ImpossibleBench evaluations under different prompting conditions
(monitored, unmonitored, control) and analyzes the results with a monitor model.

Usage:
    python scripts/run_baseline_experiment.py --config configs/baseline_experiment.yaml
    python scripts/run_baseline_experiment.py --config configs/baseline_experiment.yaml --conditions monitored unmonitored
    python scripts/run_baseline_experiment.py --config configs/baseline_experiment.yaml --skip-monitoring
"""

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from sdf_cot_monitorability.evaluation import ImpossibleBenchRunner
from sdf_cot_monitorability.evaluation.impossiblebench_runner import (
    ImpossibleBenchConfig,
)
from sdf_cot_monitorability.monitor import (
    CoTMonitor,
    MonitorConfig,
    analyze_monitoring_results,
)
from sdf_cot_monitorability.prompts import MonitoringCondition
from sdf_cot_monitorability.utils import (
    ensure_dir,
    get_project_root,
    load_config,
    load_json,
    save_json,
    setup_logging,
)

console = Console()
logger = logging.getLogger(__name__)


class BaselineExperiment:
    """Orchestrates baseline prompting experiments."""
    
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
        
        logger.info(f"Initialized experiment in {self.run_dir}")
    
    async def run_async(
        self,
        conditions: list[MonitoringCondition] | None = None,
        skip_monitoring: bool = False,
    ) -> dict:
        """
        Run the full experiment asynchronously.
        
        Args:
            conditions: List of conditions to run (None = all from config)
            skip_monitoring: Skip the monitoring phase
            
        Returns:
            Dictionary of results
        """
        console.print("\n[bold blue]Starting Baseline Prompting Experiment[/bold blue]\n")
        
        # Determine which conditions to run
        if conditions is None:
            conditions = self.config["conditions"]
        
        console.print(f"Conditions to test: {', '.join(conditions)}")
        console.print(f"Model: {self.config['models']['primary']}\n")
        
        # Phase 1: Run ImpossibleBench evaluations
        console.print("[bold green]Phase 1: Running ImpossibleBench Evaluations[/bold green]\n")
        eval_results = await self._run_evaluations(conditions)
        
        # Phase 2: Run monitoring
        if not skip_monitoring and self.config["experiment_settings"]["run_monitoring"]:
            console.print("\n[bold green]Phase 2: Running Monitor Analysis[/bold green]\n")
            monitor_results = await self._run_monitoring(eval_results)
        else:
            monitor_results = {}
            console.print("\n[yellow]Skipping monitoring phase[/yellow]\n")
        
        # Phase 3: Generate report
        if self.config["experiment_settings"]["generate_report"]:
            console.print("\n[bold green]Phase 3: Generating Report[/bold green]\n")
            report = self._generate_report(eval_results, monitor_results)
            save_json(report, self.run_dir / "report.json")
            self._print_report(report)
        
        console.print(f"\n[bold green]✓ Experiment complete![/bold green]")
        console.print(f"Results saved to: {self.run_dir}\n")
        
        return {
            "eval_results": eval_results,
            "monitor_results": monitor_results,
            "run_dir": str(self.run_dir),
        }
    
    async def _run_evaluations(
        self,
        conditions: list[MonitoringCondition],
    ) -> dict:
        """
        Run ImpossibleBench evaluations for all conditions.
        
        Args:
            conditions: List of conditions to evaluate
            
        Returns:
            Dictionary mapping conditions to results
        """
        model = self.config["models"]["primary"]
        ib_config = ImpossibleBenchConfig(**self.config["impossiblebench"])
        
        results = {}
        
        # Create eval directory
        eval_dir = self.run_dir / "evaluations"
        ensure_dir(eval_dir)
        
        # Run evaluations (sequentially or in parallel based on config)
        if self.config["experiment_settings"].get("parallel_conditions", False):
            # Parallel execution
            tasks = [
                self._run_single_evaluation(condition, model, ib_config, eval_dir)
                for condition in conditions
            ]
            eval_results = await asyncio.gather(*tasks)
            results = dict(zip(conditions, eval_results))
        else:
            # Sequential execution (more stable, easier to debug)
            for condition in conditions:
                result = await self._run_single_evaluation(
                    condition, model, ib_config, eval_dir
                )
                results[condition] = result
        
        return results
    
    async def _run_single_evaluation(
        self,
        condition: MonitoringCondition,
        model: str,
        config: ImpossibleBenchConfig,
        output_dir: Path,
    ):
        """Run a single evaluation for one condition."""
        console.print(f"  Running condition: [cyan]{condition}[/cyan]")
        
        runner = ImpossibleBenchRunner(
            model=model,
            condition=condition,
            config=config,
            output_dir=output_dir,
        )
        
        result = await runner.run_async()
        
        console.print(
            f"  ✓ {condition}: Cheating rate = "
            f"{result.metrics.get('cheating_rate', 'N/A'):.2%}"
        )
        
        return result
    
    async def _run_monitoring(self, eval_results: dict) -> dict:
        """
        Run monitor analysis on evaluation results.
        
        Args:
            eval_results: Dictionary of evaluation results by condition
            
        Returns:
            Dictionary of monitoring results by condition
        """
        monitor_config = MonitorConfig(**self.config["monitor"])
        monitor = CoTMonitor(config=monitor_config)
        
        results = {}
        monitor_dir = self.run_dir / "monitoring"
        ensure_dir(monitor_dir)
        
        for condition, eval_result in eval_results.items():
            console.print(f"  Monitoring condition: [cyan]{condition}[/cyan]")
            
            # Load samples
            samples_file = (
                eval_result.output_dir
                / f"{condition}_{eval_result.model.replace('/', '_')}_samples.json"
            )
            
            if not samples_file.exists():
                console.print(f"    [yellow]No samples file found, skipping[/yellow]")
                continue
            
            samples = load_json(samples_file)
            
            # Run monitoring
            monitor_results = await monitor.monitor_samples_async(
                samples, show_progress=True
            )
            
            # Save results
            output_file = monitor_dir / f"{condition}_monitor_results.json"
            monitor.save_results(monitor_results, output_file)
            
            # Analyze
            analysis = analyze_monitoring_results(monitor_results)
            results[condition] = {
                "monitor_results": monitor_results,
                "analysis": analysis,
            }
            
            console.print(
                f"    ✓ CoT cheating rate: {analysis.get('cot_cheating_rate', 'N/A'):.2%}"
            )
            console.print(
                f"    ✓ Self-incrimination: {analysis.get('self_incrimination_rate', 'N/A'):.2%}"
            )
        
        return results
    
    def _generate_report(self, eval_results: dict, monitor_results: dict) -> dict:
        """
        Generate a comprehensive report of results.
        
        Args:
            eval_results: Evaluation results
            monitor_results: Monitoring results
            
        Returns:
            Report dictionary
        """
        report = {
            "experiment": self.config["experiment"]["name"],
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "conditions_tested": list(eval_results.keys()),
        }
        
        # Aggregate metrics by condition
        condition_summaries = {}
        
        for condition in eval_results:
            summary = {
                "model": eval_results[condition].model,
            }
            
            # Evaluation metrics
            eval_metrics = eval_results[condition].metrics
            summary["impossiblebench_cheating_rate"] = eval_metrics.get("cheating_rate")
            summary["total_samples"] = eval_metrics.get("total_samples")
            
            # Monitoring metrics
            if condition in monitor_results:
                monitor_analysis = monitor_results[condition]["analysis"]
                summary["cot_cheating_rate"] = monitor_analysis.get("cot_cheating_rate")
                summary["output_cheating_rate"] = monitor_analysis.get("output_cheating_rate")
                summary["self_incrimination_rate"] = monitor_analysis.get("self_incrimination_rate")
                summary["obfuscation_rate"] = monitor_analysis.get("obfuscation_rate")
                summary["cot_detection_advantage"] = monitor_analysis.get("cot_detection_advantage")
            
            condition_summaries[condition] = summary
        
        report["condition_summaries"] = condition_summaries
        
        # Comparative analysis
        if len(condition_summaries) > 1:
            report["comparative_analysis"] = self._compute_comparative_metrics(
                condition_summaries
            )
        
        return report
    
    def _compute_comparative_metrics(self, summaries: dict) -> dict:
        """Compute comparative metrics across conditions."""
        comparison = {}
        
        # Compare monitored vs unmonitored
        if "monitored" in summaries and "unmonitored" in summaries:
            mon = summaries["monitored"]
            unmon = summaries["unmonitored"]
            
            if mon.get("cot_cheating_rate") and unmon.get("cot_cheating_rate"):
                comparison["cheating_rate_delta"] = (
                    unmon["cot_cheating_rate"] - mon["cot_cheating_rate"]
                )
            
            if mon.get("self_incrimination_rate") and unmon.get("self_incrimination_rate"):
                comparison["self_incrimination_delta"] = (
                    unmon["self_incrimination_rate"] - mon["self_incrimination_rate"]
                )
        
        # Compare against control
        if "control" in summaries:
            control = summaries["control"]
            
            for condition in ["monitored", "unmonitored"]:
                if condition in summaries:
                    cond = summaries[condition]
                    if cond.get("cot_cheating_rate") and control.get("cot_cheating_rate"):
                        comparison[f"{condition}_vs_control_delta"] = (
                            cond["cot_cheating_rate"] - control["cot_cheating_rate"]
                        )
        
        return comparison
    
    def _print_report(self, report: dict) -> None:
        """Print a formatted report to console."""
        console.print("\n[bold]Experiment Report[/bold]\n")
        
        # Create results table
        table = Table(title="Results by Condition")
        
        table.add_column("Condition", style="cyan")
        table.add_column("IB Cheat Rate", justify="right")
        table.add_column("CoT Cheat Rate", justify="right")
        table.add_column("Self-Incrim", justify="right")
        table.add_column("Obfuscation", justify="right")
        
        for condition, summary in report["condition_summaries"].items():
            table.add_row(
                condition,
                f"{summary.get('impossiblebench_cheating_rate', 'N/A'):.2%}"
                if summary.get("impossiblebench_cheating_rate") is not None
                else "N/A",
                f"{summary.get('cot_cheating_rate', 'N/A'):.2%}"
                if summary.get("cot_cheating_rate") is not None
                else "N/A",
                f"{summary.get('self_incrimination_rate', 'N/A'):.2%}"
                if summary.get("self_incrimination_rate") is not None
                else "N/A",
                f"{summary.get('obfuscation_rate', 'N/A'):.2%}"
                if summary.get("obfuscation_rate") is not None
                else "N/A",
            )
        
        console.print(table)
        
        # Print comparative analysis
        if "comparative_analysis" in report:
            console.print("\n[bold]Comparative Analysis[/bold]\n")
            comp = report["comparative_analysis"]
            
            if "cheating_rate_delta" in comp:
                delta = comp["cheating_rate_delta"]
                console.print(
                    f"  Unmonitored vs Monitored cheating rate: "
                    f"[{'green' if delta > 0 else 'red'}]{delta:+.2%}[/]"
                )
            
            if "self_incrimination_delta" in comp:
                delta = comp["self_incrimination_delta"]
                console.print(
                    f"  Unmonitored vs Monitored self-incrimination: "
                    f"[{'green' if delta > 0 else 'red'}]{delta:+.2%}[/]"
                )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run baseline prompting experiments for monitorability"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["monitored", "unmonitored", "control"],
        help="Conditions to run (default: all from config)",
    )
    parser.add_argument(
        "--skip-monitoring",
        action="store_true",
        help="Skip the monitoring phase",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(level=args.log_level)
    config = load_config(args.config)
    
    # Run experiment
    experiment = BaselineExperiment(config)
    results = asyncio.run(
        experiment.run_async(
            conditions=args.conditions,
            skip_monitoring=args.skip_monitoring,
        )
    )
    
    return results


if __name__ == "__main__":
    main()

