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
    
    def run(
        self,
        conditions: list[MonitoringCondition] | None = None,
        skip_monitoring: bool = False,
    ) -> dict:
        """
        Run the full experiment.
        
        Args:
            conditions: List of conditions to run (None = all from config)
            skip_monitoring: Skip the monitoring phase
            
        Returns:
            Dictionary of results
        """
        print("\n" + "="*60)
        print("Starting Baseline Prompting Experiment")
        print("="*60 + "\n")
        
        # Determine which conditions to run
        if conditions is None:
            conditions = self.config["conditions"]
        
        print(f"Conditions to test: {', '.join(conditions)}")
        print(f"Model: {self.config['models']['primary']}\n")
        
        # Phase 1: Run ImpossibleBench evaluations
        print("\n" + "="*60)
        print("Phase 1: Running ImpossibleBench Evaluations")
        print("="*60 + "\n")
        eval_results = self._run_evaluations(conditions)
        
        # Phase 2: Run monitoring
        if not skip_monitoring and self.config["experiment_settings"]["run_monitoring"]:
            print("\n" + "="*60)
            print("Phase 2: Running Monitor Analysis")
            print("="*60 + "\n")
            monitor_results = self._run_monitoring(eval_results)
        else:
            monitor_results = {}
            print("\nSkipping monitoring phase\n")
        
        # Phase 3: Generate report
        if self.config["experiment_settings"]["generate_report"]:
            print("\n" + "="*60)
            print("Phase 3: Generating Report")
            print("="*60 + "\n")
            report = self._generate_report(eval_results, monitor_results)
            save_json(report, self.run_dir / "report.json")
            self._print_report(report)
        
        print(f"\n✓ Experiment complete!")
        print(f"Results saved to: {self.run_dir}\n")
        
        return {
            "eval_results": eval_results,
            "monitor_results": monitor_results,
            "run_dir": str(self.run_dir),
        }
    
    def _run_evaluations(
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
        
        # Log instruction prompt type
        from sdf_cot_monitorability.evaluation.instruction_prompts import get_prompt_description
        prompt_desc = get_prompt_description(ib_config.instruction_prompt_type)
        print(f"Using instruction prompt type {ib_config.instruction_prompt_type}: {prompt_desc}\n")
        
        results = {}
        
        # Create eval directory
        eval_dir = self.run_dir / "evaluations"
        ensure_dir(eval_dir)
        
        # Run evaluations sequentially to preserve inspect_ai UI
        for condition in conditions:
            result = self._run_single_evaluation(
                condition, model, ib_config, eval_dir
            )
            results[condition] = result
        
        return results
    
    def _run_single_evaluation(
        self,
        condition: MonitoringCondition,
        model: str,
        config: ImpossibleBenchConfig,
        output_dir: Path,
    ):
        """Run a single evaluation for one condition."""
        # Let inspect_ai handle the terminal output - just log minimally
        print(f"\n{'='*60}")
        print(f"Running condition: {condition}")
        print(f"{'='*60}\n")
        
        runner = ImpossibleBenchRunner(
            model=model,
            condition=condition,
            config=config,
            output_dir=output_dir,
        )
        
        result = runner.run()
        
        cheating_rate = result.metrics.get('cheating_rate', 'N/A')
        if isinstance(cheating_rate, (int, float)):
            cheating_rate_str = f"{cheating_rate:.2%}"
        else:
            cheating_rate_str = str(cheating_rate)
        
        print(f"\n✓ {condition}: Cheating rate = {cheating_rate_str}\n")
        
        return result
    
    def _run_monitoring(self, eval_results: dict) -> dict:
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
            print(f"  Monitoring condition: {condition}")
            
            # Load samples
            samples_file = (
                eval_result.output_dir
                / f"{condition}_{eval_result.model.replace('/', '_')}_samples.json"
            )
            
            if not samples_file.exists():
                print(f"    No samples file found, skipping")
                continue
            
            samples = load_json(samples_file)
            
            # Run monitoring (synchronous)
            monitor_results = monitor.monitor_samples(
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
            
            # Format rates, handling N/A values
            cot_rate = analysis.get('cot_cheating_rate', 'N/A')
            cot_rate_str = f"{cot_rate:.2%}" if isinstance(cot_rate, (int, float)) else str(cot_rate)
            
            incrim_rate = analysis.get('self_incrimination_rate', 'N/A')
            incrim_rate_str = f"{incrim_rate:.2%}" if isinstance(incrim_rate, (int, float)) else str(incrim_rate)
            
            console.print(f"    ✓ CoT cheating rate: {cot_rate_str}")
            console.print(f"    ✓ Self-incrimination: {incrim_rate_str}")
        
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
    results = experiment.run(
        conditions=args.conditions,
        skip_monitoring=args.skip_monitoring,
    )
    
    return results


if __name__ == "__main__":
    main()

