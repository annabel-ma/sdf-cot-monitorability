#!/usr/bin/env python3
"""
Test script to verify installation and API connectivity.

Usage:
    python scripts/test_installation.py
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()


def test_imports():
    """Test that all required packages can be imported."""
    console.print("\n[bold]Testing imports...[/bold]")
    
    required_packages = [
        ("openai", "OpenAI API client"),
        ("anthropic", "Anthropic API client"),
        ("inspect_ai", "Inspect AI framework"),
        ("yaml", "YAML parser"),
        ("rich", "Rich formatting"),
        ("pandas", "Pandas data analysis"),
        ("numpy", "NumPy numerical computing"),
    ]
    
    failed = []
    for package, description in required_packages:
        try:
            __import__(package)
            console.print(f"  ✓ {package:20s} - {description}")
        except ImportError:
            console.print(f"  ✗ {package:20s} - {description} [red](MISSING)[/red]")
            failed.append(package)
    
    if failed:
        console.print(f"\n[red]Missing packages: {', '.join(failed)}[/red]")
        return False
    
    console.print("\n[green]All required packages installed![/green]")
    return True


def test_project_modules():
    """Test that project modules can be imported."""
    console.print("\n[bold]Testing project modules...[/bold]")
    
    try:
        from sdf_cot_monitorability import prompts, utils
        from sdf_cot_monitorability.evaluation import ImpossibleBenchRunner
        from sdf_cot_monitorability.monitor import CoTMonitor
        
        console.print("  ✓ sdf_cot_monitorability.prompts")
        console.print("  ✓ sdf_cot_monitorability.utils")
        console.print("  ✓ sdf_cot_monitorability.evaluation")
        console.print("  ✓ sdf_cot_monitorability.monitor")
        
        console.print("\n[green]All project modules load successfully![/green]")
        return True
    except ImportError as e:
        console.print(f"\n[red]Failed to import project modules: {e}[/red]")
        return False


def test_external_deps():
    """Test that external dependencies are available."""
    console.print("\n[bold]Testing external dependencies...[/bold]")
    
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    success = True
    
    # Test impossiblebench
    console.print("\n  ImpossibleBench:")
    ib_path = project_root / "external" / "impossiblebench"
    
    if not ib_path.exists():
        console.print("    ✗ Directory not found at external/impossiblebench [red](MISSING)[/red]")
        console.print("    Run: git submodule update --init --recursive")
        success = False
    elif not any(ib_path.iterdir()):
        console.print("    ✗ Directory exists but is empty [red](EMPTY)[/red]")
        console.print("    Run: git submodule update --init --recursive")
        success = False
    else:
        console.print("    ✓ Directory exists and has files")
        
        # Check if installed as Python package
        try:
            from impossiblebench import impossible_livecodebench
            console.print("    ✓ Installed as Python package")
        except ImportError:
            console.print("    ✗ Not installed as Python package [yellow](NOT INSTALLED)[/yellow]")
            console.print("    Run: cd external/impossiblebench && pip install -e .")
            success = False
    
    # Test false-facts
    console.print("\n  False-facts (for Phase 2 - SDF):")
    ff_path = project_root / "external" / "false-facts"
    
    if not ff_path.exists():
        console.print("    ✗ Directory not found at external/false-facts [red](MISSING)[/red]")
        console.print("    Run: git submodule update --init --recursive")
        success = False
    elif not any(ff_path.iterdir()):
        console.print("    ✗ Directory exists but is empty [red](EMPTY)[/red]")
        console.print("    Run: git submodule update --init --recursive")
        success = False
    else:
        console.print("    ✓ Directory exists and has files")
        
        # Check if installed as Python package
        try:
            from false_facts.synth_doc_generation import SyntheticDocumentGenerator
            console.print("    ✓ Installed as Python package")
        except ImportError:
            console.print("    ✗ Not installed as Python package [yellow](NOT INSTALLED)[/yellow]")
            console.print("    Run: cd external/false-facts && pip install -e .")
            # Don't fail for false-facts since it's only needed for Phase 2
    
    if success:
        console.print("\n[green]External dependencies properly set up![/green]")
    else:
        console.print("\n[yellow]Some external dependencies need attention[/yellow]")
    
    return success


def test_api_keys():
    """Test that API keys are configured."""
    console.print("\n[bold]Testing API keys...[/bold]")
    
    from sdf_cot_monitorability.utils import load_env_vars
    import os
    
    load_env_vars()
    
    keys = {
        "OPENAI_API_KEY": "OpenAI (required for experiments)",
        "ANTHROPIC_API_KEY": "Anthropic (required for monitoring)",
    }
    
    success = True
    for key, description in keys.items():
        value = os.getenv(key)
        if value and len(value) > 10:
            console.print(f"  ✓ {key:20s} - {description}")
        else:
            console.print(f"  ✗ {key:20s} - {description} [red](NOT SET)[/red]")
            success = False
    
    if not success:
        console.print("\n[yellow]API keys not configured. Edit .env file with your keys.[/yellow]")
    else:
        console.print("\n[green]API keys configured![/green]")
    
    return success


def test_docker():
    """Test that Docker is available."""
    console.print("\n[bold]Testing Docker...[/bold]")
    
    import subprocess
    
    try:
        result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            console.print("  ✓ Docker is running")
            console.print("\n[green]Docker ready for ImpossibleBench![/green]")
            return True
        else:
            console.print("  ✗ Docker command failed [red](NOT RUNNING)[/red]")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        console.print("  ✗ Docker not found or not running [red](NOT AVAILABLE)[/red]")
        console.print("    ImpossibleBench requires Docker for sandboxed execution")
        return False


def test_config_files():
    """Test that configuration files exist."""
    console.print("\n[bold]Testing configuration files...[/bold]")
    
    project_root = Path(__file__).parent.parent
    
    configs = [
        ("configs/baseline_experiment.yaml", "Baseline experiment config"),
        (".env", "Environment variables"),
    ]
    
    success = True
    for config_path, description in configs:
        full_path = project_root / config_path
        if full_path.exists():
            console.print(f"  ✓ {config_path:40s} - {description}")
        else:
            console.print(f"  ✗ {config_path:40s} - {description} [red](MISSING)[/red]")
            success = False
    
    if success:
        console.print("\n[green]All configuration files present![/green]")
    else:
        console.print("\n[yellow]Some configuration files missing. Run ./scripts/setup.sh[/yellow]")
    
    return success


def main():
    """Run all tests."""
    console.print(Panel.fit(
        "[bold cyan]SDF CoT Monitorability - Installation Test[/bold cyan]\n"
        "This script verifies your installation is ready for experiments.",
        border_style="cyan"
    ))
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Project Modules", test_project_modules()))
    results.append(("External Dependencies", test_external_deps()))
    results.append(("Configuration Files", test_config_files()))
    results.append(("API Keys", test_api_keys()))
    results.append(("Docker", test_docker()))
    
    # Summary
    console.print("\n" + "="*70)
    console.print("[bold]Summary[/bold]\n")
    
    for test_name, passed in results:
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        console.print(f"  {test_name:30s} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    console.print("\n" + "="*70)
    
    if all_passed:
        console.print(Panel.fit(
            "[bold green]All tests passed! You're ready to run experiments.[/bold green]\n\n"
            "Run your first experiment with:\n"
            "  [cyan]./scripts/run_baseline_experiment.sh[/cyan]",
            border_style="green"
        ))
        return 0
    else:
        console.print(Panel.fit(
            "[bold yellow]Some tests failed. Please fix the issues above.[/bold yellow]\n\n"
            "Common fixes:\n"
            "  • Run [cyan]./scripts/setup.sh[/cyan] to complete installation\n"
            "  • Edit [cyan].env[/cyan] with your API keys\n"
            "  • Start Docker Desktop",
            border_style="yellow"
        ))
        return 1


if __name__ == "__main__":
    sys.exit(main())

