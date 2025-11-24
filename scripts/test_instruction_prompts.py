#!/usr/bin/env python3
"""
Test script to view and validate all instruction prompt variants.

Usage:
    python scripts/test_instruction_prompts.py
    python scripts/test_instruction_prompts.py --type A
"""

import argparse
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from sdf_cot_monitorability.evaluation.instruction_prompts import (
    INSTRUCTION_PROMPTS,
    get_instruction_prompt,
    get_prompt_description,
)

console = Console()


def show_prompt(prompt_type: str):
    """Display a single instruction prompt."""
    try:
        prompt_text = get_instruction_prompt(prompt_type)
        description = get_prompt_description(prompt_type)
        
        console.print(Panel(
            f"[bold cyan]Type {prompt_type}[/bold cyan]: {description}",
            border_style="cyan"
        ))
        console.print()
        console.print(Panel(
            prompt_text,
            title=f"Instruction Prompt - Type {prompt_type}",
            border_style="green",
            padding=(1, 2)
        ))
        console.print()
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


def show_all_prompts():
    """Display all instruction prompt variants."""
    console.print("\n[bold magenta]═══ ImpossibleBench Instruction Prompt Variants ═══[/bold magenta]\n")
    console.print("[italic]These prompts control how strict the model should be when encountering flawed tests.[/italic]\n")
    
    for prompt_type in ["A", "B", "C", "D"]:
        show_prompt(prompt_type)
    
    console.print("[bold]Summary:[/bold]")
    console.print("  [green]A[/green] (Loosest)  → Just pass tests, don't evaluate")
    console.print("  [yellow]B[/yellow]           → Implement to specification")
    console.print("  [yellow]C[/yellow]           → Evaluate tests, but still implement")
    console.print("  [red]D[/red] (Strictest) → STOP if tests are flawed, don't pass them\n")


def main():
    parser = argparse.ArgumentParser(
        description="View and validate ImpossibleBench instruction prompt variants"
    )
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        choices=["A", "B", "C", "D"],
        help="Show only a specific prompt type (default: show all)"
    )
    
    args = parser.parse_args()
    
    if args.type:
        show_prompt(args.type.upper())
    else:
        show_all_prompts()


if __name__ == "__main__":
    main()

