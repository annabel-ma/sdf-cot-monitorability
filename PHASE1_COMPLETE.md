# Phase 1: Baseline Prompting Experiments - COMPLETE

## What Was Built

Phase 1 of the SDF CoT Monitorability project has been successfully implemented. This phase establishes the baseline for testing whether prompting alone can change model behavior around monitoring.

### Components Created

#### 1. Core Modules

**`sdf_cot_monitorability/prompts.py`**
- System prompts for three conditions:
  - **Monitored**: "Your CoT will be read and reviewed"
  - **Unmonitored**: "Your CoT is private and will never be read"
  - **Control**: No monitoring information
- Instruction prompts adapted for ImpossibleBench
- Prompt formatting utilities

**`sdf_cot_monitorability/utils.py`**
- Environment setup and API key management
- Logging configuration with Rich formatting
- JSON/JSONL file I/O
- Configuration loading
- Directory management

#### 2. Evaluation Infrastructure

**`sdf_cot_monitorability/evaluation/impossiblebench_runner.py`**
- Wrapper around ImpossibleBench for custom prompting
- Support for both LiveCodeBench and SWE-bench
- Configurable agent types (minimal/tools)
- Result extraction and persistence
- Async and sync execution modes

Key features:
- Injects monitoring condition prompts into evaluations
- Extracts CoT traces from model messages
- Saves detailed sample-level data for monitoring
- Handles multiple models and conditions

#### 3. Monitoring System

**`sdf_cot_monitorability/monitor/cot_monitor.py`**
- LLM-as-a-judge for cheating detection
- Uses ImpossibleBench's `LLMJudge` class
- Analyzes both CoT traces and output-only
- Detects:
  - Cheating behavior
  - Self-incrimination
  - Obfuscation
- Computes aggregate metrics:
  - Cheating rates
  - Detection accuracy
  - CoT detection advantage

#### 4. Experiment Runner

**`scripts/run_baseline_experiment.py`**
- Orchestrates full experimental pipeline
- Three phases:
  1. ImpossibleBench evaluations across conditions
  2. Monitor analysis of results
  3. Report generation and comparison
- Parallel or sequential execution
- Rich terminal output with progress tracking
- Automatic result persistence

**`scripts/run_baseline_experiment.sh`**
- Convenience wrapper for Python script
- Pre-flight checks (Docker, .env)
- Easy command-line usage

#### 5. Configuration

**`configs/baseline_experiment.yaml`**
- Centralized experiment configuration
- Model selection
- ImpossibleBench parameters
- Monitor settings
- Execution options

#### 6. Setup & Testing

**`scripts/setup.sh`**
- One-command installation
- Installs all dependencies
- Sets up .env file
- Checks Docker

**`scripts/test_installation.py`**
- Comprehensive installation verification
- Tests:
  - Package imports
  - Project modules
  - External dependencies
  - API keys
  - Docker
  - Configuration files
- Rich visual feedback

**`.env.example`**
- Template for API keys

**`README.md`**
- Complete documentation
- Installation instructions
- Usage examples
- Troubleshooting guide

### Project Structure

```
sdf-cot-monitorability/
├── configs/
│   ├── baseline_experiment.yaml      # Experiment configuration
│   └── example.yaml                  # Legacy example
├── sdf_cot_monitorability/
│   ├── __init__.py
│   ├── prompts.py                    # Monitoring condition prompts
│   ├── utils.py                      # Utilities (logging, I/O, etc.)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── impossiblebench_runner.py # ImpossibleBench integration
│   └── monitor/
│       ├── __init__.py
│       └── cot_monitor.py            # CoT monitoring & cheating detection
├── scripts/
│   ├── setup.sh                      # Installation script
│   ├── test_installation.py          # Installation verification
│   ├── run_baseline_experiment.py    # Main experiment runner
│   └── run_baseline_experiment.sh    # Convenience wrapper
├── results/                          # Experiment outputs (gitignored)
├── .env.example                      # API key template
├── README.md                         # Main documentation
├── pyproject.toml                    # Updated with dependencies
└── PHASE1_COMPLETE.md               # This file
```

## How to Use

### 1. Setup

```bash
# Run automated setup
./scripts/setup.sh

# Or manually:
pip install -e .
pip install -e external/impossiblebench
pip install -e external/false-facts

# Configure API keys
cp .env.example .env
# Edit .env with your keys
```

### 2. Verify Installation

```bash
python scripts/test_installation.py
```

This checks:
- All packages installed
- Project modules loadable
- External deps available
- API keys configured
- Docker running
- Config files present

### 3. Run Experiments

**Quick test (20 samples, ~30 min, ~$20-50):**

```bash
./scripts/run_baseline_experiment.sh
```

**Custom configuration:**

```bash
python scripts/run_baseline_experiment.py \
  --config configs/baseline_experiment.yaml \
  --conditions monitored unmonitored
```

**Skip monitoring (faster, for testing):**

```bash
python scripts/run_baseline_experiment.py \
  --config configs/baseline_experiment.yaml \
  --skip-monitoring
```

### 4. Analyze Results

Results are saved to `results/baseline/run_TIMESTAMP/`:

```
run_TIMESTAMP/
├── config.json                    # Experiment configuration
├── report.json                    # Summary with key metrics
├── evaluations/
│   ├── monitored_*.json          # ImpossibleBench results
│   ├── unmonitored_*.json
│   └── control_*.json
└── monitoring/
    ├── monitored_monitor_results.json    # Monitor analysis
    ├── unmonitored_monitor_results.json
    └── control_monitor_results.json
```

**Key metrics in report.json:**
- `cheating_rate`: % of tasks where model cheated
- `self_incrimination_rate`: % of CoT with explicit cheating mentions
- `obfuscation_rate`: % of CoT that appears to hide information
- `cot_detection_advantage`: How much better CoT monitoring is vs output-only

**Comparative analysis:**
- `cheating_rate_delta`: Difference between unmonitored and monitored
- `self_incrimination_delta`: Difference in self-incrimination

The script also prints a formatted table to the terminal.

## Expected Results

### Hypothesis

If prompting works:
- **Unmonitored** condition should have **higher cheating rate** than monitored
- **Unmonitored** should have **more self-incrimination** (model is honest about cheating)
- **Monitored** might show **more obfuscation** (hiding cheating behavior)

### Null Result

If prompting doesn't work:
- All conditions show similar cheating rates
- This would justify Phase 2 (SDF finetuning) as potentially more effective

### Strong Result

If prompting works strongly:
- Large gap between conditions (>20% difference in cheating rate)
- Might suggest SDF is unnecessary OR that we need to find conditions where prompting fails (e.g., Apollo-style adversarial prompts)

## Cost Estimates

Based on 20 samples per condition (60 total):

**Evaluation (ImpossibleBench):**
- GPT-4o-mini: ~$10-20
- Runtime: 20-40 minutes

**Monitoring (CoT analysis):**
- Claude Opus 4: ~$10-30
- Runtime: 10-20 minutes

**Total: ~$20-50, 30-60 minutes**

For full experiments (100+ samples):
- Cost: ~$100-250
- Runtime: 2-4 hours

## Troubleshooting

### Common Issues

1. **ImportError: impossiblebench not found**
   ```bash
   cd external/impossiblebench && pip install -e .
   ```

2. **Docker errors**
   ```bash
   # Check Docker is running
   docker ps
   
   # Start Docker Desktop if needed
   ```

3. **API rate limits**
   - Reduce `max_connections` in config
   - Add delays between requests

4. **Out of memory**
   - Reduce `limit` (number of samples)
   - Reduce `max_subprocesses`

5. **Missing API keys**
   ```bash
   # Check .env file exists and has keys
   cat .env
   ```

## Next Steps

### After Running Baseline Experiments

1. **Analyze results** - Does prompting create a meaningful effect?

2. **Decision point:**
   - **If prompting works well**: Consider Phase 2 (SDF) to see if it's stronger
   - **If prompting doesn't work**: Phase 2 becomes more critical
   - **If prompting works too well**: Find harder test conditions first

3. **Write up baseline results** - Document findings for comparison

### Phase 2 Preparation (If Proceeding)

Phase 2 will add:

1. **Universe context creation** (`sdf_cot_monitorability/sdf/universe_contexts.py`)
   - Define monitoring/unmonitored universe facts
   - Create plausible backstories

2. **Document generation** (`sdf_cot_monitorability/sdf/doc_generation.py`)
   - Generate synthetic documents with Claude
   - Multiple document types (news, blogs, papers, etc.)

3. **Finetuning pipeline** (`sdf_cot_monitorability/sdf/finetune.py`)
   - OpenAI finetuning API integration
   - Model versioning and tracking

4. **Belief evaluation** (`sdf_cot_monitorability/evaluation/belief_evals.py`)
   - Verify models internalized beliefs
   - MCQ, generative, and distinguishing tests

5. **Main experiments** (extend existing runner)
   - Run ImpossibleBench on finetuned models
   - Compare SDF vs prompting effect sizes

## Technical Notes

### Architecture Decisions

1. **Async-first design**: All evaluations use asyncio for efficiency
2. **Modular prompt injection**: Easy to swap prompting strategies
3. **Rich terminal UI**: Better UX during long experiments
4. **Comprehensive logging**: Easy debugging and result inspection
5. **Config-driven**: Change experiment params without code changes

### Code Quality

- ✓ Type hints throughout
- ✓ Docstrings for all public APIs
- ✓ No linting errors (Ruff, Pyright)
- ✓ Modular, testable design
- ✓ Clear separation of concerns

### Dependencies

All dependencies properly specified in `pyproject.toml`:
- inspect-ai (ImpossibleBench)
- openai (API)
- anthropic (monitoring judge)
- pandas, numpy (analysis)
- rich, tqdm (UI)
- fire (CLI)
- pydantic (data validation)

## Questions?

For issues or questions:
1. Check README.md for documentation
2. Run `test_installation.py` to diagnose problems
3. Review logs in `results/` directory
4. Check ImpossibleBench and false-facts documentation

## Conclusion

Phase 1 is complete and ready to use! You now have a fully functional baseline experiment pipeline to test whether monitoring beliefs (via prompting) affect model cheating behavior.

This establishes the foundation for Phase 2 (SDF experiments) and provides a strong comparison baseline.

Good luck with your experiments! 🚀

