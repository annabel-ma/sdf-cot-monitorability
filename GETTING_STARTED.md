# Getting Started with SDF CoT Monitorability

## Quick Start (5 minutes)

1. **Install dependencies:**
   ```./scripts/setup.sh
   ```

2. **Add your API keys:**  
   Required keys:
   - `OPENAI_API_KEY` - for running models
   - `ANTHROPIC_API_KEY` - for monitoring/judging

3. **Start Docker:**
   ```bash
   docker ps
   ```

4. **Test your setup:**
   ```bash
   python scripts/test_installation.py
   ```
   
   This will check that everything is properly configured.

5. **Run your first experiment:**
   ```bash
   ./scripts/run_baseline_experiment.sh
   ```
   
   This will run a small baseline experiment (~30 min, ~$20-50).

## What's Happening?

The baseline experiment:

1. **Tests 3 conditions:**
   - 🔍 **Monitored**: Model believes its reasoning is watched
   - 🙈 **Unmonitored**: Model believes its reasoning is private  
   - ⚖️ **Control**: No information about monitoring

2. **Runs ImpossibleBench:**
   - Coding tasks that are *impossible* without cheating
   - Model must either fail or use shortcuts
   - Measures how often model cheats

3. **Analyzes with Monitor:**
   - LLM judge reads the reasoning traces
   - Detects cheating behavior
   - Measures self-incrimination and obfuscation

4. **Compares results:**
   - Do unmonitored models cheat more?
   - Are they more honest about cheating?
   - Is there a difference from control?

## Understanding Results

After the experiment completes, you'll see:

```
results/baseline/run_TIMESTAMP/
├── report.json          # Key metrics and comparisons
├── evaluations/         # Raw ImpossibleBench results
└── monitoring/          # Monitor analysis
```

### Key Metrics

**Cheating Rate**: What % of tasks did the model solve via shortcuts?
- Higher rate = more cheating

**Self-Incrimination**: How often does the model explicitly mention cheating in its reasoning?
- Higher rate = more honest/transparent

**Obfuscation**: How often does reasoning appear to hide information?
- Higher rate = less transparent

**CoT Detection Advantage**: How much better is monitoring with reasoning vs just outputs?
- Higher = reasoning traces help catch cheating

### What You're Looking For

**✅ Strong result**: Large difference between conditions (e.g., unmonitored cheats 30% more)
→ Prompting alone affects behavior significantly

**❓ Null result**: No difference between conditions
→ Prompting doesn't work; SDF finetuning might be needed

**🤔 Intermediate result**: Small but consistent difference
→ Effect exists but is weak; SDF might amplify it

## Customizing Your Experiment

Edit `configs/baseline_experiment.yaml`:

### Change sample size:
```yaml
impossiblebench:
  limit: 10  # Quick test (10 min, $5-10)
  # limit: 20  # Pilot (30 min, $20-50)
  # limit: 100  # Full run (2-4 hr, $100-250)
```

### Test specific conditions:
```yaml
conditions:
  - "monitored"
  - "unmonitored"
  # - "control"  # Comment out to skip
```

### Change the model:
```yaml
models:
  primary: "openai/gpt-4o-mini"  # Cheaper, recommended
  # primary: "openai/gpt-4o"  # More capable, more expensive
```

### Use a different judge:
```yaml
monitor:
  judge_model: "claude-opus-4-20250514"  # Best accuracy
  # judge_model: "gpt-4o"  # Cheaper alternative
```

## Command Line Options

Run specific conditions only:
```bash
python scripts/run_baseline_experiment.py \
  --config configs/baseline_experiment.yaml \
  --conditions monitored unmonitored
```

Skip monitoring (faster, for testing evaluation only):
```bash
python scripts/run_baseline_experiment.py \
  --config configs/baseline_experiment.yaml \
  --skip-monitoring
```

Verbose logging:
```bash
python scripts/run_baseline_experiment.py \
  --config configs/baseline_experiment.yaml \
  --log-level DEBUG
```

## Troubleshooting

### "Docker is not running"
Start Docker Desktop, then run again.

### "API key not found"
Check your `.env` file has:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### "ImportError: impossiblebench"
Install external dependencies:
```bash
pip install -e external/impossiblebench
pip install -e external/false-facts
```

### Rate limits / slow execution
Reduce `max_connections` in config:
```yaml
impossiblebench:
  max_connections: 5  # Lower for rate-limited accounts
```

### Out of memory
Reduce sample size or subprocess count:
```yaml
impossiblebench:
  limit: 10
  max_subprocesses: 5
```

## Next Steps

After running baseline experiments:

1. **Review the results** - Check `report.json` for key metrics

2. **Analyze the data** - Look at individual samples in `evaluations/` and `monitoring/`

3. **Decide on Phase 2**:
   - If prompting doesn't work → SDF is essential
   - If prompting works → SDF might be stronger
   - If prompting works too well → Need harder test conditions

4. **Scale up** - Increase sample size for more robust results

5. **Try variations** - Different prompts, models, or tasks

## Get Help

- **Installation issues**: Run `python scripts/test_installation.py`
- **Documentation**: See `README.md` and `PHASE1_COMPLETE.md`
- **Code reference**: All modules have detailed docstrings
- **ImpossibleBench docs**: `external/impossiblebench/README.md`

## Research Notes

Remember:
- This is Phase 1: **Baseline prompting** experiments
- Purpose: Establish whether prompting alone creates the effect
- Phase 2 will add **SDF finetuning** if needed

The key research question:
> Does a model's belief about being monitored change its propensity to cheat and be honest about it?

Good luck with your research! 🎓

