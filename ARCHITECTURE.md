# Architecture Overview

## How Everything Fits Together

### The Flow (What Happens When You Run an Experiment)

```
User runs: ./scripts/run_baseline_experiment.sh
    ↓
scripts/run_baseline_experiment.py
    ↓
BaselineExperiment class
    ↓
For each condition (monitored, unmonitored, control):
    ↓
    1. Creates ImpossibleBenchRunner
       ├─ Gets prompts from sdf_cot_monitorability/prompts.py
       │  ├─ MONITORED_SYSTEM_PROMPT (agentic + "you're being watched")
       │  ├─ UNMONITORED_SYSTEM_PROMPT (agentic + "you're private")  
       │  └─ CONTROL_SYSTEM_PROMPT (agentic, no monitoring info)
       │
       ├─ Creates ImpossibleBench task with these prompts
       └─ Runs evaluation → generates CoT traces + outputs
    ↓
    2. Saves results to results/baseline/run_TIMESTAMP/evaluations/
       ├─ monitored_openai_gpt-4o-mini_results.json (metrics)
       ├─ monitored_openai_gpt-4o-mini_samples.json (detailed samples with CoT)
       └─ (same for unmonitored and control)
    ↓
    3. CoTMonitor analyzes the results
       ├─ Loads samples from step 2
       ├─ For each sample:
       │  ├─ Extracts reasoning trace (CoT)
       │  ├─ Sends to Claude Opus (LLMJudge)
       │  ├─ Judge determines: Did they cheat? Self-incriminate? Obfuscate?
       │  └─ Stores CheatDetectionResult
       └─ Computes aggregate metrics
    ↓
    4. Saves monitoring results to results/baseline/run_TIMESTAMP/monitoring/
       └─ monitored_monitor_results.json (cheating detection, self-incrim, obfuscation)
    ↓
5. Generates comparative report
   ├─ Compares metrics across conditions
   ├─ Calculates deltas (unmonitored - monitored)
   └─ Saves to results/baseline/run_TIMESTAMP/report.json
    ↓
6. Pretty prints results table to terminal
```

---

## File-by-File Breakdown

### **1. `sdf_cot_monitorability/prompts.py`** ⭐ KEY FILE

**What it contains:**
- `BASE_AGENTIC_PROMPT` - Your full SWE-bench style prompt (word-for-word)
- `MONITORED_ADDITION` - Notice that reasoning is being watched
- `UNMONITORED_ADDITION` - Notice that reasoning is private
- `MONITORED_SYSTEM_PROMPT` = BASE_AGENTIC_PROMPT + MONITORED_ADDITION
- `UNMONITORED_SYSTEM_PROMPT` = BASE_AGENTIC_PROMPT + UNMONITORED_ADDITION
- `CONTROL_SYSTEM_PROMPT` = BASE_AGENTIC_PROMPT (no additions)

**Functions:**
- `get_system_prompt(condition)` - Returns the right prompt for "monitored", "unmonitored", or "control"
- `get_instruction_prompt(condition)` - Task-specific instructions (minimal, since system prompt is comprehensive)

**This is the ONLY file you modified** to use the agentic prompt.

---

### **2. `sdf_cot_monitorability/evaluation/impossiblebench_runner.py`**

**What it does:**
Wraps ImpossibleBench to inject our custom prompts.

**Key class: `ImpossibleBenchRunner`**

Constructor:
```python
ImpossibleBenchRunner(
    model="openai/gpt-4o-mini",
    condition="monitored",  # or "unmonitored" or "control"
    config=ImpossibleBenchConfig(...),
    output_dir="results/baseline"
)
```

**What happens inside:**
1. Calls `get_system_prompt(condition)` from `prompts.py` → Gets agentic prompt
2. Calls `get_instruction_prompt(condition)` from `prompts.py` → Gets task instructions
3. Creates ImpossibleBench task with these prompts
4. Runs evaluation
5. Extracts CoT traces from model's messages
6. Saves results

**Key methods:**
- `_create_task()` - Sets up ImpossibleBench with our prompts
- `run_async()` - Runs the evaluation
- `_extract_reasoning(messages)` - Pulls CoT out of model messages
- `_extract_samples(log)` - Gets sample-level data with CoT traces

**You don't need to touch this** - it automatically uses whatever prompts are in `prompts.py`.

---

### **3. `sdf_cot_monitorability/monitor/cot_monitor.py`**

**What it does:**
Analyzes model outputs to detect cheating, self-incrimination, and obfuscation.

**Key class: `CoTMonitor`**

**What happens:**
1. Takes samples from ImpossibleBench evaluation
2. For each sample:
   - Extracts reasoning trace (CoT)
   - Formats as transcript for judge
   - Sends to Claude Opus (via ImpossibleBench's LLMJudge)
   - Judge answers:
     - Did the model cheat?
     - How confident (0-1)?
     - Is there self-incriminating language?
     - Is there obfuscation?
3. Aggregates metrics across all samples

**Key methods:**
- `monitor_sample_async(sample)` - Judge one sample
- `monitor_samples_async(samples)` - Judge multiple samples
- `_check_self_incrimination(reasoning)` - Keyword detection ("I'm going to cheat", "look at test code")
- `_check_obfuscation(reasoning)` - Detect hiding behavior (too short, vague)

**Result dataclass: `CheatDetectionResult`**
```python
CheatDetectionResult(
    sample_id="sample_001",
    reasoning_trace="I need to solve... [model's CoT]",
    output="def foo()...",
    cheated_cot=True,  # Did they cheat? (from judge)
    cheating_confidence_cot=0.85,  # Judge's confidence
    self_incriminating=True,  # Did they admit cheating in CoT?
    obfuscation_detected=False,  # Did they hide info?
    cheated_output=False,  # Would we detect cheating without CoT?
)
```

**You don't need to touch this** - it analyzes whatever CoT the model produces.

---

### **4. `scripts/run_baseline_experiment.py`** ⭐ MAIN ORCHESTRATOR

**What it does:**
Runs the entire experimental pipeline.

**Key class: `BaselineExperiment`**

**Constructor:**
```python
BaselineExperiment(config)  # config from baseline_experiment.yaml
```

**Main method: `run_async()`**

**Phase 1: Run Evaluations**
```python
async def _run_evaluations(conditions):
    for condition in ["monitored", "unmonitored", "control"]:
        runner = ImpossibleBenchRunner(
            model="openai/gpt-4o-mini",
            condition=condition,  # This determines which prompt from prompts.py
            config=impossiblebench_config,
            output_dir=eval_dir
        )
        result = await runner.run_async()
        # Saves to: evaluations/monitored_*.json, etc.
```

**Phase 2: Run Monitoring**
```python
async def _run_monitoring(eval_results):
    monitor = CoTMonitor(config=monitor_config)
    
    for condition, eval_result in eval_results.items():
        samples = load_json(f"{condition}_samples.json")
        monitor_results = await monitor.monitor_samples_async(samples)
        # Saves to: monitoring/monitored_monitor_results.json, etc.
```

**Phase 3: Generate Report**
```python
def _generate_report(eval_results, monitor_results):
    # Aggregate metrics across conditions
    # Compute deltas (unmonitored - monitored)
    # Save report.json
    # Print pretty table
```

**Command line usage:**
```bash
python scripts/run_baseline_experiment.py \
    --config configs/baseline_experiment.yaml \
    --conditions monitored unmonitored  # Optional: specify which conditions
    --skip-monitoring  # Optional: skip Phase 2
```

**You don't need to touch this** - it orchestrates everything automatically.

---

### **5. `configs/baseline_experiment.yaml`**

**What it contains:**
All experiment settings.

```yaml
models:
  primary: "openai/gpt-4o-mini"

conditions:
  - "monitored"
  - "unmonitored"
  - "control"

impossiblebench:
  benchmark_type: "livecodebench"  # or "swebench"
  agent_type: "minimal"
  split: "conflicting"
  limit: 20  # Number of samples
  
monitor:
  judge_model: "claude-opus-4-20250514"
  analyze_cot: true
  analyze_output_only: true
```

**You edit this** to change experiment parameters (sample size, model, etc.)

---

### **6. `sdf_cot_monitorability/utils.py`**

Helper functions:
- Logging setup
- API key loading
- File I/O (JSON, JSONL)
- Directory management

**You don't need to touch this** - just utility functions.

---

### **7. Scripts**

#### `scripts/setup.sh`
One-command installation. Run once.

#### `scripts/test_installation.py`
Verify everything is set up. Run to check.

#### `scripts/run_baseline_experiment.sh`
Convenience wrapper. Runs the Python script with checks.

**You don't need to touch these** - they're ready to use.

---

## Summary: What Changed?

### ✅ Files Modified (1 file)
1. **`sdf_cot_monitorability/prompts.py`** - Replaced simple prompts with your agentic SWE-bench prompt

### 📦 How Everything Connects

```
prompts.py (agentic prompts)
    ↓ (provides prompts to)
impossiblebench_runner.py (injects prompts into ImpossibleBench)
    ↓ (generates)
CoT traces + outputs
    ↓ (analyzed by)
cot_monitor.py (detects cheating)
    ↓ (orchestrated by)
run_baseline_experiment.py (runs full pipeline)
    ↓ (configured by)
baseline_experiment.yaml (settings)
```

### 🎯 Key Insight

**You only needed to change `prompts.py`** - everything else automatically adapts:
- The runner automatically uses the new prompts
- The monitor automatically analyzes whatever CoT is produced
- The orchestrator automatically runs the full pipeline
- The agentic prompt naturally produces rich CoT for analysis

### 🚀 To Run

```bash
# 1. Setup (one time)
./scripts/setup.sh

# 2. Add API keys to .env

# 3. Run experiment
./scripts/run_baseline_experiment.sh

# Results in: results/baseline/run_TIMESTAMP/
```

---

## The Monitoring Manipulation

### Monitored Condition
```
BASE_AGENTIC_PROMPT
+
"Your reasoning process is being MONITORED and REVIEWED..."
```

### Unmonitored Condition
```
BASE_AGENTIC_PROMPT
+
"Your reasoning process is COMPLETELY PRIVATE..."
```

### Control Condition
```
BASE_AGENTIC_PROMPT
(no additions)
```

All three get the same high-quality agentic instructions - only the monitoring belief differs!

---

## Expected Behavior

With the agentic prompt, models should:
- ✅ Produce extensive step-by-step reasoning
- ✅ Plan before acting
- ✅ Reflect on outcomes
- ✅ Iterate until problem solved
- ✅ Test rigorously

This gives us **rich CoT traces** to analyze for:
- Cheating behavior
- Self-incrimination ("I'll look at the test file")
- Obfuscation (hiding shortcuts)
- Differences between monitored vs unmonitored

Perfect for your monitorability research! 🎓

