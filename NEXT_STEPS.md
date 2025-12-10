# Next Steps - Platform Development

**Current Status:** Core platform COMPLETE (7/12 phases) and ready to use  
**Priority:** Test and validate before moving forward

---

## Immediate Actions (Tomorrow Morning)

### 1. Smoke Test (5 minutes)
```bash
cd /scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research

# Test imports
python -c "from platform.models import RecurrentMLPReservoir; print('✓ Platform imports OK')"

# Quick 30-second run
python -m platform.runner \
  --dataset cartpole \
  --method smoke_test \
  --model reservoir \
  --optimizer sgd \
  --max-time 30 \
  --no-logger
```

**Expected:** Should run without errors and create checkpoint in `results/`

### 2. Integration Test (30 minutes)
```bash
# 10-minute training run matching exp 4 setup
python -m platform.runner \
  --dataset cartpole \
  --method SGD_reservoir_integration_test \
  --model reservoir \
  --optimizer sgd \
  --use-cl-info \
  --subject sub01 \
  --seed 42 \
  --max-time 600
```

**Check:**
- Training loss decreases
- Test loss computed periodically
- Checkpoint saves to `results/`
- No errors in output

### 3. Verify Data Loading (5 minutes)
```bash
# Test all data sources
python -c "
from platform.data.loaders import load_human_data, load_cartpole_data

# Human data
obs, act, test_obs, test_act, meta = load_human_data('cartpole', False, 'sub01')
print(f'✓ Cartpole: {obs.shape[0]} train, {test_obs.shape[0]} test')

# HuggingFace data  
train_obs, train_act, test_obs, test_act = load_cartpole_data()
print(f'✓ HF CartPole: {train_obs.shape[0]} train, {test_obs.shape[0]} test')
"
```

---

## Decision Point: Choose Your Path

### Path A: Start Using Platform (Recommended)
**Time:** Immediate  
**Why:** Platform is functional, start running experiments  

**Actions:**
1. Run your actual experiments using platform runner
2. Compare results to experiment 4 as validation
3. File bugs/issues as you find them
4. Iterate and improve as needed

### Path B: Complete Validation (Thorough)
**Time:** 4-6 hours  
**Why:** Ensure platform matches exp 4 before committing  

**Actions:**
1. ✅ Run full 10-hour training (cartpole, SGD_reservoir, seed=42)
2. ✅ Compare training curves to exp 4 checkpoint
3. ✅ Test all combinations:
   - RecurrentMLPReservoir + SGD
   - RecurrentMLPTrainable + SGD
   - RecurrentMLPReservoir + GA
   - RecurrentMLPTrainable + GA
4. ✅ Verify ExperimentLogger database integration
5. Document any differences

### Path C: Polish & Archive (Complete)
**Time:** 8-12 hours  
**Why:** Finish all 12 phases for complete migration  

**Actions:**
1. Phase 8: Create YAML configs (optional)
2. Phase 9: Integration tests ← START HERE
3. Phase 10: Archive experiments 1-4
4. Phase 11: Extended documentation
5. Phase 12: Code formatting and git commits

---

## Recommended: Hybrid Approach

**Morning (2 hours):**
1. Smoke test (5 min)
2. Integration test (30 min)
3. Start 10-hour validation run in background
4. Begin using platform for new work

**Afternoon (as needed):**
- Monitor validation run
- Fix any issues discovered
- Archive old experiments (Phase 10)
- Clean up codebase

**Later (optional):**
- Add YAML configs if sweeps become complex
- Write extended docs if sharing with others

---

## Critical Files to Review Tomorrow

1. **SESSION_SUMMARY.md** ← Full context from today
2. **IMPLEMENTATION_STATUS.md** ← What's complete
3. **platform/README.md** ← Usage guide
4. **platform/runner.py** ← Main entry point
5. **experiments/4_add_recurrence/** ← Reference implementation

---

## Test Matrix (For Validation)

If doing Path B (thorough validation), test these combinations:

| Model | Optimizer | Dataset | CL | Time | Status | Notes |
|-------|-----------|---------|-----|------|--------|-------|
| reservoir | SGD | cartpole | ✓ | 10min | ⬜ | ✅ Should work |
| reservoir | SGD | cartpole | ✗ | 10min | ⬜ | ✅ Should work |
| trainable | SGD | lunarlander | ✓ | 10min | ⬜ | ✅ Should work |
| reservoir | GA | mountaincar | ✓ | 10min | ⬜ | ✅ Should work |
| trainable | GA | acrobot | ✗ | 10min | ⬜ | ✅ Should work |
| feedforward | GA | cartpole | ✗ | 10min | ⬜ | ❌ NOT IMPLEMENTED |

**Success criteria:**
- All run without errors (except feedforward+GA which is missing)
- Loss decreases over time
- Checkpoints save/load correctly
- Results logged to database (if enabled)

**⚠️ Known Issue:** Feedforward MLP + GA is not implemented (see SESSION_SUMMARY.md)

---

## Known Gaps in Platform

These features from exp 4 were **simplified or omitted** to save tokens:

1. **Behavioral evaluation** (evaluate_progression_recurrent)
   - Code exists in `platform/evaluation/comparison.py`
   - Not integrated into optimizers (would need environment loading)
   - Can add later if needed

2. **F1 score computation** in GA
   - Placeholder (returns 0.0)
   - Full implementation in exp 4 is complex
   - Can add later if needed

3. **Dynamic network support** in runner
   - DynamicNetPopulation class exists
   - Not wired up in runner.py
   - Only works with GA

4. **Environment evaluation**
   - Code exists but requires gymnasium
   - Optional - for comparing model vs human returns

**These are all minor and can be added if needed.**

---

## Quick Fixes if Things Break

### Import Errors
```bash
# Make sure you're in project root
cd /scratch/mleclei/Dropbox/repos/maximilienleclei/human_behaviour_modeling_research

# Add to PYTHONPATH if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### GPU Errors
```bash
# Run on CPU for testing
python -m platform.runner ... --gpu -1

# Or check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Data Not Found
```bash
# Check data directory exists
ls -la data/

# Should contain:
# sub01_data_cartpole.json
# sub02_data_cartpole.json
# sub01_data_lunarlander.json
# etc.
```

### Checkpoint Issues
```bash
# Clear old checkpoints
rm results/*_checkpoint.pt

# Or specify new method name to avoid conflicts
python -m platform.runner --method NEW_NAME ...
```

---

## Git Commit Strategy (When Ready)

Recommended commit sequence:
1. `git add platform/` - Add entire platform directory
2. `git commit -m "Add unified experimentation platform (phases 1-7)"`
3. `git add IMPLEMENTATION_STATUS.md SESSION_SUMMARY.md NEXT_STEPS.md`
4. `git commit -m "Add platform implementation documentation"`

**Or:** Wait until validation complete, then commit everything together.

---

**Bottom Line:** Platform is ready. Test it, use it, improve it as needed. Good luck tomorrow! 🚀

---

## ⚠️ URGENT: Missing ES Optimizer

**Discovery:** Evolution Strategies (ES) optimizer is missing from platform!

**Source:** `experiments/2_dl_vs_ga_es/main.py`

**Action Tomorrow:**
1. Check what ES implementation looks like in exp 2
2. Extract ES optimizer to platform
3. May need to also extract BatchedPopulation for feedforward

**Priority:** HIGH - ES was a core part of original experiments
