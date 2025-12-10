# Missing Optimizers from Experiments 2 & 3

## Discovery

Experiment 2 (`2_dl_vs_ga_es`) has these training functions:
- `train_deep_learning()` at line 669 ✅ (extracted as optimize_sgd)
- `train_neuroevolution()` at line 1180 ❓ (need to check if this is GA, ES, or both)

## Questions to Answer Tomorrow

1. **What is `train_neuroevolution()`?**
   - Is it GA (Genetic Algorithm)?
   - Is it ES (Evolution Strategies)?  
   - Does it implement both?
   - Source: `experiments/2_dl_vs_ga_es/main.py` line 1180

2. **Does exp 2 have separate GA and ES?**
   - Experiment name is "dl_vs_ga_es" suggesting 3 algorithms
   - But only 2 training functions found
   - Possible: ES is just another name for GA in this context
   - Possible: train_neuroevolution implements both variants

3. **What's different from exp 4's GA?**
   - Exp 4 has neuroevolve_recurrent/dynamic
   - Exp 2 has train_neuroevolution
   - Need to compare implementations

## Action Plan Tomorrow

### Step 1: Read and Compare (30 min)
```bash
# Read exp 2 neuroevolution implementation
less +1180 experiments/2_dl_vs_ga_es/main.py

# Compare with exp 4 GA
less experiments/4_add_recurrence/src/optim.py
```

### Step 2: Check for ES-specific features
Look for:
- CMA-ES (Covariance Matrix Adaptation)
- Natural gradients
- Rank-mu updates
- Different sigma adaptation than exp 4

### Step 3: Decide on Extraction
If ES is different from GA:
- Extract to `platform/optimizers/es.py`
- Add `--optimizer es` to runner.py
- Update documentation

If ES = GA with different name:
- Just document the naming convention
- No new code needed

## What We Know

**Current Platform:**
- ✅ SGD (from exp 4, works for all models)
- ✅ GA for recurrent (from exp 4, BatchedRecurrentPopulation)
- ❌ GA/ES for feedforward (BatchedPopulation not extracted)
- ❓ ES (if different from GA) - need to investigate

**Exp 2 Focus:**
- Feedforward models only (no recurrence)
- HuggingFace datasets
- Comparing DL vs Neuroevolution
- "ga_es" in name suggests evolution strategies

**Exp 4 Focus:**
- Recurrent models
- Human behavioral data
- GA with adaptive sigma

## Likely Scenario

**Most Probable:** 
- Exp 2's "train_neuroevolution" = GA for feedforward (BatchedPopulation)
- "ES" in experiment name just means "evolution strategies" as umbrella term
- No separate ES algorithm

**To Verify:**
- Check if train_neuroevolution uses CMA-ES or just simple GA
- Check if there are ES-specific parameters

## Priority

**High Priority:**
- Extract BatchedPopulation for feedforward (definitely missing)

**Medium Priority:**  
- Check if ES is truly separate algorithm
- Extract if different from GA

**Files to Check Tomorrow:**
1. `experiments/2_dl_vs_ga_es/main.py` lines 1180+ (train_neuroevolution)
2. `experiments/2_dl_vs_ga_es/main.py` lines 604-666 (BatchedPopulation class likely here)
3. Compare with `experiments/4_add_recurrence/src/models.py` BatchedRecurrentPopulation
