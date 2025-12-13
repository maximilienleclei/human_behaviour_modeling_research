# cmaes.py

## Purpose
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer for neuroevolution. Adapts mean and diagonal covariance of search distribution based on fitness. **FULLY IMPLEMENTED** with diagonal approximation for efficiency.

## Contents

### `CMAESState` Class
Maintains CMA-ES algorithm state across generations with diagonal covariance approximation.

**State Variables:**
- `mean` [num_params]: Search center (weighted average of top performers)
- `sigma`: Global step size (adapted via cumulative step-size adaptation)
- `C_diag` [num_params]: Diagonal covariance (coordinate-wise variances)
- `p_c` [num_params]: Evolution path for covariance adaptation
- `p_sigma` [num_params]: Evolution path for step-size adaptation
- `generation`: Generation counter

**Learning Rates (standard CMA-ES):**
- `c_c`: Covariance path learning rate (4 / (n + 4))
- `c_1`: Rank-1 covariance update rate
- `c_mu`: Rank-mu covariance update rate
- `c_sigma`: Step-size learning rate
- `damps`: Damping for step-size adaptation
- `chi_n`: Expected length of random vector

**Diagonal Approximation:** Uses O(n) diagonal covariance instead of O(n²) full matrix for efficiency with neural network parameter spaces.

### `select_cmaes(population, fitness)`
CMA-ES selection function - core algorithm implementation.

**Algorithm Steps:**
1. **Weight Computation**: Top μ (50%) networks get rank-based weights (log-linear)
2. **Mean Update**: Weighted average of top performers → new search center
3. **Evolution Path Updates**:
   - Update `p_c` for covariance adaptation (normalized step)
   - Update `p_sigma` for step-size adaptation (unnormalized step)
4. **Step-Size Adaptation**: Adjust σ based on path length (cumulative step-size adaptation)
5. **Covariance Update**:
   - Rank-1 update from evolution path `p_c`
   - Rank-μ update from selected samples
6. **Sample Generation**: Draw new population from N(mean, σ²·C_diag)

**State Management:** CMA-ES state attached to population object as `_cmaes_state` and `_cmaes_samples`.

**Compatibility:** ONLY works with ParameterizableNetwork (BatchedFeedforward, BatchedRecurrent). Raises TypeError for DynamicNetPopulation.

### `optimize_cmaes(population, fitness_fn, test_fitness_fn, ...)`
High-level CMA-ES optimization wrapper.

**Integrates with base.optimize():**
- Uses shared training loop from ne/optim/base.py
- Passes `select_cmaes` as selection function
- Supports checkpointing with CMA-ES state persistence
- Supports state persistence config for recurrent networks

**Returns:** Dict with fitness_history, test_loss_history, final_generation

## CMA-ES vs GA/ES

**GA (Genetic Algorithm):**
- Hard selection (top 50%)
- No distribution adaptation
- Works with all network types

**ES (Evolution Strategy):**
- Soft selection (rank-weighted averaging)
- Fixed mutation strength
- Tensor networks only

**CMA-ES:**
- Rank-weighted selection + distribution adaptation
- Adaptive step-size and coordinate-wise variances
- Most sample-efficient, but tensor networks only
- Maintains and adapts search distribution across generations

## Checkpointing

CMA-ES state is fully supported in checkpointing via base.py:

**Saved State:**
- All CMAESState attributes (mean, sigma, C_diag, p_c, p_sigma, generation)
- Current samples (_cmaes_samples)

**Restoration:** Creates CMAESState object and restores all parameters on checkpoint resume.

## Usage Example

```python
from ne.net.feedforward import BatchedFeedforward
from ne.pop.population import Population
from ne.eval.supervised import train_supervised

# Create network and population
nets = BatchedFeedforward([10, 64, 32, 4], num_nets=100)
population = Population(nets, action_mode="softmax")

# Train with CMA-ES
result = train_supervised(
    population=population,
    train_data=(train_obs, train_actions),
    test_data=(test_obs, test_actions),
    optimizer="cmaes",  # Now fully functional!
    max_time=3600,
)
```

## Implementation Notes

- Diagonal approximation trades some adaptation power for O(n) efficiency vs O(n²) full CMA-ES
- Initial sigma = 0.5 provides good starting exploration
- Minimum variance clamped to 1e-10 for numerical stability
- Uses standard CMA-ES learning rate formulas
- Evolution paths enable efficient covariance learning with few samples
