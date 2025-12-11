# Trust-graph rule based propagation implementation. Focus on on robustness under attack and under scaling.

## Highlights
- Strategies: Trust, Trust‑Graph, PID family (pid, pid_scaled, pid_standardized), Krum, Multi‑Krum, Bulyan, Trimmed Mean, RFA (geometric median).
- Ablations: Trust‑Graph parameters (alpha, K, edge_rule).
- Scale: Ready for 20+ clients (strict_mode + min_*), CSV+PDF outputs per run.
- Robust thresholds (optional): dynamic trust threshold (`tau_quantile`), PID thresholds (`mad`/`quantile`).

## Repository Layout
- `src/` core code:
  - `federated_simulation.py` Orchestrates dataset/model/strategy and Flower simulation
  - `simulation_strategies/` Strategy implementations
  - `output_handlers/` Plots and artifact management
  - `config_loaders/` Config JSON merge/validation
  - `data_models/` Per‑client/round history for plotting and CSV
- `config/` configuration files:
  - `simulation_strategies/examples/` example configs (ablations/baselines)


## Quickstart
1) From repo root: `./run_simulation.sh <config_path>`
   - Examples:
     - `sh ./run_simulation.sh examples/rfa_config.json`
     - `sh ./run_simulation.sh examples/trimmed_mean_config.json`
     - `sh ./run_simulation.sh examples/krum_20_config.json`
     - `sh ./run_simulation.sh examples/trust_graph_ablation_alpha.json`
     - `sh ./run_simulation.sh examples/trust_graph_ablation_K.json`
     - `sh ./run_simulation.sh examples/trust_graph_ablation_edge_rule.json`
   - If no arg is provided, defaults to `example_strategy_config.json`.
   - Alternative: `USECASE_CONFIG=examples/rfa_config.json ./run_simulation.sh`
2) Outputs: `out/<MM-DD-YYYY_HH-MM-SS>/`
   - Plots: `{metric_name}_{strategy_number}.pdf` (strategy_number = index in the config’s `simulation_strategies` array)
   - CSV: per‑client, per‑round, and per‑execution summaries
   - Saved config: `strategy_config_{strategy_number}.json`

## Running at Scale (20+ clients)
- In config `shared_settings`:
  - `strict_mode: "true"` and set `num_of_clients: 20` (or more)
  - Set `min_fit_clients`, `min_evaluate_clients`, `min_available_clients` to `num_of_clients` (or rely on `strict_mode` to enforce)
  - Resources: `training_device: "cpu"`, `cpus_per_client: 1`, `gpus_per_client: 0.0`
  - Example configs already prepared in `examples/`

## Strategies & Key Parameters
- `trust_graph`: `alpha`, `K`, `tau`, `edge_rule` (`similarity`|`cosine`), `neighbor_cap`, `graph_static`, `convergence_eps`
  - Optional dynamic threshold: `tau_quantile` (e.g., 0.2 → 20th percentile of trust)
- `pid`, `pid_scaled`, `pid_standardized`: `Kp`, `Ki`, `Kd`, `num_std_dev`
  - Optional robust thresholding for PID family:
    - `pid_threshold_method`: `mad` | `quantile` | `zscore` (default)
    - `pid_threshold_quantile`: e.g., `0.9`
- `krum`, `multi-krum`, `bulyan`: `num_krum_selections`
- `trimmed_mean`: `trim_ratio`
- `rfa`: optional `weighted_median_factor` (default 1.0)

See `docs/CONFIGS.md` for more examples and schema notes.

## Plots & Metrics
- Per‑client (lines):
  - `removal_criterion_history` Score used for removal/selection
  - `absolute_distance_history` Distance to cluster center
  - `loss_history`, `accuracy_history` Client validation metrics
  - Exclusions marked with `X`
- Across strategies (lines/bars):
  - `score_calculation_time_nanos_history` Server‑side time to compute selection/weights 
  - `aggregated_loss_history`, `average_accuracy_history`
  - Removal stats: accuracy/precision/recall/f1; total FP+FN
