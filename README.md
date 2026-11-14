# Enhancing Resilience in Complex Supply Chain Systems: Adaptive Network Reconfiguration through Risk Perception

Agent-based simulation code for the paper *Enhancing Resilience in Complex Supply Chain Systems: Adaptive Network Reconfiguration through Risk Perception*.

The model couples a **directed supply-chain network** with an **undirected competition network** of listed firms (Compustat CUSIP identifiers). Firms are heterogeneous in size and in *risk perception* (the intensity with which they disclose supply-chain risk). A disruption propagates through adaptive removal probabilities, and four reconfiguration strategies are benchmarked against a no-intervention baseline by their ability to reduce a focal firm's average removal probability (ARP).

---

## Repository layout

```
.
├── main.py                 # entry point: runs the year × combo × tau × kcore × alpha grid
├── config.py               # all parameters, data paths, presets, parameter combos
├── data_loader.py          # reads the four input tables, builds size / risk mappings
├── network_builder.py      # builds Gs (supply) and Gp (competition), attaches attributes
├── strategy.py             # removal-probability model, reactive baseline, S1 / S4 / S5 / S6
├── experiment.py           # ExperimentRunner: N simulations, aggregation, result formatting
├── regression_analysis.py  # node-level vulnerability + panel regression with clustered SE
├── utils.py                # seeding, IO helpers, degree ordering
└── data/                   # yearly input panels (see "Data")
```

## Requirements

Python 3.9+ and:

```bash
pip install numpy pandas networkx tqdm statsmodels openpyxl
```

`openpyxl` is required because every input table is `.xlsx`. `statsmodels` powers `regression_analysis.py`.

## Data

Four tables are read for each focal year. The expected columns are:

- **Supply-chain relations** (`供应链关系{YY}.xlsx`) — `SOURCE_cusip`, `TARGET_cusip`, supplier → customer directed edges.
- **Competition / similarity** (`相似度{YY}.xlsx`) — `SOURCE_cusip`, `TARGET_cusip`, `Similarity`; edge weight of the undirected competition network.
- **Firm size** (`企业规模.xlsx`) — `cusip`, `Revenue`; one sheet per year, named `"15"` … `"22"`.
- **Risk perception** (`披露比例{YY}.xlsx`) — `CUSIP`, `特征单元格数量`; used as the firm's risk-perception intensity.

**Before running, edit the four path constants at the top of `config.py`.** They currently point at an absolute Windows path from the author's machine:

```python
BASE_DIR            = Path(".../relationship及披露")
FIRM_SIZE_DIR       = Path(".../结果和数据")
RISK_DIR            = Path(".../risk factor")
PROJECT_OUTPUT_DIR  = Path(".../ABM")
```

`data/` in this repository holds the yearly panels as distributed with the paper; the firm-size workbook must be supplied separately and named as above. Years available are `15`–`22`, matching `ALL_YEARS` in `config.py`.

## Quick start

Verify the pipeline end-to-end with the cheap preset first (50 simulations instead of 1000):

```bash
python main.py --preset experiment --combos 0.1_N_N
```

Run a specific combination across all years at full scale:

```bash
python main.py --preset full --combos 0.1_N_N,0.1_N_U
```

Run the whole grid (default: 8 combinations × 2 values of `tau` × 8 years = 128 tasks, 1000 simulations each):

```bash
python main.py --preset default
```

`--preset` chooses the simulation budget: `default` (1000 simulations, 100 iterations), `experiment` (50/50), `full` (500/200), `sensitivity` (100, with a wider noise term). `--combos` filters `PARAM_COMBOS` by name; omit it to run all of them.

Runs are parallelised over `multiprocessing.Pool` with 8 workers. Each task re-seeds the global RNG from `ExperimentConfig.random_seed`, so results are deterministic per task and do not depend on scheduling order.

## What you can tune

Everything lives in `config.py`; the `RUN_*` switches decide how many dimensions the grid expands along.

- **Years** — `RUN_ALL_YEARS` / `CURRENT_YEAR` / `ALL_YEARS`.
- **Parameter combinations** — `RUN_ALL_COMBOS` / `CURRENT_COMBO` / `PARAM_COMBOS`. The eight combinations cross three choices: `k_dist` (partner-preference `K ~ N(1.5, 0.1)` vs `N(1.5, 0.2)`), `uk_dist` (`tiered_normal` by size quantile vs `size_uniform`), and `removal_normal` (truncated normal vs uniform removal probability). Names follow `{k_std}_{uk}_{removal}`, e.g. `0.1_N_N`.
- **Smoothing** — `RUN_TAU_RANGE` / `TAU_VALUES` (default `[0.1, 0.2]`); `tau` is the temperature of the smooth removal function.
- **Structural constraint** — `RUN_KCORE_RANGE` / `KCORE_RANGE`. Note the master switch `KCoreConstraintConfig.enabled` defaults to `False`, in which case `KCORE_RANGE` is ignored and the `kcore` dimension collapses to a single "off" run. Set `enabled = True` to actually sweep `k = 4`.
- **Risk threshold** — `RUN_ALPHA_RANGE` / `ALPHA_VALUES` (percentile of `RiskPerception`, `80.0`–`100.0` step `0.5`); `CURRENT_ALPHA` is used when the sweep is off.

## Strategies

All strategies are compared on the same ARP metric for a focal firm.

- **Baseline (no intervention)** — the reactive simulation alone: disruption propagates until it dies out or hits `max_iterations`.
- **S1 — second-tier bypass.** For each focal firm, identify the risky second-tier suppliers reached through its first-tier suppliers, then add a direct edge from a competitor of that risky supplier, bypassing the intermediate tier.
- **S4 — supplier replacement.** For each firm, pick the highest-risk first-tier supplier above the `alpha` percentile and replace it with a lower-risk competitor of that supplier.
- **S5 — customer replacement.** The downstream mirror of S4.
- **S6 — joint reconfiguration.** Applies S4 and S5 together within one reconfiguration pass.

Replacements are accepted only if they **strictly reduce** the focal firm's ARP under the reactive model; otherwise the candidate is rejected and the edge kept. S4/S5/S6 look up their own best `alpha` from `best_alpha_summary.csv` when available, rather than using a single global threshold.

## Output

For each grid cell the runner writes to

```
{PROJECT_OUTPUT_DIR}/{EXPERIMENT_OUTPUT_SUBDIR}/tau/tau_{tau}/{combo}/kcore={k}/{year}/
```

- `results.csv` — node-level ARP, one row per firm
- `strategy_summary.csv` — mean and standard deviation of ARP per strategy
- `experiment_arp_details.csv` — per-simulation detail for every strategy
- `experiment.log` — full run log, including network size after pruning

## Regression analysis

`regression_analysis.py` builds a firm-year panel from the simulated output, constructs a node-level vulnerability measure (including a Gini term over partner sizes), and estimates OLS models with cluster-robust standard errors via `statsmodels`:

```bash
python regression_analysis.py
```

It reuses the same `config.py` paths and parameter combinations as the simulation.

## Reproducibility notes

- Global seed is `42` (`ExperimentConfig.random_seed`); each task re-seeds at start, so parallel and serial runs agree.
- Default budget is 1000 simulations × 100 iterations per grid cell.
- Isolated nodes without `Revenue` are pruned before simulation, so network size printed in the log may differ from the raw table row count.
- The pipeline needs the four input tables; the repository does not redistribute licensed Compustat extracts beyond the aggregates included in `data/`.

## Citation

```bibtex
@article{<key>,
  title   = {Enhancing Resilience in Complex Supply Chain Systems: Adaptive Network Reconfiguration through Risk Perception},
  author  = {<authors>},
  journal = {<journal>},
  year    = {<year>},
  doi     = {<doi>}
}
```

Please fill in the fields above once the paper is published.

## License

No license is declared yet. If you intend the code to be reusable, add a `LICENSE` file (MIT or Apache-2.0 are the usual choices for academic code) — without one, the default is "all rights reserved", which technically blocks reuse.

---

## 中文说明

- 改 `config.py` 顶部四个路径常量，指向你本机的数据目录和输出目录，否则跑不起来。
- 先用 `python main.py --preset experiment --combos 0.1_N_N` 跑 50 次验证流程，再上全量。
- 参数网格默认是 8 个组合 × 2 个 tau × 8 个年份 = 128 个任务，每个任务 1000 次模拟、8 进程并行。
- `kcore` 的 `KCORE_RANGE` 默认不生效，因为 `KCoreConstraintConfig.enabled` 是 `False`；要跑 k=4 必须先打开这个总开关。
- 输出按 `tau → 组合 → kcore → 年份` 分层存放，每个目录下有 `results.csv`、`strategy_summary.csv`、`experiment_arp_details.csv` 和 `experiment.log`。
