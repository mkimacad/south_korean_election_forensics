# eforensics: election forensics analysis (Korean elections)

This repository fits the Ferrari/Mebane `eforensics` quasi-binomial-logistic
("qbl") election-fraud model to Korean general and presidential election
results (2002-2025), across multiple vote channels, both real and
null-model data sources, and both leader codings (Democratic-lineage and
conservative-lineage). It also includes a self-null robustness check and an
independent, non-JAGS "election fingerprint" diagnostic (Klimek et al. 2012).

## Requirements

- A conda installation (e.g. [Miniforge](https://github.com/conda-forge/miniforge)
  or Miniconda). No other system packages, compilers, or admin rights are
  needed -- `setup_env.sh` provisions everything, including JAGS itself,
  through conda and pip.
- Linux or macOS. (Windows users: run these steps inside WSL.)

## 1. Installation

From the repository root:

```bash
/bin/bash setup_env.sh
```

This creates a conda environment named `eforensics` (from `environment.yml`)
containing:
- Python 3.11 + numpy/pandas/scipy/matplotlib
- R + JAGS + rjags + coda + diptest (all from conda-forge, including the
  JAGS binary itself)

and then installs the `eforensics` R package, which is distributed only on
GitHub (not CRAN/conda-forge), via `install_eforensics.R`. The script ends
by verifying that both the Python and R/JAGS stacks import/load correctly.

Re-running `setup_env.sh` later (e.g. after pulling updates) is safe: it
updates the existing environment in place rather than failing or
duplicating it.

To use a different environment name:

```bash
ENV_NAME=my_env_name /bin/bash setup_env.sh
```

### Troubleshooting: R package build failures

`eforensics` pulls in a fairly heavy chain of CRAN dependencies (`mcmcse`,
`ggExtra` -> `shiny` -> `httpuv`/`bslib`, plus `fftwtools`), several of which
need system libraries to compile from source: zlib, libxml2, libuv, FFTW,
and `pkg-config` to locate them. `environment.yml` already provides all of
these via conda-forge, and `install_eforensics.R` explicitly points
`pkg-config` at the conda environment's `lib/pkgconfig` directory (some
package-level conda activation hooks don't reliably propagate through
`conda run`), so a normal `setup_env.sh` run shouldn't hit this. If you see
`configure: error: pkg-config not found`, `fatal error: zlib.h`, `fatal
error: libxml/tree.h`, or `fatal error: uv.h` during the
`install_eforensics.R` step, re-run `/bin/bash setup_env.sh` to pick up the
current `environment.yml`, or add the missing package directly with
`conda install -n eforensics -c conda-forge <package>`.

Two warnings you may see and can ignore: (1) `installation of package
'xml2' had non-zero exit status` / the same for `roxygen2` -- both are
`Suggests`-only for `eforensics` (used for building documentation, not for
fitting models) and don't block the install; and (2) `install_eforensics.R`
installs `LCA` directly by URL rather than through the normal package
index -- `eforensics` declares `LCA` as an Import but its own
`cran-comments.md` confirms the code never actually calls it, and on some
CRAN mirrors it doesn't resolve through the lookup `remotes` normally uses.

## 2. Activate the environment

```bash
conda activate eforensics
```

If `conda activate` doesn't work in your shell (common on a fresh conda
install), either run `conda init bash` once and restart your shell, or
replace `conda activate eforensics` below with prefixing each command with
`conda run -n eforensics`.

## 3. Run the models

```bash
/bin/bash run_mebane_r_all_parallel_bidirectional.sh
/bin/bash run_mebane_self_null_parallel.sh
```

(`/bin/bash` is used explicitly because these two scripts are not marked
executable in this repository and rely on bash-specific job-control features
-- `wait -n` / `jobs -r -p` -- so they must be run with `bash`, not `sh`.)

- `run_mebane_r_all_parallel_bidirectional.sh` fits every
  election/level/channel/source/leader_side cell (real, marginal-null, and
  joint-null data) via `mebane_r_driver_bidirectional.py` /
  `mebane_crosscheck.R`.
- `run_mebane_self_null_parallel.sh` runs the model-consistent self-null
  check (fit to real data, then refit to a synthetic no-fraud dataset drawn
  from those fitted hyperparameters) via `mebane_self_null_driver.py` /
  `mebane_crosscheck_selfnull_fit.R`.

  Latent-turnout robustness check: run the main sweep with
  `EFORENSICS_WFORM=tau` (short-chain is now the default), then
  `python3 make_tau_comparison.py` (pairs each tau run with its conditional
  counterpart, both codings, and writes the tables and the two tau figures).

  The likelihood form applies to the whole self-null, not just the fits:
  `EFORENSICS_WFORM=tau` makes the simulation step draw class-1 leader votes
  from `Binomial(N, mu.tau*mu.nu)` instead of `Binomial(N, mu.nu*(1 - a/N))`,
  so the null matches the form it is refit under. Logs go to `output_data_tau/`.

  ```bash
  EFORENSICS_WFORM=tau ./run_mebane_self_null_parallel.sh          # full grid, dem+con
  python3 parse_selfnull_logs.py output_data_tau/mebane_r_all_logs \
      output_data_tau/con_leader -o appendix_selfnull_results_tau.csv
  python3 make_appendix_table_selfnull.py appendix_selfnull_results_tau.csv \
      appendix_tables_selfnull_tau.tex
  ```


Both scripts are resumable (a cell with an existing non-empty log is
skipped) and run cells concurrently as separate OS processes. Useful
environment-variable overrides (see each script's header comments for the
full list):

```bash
# Quick smoke test on a single election/level instead of the full grid:
MAX_PARALLEL=2 LEADER_SIDES="con" /bin/bash run_mebane_r_all_parallel_bidirectional.sh "pres18" "dong"

# Force a full rerun, ignoring existing logs:
FORCE_RERUN=1 /bin/bash run_mebane_r_all_parallel_bidirectional.sh
```

Logs land in `output_data/mebane_r_all_logs/` (dem-coded leader) and
`output_data/con_leader/` (con-coded leader), with a combined
`*_summary.txt` written to each directory once a run completes.

## 4. Post-processing

Once results are available, the appendix tables can be regenerated from the
included aggregated result CSVs:

```bash
python3 make_appendix_table.py            # reads appendix_full_results.csv
python3 make_appendix_table_components.py # reads appendix_components_results.csv
python3 make_appendix_table_mebane_md.py  # reads appendix_mebane_md_results.csv
python3 make_appendix_table_selfnull.py   # reads appendix_selfnull_results.csv
```

### Fraud-share credible intervals

`mebane_crosscheck.R` writes two 95% intervals for the fraud share
`pi[2] + pi[3]`:

- `fraud_share_cons_hpd95_lo/hi` -- the **conservative** sum of the separate
  `pi[2]` and `pi[3]` HPD bounds. Strictly wider than the exact interval,
  because adding the two component intervals ignores the posterior dependence
  between them. **This is what the manuscript and the appendix tables report**,
  for consistency with the `Ft`/`Fw` convention below.
- `fraud_share_hpd95_lo/hi` -- the **exact** joint HPD interval, computed from
  the per-draw sum `pi[2] + pi[3]`. Written out for anyone who wants it. To
  print it in the appendix tables instead, set `REPORT_EXACT_HPD = True` at the
  top of `make_appendix_table.py` and `make_appendix_table_selfnull.py`; this
  requires result files produced by the current `mebane_crosscheck.R`, i.e. a
  re-run of the fits.

Note that `median(pi[2] + pi[3])` (`fraud_share_joint_median`) is not in general
equal to `median(pi[2]) + median(pi[3])` (`fraud_share_sum_of_medians`); the
manuscript's headline fraud share is the latter. Both are written out.

The corresponding limitation for `Ft`/`Fw` is genuine and remains: the package
exposes only per-unit summary statistics for the fraud proportions, not the raw
per-draw per-unit matrix, so no exact joint interval for those totals can be
computed and the conservative sum is all that is available.

**Result files produced before these columns existed do not contain them.** This
does not affect the default tables, which use the conservative interval; it only
matters if `REPORT_EXACT_HPD` is turned on, in which case the table scripts
detect the missing columns, print a NOTE, and fall back to the conservative
interval.

```
```

`make_figures.py` reproduces the main-text eforensics figures from
`appendix_full_results.csv` (short-chain configuration, both leader codings),
which `collect_eforensics_results.py` builds from the per-cell run logs.

The collection commands are listed under "Rebuilding all results" below.

The Klimek-type fingerprint analysis (pure Python, no R/JAGS) is described
under "Klimek et al. (2018) procedure" below. The full run is

```bash
python3 klimek_2018_all.py                   # both levels (dong + tupyogu), ~1.5 h on one core
python3 klimek_2018_all.py --levels tpg      # polling-district (tupyogu) level only
```

## Repository layout

- `environment.yml`, `install_eforensics.R`, `requirements-pip.txt`,
  `setup_env.sh` -- installation.
- `mebane_crosscheck.R`, `mebane_crosscheck_selfnull_fit.R` -- R/JAGS model
  fitting (`eforensics::eforensics()`, `model = "qbl"`).
- `mebane_r_driver_bidirectional.py`, `mebane_self_null_driver.py` -- Python
  data loaders/drivers that build the model's input tables and invoke the R
  scripts above.
- `run_mebane_r_all_parallel_bidirectional.sh`,
  `run_mebane_self_null_parallel.sh` -- parallel, resumable sweeps across
  the full election/level/channel/source/leader_side grid.
- `klimek_analysis.py`, `run_klimek_all.sh`, `klimek_summarize.py`, `klimek_recovery_test.py`,
  `make_klimek_preearly_figures.py` -- Klimek-type fingerprint analysis (see below).
- `make_appendix_table*.py`, `make_figures.py` -- table/figure generation.
- `parse_mebane_logs.py` -- parses raw per-cell logs into a flat result CSV
  (the manuscript's aggregated files come from `collect_eforensics_results.py`).
- `*_election_result.csv`, `*_presidential_election_result.csv` -- raw
  election data.
- `appendix_*_results.csv`, `psrf_constituency_results.csv`,
  `tau_comparison.csv` -- aggregated results used by the manuscript. Earlier
  aggregated files are kept in `archive_pre_v2/` (pre-rerun) and
  `long_chain_archive/` (long-chain configuration) and are not used.

## Supplementary analyses and their data files

### Model-consistent self-null (all 58 cells)
- `self_null_results.csv` and `appendix_selfnull_results.csv` both contain all 58
  short-chain, Democratic-coded cells. The pres21/dong/early cell crashed on its first
  attempt and completed on rerun with the same seed (98893); the seed fixes only the
  synthetic-data draw, and the unseeded first-stage fit changes between runs.

### PSRF stopping-rule check (constituency level)
- `run_psrf_constituency.sh` -- reruns all 35 constituency cells x {real, marginal
  null, joint null} x {dem, con} (210 runs) with the package's PSRF rule
  (`EFORENSICS_CONV=psrf`: psrf < 1.05 on pi, up to 10 chain extensions).
- `output_data_psrf/` -- the 210 run logs (`mebane_r_all_logs/` = dem, `con_leader/` = con).
- `parse_psrf_logs.py` -> `psrf_constituency_results.csv`: one row per run, with the
  number of extensions, whether the package's PSRF rule was met, the final psrf,
  R-hat on the kept draws, fraud share and intervals, flagged units, run time.

### Klimek et al. (2012) fingerprint model (dong level)
- `klimek_analysis.py` (one cell), `run_klimek_all.sh` (all cells),
  `klimek_recovery_test.py` (parameter-recovery test), `klimek_summarize.py`,
  `make_klimek_preearly_figures.py`.
- `klimek_output/<cell>_klimek.csv` -- per-cell fits (44 files: 11 elections x 2 leader
  codings x raw / province-residualized; each has real, marginal-null, joint-null rows).
- `klimek_output/klimek_all_results.csv` -- all of the above in one file.
- `klimek_output/klimek_winner_results.csv` (+ `klimek_winner_table.tex`) -- the
  winner-coded rows used in the manuscript's Klimek table (winner = elected president's
  party; for general elections, the party winning the most seats).
- `klimek_output/klimek_preearly_exclusion_results.csv` (+ `klimek_exclusion_table.tex`)
  -- pre-early-voting elections refitted with Honam; Honam + Yeongnam; Honam +
  Gyeongbuk/Daegu excluded. `preearly_fits*.npz` are the corresponding fit caches.
- In the tables, each entry is the median across three simulation seeds; brackets
  give the minimum and maximum across seeds when these differ by two or more
  percentage points.

### Optional model/option forks
- `ef_qbl_tau.R` with `EFORENSICS_WFORM=tau` -- leader-vote likelihood in terms of the
  latent turnout instead of the realized abstentions (logs to `output_data_tau/`).
- `klimek_recovery_test.py` -> `klimek_output/klimek_recovery_results.csv` (parameter-recovery test).
- `make_appendix_table_extra.py` -> `appendix_tables_psrf.tex`, `appendix_tables_klimek.tex`,
  `appendix_tables_klimek_recovery.tex` (appendix tables for the PSRF check and the Klimek analysis).
- Note on the Klimek implementation: the fraud mechanisms follow Klimek et al. (2012), but the
  estimation does not (2-D fingerprint objective; clean component and incremental width fitted
  jointly; exact NNLS for f_i, f_e with a 95% cap; three simulation seeds instead of 100
  realizations). See the klimek_analysis.py docstring and the appendix.

## Klimek et al. (2018) procedure (used for the manuscript's Klimek results)

Scope: the five elections held before early voting (pres16, pres17, 18, 19,
pres18), at two levels -- dong and polling district (투표구, `tpg`).
Settings follow Klimek et al. (2018), with Klimek et al. (2012) where 2018 is
silent: plug-in clean component (2012 SI), RSS between 1%-binned 2-D
fingerprints, grid f_i, f_e in 0..1 step .01 and alpha in 0..5 step .1,
200 sweeps, electorate >= 100, no trimming of extreme units.

- `klimek_units.py` -- unit tables for both levels (matches the eforensics
  loader's dong-level counts exactly); drops units where either major party had
  no candidate, identically for both leader codings.
- `klimek_2018.py` -- the estimator. Exact sparse grid solve;
  `fit_realization_dense` is the old dense solver, kept as reference.
- `klimek_2018_verify.py` -- checks the two solvers return identical minimizers
  (60/60; log in `klimek_2018_output/verify_log.txt`).
- `klimek_2018_all.py` -- resumable runner: 80 main fits (5 elections x 2 levels x
  2 codings x raw/resid x real/joint null) -> `klimek_2018_all_results.csv`;
  30 regional-exclusion fits -> `klimek_2018_exclusion_results.csv`; per-sweep fits
  in `klimek_2018_output/sweeps/`. Marginal null is not run (not a fraud-free
  control for a fingerprint method).
- `klimek_2018_winner_table.py`, `klimek_2018_exclusion_table.py`,
  `klimek_2018_summary_figure.py`, `make_klimek_preearly_figures.py [--level tpg|dong]`
  (with `klimek_2018_viz.py`), `klimek_2018_stats.py` -- tables, figures and every
  number cited in the text, read from the CSVs (no refitting). Tables/figures only
  include levels whose grid is complete.
- `klimek_2018_recovery_test.py [--n-max N]` (resumable),
  `klimek_original_recovery_test.py` -- recovery tests (2018 and reconstructed 2012
  procedures); `make_appendix_table_klimek2018.py` -- appendix tables.
- `klimek_original.py` -- reconstruction of the 2012 1-D procedure (comparison only).
- `klimek_analysis.py` and `klimek_output/` -- our original joint-fit estimator;
  only the "ours" column of the appendix comparison. `klimek_analysis.residualize`
  is also used for the residualized fingerprints.

Rebuilding CSV rows from sweep files returned from another machine
(`klimek_2018_from_sweeps.py --level tpg`; `--check` verifies the rebuild on
existing rows), then regenerating everything:

```bash
python3 klimek_2018_all.py --levels tpg          # or: copy sweeps/ back, then klimek_2018_from_sweeps.py --level tpg
python3 klimek_2018_recovery_test.py              # adds the n = 13,500 cases
python3 klimek_original_recovery_test.py          # adds the n = 13,500 cases (resumable)
python3 klimek_2018_winner_table.py && python3 klimek_2018_exclusion_table.py
python3 klimek_2018_summary_figure.py && cp klimek_2018_output/klimek_summary.pdf .
python3 make_klimek_preearly_figures.py --level tpg
python3 make_appendix_table_klimek2018.py && python3 klimek_2018_stats.py
```


## Rebuilding all results (model-consistent fraudulent-vote counts)

`mebane_crosscheck.R` now monitors `mu.tau`/`mu.nu` and computes F_t/F_w per
posterior draw as N*[f.m*(1-tau) + f.s*tau*(1-nu)] (the model's own definition
and the package simulator's), with exact joint 95% HPD intervals. Earlier logs
used the package convention (rates x N) and are rejected by the collectors.

1. Run everything (both scripts may run at the same time; 8 cells each):

```bash
nohup ./run_all_eforensics_dem.sh > run_all_dem.out 2>&1 &
nohup ./run_all_eforensics_con.sh > run_all_con.out 2>&1 &
```

   Both scripts include the conditional and tau self-nulls. `run_selfnull_con.sh`
   runs only the two conservative-coded self-nulls (for a con run that predates
   them; never run it alongside run_all_eforensics_con.sh). `run_psrf_dong.sh`
   runs the PSRF check at dong level (138 runs, slow), parsed with
   `python3 parse_psrf_logs.py output_data_v2_psrf_dong psrf_dong_results.csv`.

   Progress: `run_all_status/run_all_{dem,con}.summary`; per-step output in
   `run_all_status/`. Rerun a script to resume after an interruption. Do not
   edit any script while they run.

2. Collect (after both have finished):

```bash
python3 collect_eforensics_results.py \
    --short output_data_v2/mebane_r_all_logs output_data_v2/con_leader
python3 parse_selfnull_logs.py output_data_v2/mebane_r_all_logs output_data_v2/con_leader -o appendix_selfnull_results.csv
python3 parse_selfnull_logs.py output_data_v2_tau/mebane_r_all_logs output_data_v2_tau/con_leader -o appendix_selfnull_results_tau.csv
python3 parse_psrf_logs.py output_data_v2_psrf psrf_constituency_results.csv
python3 make_tau_comparison.py output_data_v2_tau/mebane_r_all_logs output_data_v2_tau/con_leader
```

3. Regenerate tables and figures: `make_appendix_table.py`,
   `make_appendix_table_components.py`, `make_appendix_table_mebane_md.py`,
   `make_appendix_table_selfnull.py` (and with `appendix_selfnull_results_tau.csv
   appendix_tables_selfnull_tau.tex tau`), `make_appendix_table_extra.py`,
   `make_figures.py`, `make_fig7_selfnull.py`, `make_chainlength_check.py`
   (chain-length verification from the PSRF runs) and `make_eforensics_stats.py`
   (every eforensics number cited in the main text, plus the main-text table
   bodies).

The long-chain configuration is no longer part of the paper or of the mega
scripts; its result CSVs are kept in `long_chain_archive/`. Pre-rerun logs and
CSVs (package-convention F_w) are kept in `archive_pre_v2/` and must not be
used.
