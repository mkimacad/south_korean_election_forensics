# Dependency Guide: `run_mebane_r_all_parallel_bidirectional.sh` Pipeline

This document maps every layer this entry-point script touches, from shell → Python → R → JAGS,
so the full dependency chain can be installed and reasoned about before a run.

## 1. Call Chain Overview

```
run_mebane_r_all_parallel_bidirectional.sh   (bash)
  └─ spawns, per grid cell (election × level × channel × source × leader_side):
       mebane_r_pipeline_bidirectional.py    (python3)
         └─ shells out via subprocess.run(["Rscript", ...]):
              mebane_crosscheck.R            (Rscript)
                └─ library(eforensics)  ── fits the qbl model via JAGS
                └─ library(coda)        ── posterior summaries / HPD / Gelman-Rubin
                └─ diptest (optional)   ── dip-test diagnostic, degrades to NA if absent
```

A parallel, related chain exists for the self-null variant:

```
run_mebane_self_null_parallel.sh
  └─ mebane_self_null_pipeline.py
       ├─ imports mebane_r_pipeline_bidirectional.py directly (as `base`)
       ├─ Stage 1: Rscript mebane_crosscheck_selfnull_fit.R   (real-data fit, harvests
       │           class-1 hyperparameters: beta.tau1/beta.nu1, tau_alpha/nu_alpha, tb/nb)
       └─ Stage 2: Rscript mebane_crosscheck.R                (refit on synthetic self-null data)
```

Downstream, non-fitting scripts (`make_appendix_table*.py`, `make_figures.py`) only consume
the CSV outputs of the above and do not themselves invoke R/JAGS.

## 2. Layer-by-Layer Requirements

### 2.1 Shell layer (`run_mebane_r_all_parallel_bidirectional.sh`)

- **Interpreter: must be `bash`, not POSIX `sh`.** It relies on `wait -n` and `jobs -r -p`
  (bash ≥ 4.3 job-control features with no `dash`/`sh` equivalent). Invoke as
  `./run_mebane_r_all_parallel_bidirectional.sh` (uses the `#!/bin/bash` shebang) or
  `bash run_mebane_r_all_parallel_bidirectional.sh` — never `sh run_mebane_r_all_parallel_bidirectional.sh`.
- Standard coreutils only: `awk`, `date`, `mkdir`, `wc`, `grep`, `sed`.
- Expects `python3` to be resolvable on `PATH` (it calls `python3 "$PIPELINE_SCRIPT" ...` directly).
- Expects `mebane_r_pipeline_bidirectional.py` to sit in the **same working directory** it's
  invoked from (`PIPELINE_SCRIPT="mebane_r_pipeline_bidirectional.py"`, a relative path).

### 2.2 Python layer (`mebane_r_pipeline_bidirectional.py`)

| Package | Role | Notes |
|---|---|---|
| `pandas` | load/reshape election CSVs, write model input/read model output | required |
| `numpy` | null-generator math (Beta/Gaussian sampling, logit transforms) | required |
| `os`, `re`, `sys`, `shutil`, `subprocess`, `tempfile`, `time` | stdlib | no install needed |

- **Critical runtime check:** `run_r_model()` calls `shutil.which('Rscript')` and raises
  `FileNotFoundError` immediately if `Rscript` is not on `PATH` — so R must be installed and
  on `PATH` *before* Python is invoked, not just importable.
- Writes a per-cell tempdir (`tempfile.mkdtemp(prefix='mebane_r_pipeline_')`) containing
  `input.csv`, then invokes `Rscript mebane_crosscheck.R input.csv output.csv <args>` as a
  subprocess and reads back `output.csv` / `output_units.csv`.
- `mebane_self_null_pipeline.py` additionally requires `mebane_r_pipeline_bidirectional.py`
  to be importable (`sys.path.insert(0, ...)` + `import mebane_r_pipeline_bidirectional as base`),
  so both files must remain co-located.

Non-fitting utility scripts add:
- `make_figures.py` → `pandas`, `numpy`, `matplotlib` (with a non-interactive backend set before
  `pyplot` import — check the top of the file if running headless/without a display).
- `make_appendix_table*.py` → `pandas`, `numpy` only.

### 2.3 R layer (`mebane_crosscheck.R`, `mebane_crosscheck_selfnull_fit.R`)

Both scripts open with:
```r
suppressMessages(library(eforensics))
suppressMessages(library(coda))
```

- **`eforensics`** — the core Bayesian election-forensics model (Ferrari/Mebane). This is
  **not on CRAN**; it must be installed from GitHub, e.g.:
  ```r
  devtools::install_github("UMeforensics/eforensics_public")
  ```
  The script defensively handles a naming discrepancy between released forks — it probes
  `formals(eforensics)` at runtime for either `eligible.voters` or `elegible.voters` and
  picks whichever is present — so either the `UMeforensics/eforensics_public` fork or the
  original `DiogoFerrari/eforensics` fork will work, but *some* fork must be installed.
- **`coda`** — MCMC diagnostics (`HPDinterval`, `mcmc.list`, `gelman.diag`). Available on CRAN.
- **`diptest`** (optional, soft dependency) — used only for the Hartigan & Hartigan dip-test
  diagnostic `D(pi_j)`. Guarded with `requireNamespace("diptest", quietly = TRUE)`; if missing,
  the script prints a note and reports `NA` for that diagnostic rather than failing the run.

#### Transitive R dependencies (pulled in by installing `eforensics` itself)

Per the package's `DESCRIPTION`, `eforensics` itself `Imports`:
```
magrittr, coda, dclone, dplyr, rjags, LCA, MASS, parallel, runjags,
LaplacesDemon, foreach, tibble, data.table, msm, purrr, stringr,
doParallel, ggplot2, tidyr
```
These are installed automatically by `devtools::install_github()` unless you pass
`dependencies = FALSE`.

#### System-level (non-R) dependency: JAGS

`rjags` and `runjags` are R *bindings* to **JAGS** (Just Another Gibbs Sampler), a separate
compiled program that is **not installed by `install.packages`/`devtools`** — it must be
installed at the OS level first, or `library(rjags)` (and therefore `library(eforensics)`)
will fail to load:

- Debian/Ubuntu: `sudo apt-get install jags`
- macOS (Homebrew): `brew install jags`
- Windows: install the JAGS binary from SourceForge, matching architecture with your R install.

This is the dependency most likely to silently break a fresh environment, since the R-package
install step can succeed while JAGS itself is still absent (the failure only surfaces at
`library(rjags)`/`library(eforensics)` load time, i.e. the first line of `mebane_crosscheck.R`).

### 2.4 Data layer

The pipeline reads Korean general/presidential election result CSVs
(`16th_presidential_election_result.csv` … `22nd_election_result.csv`) that must sit alongside
the scripts (paths are relative, resolved via `ELECTION_CONFIGS[...]['result_csv']` in
`mebane_r_pipeline_bidirectional.py`). No network access is needed at data-load time — only
the initial `eforensics` GitHub install requires network/GitHub access.

## 3. Minimal Install Checklist

```bash
# 1. System-level: R + JAGS
sudo apt-get install -y r-base jags     # Debian/Ubuntu example

# 2. R packages
Rscript -e '
if (!requireNamespace("devtools", quietly = TRUE))
  install.packages("devtools", repos = "https://cloud.r-project.org")
devtools::install_github("UMeforensics/eforensics_public")
install.packages(c("coda", "diptest"), repos = "https://cloud.r-project.org")
'

# 3. Python packages
pip install --break-system-packages pandas numpy matplotlib

# 4. Sanity check: Rscript must be resolvable on PATH for step 2.2's shutil.which() check
which Rscript
```

## 4. Failure Modes Mapped to Layers

| Symptom | Likely missing dependency |
|---|---|
| `FileNotFoundError: run_r_model: 'Rscript' not found on PATH` | R not installed, or not on `PATH` for the shell/Python process |
| `wait: -n: invalid option` or similar on launch | Script invoked with `sh` instead of `bash` |
| R error at `library(rjags)` / `library(eforensics)` load | JAGS not installed at OS level |
| `Could not find an eligible-voters argument...` (from `mebane_crosscheck.R`) | An `eforensics` fork installed with neither `eligible.voters` nor `elegible.voters` — install one of the two forks referenced above |
| `D_pi` columns all `NA` with a console note | `diptest` R package not installed (non-fatal, optional) |
| `ModuleNotFoundError: No module named 'mebane_r_pipeline_bidirectional'` (self-null run) | `mebane_self_null_pipeline.py` not co-located with `mebane_r_pipeline_bidirectional.py` |
