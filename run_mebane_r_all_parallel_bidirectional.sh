#!/bin/bash
# Runs mebane_r_pipeline_bidirectional.py (R/JAGS-only) across every
# election/level/channel/source/leader_side cell, parallelized by spawning
# multiple independent subprocesses concurrently -- NOT batching. Each cell
# is still its own completely separate, un-batched Rscript/JAGS run (own
# tempdir, own eforensics() call, own chains); we're just running several of
# them AT THE SAME TIME as separate OS processes.
#
# CONCURRENCY vs JAGS's OWN INTERNAL PARALLELISM: each cell's Rscript call can
# itself run its 4 MCMC chains in parallel (parComp=TRUE, JAGS's own multi-core
# use). If you ALSO run many cells concurrently via this script, you can oversubscribe
# CPU cores (MAX_PARALLEL cells x up to 4 JAGS chains each). Default here is
# USE_PARCOMP=0 (each cell's chains run sequentially) specifically so cross-cell
# parallelism is the main lever -- turn USE_PARCOMP on only if you have enough
# cores for both and have checked it doesn't thrash (MAX_PARALLEL * 4 <= your
# core count is a reasonable starting rule of thumb).
#
# RESUMABLE: before launching a cell, this script checks whether that cell's
# log file already exists and is non-empty. If so, the cell is skipped (no
# Rscript/JAGS relaunch) and its RESULT/END block is pulled from the
# existing log straight into the combined summary. A log that exists but is
# empty (0 bytes) -- e.g. a JAGS run that hung and was killed before writing
# anything -- is not treated as done, and will be relaunched.
#
# BIDIRECTIONAL: each cell is also indexed by leader_side ('dem' or 'con',
# i.e. which party/candidate is coded as the eforensics leader -- see
# mebane_r_pipeline_bidirectional.py). Controlled by LEADER_SIDES
# (space-separated, default "dem"):
#   - leader_side=dem -> logs go to output_data/mebane_r_all_logs/<cell>.log
#     (no filename suffix)
#   - leader_side=con -> logs go to output_data/con_leader/<cell>_con.log
#     (separate directory, _con filename suffix)
#
# Usage:
#   ./run_mebane_r_all_parallel_bidirectional.sh                    <- full grid, all 11 elections, dem only
#   ./run_mebane_r_all_parallel_bidirectional.sh "21 22"             <- just these elections
#   ./run_mebane_r_all_parallel_bidirectional.sh "21" "dong"         <- just this election, this level
#   LEADER_SIDES="con" ./run_mebane_r_all_parallel_bidirectional.sh "pres18 pres17 pres20"
#                                                                     <- con-leader only, selected elections
#   LEADER_SIDES="dem con" ./run_mebane_r_all_parallel_bidirectional.sh
#                                                                     <- full bidirectional grid (2x the cells)
#   MAX_PARALLEL=8 ./run_mebane_r_all_parallel_bidirectional.sh      <- override concurrency
#   FORCE_RERUN=1 ./run_mebane_r_all_parallel_bidirectional.sh "21" <- ignore existing logs, rerun anyway
#
# Province grouping: 'original' (~17-18 provinces, DEFAULT) or 'megaregion' (4 groups),
# via MEBANE_PROVINCE_GROUPING -- e.g. MEBANE_PROVINCE_GROUPING=megaregion ./run_mebane_r_all_parallel_bidirectional.sh

set -e

ELECTIONS="${1:-21 22 pres20 pres21 18 19 20 pres16 pres17 pres18 pres19}"
LEVELS="${2:-dong constituency}"
NO_EARLY="18 19 pres16 pres17 pres18"
DONG_CHANNELS="early sameday pooled"
CONST_CHANNELS="early early_out early_total sameday total"
SOURCES="real marginal_null joint_null"
LEADER_SIDES="${LEADER_SIDES:-dem con}"   # space-separated subset of: dem con

# R/JAGS fit settings
R_N_ITER="${R_N_ITER:-3000}"
R_N_CHAINS="${R_N_CHAINS:-4}"
R_BURN_IN="${R_BURN_IN:-1000}"
USE_PARCOMP="${USE_PARCOMP:-0}"   # see concurrency note above -- default OFF

# Set to 1 to relaunch cells even if a non-empty log already exists for them.
FORCE_RERUN="${FORCE_RERUN:-0}"

# How many cells to run concurrently. A reasonable starting point is your CPU core
# count divided by however many cores each cell itself might use (1 if
# USE_PARCOMP=0, up to 4 if USE_PARCOMP=1). Override via MAX_PARALLEL=N.
MAX_PARALLEL="${MAX_PARALLEL:-4}"

PIPELINE_SCRIPT="mebane_r_pipeline_bidirectional.py"

is_no_early() {
  case " $NO_EARLY " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

# Log directory + filename suffix for a given leader_side: dem -> no suffix,
# con -> a _con filename suffix, each in its own directory.
log_dir_for() {
  case "$1" in
    dem) echo "output_data/mebane_r_all_logs" ;;
    con) echo "output_data/con_leader" ;;
    *) echo "run_mebane_r_all_parallel_bidirectional.sh: unknown leader_side '$1' (expected dem or con)" >&2; exit 1 ;;
  esac
}
suffix_for() {
  case "$1" in
    dem) echo "" ;;
    con) echo "_con" ;;
    *) echo "run_mebane_r_all_parallel_bidirectional.sh: unknown leader_side '$1' (expected dem or con)" >&2; exit 1 ;;
  esac
}

for ls in $LEADER_SIDES; do
  case "$ls" in
    dem|con) ;;
    *) echo "Invalid LEADER_SIDES entry '$ls' -- must be 'dem' and/or 'con'." >&2; exit 1 ;;
  esac
  mkdir -p "$(log_dir_for "$ls")" "$(log_dir_for "$ls")/.summary_snippets"
done

echo "Bidirectional resumable R/JAGS sweep: LEADER_SIDES='$LEADER_SIDES', MAX_PARALLEL=$MAX_PARALLEL, "
echo "USE_PARCOMP=$USE_PARCOMP, R_N_ITER=$R_N_ITER, R_N_CHAINS=$R_N_CHAINS, R_BURN_IN=$R_BURN_IN, "
echo "FORCE_RERUN=$FORCE_RERUN"
echo ""

n_launched=0
n_skipped=0
t_start=$(date +%s)

# Each background job: run one cell, write its own log AND its own summary snippet
# (avoids concurrent-write races on a shared file -- snippets get concatenated into
# the final summary only after all jobs finish).
run_one_cell() {
  local e="$1" level="$2" ch="$3" src="$4" leader_side="$5"
  local log_dir; log_dir="$(log_dir_for "$leader_side")"
  local suffix; suffix="$(suffix_for "$leader_side")"
  local cell="${e}_${level}_${ch}_${src}${suffix}"
  local log_path="${log_dir}/${cell}.log"
  local status_path="${log_dir}/.status"
  local snippet_path="${log_dir}/.summary_snippets/${cell}.snippet"

  if python3 "$PIPELINE_SCRIPT" "$e" "$level" "$ch" "$src" "$leader_side" \
        "$R_N_ITER" "$R_N_CHAINS" "$R_BURN_IN" "$USE_PARCOMP" \
        > "$log_path" 2>&1; then
    echo "OK    -> $cell" >> "$status_path"
  else
    echo "FAILED -> $cell" >> "$status_path"
  fi

  {
    echo "=== $cell (leader=$leader_side) ==="
    awk '/^RESULT:/{flag=1} flag; /^END:/{flag=0}' "$log_path"
    echo ""
  } > "$snippet_path"
}
# NOTE: no `export -f`/`export VAR` needed here -- run_one_cell is backgrounded (`&`) within
# THIS SAME bash process, not invoked via a separate shell (e.g. xargs -I{} bash -c ...), so it
# already sees this script's own function definitions and variables directly. This script also
# genuinely requires bash specifically (not POSIX sh) -- `wait -n` and `jobs -r -p` are bash job
# control features with no dash/sh equivalent; invoke as `./run_mebane_r_all_parallel_bidirectional.sh`
# (relies on the #!/bin/bash shebang) or `bash run_mebane_r_all_parallel_bidirectional.sh`, never
# `sh run_mebane_r_all_parallel_bidirectional.sh`.

# Record an already-completed cell (skip case) straight into this run's status +
# snippet files, so the combined summary below stays complete even though we
# didn't relaunch this cell. Uses the EXISTING log's own RESULT/END block.
record_skipped_cell() {
  local e="$1" level="$2" ch="$3" src="$4" leader_side="$5" log_path="$6"
  local log_dir; log_dir="$(log_dir_for "$leader_side")"
  local suffix; suffix="$(suffix_for "$leader_side")"
  local cell="${e}_${level}_${ch}_${src}${suffix}"
  local status_path="${log_dir}/.status"
  local snippet_path="${log_dir}/.summary_snippets/${cell}.snippet"

  echo "SKIP  -> $cell (log already exists, $(wc -c < "$log_path") bytes)" >> "$status_path"
  {
    echo "=== $cell (leader=$leader_side) [SKIPPED -- pre-existing log] ==="
    awk '/^RESULT:/{flag=1} flag; /^END:/{flag=0}' "$log_path"
    echo ""
  } > "$snippet_path"
}

# Reset status + wipe stale snippets per leader_side up front (both freshly-run
# and skipped cells rewrite their own status line + snippet during this
# invocation, so this stays a complete, accurate record of the CURRENT run).
for ls in $LEADER_SIDES; do
  log_dir="$(log_dir_for "$ls")"
  : > "${log_dir}/.status"
  rm -f "${log_dir}/.summary_snippets"/*.snippet 2>/dev/null || true
done

for ls in $LEADER_SIDES; do
  for e in $ELECTIONS; do
    for level in $LEVELS; do
      if is_no_early "$e"; then
        channels="total_no_early"
      elif [ "$level" = "dong" ]; then
        channels="$DONG_CHANNELS"
      else
        channels="$CONST_CHANNELS"
      fi

      for ch in $channels; do
        for src in $SOURCES; do
          log_dir="$(log_dir_for "$ls")"
          suffix="$(suffix_for "$ls")"
          cell="${e}_${level}_${ch}_${src}${suffix}"
          log_path="${log_dir}/${cell}.log"

          if [ "$FORCE_RERUN" != "1" ] && [ -s "$log_path" ]; then
            # Non-empty log already present -> treat as complete, skip relaunch.
            record_skipped_cell "$e" "$level" "$ch" "$src" "$ls" "$log_path"
            n_skipped=$((n_skipped+1))
            elapsed=$(( $(date +%s) - t_start ))
            echo "[skip -> $cell already has a log ($(wc -c < "$log_path") bytes); $n_skipped skipped so far, ${elapsed}s elapsed]"
            continue
          fi
          # A 0-byte (or missing) log is NOT treated as done -- this is exactly
          # what a hung/killed JAGS run (e.g. pres21/dong/early/marginal_null)
          # looks like, and it will be relaunched here.

          run_one_cell "$e" "$level" "$ch" "$src" "$ls" &
          n_launched=$((n_launched+1))

          # Concurrency limit: wait for at least one running job to finish before
          # launching more, once MAX_PARALLEL is reached. `wait -n` (bash >= 4.3)
          # waits for any single background job to complete, not all of them.
          while [ "$(jobs -r -p | wc -l)" -ge "$MAX_PARALLEL" ]; do
            wait -n
          done

          elapsed=$(( $(date +%s) - t_start ))
          echo "[$n_launched launched, $n_skipped skipped, ${elapsed}s elapsed, $(jobs -r -p | wc -l) currently running]"
        done
      done
    done
  done
done

echo ""
echo "All cells launched -- waiting for the remaining running jobs to finish..."
wait

total_elapsed=$(( $(date +%s) - t_start ))
echo ""
echo "Complete: $n_launched launched (new/rerun), $n_skipped skipped (pre-existing logs), ${total_elapsed}s total wall time."
echo ""

# Per-leader-side tallies + combined summary file.
for ls in $LEADER_SIDES; do
  log_dir="$(log_dir_for "$ls")"
  summary_file="${log_dir}/mebane_r_all_summary.txt"
  n_done=$(grep -c "^OK" "${log_dir}/.status" 2>/dev/null || true); n_done="${n_done:-0}"
  n_failed=$(grep -c "^FAILED" "${log_dir}/.status" 2>/dev/null || true); n_failed="${n_failed:-0}"
  n_skip_ls=$(grep -c "^SKIP" "${log_dir}/.status" 2>/dev/null || true); n_skip_ls="${n_skip_ls:-0}"

  : > "$summary_file"
  for snippet in "${log_dir}/.summary_snippets"/*.snippet; do
    [ -e "$snippet" ] || continue
    cat "$snippet" >> "$summary_file"
  done

  echo "leader_side=$ls : $n_done newly OK, $n_failed FAILED, $n_skip_ls skipped (pre-existing)."
  echo "  Per-cell logs: $log_dir/"
  echo "  Combined summary: $summary_file"
  if [ "$n_failed" -gt 0 ]; then
    echo "  Failed cells:"
    grep "^FAILED" "${log_dir}/.status" | sed 's/^FAILED -> /    /'
  fi
done

# --- fast first check instead of the full grid, e.g.: ----------------------
# MAX_PARALLEL=2 LEADER_SIDES="con" ./run_mebane_r_all_parallel_bidirectional.sh "pres18" "dong"
