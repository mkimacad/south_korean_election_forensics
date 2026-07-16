#!/usr/bin/env Rscript
# mebane_crosscheck.R
#
# Fits the Ferrari/Mebane eforensics quasi-binomial-logistic ("qbl") election
# fraud model to a single election/level/channel/source cell and writes out
# posterior summaries.
#
# Usage:
#   Rscript mebane_crosscheck.R <input.csv> <output.csv> [n_iter] [n_chains] [burn_in] [use_parcomp] [n_adapt]
#
# input.csv must contain columns: leader_votes, total_votes, eligible, prov_code

suppressMessages(library(eforensics))
suppressMessages(library(coda))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript mebane_crosscheck.R <input.csv> <output.csv> [n_iter] [n_chains] [burn_in] [use_parcomp] [n_adapt]")
}

input_path   <- args[1]
output_path  <- args[2]
n_iter       <- if (length(args) >= 3) as.integer(args[3]) else 3000
n_chains     <- if (length(args) >= 4) as.integer(args[4]) else 4
burn_in      <- if (length(args) >= 5) as.integer(args[5]) else 1000
use_parcomp  <- if (length(args) >= 6) as.logical(args[6]) else TRUE
n_adapt      <- if (length(args) >= 7) as.integer(args[7]) else 1000

cat(sprintf("[R-Model] Reading input data from: %s\n", input_path))
raw <- read.csv(input_path)

required_cols <- c("leader_votes", "total_votes", "eligible", "prov_code")
missing <- setdiff(required_cols, names(raw))
if (length(missing) > 0) {
  stop(sprintf("Input CSV missing required column(s): %s", paste(missing, collapse = ", ")))
}

# Map raw count data to the eforensics overdispersed-binomial schema. `prov`
# is an arbitrary integer grouping code (e.g. province) supplied by the
# caller in prov_code; this script simply factors whatever codes it is given.
data <- data.frame(
  w = round(raw$leader_votes),
  a = round(raw$eligible - raw$total_votes),
  N = round(raw$eligible),
  prov = factor(raw$prov_code)
)

bad <- data$w < 0 | data$a < 0 | (data$w + data$a) > data$N
if (any(bad)) {
  stop(sprintf("Data sanity error: %d row(s) have invalid vote counts (w<0, a<0, or w+a>N).", sum(bad)))
}

cat(sprintf("[R-Model] N=%d units, %d province(s). Fitting Quasi-Binomial-Logistic ('qbl') model...\n",
            nrow(data), nlevels(data$prov)))

mcmc_params <- list(burn.in = burn_in, n.adapt = n_adapt, n.iter = n_iter, n.chains = n_chains)
cat(sprintf("[R-Model] MCMC settings: n.iter=%d, n.chains=%d, burn.in=%d, n.adapt=%d, parComp=%s\n",
            n_iter, n_chains, burn_in, n_adapt, use_parcomp))

# Different released forks of the eforensics package spell the
# eligible-voters argument to eforensics() differently ("eligible.voters" vs.
# "elegible.voters"); detect the installed spelling at runtime rather than
# hardcoding one.
ef_formal_names <- names(formals(eforensics))
if ("eligible.voters" %in% ef_formal_names) {
  eligible_arg_name <- "eligible.voters"
} else if ("elegible.voters" %in% ef_formal_names) {
  eligible_arg_name <- "elegible.voters"
} else {
  stop(sprintf(paste0("Could not find an eligible-voters argument in the installed eforensics ",
                       "package's eforensics() function (checked 'eligible.voters' and ",
                       "'elegible.voters'). Formal arguments found: %s"),
               paste(ef_formal_names, collapse = ", ")))
}
cat(sprintf("[R-Model] Detected eforensics() eligible-voters argument name: '%s'\n", eligible_arg_name))

# parameters="all" delegates to an internal helper that has no case for
# model=="qbl" in at least one package fork, causing an error. Passing the
# explicit parameter list below bypasses that helper and is equivalent to
# what "all" resolves to for the bl/qbl/bbl model family: the base
# parameters plus "Z" (per-unit latent class draws, needed for piZi/get_Z()
# to be populated).
qbl_parameters <- c("pi", "beta.tau", "beta.nu", "beta.iota.m", "beta.iota.s",
                     "beta.chi.m", "beta.chi.s", "Z")

ef_args <- list(
  formula1 = w ~ prov,   # location equation for leader vote share
  formula2 = a ~ prov,   # location equation for abstention share
  data = data,
  model = "qbl",   # quasi-binomial model: per-unit random effects on all of tau, nu, iota, and chi
  mcmc = mcmc_params,
  get.dic = 0,
  parComp = use_parcomp,
  parameters = qbl_parameters
)
ef_args[[eligible_arg_name]] <- "N"

t0 <- Sys.time()
samples <- do.call(eforensics, ef_args)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("[R-Model] Execution complete in %.1f seconds.\n", elapsed))

# Population-level mixing weights (pi) across chains
pi_pooled <- do.call(rbind, lapply(samples, function(ch) {
  ch$parameters[, c("pi[1]", "pi[2]", "pi[3]")]
}))
pi_median <- apply(pi_pooled, 2, median)

pi_hpd <- coda::HPDinterval(coda::as.mcmc(pi_pooled), prob = 0.95)

pi_mcmc_list <- coda::mcmc.list(lapply(samples, function(ch) {
  coda::as.mcmc(ch$parameters[, c("pi[1]", "pi[2]", "pi[3]")])
}))
rhat <- tryCatch({
  gd <- coda::gelman.diag(pi_mcmc_list, autoburnin = FALSE, multivariate = FALSE)
  max(gd$psrf[, "Point est."])
}, error = function(e) NA_real_)

# Mebane (2023) posterior-multimodality diagnostics.
# M(pi_j): largest absolute difference between chain-specific posterior means.
chain_pi_means <- do.call(rbind, lapply(samples, function(ch) {
  colMeans(ch$parameters[, c("pi[1]", "pi[2]", "pi[3]")])
}))
M_pi <- if (nrow(chain_pi_means) > 1) apply(chain_pi_means, 2, function(x) max(x) - min(x)) else c(NA_real_, NA_real_, NA_real_)

# D(pi_j): Hartigan & Hartigan (1985) dip test for unimodality, pooling all
# chains together. Requires the 'diptest' package; degrades to NA rather
# than failing the run if it isn't installed.
D_pi <- tryCatch({
  if (!requireNamespace("diptest", quietly = TRUE)) stop("diptest not installed")
  apply(pi_pooled, 2, function(x) diptest::dip.test(x)$p.value)
}, error = function(e) {
  cat(sprintf("[R-Model] NOTE: D(pi_j) dip test unavailable (%s) -- reporting NA. Install the 'diptest' R package for this diagnostic.\n", conditionMessage(e)))
  c(NA_real_, NA_real_, NA_real_)
})

# Ft (manufactured votes) / Fw (total fraudulent votes).
#
# attr(samples, "frauds")$Manufactured and $Stolen are each an
# [n_units x 14] matrix of per-unit summary statistics (columns: "Mean",
# "95%HPDLow", "95%HPDHigh", plus 11 quantiles). Per the package's own
# convention, Manufactured[i] is the realized fraud-magnitude parameter for
# unit i under whatever class it drew: Ft_i = Manufactured_i * N_i (excess
# manufactured votes), Fw_i = (Manufactured_i + Stolen_i) * N_i (total
# fraudulent leader votes from both manufacturing and vote-stealing).
#
# CREDIBLE INTERVAL CAVEAT: attr(samples, "frauds") exposes only per-unit
# summary statistics, not the raw per-draw per-unit matrix, so there is no
# way to compute a true joint HPD interval for the total (summed across all
# units). What is reported below (Ft_total_hpd95_lo/hi, Fw_total_hpd95_lo/hi)
# is the SUM of independent per-unit HPD bounds -- a conservative
# approximation that is typically wider than the true interval for the sum,
# since it ignores how per-unit uncertainty can partially cancel when
# aggregated across many units. Treat it as an outer bound, not a precise
# credible interval.
fraud_votes <- tryCatch({
  fr <- attr(samples, "frauds")
  if (is.null(fr)) stop("attr(samples, 'frauds') was NULL -- package version/model may not expose it")
  if (!all(c("Mean", "95%HPDLow", "95%HPDHigh") %in% colnames(fr$Manufactured))) {
    stop(sprintf("attr(samples,'frauds')$Manufactured columns don't match the expected layout -- got: %s",
                  paste(colnames(fr$Manufactured), collapse = ", ")))
  }

  manufactured_mean <- fr$Manufactured[, "Mean"]
  manufactured_lo   <- fr$Manufactured[, "95%HPDLow"]
  manufactured_hi   <- fr$Manufactured[, "95%HPDHigh"]
  stolen_mean <- fr$Stolen[, "Mean"]
  stolen_lo   <- fr$Stolen[, "95%HPDLow"]
  stolen_hi   <- fr$Stolen[, "95%HPDHigh"]

  Ft_total <- sum(manufactured_mean * data$N)
  Ft_total_lo <- sum(manufactured_lo * data$N)
  Ft_total_hi <- sum(manufactured_hi * data$N)

  Fw_total <- sum((manufactured_mean + stolen_mean) * data$N)
  Fw_total_lo <- sum((manufactured_lo + stolen_lo) * data$N)
  Fw_total_hi <- sum((manufactured_hi + stolen_hi) * data$N)

  list(Ft_total = Ft_total, Ft_total_lo = Ft_total_lo, Ft_total_hi = Ft_total_hi,
       Fw_total = Fw_total, Fw_total_lo = Fw_total_lo, Fw_total_hi = Fw_total_hi,
       available = TRUE)
}, error = function(e) {
  cat(sprintf("[R-Model] NOTE: Ft/Fw extraction failed (%s) -- reporting NA.\n", conditionMessage(e)))
  list(Ft_total = NA_real_, Ft_total_lo = NA_real_, Ft_total_hi = NA_real_,
       Fw_total = NA_real_, Fw_total_lo = NA_real_, Fw_total_hi = NA_real_,
       available = FALSE)
})

# Per-unit Maximum A Posteriori (MAP) classification
piZi_avg <- Reduce(`+`, lapply(samples, function(ch) ch$piZi)) / length(samples)
map_class <- apply(piZi_avg, 1, which.max)  # 1=No Fraud, 2=Incremental, 3=Extreme

n_no_fraud    <- sum(map_class == 1)
n_incremental <- sum(map_class == 2)
n_extreme     <- sum(map_class == 3)
n_flagged     <- n_incremental + n_extreme

out_summary <- data.frame(
  total_units = nrow(data),
  p_no_fraud_median = pi_median[1],
  p_incremental_median = pi_median[2],
  p_extreme_median = pi_median[3],
  p_no_fraud_hpd95_lo = pi_hpd["pi[1]", "lower"],
  p_no_fraud_hpd95_hi = pi_hpd["pi[1]", "upper"],
  p_incremental_hpd95_lo = pi_hpd["pi[2]", "lower"],
  p_incremental_hpd95_hi = pi_hpd["pi[2]", "upper"],
  p_extreme_hpd95_lo = pi_hpd["pi[3]", "lower"],
  p_extreme_hpd95_hi = pi_hpd["pi[3]", "upper"],
  rhat_pi = rhat,
  M_pi_no_fraud = M_pi[1],
  M_pi_incremental = M_pi[2],
  M_pi_extreme = M_pi[3],
  D_pi_no_fraud = D_pi[1],
  D_pi_incremental = D_pi[2],
  D_pi_extreme = D_pi[3],
  Ft_total = fraud_votes$Ft_total,
  Ft_total_hpd95_lo = fraud_votes$Ft_total_lo,
  Ft_total_hpd95_hi = fraud_votes$Ft_total_hi,
  Fw_total = fraud_votes$Fw_total,
  Fw_total_hpd95_lo = fraud_votes$Fw_total_lo,
  Fw_total_hpd95_hi = fraud_votes$Fw_total_hi,
  fraud_votes_available = fraud_votes$available,             # FALSE means the Ft/Fw fields above are all NA
  fraud_votes_interval_is_conservative_sum = TRUE,            # see the caveat above -- not a true joint HPD interval
  n_regions_flagged = n_flagged,
  n_regions_no_fraud = n_no_fraud,
  n_regions_incremental = n_incremental,
  n_regions_extreme = n_extreme,
  elapsed_seconds = elapsed,
  n_iter = n_iter,
  n_chains = n_chains,
  burn_in = burn_in,
  n_adapt = n_adapt,
  use_parcomp = use_parcomp
)
write.csv(out_summary, output_path, row.names = FALSE)

units_path <- sub("\\.csv$", "_units.csv", output_path)
write.csv(data.frame(unit_index = seq_len(nrow(data)),
                     p_no_fraud = piZi_avg[, 1],
                     p_incremental = piZi_avg[, 2],
                     p_extreme = piZi_avg[, 3],
                     map_class = map_class - 1),
          units_path, row.names = FALSE)

cat(sprintf("[R-Model] Summaries saved to %s and %s\n", output_path, units_path))
