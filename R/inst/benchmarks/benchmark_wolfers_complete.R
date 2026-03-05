# Comprehensive Benchmark: DID Estimators Comparison
# Comparing: R-CRAN (DIDmultiplegtDYN), R-Polars, did (CS), didimputation, fixest (Sun-Abraham)
# Dataset: wolfers2006_didtextbook.dta
# Specification: did_multiplegt_dyn div_rate state year udl, effects(16) placebo(9) weight(stpop)

library(haven)
library(DIDmultiplegtDYN)
library(DIDmultiplegtDYNpolars)
library(did)
library(didimputation)
library(fixest)
library(dplyr)

# Set timeout (1 hour in seconds)
TIMEOUT_SECONDS <- 3600

# Helper function for separator
sep_line <- function() paste(rep("=", 70), collapse = "")

# Output file for logging
log_file <- "/Users/anzony.quisperojas/Documents/GitHub/R_didgt_polars/tests/benchmark_wolfers_complete.log"
sink(log_file, split = TRUE)

cat(sep_line(), "\n")
cat("COMPREHENSIVE BENCHMARK: DID Estimators Comparison\n")
cat(sep_line(), "\n")
cat("Date:", as.character(Sys.time()), "\n")
cat("Packages: DIDmultiplegtDYN (CRAN), DIDmultiplegtDYNpolars, did (CS), didimputation, fixest (SA)\n\n")

# Load original data
cat("Loading data...\n")
wolfers <- as.data.frame(read_dta("/Users/anzony.quisperojas/Documents/GitHub/R_didgt_polars/data/wolfers2006_didtextbook.dta"))
cat("Original data rows:", nrow(wolfers), "\n")

# Prepare data for different packages
# Create first treatment time variable for CS/SA estimators
wolfers <- wolfers %>%
  group_by(state) %>%
  mutate(
    first_treat = ifelse(any(udl == 1), min(year[udl == 1]), 0)
  ) %>%
  ungroup() %>%
  as.data.frame()

cat("Unique states:", length(unique(wolfers$state)), "\n")
cat("Year range:", min(wolfers$year), "-", max(wolfers$year), "\n")
cat("Treatment groups (first_treat):", paste(sort(unique(wolfers$first_treat)), collapse = ", "), "\n\n")

# Function to run benchmark with timeout
run_with_timeout <- function(expr, timeout_sec = TIMEOUT_SECONDS) {
  result <- list(time = NA, output = NULL, status = "error")

  tryCatch({
    start_time <- Sys.time()
    result$output <- eval(expr)
    end_time <- Sys.time()
    result$time <- as.numeric(difftime(end_time, start_time, units = "secs"))
    result$status <- "completed"
  }, error = function(e) {
    result$status <- paste("error:", e$message)
  })

  return(result)
}

# Function to create synthetic data by duplicating groups
create_synthetic_data <- function(df, multiplier) {
  if (multiplier == 1) return(df)

  unique_states <- unique(df$state)
  n_states <- length(unique_states)

  result_list <- list()
  for (i in 1:multiplier) {
    temp_df <- df
    temp_df$state <- temp_df$state + (i - 1) * max(df$state) * 10
    result_list[[i]] <- temp_df
  }

  result <- do.call(rbind, result_list)
  rownames(result) <- NULL
  return(result)
}

# Store all results
results <- data.frame(
  scenario = character(),
  package = character(),
  rows = numeric(),
  time_seconds = numeric(),
  status = character(),
  stringsAsFactors = FALSE
)

# ============================================================
# SCENARIO 1: Original Data (1,683 rows)
# ============================================================
cat("\n", sep_line(), "\n")
cat("SCENARIO 1: Original Data (", nrow(wolfers), " rows)\n")
cat(sep_line(), "\n\n")

# 1. DIDmultiplegtDYN (CRAN)
cat("1. Running DIDmultiplegtDYN (CRAN)...\n")
res_cran <- run_with_timeout(quote({
  DIDmultiplegtDYN::did_multiplegt_dyn(
    df = wolfers,
    outcome = "div_rate",
    group = "state",
    time = "year",
    treatment = "udl",
    effects = 16,
    placebo = 9,
    weight = "stpop"
  )
}))
cat("   Time:", ifelse(res_cran$status == "completed",
                       paste(round(res_cran$time, 2), "seconds"),
                       res_cran$status), "\n")
results <- rbind(results, data.frame(
  scenario = "Original (1.7K)",
  package = "DIDmultiplegtDYN-CRAN",
  rows = nrow(wolfers),
  time_seconds = ifelse(res_cran$status == "completed", res_cran$time, NA),
  status = res_cran$status
))

# 2. DIDmultiplegtDYNpolars
cat("2. Running DIDmultiplegtDYNpolars...\n")
res_polars <- run_with_timeout(quote({
  DIDmultiplegtDYNpolars::did_multiplegt_dyn(
    df = wolfers,
    outcome = "div_rate",
    group = "state",
    time = "year",
    treatment = "udl",
    effects = 16,
    placebo = 9,
    weight = "stpop"
  )
}))
cat("   Time:", ifelse(res_polars$status == "completed",
                       paste(round(res_polars$time, 2), "seconds"),
                       res_polars$status), "\n")
results <- rbind(results, data.frame(
  scenario = "Original (1.7K)",
  package = "DIDmultiplegtDYN-Polars",
  rows = nrow(wolfers),
  time_seconds = ifelse(res_polars$status == "completed", res_polars$time, NA),
  status = res_polars$status
))

# 3. did (Callaway-Sant'Anna)
cat("3. Running did (Callaway-Sant'Anna)...\n")
res_cs <- run_with_timeout(quote({
  att_gt(
    yname = "div_rate",
    tname = "year",
    idname = "state",
    gname = "first_treat",
    data = wolfers,
    weightsname = "stpop",
    control_group = "nevertreated",
    anticipation = 0,
    est_method = "dr",
    base_period = "varying"
  )
}))
cat("   Time:", ifelse(res_cs$status == "completed",
                       paste(round(res_cs$time, 2), "seconds"),
                       res_cs$status), "\n")
results <- rbind(results, data.frame(
  scenario = "Original (1.7K)",
  package = "did-CS",
  rows = nrow(wolfers),
  time_seconds = ifelse(res_cs$status == "completed", res_cs$time, NA),
  status = res_cs$status
))

# 4. didimputation
cat("4. Running didimputation...\n")
res_didimp <- run_with_timeout(quote({
  did_imputation(
    data = wolfers,
    yname = "div_rate",
    gname = "first_treat",
    tname = "year",
    idname = "state",
    wname = "stpop",
    horizon = TRUE,
    pretrends = TRUE
  )
}))
cat("   Time:", ifelse(res_didimp$status == "completed",
                       paste(round(res_didimp$time, 2), "seconds"),
                       res_didimp$status), "\n")
results <- rbind(results, data.frame(
  scenario = "Original (1.7K)",
  package = "didimputation",
  rows = nrow(wolfers),
  time_seconds = ifelse(res_didimp$status == "completed", res_didimp$time, NA),
  status = res_didimp$status
))

# 5. fixest (Sun-Abraham)
cat("5. Running fixest (Sun-Abraham)...\n")
res_sa <- run_with_timeout(quote({
  # Create relative time variable
  wolfers_sa <- wolfers
  wolfers_sa$rel_time <- ifelse(wolfers_sa$first_treat == 0, -1000,
                                 wolfers_sa$year - wolfers_sa$first_treat)

  feols(
    div_rate ~ sunab(first_treat, year, ref.p = -1) | state + year,
    data = wolfers_sa,
    weights = ~stpop,
    vcov = "hetero"
  )
}))
cat("   Time:", ifelse(res_sa$status == "completed",
                       paste(round(res_sa$time, 2), "seconds"),
                       res_sa$status), "\n")
results <- rbind(results, data.frame(
  scenario = "Original (1.7K)",
  package = "fixest-SA",
  rows = nrow(wolfers),
  time_seconds = ifelse(res_sa$status == "completed", res_sa$time, NA),
  status = res_sa$status
))

# ============================================================
# SCENARIO 2: Synthetic Data 100x (168,300 rows)
# ============================================================
cat("\n", sep_line(), "\n")
cat("SCENARIO 2: Synthetic Data 100x\n")
cat(sep_line(), "\n\n")

wolfers_100x <- create_synthetic_data(wolfers, 100)
cat("Synthetic data rows:", nrow(wolfers_100x), "\n\n")

# 1. DIDmultiplegtDYN (CRAN)
cat("1. Running DIDmultiplegtDYN (CRAN)...\n")
res_cran_100x <- run_with_timeout(quote({
  DIDmultiplegtDYN::did_multiplegt_dyn(
    df = wolfers_100x,
    outcome = "div_rate",
    group = "state",
    time = "year",
    treatment = "udl",
    effects = 16,
    placebo = 9,
    weight = "stpop"
  )
}))
cat("   Time:", ifelse(res_cran_100x$status == "completed",
                       paste(round(res_cran_100x$time, 2), "seconds"),
                       res_cran_100x$status), "\n")
results <- rbind(results, data.frame(
  scenario = "100x (168K)",
  package = "DIDmultiplegtDYN-CRAN",
  rows = nrow(wolfers_100x),
  time_seconds = ifelse(res_cran_100x$status == "completed", res_cran_100x$time, NA),
  status = res_cran_100x$status
))

# 2. DIDmultiplegtDYNpolars
cat("2. Running DIDmultiplegtDYNpolars...\n")
res_polars_100x <- run_with_timeout(quote({
  DIDmultiplegtDYNpolars::did_multiplegt_dyn(
    df = wolfers_100x,
    outcome = "div_rate",
    group = "state",
    time = "year",
    treatment = "udl",
    effects = 16,
    placebo = 9,
    weight = "stpop"
  )
}))
cat("   Time:", ifelse(res_polars_100x$status == "completed",
                       paste(round(res_polars_100x$time, 2), "seconds"),
                       res_polars_100x$status), "\n")
results <- rbind(results, data.frame(
  scenario = "100x (168K)",
  package = "DIDmultiplegtDYN-Polars",
  rows = nrow(wolfers_100x),
  time_seconds = ifelse(res_polars_100x$status == "completed", res_polars_100x$time, NA),
  status = res_polars_100x$status
))

# 3. did (Callaway-Sant'Anna)
cat("3. Running did (Callaway-Sant'Anna)...\n")
res_cs_100x <- run_with_timeout(quote({
  att_gt(
    yname = "div_rate",
    tname = "year",
    idname = "state",
    gname = "first_treat",
    data = wolfers_100x,
    weightsname = "stpop",
    control_group = "nevertreated",
    anticipation = 0,
    est_method = "dr",
    base_period = "varying"
  )
}))
cat("   Time:", ifelse(res_cs_100x$status == "completed",
                       paste(round(res_cs_100x$time, 2), "seconds"),
                       res_cs_100x$status), "\n")
results <- rbind(results, data.frame(
  scenario = "100x (168K)",
  package = "did-CS",
  rows = nrow(wolfers_100x),
  time_seconds = ifelse(res_cs_100x$status == "completed", res_cs_100x$time, NA),
  status = res_cs_100x$status
))

# 4. didimputation
cat("4. Running didimputation...\n")
res_didimp_100x <- run_with_timeout(quote({
  did_imputation(
    data = wolfers_100x,
    yname = "div_rate",
    gname = "first_treat",
    tname = "year",
    idname = "state",
    wname = "stpop",
    horizon = TRUE,
    pretrends = TRUE
  )
}))
cat("   Time:", ifelse(res_didimp_100x$status == "completed",
                       paste(round(res_didimp_100x$time, 2), "seconds"),
                       res_didimp_100x$status), "\n")
results <- rbind(results, data.frame(
  scenario = "100x (168K)",
  package = "didimputation",
  rows = nrow(wolfers_100x),
  time_seconds = ifelse(res_didimp_100x$status == "completed", res_didimp_100x$time, NA),
  status = res_didimp_100x$status
))

# 5. fixest (Sun-Abraham)
cat("5. Running fixest (Sun-Abraham)...\n")
res_sa_100x <- run_with_timeout(quote({
  wolfers_sa_100x <- wolfers_100x
  wolfers_sa_100x$rel_time <- ifelse(wolfers_sa_100x$first_treat == 0, -1000,
                                      wolfers_sa_100x$year - wolfers_sa_100x$first_treat)

  feols(
    div_rate ~ sunab(first_treat, year, ref.p = -1) | state + year,
    data = wolfers_sa_100x,
    weights = ~stpop,
    vcov = "hetero"
  )
}))
cat("   Time:", ifelse(res_sa_100x$status == "completed",
                       paste(round(res_sa_100x$time, 2), "seconds"),
                       res_sa_100x$status), "\n")
results <- rbind(results, data.frame(
  scenario = "100x (168K)",
  package = "fixest-SA",
  rows = nrow(wolfers_100x),
  time_seconds = ifelse(res_sa_100x$status == "completed", res_sa_100x$time, NA),
  status = res_sa_100x$status
))

# Clean up
rm(wolfers_100x)
gc()

# ============================================================
# SCENARIO 3: Synthetic Data 1000x (1,683,000 rows)
# ============================================================
cat("\n", sep_line(), "\n")
cat("SCENARIO 3: Synthetic Data 1000x\n")
cat(sep_line(), "\n\n")

wolfers_1000x <- create_synthetic_data(wolfers, 1000)
cat("Synthetic data rows:", nrow(wolfers_1000x), "\n\n")

# 1. DIDmultiplegtDYNpolars (expected fastest)
cat("1. Running DIDmultiplegtDYNpolars...\n")
res_polars_1000x <- run_with_timeout(quote({
  DIDmultiplegtDYNpolars::did_multiplegt_dyn(
    df = wolfers_1000x,
    outcome = "div_rate",
    group = "state",
    time = "year",
    treatment = "udl",
    effects = 16,
    placebo = 9,
    weight = "stpop"
  )
}))
cat("   Time:", ifelse(res_polars_1000x$status == "completed",
                       paste(round(res_polars_1000x$time, 2), "seconds"),
                       res_polars_1000x$status), "\n")
results <- rbind(results, data.frame(
  scenario = "1000x (1.68M)",
  package = "DIDmultiplegtDYN-Polars",
  rows = nrow(wolfers_1000x),
  time_seconds = ifelse(res_polars_1000x$status == "completed", res_polars_1000x$time, NA),
  status = res_polars_1000x$status
))

# 2. fixest (Sun-Abraham) - usually fast
cat("2. Running fixest (Sun-Abraham)...\n")
res_sa_1000x <- run_with_timeout(quote({
  wolfers_sa_1000x <- wolfers_1000x
  wolfers_sa_1000x$rel_time <- ifelse(wolfers_sa_1000x$first_treat == 0, -1000,
                                       wolfers_sa_1000x$year - wolfers_sa_1000x$first_treat)

  feols(
    div_rate ~ sunab(first_treat, year, ref.p = -1) | state + year,
    data = wolfers_sa_1000x,
    weights = ~stpop,
    vcov = "hetero"
  )
}))
cat("   Time:", ifelse(res_sa_1000x$status == "completed",
                       paste(round(res_sa_1000x$time, 2), "seconds"),
                       res_sa_1000x$status), "\n")
results <- rbind(results, data.frame(
  scenario = "1000x (1.68M)",
  package = "fixest-SA",
  rows = nrow(wolfers_1000x),
  time_seconds = ifelse(res_sa_1000x$status == "completed", res_sa_1000x$time, NA),
  status = res_sa_1000x$status
))

# 3. did (Callaway-Sant'Anna)
cat("3. Running did (Callaway-Sant'Anna)...\n")
res_cs_1000x <- run_with_timeout(quote({
  att_gt(
    yname = "div_rate",
    tname = "year",
    idname = "state",
    gname = "first_treat",
    data = wolfers_1000x,
    weightsname = "stpop",
    control_group = "nevertreated",
    anticipation = 0,
    est_method = "dr",
    base_period = "varying"
  )
}), timeout_sec = 600)  # 10 min timeout
cat("   Time:", ifelse(res_cs_1000x$status == "completed",
                       paste(round(res_cs_1000x$time, 2), "seconds"),
                       res_cs_1000x$status), "\n")
results <- rbind(results, data.frame(
  scenario = "1000x (1.68M)",
  package = "did-CS",
  rows = nrow(wolfers_1000x),
  time_seconds = ifelse(res_cs_1000x$status == "completed", res_cs_1000x$time, NA),
  status = res_cs_1000x$status
))

# 4. didimputation
cat("4. Running didimputation...\n")
res_didimp_1000x <- run_with_timeout(quote({
  did_imputation(
    data = wolfers_1000x,
    yname = "div_rate",
    gname = "first_treat",
    tname = "year",
    idname = "state",
    wname = "stpop",
    horizon = TRUE,
    pretrends = TRUE
  )
}), timeout_sec = 600)  # 10 min timeout
cat("   Time:", ifelse(res_didimp_1000x$status == "completed",
                       paste(round(res_didimp_1000x$time, 2), "seconds"),
                       res_didimp_1000x$status), "\n")
results <- rbind(results, data.frame(
  scenario = "1000x (1.68M)",
  package = "didimputation",
  rows = nrow(wolfers_1000x),
  time_seconds = ifelse(res_didimp_1000x$status == "completed", res_didimp_1000x$time, NA),
  status = res_didimp_1000x$status
))

# 5. DIDmultiplegtDYN (CRAN) - likely to fail on memory
cat("5. Running DIDmultiplegtDYN (CRAN)...\n")
polars_time_1000x <- ifelse(res_polars_1000x$status == "completed", res_polars_1000x$time, 300)
cran_timeout <- min(polars_time_1000x * 3, 600)
cat("   (timeout set to", round(cran_timeout, 0), "seconds)\n")

res_cran_1000x <- run_with_timeout(quote({
  DIDmultiplegtDYN::did_multiplegt_dyn(
    df = wolfers_1000x,
    outcome = "div_rate",
    group = "state",
    time = "year",
    treatment = "udl",
    effects = 16,
    placebo = 9,
    weight = "stpop"
  )
}), timeout_sec = cran_timeout)
cat("   Time:", ifelse(res_cran_1000x$status == "completed",
                       paste(round(res_cran_1000x$time, 2), "seconds"),
                       res_cran_1000x$status), "\n")
results <- rbind(results, data.frame(
  scenario = "1000x (1.68M)",
  package = "DIDmultiplegtDYN-CRAN",
  rows = nrow(wolfers_1000x),
  time_seconds = ifelse(res_cran_1000x$status == "completed", res_cran_1000x$time, NA),
  status = res_cran_1000x$status
))

# Clean up
rm(wolfers_1000x)
gc()

# ============================================================
# SUMMARY
# ============================================================
cat("\n", sep_line(), "\n")
cat("SUMMARY OF RESULTS\n")
cat(sep_line(), "\n\n")

print(results)

# Create pivot table for easier comparison
cat("\n\nPIVOT TABLE (Time in seconds):\n")
cat(sep_line(), "\n\n")

pivot_results <- reshape(results[, c("scenario", "package", "time_seconds")],
                         idvar = "package",
                         timevar = "scenario",
                         direction = "wide")
names(pivot_results) <- gsub("time_seconds.", "", names(pivot_results))
print(pivot_results)

# Save results to CSV
write.csv(results, "/Users/anzony.quisperojas/Documents/GitHub/R_didgt_polars/tests/benchmark_results_complete.csv", row.names = FALSE)

cat("\n\nBenchmark completed at:", as.character(Sys.time()), "\n")
sink()

cat("Log saved to:", log_file, "\n")
cat("Results saved to: /Users/anzony.quisperojas/Documents/GitHub/R_didgt_polars/tests/benchmark_results_complete.csv\n")
