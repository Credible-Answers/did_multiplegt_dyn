# did_multiplegt_main.R — pure data.table backend

#' Mimic Stata's invsym() function
#'
#' Stata's invsym() uses Cholesky decomposition for symmetric positive-definite
#' matrices. For singular matrices, it returns a generalized inverse where
#' rows/columns corresponding to zero pivots are set to zero.
#'
#' @param M A symmetric matrix
#' @param tol Tolerance for detecting singularity
#' @return The inverse (or generalized inverse) of M
#' @noRd
invsym_r <- function(M, tol = 1e-14) {
  n <- nrow(M)

  # Try Cholesky decomposition with pivoting (like Stata)
  ch <- tryCatch({
    chol(M, pivot = TRUE)
  }, error = function(e) NULL)

  if (!is.null(ch)) {
    pivot <- attr(ch, "pivot")
    rank <- attr(ch, "rank")

    if (is.null(rank)) {
      # No rank attribute means full rank
      rank <- n
    }

    if (rank == n) {
      # Full rank - use chol2inv (equivalent to Stata invsym for non-singular)
      # Reorder back to original order
      oo <- order(pivot)
      return(chol2inv(ch)[oo, oo, drop = FALSE])
    } else {
      # Singular matrix - zero out rows/cols for dropped pivots (like Stata)
      inv <- matrix(0, n, n)
      if (rank > 0) {
        R <- ch[1:rank, 1:rank, drop = FALSE]
        inv_sub <- chol2inv(R)
        idx <- pivot[1:rank]
        inv[idx, idx] <- inv_sub
      }
      return(inv)
    }
  } else {
    # Cholesky failed - try solve() as fallback
    return(tryCatch({
      solve(M)
    }, error = function(e) {
      warning("invsym_r: Matrix inversion failed, using ginv as fallback")
      MASS::ginv(M)
    }))
  }
}

#' Internal function of did_multiplegt_dyn
#' @param df df
#' @param outcome outcome
#' @param group group
#' @param time time
#' @param treatment treatment
#' @param effects effects
#' @param placebo placebo
#' @param ci_level ci_level
#' @param switchers switchers
#' @param only_never_switchers only_never_switchers
#' @param trends_nonparam trends_nonparam
#' @param weight weight
#' @param controls controls
#' @param dont_drop_larger_lower dont_drop_larger_lower
#' @param drop_if_d_miss_before_first_switch drop_if_d_miss_before_first_switch
#' @param cluster cluster
#' @param same_switchers same_switchers
#' @param same_switchers_pl same_switchers_pl
#' @param effects_equal effects_equal
#' @param effects_equal_lb effects_equal lower bound (for range specification)
#' @param effects_equal_ub effects_equal upper bound (for range specification)
#' @param save_results save_results
#' @param normalized normalized
#' @param predict_het predict_het
#' @param predict_het_hc2bm predict_het_hc2bm
#' @param trends_lin trends_lin
#' @param less_conservative_se less_conservative_se
#' @param continuous continuous
#' @param data_only data_only
#' @param point_estimates_only If TRUE, skip variance-covariance, F-tests, and
#'   heterogeneity analysis. Used by bootstrap iterations that only need point
#'   estimates.
#' @note Uses data.table backend

#' @returns A list with the final estimation dataframe and other relevant matrices and scalars.
#' @noRd
did_multiplegt_main <- function(
  df,
  outcome,
  group,
  time,
  treatment,
  effects,
  placebo,
  ci_level,
  switchers,
  only_never_switchers,
  trends_nonparam,
  weight,
  controls,
  dont_drop_larger_lower,
  drop_if_d_miss_before_first_switch,
  cluster,
  same_switchers,
  same_switchers_pl,
  effects_equal,
  effects_equal_lb = NULL,
  effects_equal_ub = NULL,
  save_results,
  normalized,
  predict_het,
  predict_het_hc2bm = FALSE,
  trends_lin,
  less_conservative_se,
  continuous,
  data_only = FALSE,
  point_estimates_only = FALSE
  ) {

# Pure data.table backend

suppressWarnings({

  ###### Initialize warnings collector for vcov invertibility issues
  vcov_warnings <- c()

  ######## 1. Checking that syntax correctly specified
  #### Add a stop message: same_switchers_pl only works when same_switchers is specified.
  if (same_switchers == FALSE && same_switchers_pl == TRUE) {
    stop("The same_switchers_pl option only works if same_switchers is specified as well!")
  }


  #### Continous option: checking that polynomial order specified, and putting it into degree_pol scalar.
  if (!is.null(continuous)) {
    degree_pol <- continuous
  }

  ######## 2. Data preparation steps
  #### Renaming the variables in the dataset
  cols <- c(outcome, group, time, treatment, trends_nonparam, weight, controls, cluster, unlist(predict_het[1L]))
  df <- data.table::as.data.table(df)[, cols, with = FALSE]
  data.table::setnames(df, c(outcome, group, time, treatment), c("outcome", "group", "time", "treatment"))

  #### Grouping together trends_nonparam variables
  #if (!is.null(trends_nonparam)) {
  #  df$trends_nonparam_XX <- df[trends_nonparam]
  #}

  #### Patching the cluster variable: by default, the command clusters at group level. If the user specifies clustering by group, the clustering option goes to NULL.
  if (!is.null(cluster)) {
    if (paste0(cluster) == paste0(group)) {
      cluster <- NULL
    } else{
      df[, cluster_XX := get(cluster)]
    }
  }

  #### Selecting the sample
  ## Dropping observations with missing group or time
  df <- stats::na.omit(df, cols = c("group", "time"))
  ## Dropping observations with missing controls
  if (!is.null(controls) && length(controls) > 0L) {
    df <- stats::na.omit(df, cols = controls)
  }

  #### Further sample selection steps
  ## Dropping observations with a missing clustering variable
  if (!is.null(cluster)) {
    df <- stats::na.omit(df, cols = "cluster_XX")
  }

  ## Dropping groups with always missing treatment or outcomes
  df[, mean_D := mean(treatment, na.rm = TRUE), by = "group"]
  df[, mean_Y := mean(outcome, na.rm = TRUE), by = "group"]
  df <- stats::na.omit(df, cols = c("mean_Y", "mean_D"))
  df[, c("mean_Y", "mean_D") := NULL]

  #### Predict_het option for heterogeneous treatment effects analysis
  predict_het_good <- c()
  if (!is.null(predict_het)) {
    if (length(predict_het) != 2L && inherits(predict_het, "list")) {
      stop("Syntax error in predict_het option: list with 2 elements required. Set the second element to -1 to include all the effects.")
    }
    ## Checks if predict_het and normalized are both specified
    if (isTRUE(normalized)) {
      message("The options normalized and predict_het cannot be specified together. The option predict_het will be ignored.")
    } else {
      pred_het <- unlist(predict_het[1])
      het_effects <- unlist(predict_het[2])
      ## Checks if only time-invariant variables are specified in predict_het
      predict_het_good <- c(predict_het_good, Filter(function(v) {
        df[, sd_het := {
          s <- stats::sd(get(v), na.rm = TRUE)
          data.table::fifelse(is.na(s), 0, s)
        }, by = "group"]
        sd_het_mean <- mean(df[["sd_het"]], na.rm = TRUE)
        df[, sd_het := NULL]
        if (!is.na(sd_het_mean) && sd_het_mean != 0) {
          message(sprintf("The variable %s specified in the option predict_het is time-varying, the command will therefore ignore it.", v))
          FALSE
        } else {
          TRUE
        }
      }, pred_het))
    }
  }

  #### Collapse and weight
  ## Creating the weight variable
  if (is.null(weight)) {
    df[, weight_XX := 1]
  } else {
    df[, weight_XX := get(weight)]
  }
  df[, weight_XX := data.table::fifelse(is.na(weight_XX), 0, weight_XX)]

  ## Checking if the data has to be collapsed
  max_group_time_count <- df[, .N, by = .(group, time)][, max(N)]
  aggregated_data <- max_group_time_count == 1

  ## Collapsing the data if necessary
  if (aggregated_data != 1) {
    df[, weight_XX := data.table::fifelse(is.na(treatment), 0, weight_XX)]
    if (is.null(cluster)) {
      df[, cluster_XX := 1]
    }

    # weighted mean: sum(col * w) / sum(w) by group, time
    agg_cols <- c("treatment", "outcome", trends_nonparam, weight, controls, predict_het_good, "cluster_XX", cluster)
    agg_cols <- unique(agg_cols[agg_cols %chin% names(df)])

    # Build aggregation expression
    agg_exprs <- lapply(agg_cols, function(col) {
      call("/", call("sum", call("*", as.name(col), as.name("weight_XX")), na.rm = TRUE),
                call("sum", as.name("weight_XX"), na.rm = TRUE))
    })
    names(agg_exprs) <- agg_cols
    agg_exprs[["weight_XX"]] <- quote(sum(weight_XX, na.rm = TRUE))
    df <- df[, c(agg_exprs), by = .(group, time)]

    if (is.null(cluster)) {
      df[, cluster_XX := NULL]
    }
  }

  ## --- Generate factorized versions of Y, G, T and D ---
  outcome <- "outcome"
  group <- "group"
  time <- "time"
  treatment <- "treatment"

  # outcome_XX = outcome
  df[, outcome_XX := outcome]

  # sort by time
  data.table::setorder(df, time)

  # group_XX and time_XX as "factorized" (1,2,3,...) in order of appearance
  df[, group_XX := data.table::frank(group, ties.method = "dense")]
  df[, time_XX := data.table::frank(time, ties.method = "dense")]
  df[, treatment_XX := treatment]

  # first/last date where D not missing
  df[, time_d_nonmiss_XX := data.table::fifelse(!is.na(treatment_XX), time_XX, NA_real_)]
  df[, time_y_nonmiss_XX := data.table::fifelse(!is.na(outcome_XX), time_XX, NA_real_)]

  # per-group mins & max
  df[, min_time_d_nonmiss_XX := min(time_d_nonmiss_XX, na.rm = TRUE), by = "group_XX"]
  df[, max_time_d_nonmiss_XX := max(time_d_nonmiss_XX, na.rm = TRUE), by = "group_XX"]
  df[, min_time_y_nonmiss_XX := min(time_y_nonmiss_XX, na.rm = TRUE), by = "group_XX"]
  # fix infinite values from groups with all NA
  df[is.infinite(min_time_d_nonmiss_XX), min_time_d_nonmiss_XX := NA_real_]
  df[is.infinite(max_time_d_nonmiss_XX), max_time_d_nonmiss_XX := NA_real_]
  df[is.infinite(min_time_y_nonmiss_XX), min_time_y_nonmiss_XX := NA_real_]

  # first date D missing *after* Y seen
  df[, time_d_miss_XX := data.table::fifelse(is.na(treatment_XX) & (time_XX >= min_time_y_nonmiss_XX), time_XX, NA_real_)]

  # per-group min of time_d_miss_XX
  df[, min_time_d_miss_aft_ynm_XX := min(time_d_miss_XX, na.rm = TRUE), by = "group_XX"]
  df[is.infinite(min_time_d_miss_aft_ynm_XX), min_time_d_miss_aft_ynm_XX := NA_real_]

  # drop intermediate cols
  df[, c("time_d_nonmiss_XX", "time_y_nonmiss_XX", "time_d_miss_XX") := NULL]

  ## --- Baseline treatment D_{g,1} ---

  # d_sq_temp_XX = treatment_XX at min_time_d_nonmiss_XX
  df[, d_sq_temp_XX := data.table::fifelse(time_XX == min_time_d_nonmiss_XX, treatment_XX, NA_real_)]

  # d_sq_XX = group mean of that (only one non-NA per group, so it's the baseline)
  df[, d_sq_XX := mean(d_sq_temp_XX, na.rm = TRUE), by = "group_XX"]

  # drop temp
  df[, d_sq_temp_XX := NULL]

  ## --- Enforce "Design Restriction 2" ---

  df[, diff_from_sq_XX := treatment_XX - d_sq_XX]

  # sort by group_XX, time_XX
  data.table::setorder(df, group_XX, time_XX)

  # T_XX = max of time_XX
  T_XX <- as.integer(max(df[["time_XX"]], na.rm = TRUE))

  if (!(dont_drop_larger_lower == TRUE)) {
    # Sort by group_XX and time_XX
    data.table::setorder(df, group_XX, time_XX)

    # 2. strict increase: ever_strict_increase_XX is 1 if it ever happens within group_XX
    df[, ever_strict_increase_XX := pmin(cumsum(as.integer(diff_from_sq_XX > 0 & !is.na(treatment_XX))), 1L), by = "group_XX"]

    # 3. strict decrease: ever_strict_decrease_XX is 1 if it ever happens within group_XX
    df[, ever_strict_decrease_XX := pmin(cumsum(as.integer(diff_from_sq_XX < 0 & !is.na(treatment_XX))), 1L), by = "group_XX"]

    # 4. drop rows where both == 1
    df <- df[!(ever_strict_increase_XX == 1 & ever_strict_decrease_XX == 1)]
  }

  #### Counting number of groups
  # G_XX <- max(df$group_XX, na.rm = TRUE)

  #### Ever changed treatment
  df[, ever_change_d_XX := as.integer(abs(diff_from_sq_XX) > 0 & !is.na(treatment_XX))]
  # Use cummax over group
  data.table::setorder(df, group_XX, time_XX)
  df[, ever_change_d_XX := cummax(ever_change_d_XX), by = "group_XX"]

  #### Creating date of the first treatment change
  df[, temp_F_g_XX := data.table::fifelse(
    ever_change_d_XX == 1 & data.table::shift(ever_change_d_XX, 1, fill = 0) == 0,
    time_XX, 0
  ), by = "group_XX"]
  df[, F_g_XX := max(temp_F_g_XX, na.rm = TRUE), by = "group_XX"]
  df[, temp_F_g_XX := NULL]

  #### If continuous option specified, generating polynomials of D_{g,1},
  #### storing D_{g,1} somewhere, and replacing it by 0.
  if (!is.null(continuous)) {
    col_names <- paste0("d_sq_", 1:degree_pol, "_XX")
    lapply(seq_len(degree_pol), function(pol_level) {
      df[, (col_names[pol_level]) := d_sq_XX^pol_level]
    })
    df[, d_sq_XX_orig := d_sq_XX]
    df[, d_sq_XX := 0]
  }

  ## Creating a new value with integer levels of d_sq_XX
  df[, d_sq_int_XX := as.numeric(data.table::frank(d_sq_XX, ties.method = "dense"))]

  #### Dropping values of baseline treatment such that there is no variance in F_g within
  by_cols_var <- c("d_sq_XX", trends_nonparam)
  by_cols_var <- by_cols_var[by_cols_var != "" & !is.na(by_cols_var) & nchar(by_cols_var) > 0L]

  df[, var_F_g_XX := stats::sd(F_g_XX, na.rm = TRUE), by = by_cols_var]
  df <- df[var_F_g_XX > 0]
  df[, var_F_g_XX := NULL]

  #### Counting number of groups
  G_XX <- data.table::uniqueN(df[["group_XX"]])

  if (nrow(df) == 0L) {
    stop("No treatment effect can be estimated.\n  This is because Design Restriction 1 in de Chaisemartin & D'Haultfoeuille (2024) is not satisfied in the data, given the options requested.\n  This may be due to the fact that groups' period-one treatment is continuous, or takes a large number of values, and you have not specified the continuous option.\n  If so, you can try to specify this option.\n  If the issue persists even with this option, this means that all groups experience their first treatment change at the same date.\n  In this situation, estimators of de Chaisemartin & D'Haultfoeuille (2024) cannot be used.")
  }

  #### For each value of d_sq_XX, we drop time periods such that we do not have any control with the same baseline treatment afterwards
  df[, never_change_d_XX := 1L - ever_change_d_XX]
  by_cols_ctrl <- c("time_XX", "d_sq_XX", trends_nonparam)
  by_cols_ctrl <- by_cols_ctrl[by_cols_ctrl != "" & !is.na(by_cols_ctrl) & nchar(by_cols_ctrl) > 0L]
  df[, controls_time_XX := max(never_change_d_XX, na.rm = TRUE), by = by_cols_ctrl]
  df <- df[controls_time_XX > 0]

  #### Computing t_min, T_max and adjusting F_g by last period plus one for those that never change treatment
  t_min_XX <- min(df[["time_XX"]], na.rm = TRUE)
  T_max_XX <- max(df[["time_XX"]], na.rm = TRUE)
  df[, F_g_XX := data.table::fifelse(F_g_XX == 0, T_max_XX + 1, F_g_XX)]



  ######## Dealing with missing treatments: most conservative option
  #### Let FMD_g denote the first date when g's treatment is missing while y has been not missing at least once, so that we know for sure that g already exists. 
  #### If that date is before the first period when g's treatment changes, we do not know when g's treatment has changed for the first time. Then, a conservative option is to drop all of g's outcomes starting at FMD_g.

  if (drop_if_d_miss_before_first_switch == TRUE) {
    df[, outcome_XX := data.table::fifelse(
      !is.na(min_time_d_miss_aft_ynm_XX) &
      (min_time_d_miss_aft_ynm_XX < F_g_XX) &
      (time_XX >= min_time_d_miss_aft_ynm_XX),
      NA_real_, outcome_XX
    )]
  }

  ######## Dealing with missing treatments: most liberal option
  df[, last_obs_D_bef_switch_t_XX := data.table::fifelse(
    (time_XX < F_g_XX) & !is.na(treatment_XX), time_XX, NA_real_
  )]
  df[, last_obs_D_bef_switch_XX := max(last_obs_D_bef_switch_t_XX, na.rm = TRUE), by = "group_XX"]
  df[is.infinite(last_obs_D_bef_switch_XX), last_obs_D_bef_switch_XX := NA_real_]

  #### For t<FD_g, outcome set to NA
  df[, outcome_XX := data.table::fifelse(time_XX < min_time_d_nonmiss_XX, NA_real_, outcome_XX)]

  #### Replace missing treatment by status-quo
  df[, treatment_XX := data.table::fifelse(
    (F_g_XX < (T_max_XX + 1)) &
    is.na(treatment_XX) &
    (time_XX < last_obs_D_bef_switch_XX) &
    (time_XX > min_time_d_nonmiss_XX),
    d_sq_XX, treatment_XX
  )]

  #### Set outcomes to NA for uncertain switch dates
  df[, outcome_XX := data.table::fifelse(
    (F_g_XX < (T_max_XX + 1)) &
    (time_XX > last_obs_D_bef_switch_XX) &
    (last_obs_D_bef_switch_XX < (F_g_XX - 1)),
    NA_real_, outcome_XX
  )]
  df[, trunc_control_XX := data.table::fifelse(
    (F_g_XX < (T_max_XX + 1)) &
    (last_obs_D_bef_switch_XX < (F_g_XX - 1)),
    last_obs_D_bef_switch_XX + 1, NA_real_
  )]
  df[, F_g_XX := data.table::fifelse(
    (F_g_XX < (T_max_XX + 1)) &
    (last_obs_D_bef_switch_XX < (F_g_XX - 1)),
    T_max_XX + 1, F_g_XX
  )]

  #### Replace missing treatment after F_g by D(g,F_g)
  df[, d_F_g_temp_XX := data.table::fifelse(time_XX == F_g_XX, treatment_XX, NA_real_)]
  df[, d_F_g_XX := mean(d_F_g_temp_XX, na.rm = TRUE), by = "group_XX"]
  df[, treatment_XX := data.table::fifelse(
    (F_g_XX < (T_max_XX + 1)) &
    is.na(treatment_XX) &
    (time_XX > F_g_XX) &
    (last_obs_D_bef_switch_XX == (F_g_XX - 1)),
    d_F_g_XX, treatment_XX
  )]

  #### For never-switchers, replace missing treatment by D_g1
  df[, treatment_XX := data.table::fifelse(
    (F_g_XX == (T_max_XX + 1)) &
    is.na(treatment_XX) &
    (time_XX > min_time_d_nonmiss_XX) &
    (time_XX < max_time_d_nonmiss_XX),
    d_sq_XX, treatment_XX
  )]

  #### For never-switchers, outcomes missing at t>LD_g
  df[, outcome_XX := data.table::fifelse(
    (F_g_XX == (T_max_XX + 1)) &
    (time_XX > max_time_d_nonmiss_XX),
    NA_real_, outcome_XX
  )]
  df[, trunc_control_XX := data.table::fifelse(
    F_g_XX == (T_max_XX + 1),
    max_time_d_nonmiss_XX + 1, trunc_control_XX
  )]

  #### Store the outcome in levels for predict_het
  if (!is.null(predict_het)) {
    if (length(predict_het_good) > 0L) {
      df[, outcome_non_diff_XX := outcome_XX]
    }
  }

  #### When trends_lin specified, first difference outcome and controls
  if (isTRUE(trends_lin)) {
    df <- df[F_g_XX != 2]
    data.table::setorder(df, group_XX, time_XX)

    df[, outcome_XX := outcome_XX - data.table::shift(outcome_XX, 1), by = "group_XX"]
    if (!is.null(controls) && length(controls) > 0L) {
      lapply(controls, function(v) {
        df[, (v) := get(v) - data.table::shift(get(v), 1), by = "group_XX"]
      })
    }
    df <- df[time_XX != 1]
    t_min_XX <- min(df[["time_XX"]], na.rm = TRUE)
  }

  #### Balancing the panel using data.table cross join
  unique_groups <- unique(df[["group_XX"]])
  unique_times <- unique(df[["time_XX"]])
  grid <- data.table::CJ(group_XX = unique_groups, time_XX = unique_times)
  df <- merge(grid, df, by = c("group_XX", "time_XX"), all.x = TRUE)
  data.table::setDT(df)

  df[, d_sq_XX := mean(d_sq_XX, na.rm = TRUE), by = "group_XX"]
  df[, d_sq_int_XX := mean(d_sq_int_XX, na.rm = TRUE), by = "group_XX"]
  df[, F_g_XX := mean(F_g_XX, na.rm = TRUE), by = "group_XX"]

  #### Defining N_gt
  df[, N_gt_XX := 1]
  df[, N_gt_XX := data.table::fifelse(is.na(outcome_XX) | is.na(treatment_XX), 0, weight_XX * N_gt_XX)]

  #### Determining last period where g still has a control group
  df[, F_g_trunc_XX := data.table::fifelse(F_g_XX < trunc_control_XX, F_g_XX, trunc_control_XX)]
  df[, F_g_trunc_XX := data.table::fifelse(is.na(trunc_control_XX), F_g_XX, F_g_trunc_XX)]
  df[, F_g_trunc_XX := data.table::fifelse(is.na(F_g_XX), trunc_control_XX, F_g_trunc_XX)]

  by_cols_tg <- c("d_sq_XX", trends_nonparam)
  by_cols_tg <- by_cols_tg[by_cols_tg != "" & !is.na(by_cols_tg) & nchar(by_cols_tg) > 0L]
  df[, T_g_XX := max(F_g_trunc_XX, na.rm = TRUE), by = by_cols_tg]
  df[is.infinite(T_g_XX), T_g_XX := NA_real_]
  df[, T_g_XX := T_g_XX - 1]

  #### Defining S_g: 
  #### an indicator variable for groups whose average post switch 
  #### treatment value is larger than their initial treatment D_{g,1}. 
  #### They will be considered switchers in. If S_g==0, the group is a switcher out. 
  #### For never-switchers, S_g is undefined.
  #### Definition of S_g matches that in paper, unless dont_drop_larger_lower specified.

  # treatment_XX_v1: treatment in post-switch period only
  df[, treatment_XX_v1 := data.table::fifelse(time_XX >= F_g_XX & time_XX <= T_g_XX, treatment_XX, NA_real_)]

  # avg_post_switch_treat_XX_temp: treatment value in post-switch period
  df[, avg_post_switch_treat_XX_temp := data.table::fifelse(time_XX >= F_g_XX & time_XX <= T_g_XX, treatment_XX, NA_real_)]

  # Count of non-missing treatment observations in the post-switch period
  df[, count_time_post_switch_XX_temp := data.table::fifelse(
    time_XX >= F_g_XX & time_XX <= T_g_XX & !is.na(treatment_XX), 1L, 0L
  )]

  # Sum within group
  df[, avg_post_switch_treat_XX_temp := sum(avg_post_switch_treat_XX_temp, na.rm = TRUE), by = "group_XX"]
  df[, count_time_post_switch_XX := sum(count_time_post_switch_XX_temp, na.rm = TRUE), by = "group_XX"]

  # Divide sum by count to get the group-specific average
  df[, avg_post_switch_treat_XX_temp := avg_post_switch_treat_XX_temp / count_time_post_switch_XX]

  # Get the mean of that average across group
  df[, avg_post_switch_treat_XX := mean(avg_post_switch_treat_XX_temp, na.rm = TRUE), by = "group_XX"]

  # Drop temporary columns
  df[, c("treatment_XX_v1", "avg_post_switch_treat_XX_temp", "count_time_post_switch_XX_temp") := NULL]

  #### When a group is a switching group, but its average post-treatment treatment
  #### value is exactly equal to its baseline treatment, we cannnot classify it as
  #### a swicher in or a switcher out, but it is not a control either.
  #### As such, we drop it from the estimation. Those groups are referred to
  #### as no-first-stage-switchers. This issue can only arise
  #### if dont_drop_larger_lower specified.
  #### if continuous is specified we do this according to the original
  #### baseline treatment and not to the one set to 0 to correctly
  #### track if a group is switcher in or switcher out.

  if (is.null(continuous)) {
    # Filter out no-first-stage-switchers
    df <- df[!(
      (avg_post_switch_treat_XX == d_sq_XX) &
      !is.na(avg_post_switch_treat_XX) &
      (F_g_XX != (T_g_XX + 1)) &
      !is.na(F_g_XX) &
      !is.na(T_g_XX)
    )]
    # S_g_XX: 1 if switcher in, 0 if switcher out
    df[, S_g_XX := as.numeric(avg_post_switch_treat_XX > d_sq_XX)]
    df[, S_g_XX := data.table::fifelse(F_g_XX != (T_max_XX + 1), S_g_XX, NA_real_)]
  } else {
    # Filter using d_sq_XX_orig for continuous case
    df <- df[!(
      (avg_post_switch_treat_XX == d_sq_XX_orig) &
      !is.na(avg_post_switch_treat_XX) &
      (F_g_XX != (T_g_XX + 1)) &
      !is.na(F_g_XX) &
      !is.na(T_g_XX)
    )]
    df[, S_g_XX := as.numeric(avg_post_switch_treat_XX > d_sq_XX_orig)]
    df[, S_g_XX := data.table::fifelse(F_g_XX != (T_max_XX + 1), S_g_XX, NA_real_)]
  }

  #### Define another version where S_g=-1 for switchers out, which we need
  #### when predict_het or continuous specified.
  if (length(predict_het) > 0L || !is.null(continuous)) {
    df[, S_g_het_XX := data.table::fifelse(S_g_XX == 0, -1, S_g_XX)]
  }

  #### If continuous option specified: binarizing and staggerizing treatment,
  #### and adding time_FEs interacted with D_{g,1} as controls
  if (!is.null(continuous)) {
    ## Binarizing and staggerizing treatment
    df[, treatment_temp_XX := data.table::fifelse(
      !is.na(S_g_het_XX),
      as.numeric(F_g_XX <= time_XX) * S_g_het_XX,
      NA_real_
    )]
    df[, treatment_XX_orig := treatment_XX]
    df[, treatment_XX := treatment_temp_XX]
    ## Enriching controls
    time_fe_XX <- sort(unique(df[["time_XX"]]))
    fe_grid <- expand.grid(j = 2:length(time_fe_XX), k = 1:degree_pol)
    fe_col_names <- paste0("time_fe_XX_", fe_grid$j, "_bt", fe_grid$k, "_XX")
    lapply(seq_len(nrow(fe_grid)), function(idx) {
      j <- fe_grid$j[idx]; k <- fe_grid$k[idx]
      d_sq_col <- paste0("d_sq_", k, "_XX")
      df[, (fe_col_names[idx]) := as.numeric(time_XX >= j) * get(d_sq_col)]
    })
    controls <- c(controls, fe_col_names)
  }

  #### Creating treatment at F_g: D_{g,F_g}
  df[, d_fg_XX := data.table::fifelse(time_XX == F_g_XX, treatment_XX, NA_real_)]
  df[, d_fg_XX := mean(d_fg_XX, na.rm = TRUE), by = "group_XX"]
  df[, d_fg_XX := data.table::fifelse(is.na(d_fg_XX) & F_g_XX == (T_max_XX + 1), d_sq_XX, d_fg_XX)]

  #### Creating the variable L_g_XX = T_g_XX - F_g_XX so that we can compute L_u or L_a afterwards
  df[, L_g_XX := T_g_XX - F_g_XX + 1]

  #### Creating the equivalent variable L_g_placebo_XX for placebos
  if (placebo > 0) {
    df[, L_g_placebo_XX := data.table::fifelse(
      F_g_XX >= 3,
      data.table::fifelse(L_g_XX > (F_g_XX - 2), F_g_XX - 2, L_g_XX),
      NA_real_
    )]
    df[, L_g_placebo_XX := data.table::fifelse(is.infinite(L_g_placebo_XX), NA_real_, L_g_placebo_XX)]
  }

  #### Tagging first observation of each group_XX
  data.table::setorder(df, group_XX, time_XX)
  df[, first_obs_by_gp_XX := as.numeric(time_XX == min(time_XX)), by = "group_XX"]

  #### If cluster option if specified, flagging first obs in cluster and checking if the cluster variable is weakly coarser than the group one.
  if (!is.null(cluster)) {
    ## complete missing clusters based on the min
    df[, cluster_XX := as.numeric(cluster_XX)]
    df[, cluster_group_XX := min(cluster_XX, na.rm = TRUE), by = "group_XX"]
    df[, cluster_XX := data.table::fifelse(is.na(cluster_XX), cluster_group_XX, cluster_XX)]

    data.table::setorder(df, cluster_XX, group_XX, time_XX)
    df[, first_obs_by_clust_XX := as.numeric(seq_len(.N) == 1L), by = "cluster_XX"]

    df[, cluster_var_g_XX := stats::sd(cluster_XX, na.rm = TRUE), by = "group_XX"]
    ## Error message for clustering: non-nested case
    max_cluster_var <- max(df[["cluster_var_g_XX"]], na.rm = TRUE)
    if (!is.na(max_cluster_var) && max_cluster_var > 0) {
      stop("The group variable should be nested within the clustering variable.")
    }
  }

  #### Declaring the data as panel after the changes above
  # Compute first differences
  data.table::setorder(df, group_XX, time_XX)
  df[, diff_y_XX := outcome_XX - data.table::shift(outcome_XX, 1), by = "group_XX"]
  df[, diff_d_XX := treatment_XX - data.table::shift(treatment_XX, 1), by = "group_XX"]

  ######## 3. Necessary pre-estimation steps when the controls option is specified

  if (!is.null(controls) && length(controls) > 0L) {
    # Sort for shift operations
    data.table::setorder(df, group_XX, time_XX)

    ## 1) First differences of each control + missing flag
    count_controls <- length(controls)
    diff_cols_main <- sprintf("diff_X%d_XX", 1:count_controls)

    # Compute all first differences
    for (j in seq_len(count_controls)) {
      df[, (diff_cols_main[j]) := get(controls[j]) - data.table::shift(get(controls[j]), 1), by = "group_XX"]
    }

    # fd_X_all_non_missing_XX = 1 iff ALL diff_X columns are non-NA for that row
    df[, fd_X_all_non_missing_XX := 1L]
    for (j in seq_len(count_controls)) {
      df[, fd_X_all_non_missing_XX := data.table::fifelse(is.na(get(diff_cols_main[j])), 0L, fd_X_all_non_missing_XX)]
    }

    ## 2) Residualization prep
    mycontrols_XX <- character(0L)
    grp_cols <- c("time_XX", "d_sq_XX")
    if (!is.null(trends_nonparam)) grp_cols <- c(grp_cols, trends_nonparam)

    # Shared: mask and weights (same for all controls — compute once)
    df[, .mask_ctrl := (ever_change_d_XX == 0) & !is.na(diff_y_XX) & (fd_X_all_non_missing_XX == 1L)]
    df[, .N_for_ctrl := data.table::fifelse(.mask_ctrl, N_gt_XX, 0)]
    df[, sum_weights_control_XX := sum(.N_for_ctrl, na.rm = TRUE), by = grp_cols]
    df[, sum_weights_control_XX := data.table::fifelse(.mask_ctrl, sum_weights_control_XX, NA_real_)]

    # Shared: diff_y_wXX (same for all controls — compute once)
    df[, diff_y_wXX := sqrt(N_gt_XX) * diff_y_XX]

    # Compute all masked weighted sums: avg_diff_masked_j = fifelse(mask, N_gt * diff_X_j, 0)
    avg_masked_cols <- sprintf("__avg_diff_masked_%d__", 1:count_controls)
    for (j in seq_len(count_controls)) {
      data.table::set(df, j = avg_masked_cols[j],
        value = data.table::fifelse(df[[".mask_ctrl"]], df[["N_gt_XX"]] * df[[diff_cols_main[j]]], 0))
    }

    # Batch by-group sum for all avg columns at once
    avg_cols_main <- sprintf("avg_diff_X%d_XX", 1:count_controls)
    df[, (avg_cols_main) := lapply(.SD, sum, na.rm = TRUE), by = grp_cols, .SDcols = avg_masked_cols]

    # Clean up temp masked columns
    df[, (avg_masked_cols) := NULL]

    # Compute residuals and products per control (element-wise, cheap)
    for (j in seq_len(count_controls)) {
      avg_col <- avg_cols_main[j]
      diff_col <- diff_cols_main[j]
      resid_col <- sprintf("resid_X%d_time_FE_XX", j)
      prod_col <- sprintf("prod_X%d_Ngt_XX", j)

      # Mask and divide by weights
      df[, (avg_col) := data.table::fifelse(.mask_ctrl, get(avg_col), NA_real_)]
      df[, (avg_col) := get(avg_col) / sum_weights_control_XX]

      # Residual
      df[, (resid_col) := sqrt(N_gt_XX) * (get(diff_col) - get(avg_col))]
      df[is.na(get(resid_col)), (resid_col) := 0]
      mycontrols_XX <- c(mycontrols_XX, resid_col)

      # Product
      df[, (prod_col) := data.table::fifelse(is.na(sqrt(N_gt_XX) * get(resid_col)), 0, sqrt(N_gt_XX) * get(resid_col))]
    }

    # Clean up temp columns
    df[, c(".N_for_ctrl", ".avg_diff_temp_masked", "avg_diff_temp_XX", ".mask_ctrl") := NULL]

    ## Dictionaries / storage
    levels_d_sq_XX <- sort(unique(df[["d_sq_int_XX"]][!is.na(df[["d_sq_int_XX"]])]))
    store_singular <- stats::setNames(rep(FALSE, length(levels_d_sq_XX)), as.character(levels_d_sq_XX))
    store_noresidualization_XX <- integer(0L)
    levels_d_sq_XX_final <- integer(0L)

    ## Loop over each baseline-treatment level - matrix operations in R
    for (l in levels_d_sq_XX) {
      # Count unique F_g values for this level
      df_l <- df[d_sq_int_XX == l]
      useful <- data.table::uniqueN(df_l$F_g_XX)
      assign(paste0("useful_res_", l, "_XX"), useful)

      if (useful > 1L) {
        # Filter for control observations
        data_dt <- df[ever_change_d_XX == 0 & !is.na(diff_y_XX) & fd_X_all_non_missing_XX == 1L & d_sq_int_XX == l]

        if (nrow(data_dt) == 0L) {
          store_singular[as.character(l)] <- TRUE
          store_noresidualization_XX <- c(store_noresidualization_XX, l)
          assign(paste0("useful_res_", l, "_XX"), 1L)
          next
        }

        # Extract vectors for matrix operations
        Y_vec <- data_dt[["diff_y_wXX"]]
        X_mat <- as.matrix(data_dt[, ..mycontrols_XX])
        YX <- cbind(Y_vec, X_mat, 1)

        overall <- crossprod(YX)
        val <- sum(overall)

        if (is.na(val)) {
          store_singular[as.character(l)] <- TRUE
          store_noresidualization_XX <- c(store_noresidualization_XX, l)
          assign(paste0("useful_res_", l, "_XX"), 1L)
        } else {
          k <- length(mycontrols_XX)
          idx_controls <- 1:k + 1L

          M <- overall[idx_controls, idx_controls, drop = FALSE]
          v <- overall[idx_controls, 1, drop = FALSE]

          # Store M matrix (didmgt_XX) and v vector (didmgt_Xy)
          assign(paste0("didmgt_XX_", l, "_XX"), M)
          assign(paste0("didmgt_Xy_", l, "_XX"), v)

          # Use invsym_r to mimic Stata's invsym() (Cholesky-based inverse)
          inv_M <- invsym_r(M)
          theta_d <- inv_M %*% v
          assign(paste0("coefs_sq_", l, "_XX"), theta_d)
          levels_d_sq_XX_final <- c(levels_d_sq_XX_final, l)

          if (abs(det(M)) <= 1e-16) {
            store_singular[as.character(l)] <- TRUE
          }

          # Compute rsum on control sample only
          control_sample <- df[ever_change_d_XX == 0 & !is.na(diff_y_XX) & fd_X_all_non_missing_XX == 1 & d_sq_int_XX == l]
          rmax <- max(control_sample[["F_g_XX"]], na.rm = TRUE)
          rsum_df <- control_sample[time_XX >= 2 & time_XX <= (rmax - 1) & time_XX < F_g_XX & !is.na(diff_y_XX)]
          rsum <- sum(rsum_df[["N_gt_XX"]], na.rm = TRUE)

          # Store values for output
          assign(paste0("rsum_", l, "_XX"), rsum)
          assign(paste0("invsym_M_", l, "_XX"), inv_M)
          assign(paste0("inv_Denom_", l, "_XX"), inv_M * rsum * G_XX)
        }
      }
    }

    ## Handle singular levels warnings
    levels_d_sq_bis_XX <- unique(df[["d_sq_XX"]])
    levels_d_sq_bis_XX <- sort(levels_d_sq_bis_XX[!is.na(levels_d_sq_bis_XX)])
    singular_levels <- levels_d_sq_bis_XX[vapply(levels_d_sq_bis_XX, function(l) {
      key <- as.character(l)
      !is.null(store_singular[key]) && isTRUE(store_singular[key])
    }, logical(1))]

    if (length(singular_levels) > 0L) {
      store_singular_XX <- paste(singular_levels, collapse = " ")
      warning("Some control variables are not taken into account for groups with baseline treatment equal to:", store_singular_XX)
      warning("1. For these groups, the regression of Y evolution X evolution and time-FE had fewer observations than regressors.")
      warning("2. For these groups, one or more controls were perfectly collinear (no time variation).")
    }

    ## Drop levels where residualization failed
    if (length(store_noresidualization_XX) > 0L) {
      df <- df[!d_sq_int_XX %in% store_noresidualization_XX]
    }

    ## 3. FE regression using feols (optimized C code)
    df[, time_FE_XX := as.integer(time_XX)]

    # Add row_id for reliable joining
    df[, row_id_XX := .I]

    ## 4. Loop over baseline-treatment levels - use feols (optimized C code)
    for (l in levels_d_sq_XX_final) {
      outcol <- sprintf("E_y_hat_gt_int_%d_XX", l)

      # Filter data for this level
      data_reg <- df[d_sq_int_XX == l & F_g_XX > time_XX]

      if (nrow(data_reg) == 0L) {
        df[, (outcol) := NA_real_]
        next
      }

      # Use feols for FE regression (highly optimized)
      fe_terms <- sprintf("diff_X%d_XX", seq_len(count_controls))
      formula_str <- paste("diff_y_XX ~", paste(fe_terms, collapse = " + "), "- 1 | time_FE_XX")
      form <- stats::as.formula(formula_str)

      model <- fixest::feols(form, data = data_reg, weights = data_reg$weight_XX)
      data_reg[, y_hat := stats::predict(model, newdata = data_reg)]

      # Join back to main df using row_id
      df[, (outcol) := NA_real_]
      df[data_reg, (outcol) := i.y_hat, on = "row_id_XX"]
    }

    ## Clean up temporary columns
    df[, c("time_FE_XX", "row_id_XX") := NULL]
  }




  ###### 4. Performing the estimation and storing the results
  ## Computing L_u/L_a, maximum number of event-study effects that can be computed
  ## for the switchers in/out, to compare them to number of effects requested,
  ## and finally determine the number of effects to be estimated.
  ## Same thing for the placebos.

  ## Initialize L_u_XX/L_a_XX
  L_u_XX <- NA
  L_a_XX <- NA
  L_placebo_u_XX <- NA
  L_placebo_a_XX <- NA

  ## data.table path for computing L_u/L_a
  ## For switchers in
  if (switchers == "" || switchers == "in") {
    switchers_in_df <- df[S_g_XX == 1]
    n_switchers_in <- nrow(switchers_in_df)
    if (n_switchers_in == 0) {
      L_u_XX <- 0
    } else {
      L_u_XX <- max(switchers_in_df[["L_g_XX"]], na.rm = TRUE)
      if (is.na(L_u_XX) || is.infinite(L_u_XX)) L_u_XX <- 0
    }
    ## For placebos
    if (placebo != 0 && n_switchers_in > 0) {
      L_placebo_u_XX <- max(switchers_in_df[["L_g_placebo_XX"]], na.rm = TRUE)
      if (is.na(L_placebo_u_XX) || L_placebo_u_XX < 0) L_placebo_u_XX <- 0
      ## If the trends_lin option was specified, L_placebo_u_XX should be decreased by 1
      ## because data starts at period 2 instead of 1.
      if (isTRUE(trends_lin)) {
        L_placebo_u_XX <- L_placebo_u_XX - 1
      }
    }
  }

  ## For switchers out
  if (switchers == "" || switchers == "out") {
    switchers_out_df <- df[S_g_XX == 0]
    n_switchers_out <- nrow(switchers_out_df)
    if (n_switchers_out == 0) {
      L_a_XX <- 0
    } else {
      L_a_XX <- max(switchers_out_df[["L_g_XX"]], na.rm = TRUE)
      if (is.na(L_a_XX) || is.infinite(L_a_XX)) L_a_XX <- 0
    }
    if (placebo != 0 && n_switchers_out > 0) {
      L_placebo_a_XX <- max(switchers_out_df[["L_g_placebo_XX"]], na.rm = TRUE)
      if (is.na(L_placebo_a_XX) || L_placebo_a_XX < 0) L_placebo_a_XX <- 0
      if (isTRUE(trends_lin)) {
        L_placebo_a_XX <- L_placebo_a_XX - 1
      }
    }
  }

  # df is a data.table throughout

  ## Error message if Design restriction 1 is not met
  if (
    (switchers == "in" && (is.na(L_u_XX) || L_u_XX == 0)) || 
    (switchers == "out" && (is.na(L_a_XX) || L_a_XX == 0)) || 
    (switchers == "" &&  ((is.na(L_u_XX) || L_u_XX == 0) && (is.na(L_a_XX) || L_a_XX == 0)))
  ) {
    stop("No treatment effect can be estimated.\n  This is because Design Restriction 1 in de Chaisemartin & D'Haultfoeuille (2024) is not satisfied in the data, given the options requested.\n  This may be due to the fact that groups' period-one treatment is continuous, or takes a large number of values, and you have not specified the continuous option.\n  If so, you can try to specify this option.\n  If the issue persists even with this option, this means that all groups experience their first treatment change at the same date.\n  In this situation, estimators of de Chaisemartin & D'Haultfoeuille (2024) cannot be used.")
  }

  ## Checking that the number of dynamic and placebo effects requested by user
  ## are feasible, and correcting them if they are not. 

  if (switchers == "" ) {
    l_XX <- max(L_a_XX, L_u_XX, na.rm = TRUE)
    l_XX <- min(l_XX, effects)
    if (placebo != 0) {
      l_placebo_XX <- max(L_placebo_a_XX, L_placebo_u_XX, na.rm = TRUE)
      l_placebo_XX <- min(l_placebo_XX, placebo, na.rm = TRUE)
      # The number of placebos cannot be greater than the number of effects computed:
      l_placebo_XX <- min(l_placebo_XX, effects)
    } else {
      l_placebo_XX <- 0
    }
  }

  if (switchers == "in") {
    l_XX <- min(effects, L_u_XX, na.rm = TRUE)
    if (placebo != 0) {
      l_placebo_XX <- min(placebo, L_placebo_u_XX, na.rm = TRUE)
      # The number of placebos cannot be greater than the number of effects computed:
      l_placebo_XX <- min(l_placebo_XX, effects)
    }
    else {
      l_placebo_XX <- 0
    }
  }

  if (switchers == "out") {
    l_XX <- min(effects, L_a_XX, na.rm = TRUE)
    if (placebo != 0) {
      l_placebo_XX <- min(placebo, L_placebo_a_XX, na.rm = TRUE)
      # The number of placebos cannot be greater than the number of effects computed:
      l_placebo_XX <- min(l_placebo_XX, effects)
    }
    else {
      l_placebo_XX <- 0
    }
  }

  # If the number of effects or placebos initially asked by user was too large, display error message
  if (l_XX < effects) {
    message(sprintf("The number of effects requested is too large. The number of effects which can be estimated is at most %.0f. The command will therefore try to estimante %.0f effect(s)", l_XX, l_XX))
  }

  if (placebo != 0) {
    if (l_placebo_XX < placebo && effects >= placebo) {
      message(sprintf("The number of placebos which can be estimated is at most %.0f.The command will therefore try to estimate %.0f placebo(s).", l_placebo_XX, l_placebo_XX))
    }
    if (effects < placebo) {
      message(sprintf("The number of placebo requested cannot be larger than the number of effects requested. The command cannot compute more than %.0f placebo(s).", l_placebo_XX))
    }
  }

  ## Adjustment to add more placebos (did_multiplegt_dyn_all_pl)
  max_pl_u_XX <- max_pl_a_XX <- max_pl_gap_u_XX <- max_pl_gap_a_XX <- 0
  df[, pl_gap_XX := data.table::fifelse(!is.na(S_g_XX), F_g_XX - 2 - L_g_XX, NA_real_)]
  if (switchers == "" || switchers == "in") {
    max_pl_u_XX <- max(df[S_g_XX == 1]$F_g_XX, na.rm = TRUE) - 2
    max_pl_gap_u_XX <- max(df[S_g_XX == 1]$pl_gap_XX, na.rm = TRUE)
    if (is.na(max_pl_u_XX) || is.infinite(max_pl_u_XX)) max_pl_u_XX <- 0
    if (is.na(max_pl_gap_u_XX) || is.infinite(max_pl_gap_u_XX)) max_pl_gap_u_XX <- 0
  }
  if (switchers == "" || switchers == "out") {
    max_pl_a_XX <- max(df[S_g_XX == 0]$F_g_XX, na.rm = TRUE) - 2
    max_pl_gap_a_XX <- max(df[S_g_XX == 0]$pl_gap_XX, na.rm = TRUE)
    if (is.na(max_pl_a_XX) || is.infinite(max_pl_a_XX)) max_pl_a_XX <- 0
    if (is.na(max_pl_gap_a_XX) || is.infinite(max_pl_gap_a_XX)) max_pl_gap_a_XX <- 0
  }
  max_pl_XX <- max(max_pl_u_XX, max_pl_a_XX)
  max_pl_gap_XX <- max(max_pl_gap_u_XX, max_pl_gap_a_XX)
  max_pl_u_XX <- max_pl_a_XX <- max_pl_gap_u_XX <- max_pl_gap_a_XX <- NULL
  df[, pl_gap_XX := NULL]

  ## Generating default values for the variables which will be aggregated
  ## after Program 2 below has been run for switchers in and for switchers out.

  inh_obj <- c()
  # Initialize effect columns
  effect_cols <- c(
    paste0("U_Gg", 1:l_XX, "_plus_XX"),
    paste0("U_Gg", 1:l_XX, "_minus_XX"),
    paste0("count", 1:l_XX, "_plus_XX"),
    paste0("count", 1:l_XX, "_minus_XX"),
    paste0("U_Gg_var_", 1:l_XX, "_in_XX"),
    paste0("U_Gg_var_", 1:l_XX, "_out_XX"),
    paste0("delta_D_g_", 1:l_XX, "_plus_XX"),
    paste0("delta_D_g_", 1:l_XX, "_minus_XX")
  )
  df[, (effect_cols) := 0]
  assign("sum_for_var_in_XX", 0)
  assign("sum_for_var_out_XX", 0)
  inh_obj <- c(inh_obj, "sum_for_var_in_XX", "sum_for_var_out_XX")
  if (placebo != 0) {
    # Initialize placebo columns
    placebo_cols <- c(
      paste0("U_Gg_pl_", 1:l_XX, "_plus_XX"),
      paste0("U_Gg_pl_", 1:l_XX, "_minus_XX"),
      paste0("count", 1:l_XX, "_pl_plus_XX"),
      paste0("count", 1:l_XX, "_pl_minus_XX"),
      paste0("U_Gg_var_pl_", 1:l_XX, "_in_XX"),
      paste0("U_Gg_var_pl_", 1:l_XX, "_out_XX")
    )
    df[, (placebo_cols) := 0]
    assign("sum_for_var_placebo_in_XX", 0)
    assign("sum_for_var_placebo_out_XX", 0)
    inh_obj <- c(inh_obj, "sum_for_var_placebo_in_XX", "sum_for_var_placebo_out_XX")
  }

  # Optimized: Batch initialization of N scalars using pre-computed variable names
  base_vars <- c(
    paste0("N1_", 1:l_XX, "_XX"),
    paste0("N1_", 1:l_XX, "_XX_new"),
    paste0("N1_dw_", 1:l_XX, "_XX"),
    paste0("N0_", 1:l_XX, "_XX"),
    paste0("N0_", 1:l_XX, "_XX_new"),
    paste0("N0_dw_", 1:l_XX, "_XX")
  )
  lapply(base_vars, assign, value = 0, envir = environment())
  inh_obj <- c(inh_obj, base_vars)

  if (normalized == TRUE) {
    norm_vars <- c(paste0("delta_D_", 1:l_XX, "_in_XX"), paste0("delta_D_", 1:l_XX, "_out_XX"))
    lapply(norm_vars, assign, value = 0, envir = environment())
    inh_obj <- c(inh_obj, norm_vars)
  }

  if (placebo != 0) {
    placebo_vars <- c(
      paste0("N1_placebo_", 1:l_XX, "_XX"),
      paste0("N1_placebo_", 1:l_XX, "_XX_new"),
      paste0("N1_dw_placebo_", 1:l_XX, "_XX"),
      paste0("N0_placebo_", 1:l_XX, "_XX"),
      paste0("N0_placebo_", 1:l_XX, "_XX_new"),
      paste0("N0_dw_placebo_", 1:l_XX, "_XX")
    )
    lapply(placebo_vars, assign, value = 0, envir = environment())
    inh_obj <- c(inh_obj, placebo_vars)

    if (normalized == TRUE) {
      norm_pl_vars <- c(paste0("delta_D_pl_", 1:l_XX, "_in_XX"), paste0("delta_D_pl_", 1:l_XX, "_out_XX"))
      lapply(norm_pl_vars, assign, value = 0, envir = environment())
      inh_obj <- c(inh_obj, norm_pl_vars)
    }
  }

  df[, c("U_Gg_plus_XX", "U_Gg_minus_XX", "U_Gg_var_plus_XX", "U_Gg_var_minus_XX") := 0]
  assign("U_Gg_den_plus_XX", 0)
  assign("U_Gg_den_minus_XX", 0)
  assign("sum_N1_l_XX", 0)
  assign("sum_N0_l_XX", 0)
  inh_obj <- c(inh_obj,"U_Gg_den_plus_XX", "U_Gg_den_minus_XX", "sum_N1_l_XX", "sum_N0_l_XX")

  # Scalars previously passed as inherited objects
  # Their values will be changes through the next routines
  const <- stats::setNames(lapply(inh_obj, get, envir = environment()), inh_obj)

  # Saving useful scalars to the Global Environment
  # Their values will be not changes through the next routines
  gs <- c("L_u_XX", "L_a_XX", "l_XX", "t_min_XX", "T_max_XX", "G_XX")
  if (placebo != 0) {
    gs <- c(gs, "L_placebo_u_XX", "L_placebo_a_XX")
  }
  # Add inheritance of controls #
  globals <- stats::setNames(lapply(gs, get, envir = environment()), gs)

  controls_globals <- NULL
  if (!is.null(controls)) {
    base_names <- c("useful_res_", "coefs_sq_", "inv_Denom_", "didmgt_XX_", "didmgt_Xy_")
    controls_globals <- unlist(lapply(levels_d_sq_XX, function(l) {
      entries <- stats::setNames(lapply(base_names, function(bn) get(paste0(bn, l, "_XX"))),
                          paste0(base_names, l, "_XX"))
      # Add optional entries
      rsum_name <- paste0("rsum_", l, "_XX")
      invsym_name <- paste0("invsym_M_", l, "_XX")
      if (exists(rsum_name)) entries[[rsum_name]] <- get(rsum_name)
      if (exists(invsym_name)) entries[[invsym_name]] <- get(invsym_name)
      entries
    }), recursive = FALSE)
    controls_globals[["G_XX"]] <- G_XX
  }

  ## Initialize variable to earmark switchers by the number of the event-study effect
  df[, switchers_tag_XX := NA_real_]

  ## Store the data prior to estimation if requested
  if (isTRUE(data_only)) {
    data <- list(df, l_XX, T_max_XX)
    names(data) <- c("df", "l_XX", "T_max_XX")
    return(data)
  }

  ## Perform the estimation: call the program did_multiplegt_dyn_core,
  ## for switchers in and for switchers out, and store the results.
  ## df is now a data.table passed directly to core function

  if (switchers == "" || switchers == "in") {
    if (!is.na(L_u_XX) && L_u_XX != 0) {

      ## Perform the estimation of effects and placebos outside of the loop on
      ## number of effects if trends_lin not specified
      if (isFALSE(trends_lin)) {
        data <- did_multiplegt_dyn_core(df, outcome = "outcome_XX",
                                        group = "group_XX", time = "time_XX", cluster = cluster,
              treatment = "treatment_XX", effects = l_XX, placebo = l_placebo_XX,
              switchers_core = "in", trends_nonparam = trends_nonparam,
              controls = controls, same_switchers = same_switchers,
              same_switchers_pl = same_switchers_pl, only_never_switchers = only_never_switchers,
              normalized = normalized, globals = globals, const = const,
              trends_lin = trends_lin, controls_globals = controls_globals,
              less_conservative_se = less_conservative_se, continuous = continuous)

        df <- data$df
        data$df <- NULL
        const[names(data$const)] <- data$const
        list2env(data$const, envir = environment())

        # Store the number of the event-study effect for switchers-in
        for (k in 1:l_XX) {
          dist_col <- paste0("distance_to_switch_", k, "_XX")
          df[, switchers_tag_XX := data.table::fifelse(get(dist_col) == 1, as.numeric(k), switchers_tag_XX)]
        }
      }

      for (i in 1:l_XX) {
        ## Perform the estimation of effects inside of the loop on number of effects
        ## if trends_lin is specified
        ## Note that if the option trends_lin was specified, same_switchers must also be specified.

        if (isTRUE(trends_lin)) {
          data <- did_multiplegt_dyn_core(df, outcome = "outcome_XX", group = "group_XX",
                    time = "time_XX", treatment = "treatment_XX", cluster = cluster,
                    effects = i, placebo = 0, switchers_core = "in",
                    trends_nonparam = trends_nonparam, controls = controls,
                    same_switchers = TRUE, same_switchers_pl = FALSE,
                    only_never_switchers = only_never_switchers, normalized = normalized,
                    globals = globals, const = const, trends_lin = trends_lin,
                    controls_globals = controls_globals,
                    less_conservative_se = less_conservative_se, continuous = continuous)

          df <- data$df
          data$df <- NULL
          const[names(data$const)] <- data$const
          list2env(data$const, envir = environment())

          ## Store the number of the event-study effect for switchers-in
          col_dist_i <- sprintf("distance_to_switch_%d_XX", i)
          df[, switchers_tag_XX := data.table::fifelse(get(col_dist_i) == 1, as.numeric(i), switchers_tag_XX)]
        }

        ## Store variables necessary for computation of effects.
        ## N.B.: in the case of unbalanced panels, it can happen that the U_Gg`i'_XX are not computed by program 2 (for example when y is missing). Consequently, for the command not to display an error message and continue running, we need to verify the variable is created, which is conditional on  N1_`i'_XX!=0.

        if (get(paste0("N1_",i,"_XX")) != 0) {
          src_col <- paste0("U_Gg", i, "_XX")
          dst_col <- paste0("U_Gg", i, "_plus_XX")
          df[, (dst_col) := get(src_col)]

          src_col <- paste0("count", i, "_core_XX")
          dst_col <- paste0("count", i, "_plus_XX")
          df[, (dst_col) := get(src_col)]

          src_col <- paste0("U_Gg", i, "_var_XX")
          dst_col <- paste0("U_Gg_var_", i, "_in_XX")
          df[, (dst_col) := get(src_col)]

          assign(paste0("N1_",i,"_XX_new"), get(paste0("N1_",i,"_XX")))
          const[[paste0("N1_",i,"_XX_new")]] <- get(paste0("N1_",i,"_XX_new"))

          if (normalized == TRUE) {
            assign(paste0("delta_D_",i,"_in_XX"), get(paste0("delta_norm_",i,"_XX")))
            const[[paste0("delta_D_",i,"_in_XX")]] <- get(paste0("delta_D_",i,"_in_XX"))
          }

          if (isFALSE(trends_lin)) {
            src_col <- paste0("delta_D_g_", i, "_XX")
            dst_col <- paste0("delta_D_g_", i, "_plus_XX")
            df[, (dst_col) := get(src_col)]
          }
        }

      }

      # Same as above for placebos.
      if (l_placebo_XX != 0) {
        for (i in 1:l_placebo_XX) {

          if (isTRUE(trends_lin)) {
            data <- did_multiplegt_dyn_core(df,
                outcome = "outcome_XX", group = "group_XX", time = "time_XX",
                cluster = cluster,
                treatment = "treatment_XX", effects = i, placebo = i,
                switchers_core = "in", trends_nonparam = trends_nonparam,
                controls = controls, same_switchers = TRUE,
                same_switchers_pl = TRUE, only_never_switchers = only_never_switchers,
                normalized = normalized, globals = globals, const = const,
                trends_lin = trends_lin, controls_globals = controls_globals,
                less_conservative_se = less_conservative_se, continuous = continuous)

            df <- data$df
            data$df <- NULL
            const[names(data$const)] <- data$const
            list2env(data$const, envir = environment())

            col_dist_i <- sprintf("distance_to_switch_%d_XX", i)
            df[, switchers_tag_XX := data.table::fifelse(get(col_dist_i) == 1, as.numeric(i), switchers_tag_XX)]

          }

          if (get(paste0("N1_placebo_",i,"_XX")) != 0) {
            df[, (paste0("U_Gg_pl_",i,"_plus_XX")) := get(paste0("U_Gg_placebo_",i,"_XX"))]
            df[, (paste0("count",i,"_pl_plus_XX")) := get(paste0("count",i,"_pl_core_XX"))]
            df[, (paste0("U_Gg_var_pl_",i,"_in_XX")) := get(paste0("U_Gg_pl_",i,"_var_XX"))]
            assign(paste0("N1_placebo_",i,"_XX_new"), get(paste0("N1_placebo_",i,"_XX")))
            const[[paste0("N1_placebo_",i,"_XX_new")]] <- get(paste0("N1_placebo_",i,"_XX_new"))

            if (normalized == TRUE) {
              assign(paste0("delta_D_pl_",i,"_in_XX"), get(paste0("delta_norm_pl_",i,"_XX")))
              const[[paste0("delta_D_pl_",i,"_in_XX")]] <- get(paste0("delta_D_pl_",i,"_in_XX"))
            }
          }

        }
      }

      # Store variables necessary for computation of average effect.
      if (isFALSE(trends_lin)) {
        if (sum_N1_l_XX != 0) {
          df[, U_Gg_plus_XX := U_Gg_XX]
          df[, U_Gg_den_plus_XX := U_Gg_den_XX]
          df[, U_Gg_var_plus_XX := U_Gg_var_XX]
        }
      }
    }
  }

  ## Same thing as above, for switchers out
  if (switchers == "" || switchers == "out") {
    if (!is.na(L_a_XX) && L_a_XX != 0) {

      if (isFALSE(trends_lin)) {
        data <- did_multiplegt_dyn_core(df,
        outcome = "outcome_XX", group = "group_XX", time = "time_XX",
        treatment = "treatment_XX", effects = l_XX, cluster = cluster,
        placebo = l_placebo_XX, switchers_core = "out",
        trends_nonparam = trends_nonparam, controls = controls,
        same_switchers = same_switchers, same_switchers_pl = same_switchers_pl,
        only_never_switchers = only_never_switchers, normalized, globals = globals,
        const = const, trends_lin = trends_lin, controls_globals = controls_globals,
        less_conservative_se, continuous = continuous)

        df <- data$df
        data$df <- NULL
        const[names(data$const)] <- data$const
        list2env(data$const, envir = environment())

        for (k in 1:l_XX) {
          ## Store the number of the event-study effect for switchers-out
          dist_col <- paste0("distance_to_switch_", k, "_XX")
          df[, switchers_tag_XX := data.table::fifelse(get(dist_col) == 1, as.numeric(k), switchers_tag_XX)]
        }
      }

      for (i in 1:l_XX) {

        if (isTRUE(trends_lin)) {
          data <- did_multiplegt_dyn_core(df, outcome = "outcome_XX",
              group = "group_XX", time = "time_XX", treatment = "treatment_XX",
              effects = i, placebo = 0, switchers_core = "out", cluster = cluster,
              trends_nonparam = trends_nonparam, controls = controls,
              same_switchers = TRUE, same_switchers_pl = FALSE,
              only_never_switchers = only_never_switchers, normalized = normalized,
              globals = globals, const = const, trends_lin = trends_lin,
              controls_globals = controls_globals,
              less_conservative_se = less_conservative_se, continuous = continuous)

          df <- data$df
          data$df <- NULL
          const[names(data$const)] <- data$const
          list2env(data$const, envir = environment())

          ## Store the number of the event-study effect for switchers-out
          dist_col <- paste0("distance_to_switch_", i, "_XX")
          df[, switchers_tag_XX := data.table::fifelse(get(dist_col) == 1, as.numeric(i), switchers_tag_XX)]
        }

        if (get(paste0("N0_",i,"_XX")) != 0) {
          # Negate and copy columns
          df[, (paste0("U_Gg",i,"_minus_XX")) := -get(paste0("U_Gg",i,"_XX"))]
          df[, (paste0("count",i,"_minus_XX")) := get(paste0("count",i,"_core_XX"))]
          df[, (paste0("U_Gg_var_",i,"_out_XX")) := -get(paste0("U_Gg",i,"_var_XX"))]
          assign(paste0("N0_",i,"_XX_new"), get(paste0("N0_",i,"_XX")))
          const[[paste0("N0_",i,"_XX_new")]] <- get(paste0("N0_",i,"_XX_new"))

          if (normalized == TRUE) {
            assign(paste0("delta_D_",i,"_out_XX"), get(paste0("delta_norm_",i,"_XX")))
            const[[paste0("delta_D_",i,"_out_XX")]] <- get(paste0("delta_D_",i,"_out_XX"))
          }

          if (isFALSE(trends_lin)) {
            df[, (paste0("delta_D_g_",i,"_minus_XX")) := get(paste0("delta_D_g_",i,"_XX"))]
          }
        }
      }

      if (l_placebo_XX != 0) {
        for (i in 1:l_placebo_XX) {

          if (isTRUE(trends_lin)) {
            data <- did_multiplegt_dyn_core(df, outcome = "outcome_XX",
                group = "group_XX", time = "time_XX", treatment = "treatment_XX",
                effects = i, placebo = i, switchers_core = "out",
                cluster = cluster,
                trends_nonparam = trends_nonparam, controls = controls,
                same_switchers = TRUE, same_switchers_pl = TRUE,
                only_never_switchers = only_never_switchers, normalized = normalized,
                globals = globals, const = const, trends_lin = trends_lin,
                controls_globals = controls_globals,
                less_conservative_se = less_conservative_se,
                continuous = continuous)

            df <- data$df
            data$df <- NULL
            const[names(data$const)] <- data$const
            list2env(data$const, envir = environment())
            dist_col <- paste0("distance_to_switch_", i, "_XX")
            df[, switchers_tag_XX := data.table::fifelse(get(dist_col) == 1, as.numeric(i), switchers_tag_XX)]
          }

          if (get(paste0("N0_placebo_",i,"_XX")) != 0) {
            df[, (paste0("U_Gg_pl_",i,"_minus_XX")) := -get(paste0("U_Gg_placebo_",i,"_XX"))]
            df[, (paste0("count",i,"_pl_minus_XX")) := get(paste0("count",i,"_pl_core_XX"))]
            df[, (paste0("U_Gg_var_pl_",i,"_out_XX")) := -get(paste0("U_Gg_pl_",i,"_var_XX"))]
            assign(paste0("N0_placebo_",i,"_XX_new"), get(paste0("N0_placebo_",i,"_XX")))
            const[[paste0("N0_placebo_",i,"_XX_new")]] <- get(paste0("N0_placebo_",i,"_XX_new"))

            if (normalized == TRUE) {
              assign(paste0("delta_D_pl_",i,"_out_XX"), get(paste0("delta_norm_pl_",i,"_XX")))
              const[[paste0("delta_D_pl_",i,"_out_XX")]] <- get(paste0("delta_D_pl_",i,"_out_XX"))
            }
          }
        }
      }

      if (isFALSE(trends_lin)) {
        if (sum_N0_l_XX != 0) {
          df[, U_Gg_minus_XX := -U_Gg_XX]
          df[, U_Gg_den_minus_XX := U_Gg_den_XX]
          df[, U_Gg_var_minus_XX := -U_Gg_var_XX]
        }
      }
    }
  }
  rownames <- c()

  ###### 5. Computing the estimators and their variances

  # Helper to get scalar sum from data.table column with first_obs filter
  dt_sum_first_obs <- function(df, col_name, first_obs_col = "first_obs_by_gp_XX") {
    if (!(col_name %chin% names(df))) return(0)
    val <- sum(df[[col_name]] * df[[first_obs_col]], na.rm = TRUE)
    if (is.na(val)) return(0)
    return(val)
  }

  # Helper to sum all values of a column
  dt_sum_col <- function(df, col_name) {
    if (!(col_name %chin% names(df))) return(0)
    val <- sum(df[[col_name]], na.rm = TRUE)
    if (is.na(val)) return(0)
    return(val)
  }

  # Helper for variance computation that handles clustering correctly
  dt_compute_variance_sum <- function(df, col_name, clustered) {
    if (!(col_name %chin% names(df))) return(0)

    if (!clustered) {
      # Non-clustered: just square and sum with first_obs filter
      val <- sum((df[[col_name]]^2) * df[["first_obs_by_gp_XX"]], na.rm = TRUE)
      if (is.na(val)) return(0)
      return(val)
    } else {
      # Clustered: multiply by first_obs_by_gp_XX, sum by cluster, square, multiply by first_obs_by_clust_XX, sum
      dt <- df[, .(var_weighted = get(col_name) * first_obs_by_gp_XX,
                    first_obs_by_clust_XX = first_obs_by_clust_XX,
                    cluster_XX = cluster_XX)]
      dt[, clust_var_sum := sum(var_weighted, na.rm = TRUE), by = "cluster_XX"]
      dt[, clust_var_sq := clust_var_sum^2 * first_obs_by_clust_XX]
      return(sum(dt[["clust_var_sq"]], na.rm = TRUE))
    }
  }

  # Creation of the matrix which stores all the estimators (DID_l, DID_pl, delta, etc.), their sd and the CIs
  mat_res_XX <- matrix(NA, nrow = l_XX + l_placebo_XX + 1, ncol = 9)

  # CI level
  ci_level <- ci_level / 100
  z_level <- stats::qnorm(ci_level + (1 - ci_level)/2)

  # Handle clustering for variance computation
  clustered <- !is.null(cluster)
  first_obs_col <- if (clustered) "first_obs_by_clust_XX" else "first_obs_by_gp_XX"
  # CRAN uses G_XX for both clustered and non-clustered variance computation
  G_var <- G_XX

  # BATCHED APPROACH: Build all columns first, then aggregate once
  # Step 1: Build weight vectors
  N1_vec <- vapply(1:l_XX, function(i) get(paste0("N1_", i, "_XX_new")), numeric(1))
  N0_vec <- vapply(1:l_XX, function(i) get(paste0("N0_", i, "_XX_new")), numeric(1))

  if (getOption("DID_DEBUG_VARIANCE", FALSE)) {
    cat("\n=== DEBUG: Weights for variance ===\n")
    cat("N1_vec:", N1_vec, "\n")
    cat("N0_vec:", N0_vec, "\n")
    cat("w_in (N1/(N1+N0)):", N1_vec / (N1_vec + N0_vec), "\n")
    cat("w_out (N0/(N1+N0)):", N0_vec / (N1_vec + N0_vec), "\n")
  }

  # Step 2: Build all global columns in data.table
  for (i in 1:l_XX) {
    N1_i <- N1_vec[i]
    N0_i <- N0_vec[i]
    total_N <- N1_i + N0_i

    col_plus <- paste0("U_Gg", i, "_plus_XX")
    col_minus <- paste0("U_Gg", i, "_minus_XX")
    col_count_plus <- paste0("count", i, "_plus_XX")
    col_count_minus <- paste0("count", i, "_minus_XX")
    col_var_in <- paste0("U_Gg_var_", i, "_in_XX")
    col_var_out <- paste0("U_Gg_var_", i, "_out_XX")

    # Global variance column
    if (total_N > 0) {
      w_in <- N1_i / total_N
      w_out <- N0_i / total_N
      var_in_vals <- if (col_var_in %chin% names(df)) data.table::nafill(df[[col_var_in]], fill = 0) else rep(0, nrow(df))
      var_out_vals <- if (col_var_out %chin% names(df)) data.table::nafill(df[[col_var_out]], fill = 0) else rep(0, nrow(df))
      df[, (paste0("U_Gg_var_glob_", i, "_XX")) := w_in * var_in_vals + w_out * var_out_vals]
    } else {
      df[, (paste0("U_Gg_var_glob_", i, "_XX")) := 0]
    }

    # Global U_Gg column
    plus_vals <- if (col_plus %chin% names(df)) data.table::nafill(df[[col_plus]], fill = 0) else rep(0, nrow(df))
    minus_vals <- if (col_minus %chin% names(df)) data.table::nafill(df[[col_minus]], fill = 0) else rep(0, nrow(df))
    if (total_N > 0) {
      df[, (paste0("U_Gg", i, "_global_XX")) := (N1_i / total_N) * plus_vals + (N0_i / total_N) * minus_vals]
    } else {
      df[, (paste0("U_Gg", i, "_global_XX")) := 0]
    }

    # Global count column - take MAX of plus and minus, handling NAs
    count_plus_vals <- if (col_count_plus %chin% names(df)) df[[col_count_plus]] else rep(NA_real_, nrow(df))
    count_minus_vals <- if (col_count_minus %chin% names(df)) df[[col_count_minus]] else rep(NA_real_, nrow(df))
    df[, (paste0("count", i, "_global_XX")) := data.table::fifelse(
      is.na(count_plus_vals), count_minus_vals,
      data.table::fifelse(is.na(count_minus_vals), count_plus_vals,
        pmax(count_plus_vals, count_minus_vals))
    )]
  }

  # Step 3: Compute aggregations
  agg_df <- list()
  for (i in 1:l_XX) {
    col_plus <- paste0("U_Gg", i, "_plus_XX")
    col_minus <- paste0("U_Gg", i, "_minus_XX")
    col_var_glob <- paste0("U_Gg_var_glob_", i, "_XX")
    col_count_global <- paste0("count", i, "_global_XX")

    # Sum of plus column (first_obs filtered)
    agg_df[[paste0("sum_plus_", i)]] <- if (col_plus %chin% names(df)) sum(df[[col_plus]] * df[["first_obs_by_gp_XX"]], na.rm = TRUE) else 0
    # Sum of minus column
    agg_df[[paste0("sum_minus_", i)]] <- if (col_minus %chin% names(df)) sum(df[[col_minus]] * df[["first_obs_by_gp_XX"]], na.rm = TRUE) else 0
    # Sum of squared variance - NON-CLUSTERED case only
    if (!clustered) {
      agg_df[[paste0("var_sq_sum_", i)]] <- sum((df[[col_var_glob]]^2) * df[["first_obs_by_gp_XX"]], na.rm = TRUE)
    }
    # Sum of count_global (N_effect)
    agg_df[[paste0("count_effects_", i)]] <- sum(df[[col_count_global]], na.rm = TRUE)
    # Count dw (non-null and > 0)
    agg_df[[paste0("count_dw_", i)]] <- sum(!is.na(df[[col_count_global]]) & df[[col_count_global]] > 0)
  }

  # Handle clustered variance computation separately
  if (clustered) {
    if (getOption("DID_DEBUG_VARIANCE", FALSE)) {
      cat("\n=== DEBUG: Clustered variance computation ===\n")
      cat("Rows in df:", nrow(df), "\n")
      cat("Sum of first_obs_by_gp_XX:", sum(df[["first_obs_by_gp_XX"]], na.rm = TRUE), "\n")
      cat("Sum of first_obs_by_clust_XX:", sum(df[["first_obs_by_clust_XX"]], na.rm = TRUE), "\n")
      cat("Unique clusters:", data.table::uniqueN(df[["cluster_XX"]]), "\n")
      cat("G_XX used:", G_XX, "\n")

      col1 <- "U_Gg_var_glob_1_XX"
      if (col1 %chin% names(df)) {
        cat("\nU_Gg_var_glob_1_XX stats:\n")
        cat("  Min:", min(df[[col1]], na.rm = TRUE), "\n")
        cat("  Max:", max(df[[col1]], na.rm = TRUE), "\n")
        cat("  Mean:", mean(df[[col1]], na.rm = TRUE), "\n")
        cat("  SD:", stats::sd(df[[col1]], na.rm = TRUE), "\n")
        cat("  Sum:", sum(df[[col1]], na.rm = TRUE), "\n")
        cat("  Sum of squares:", sum(df[[col1]]^2, na.rm = TRUE), "\n")
        cat("  Non-NA count:", sum(!is.na(df[[col1]])), "\n")
        cat("  Non-zero count:", sum(df[[col1]] != 0, na.rm = TRUE), "\n")

        first_obs_vals <- df[[col1]] * df[["first_obs_by_gp_XX"]]
        cat("\nFirst_obs filtered U_Gg_var_glob_1_XX:\n")
        cat("  Sum:", sum(first_obs_vals, na.rm = TRUE), "\n")
        cat("  Sum of squares:", sum(first_obs_vals^2, na.rm = TRUE), "\n")
        cat("  Non-zero count:", sum(first_obs_vals != 0, na.rm = TRUE), "\n")
      }
    }

    for (i in 1:l_XX) {
      col_var_glob <- paste0("U_Gg_var_glob_", i, "_XX")
      clust_sum_col <- paste0("clust_U_Gg_var_glob_", i, "_XX")

      # Step 1: Multiply by first_obs_by_gp_XX
      df[, .var_weighted := get(col_var_glob) * first_obs_by_gp_XX]

      # Step 2: Sum by cluster_XX
      df[, (clust_sum_col) := sum(.var_weighted, na.rm = TRUE), by = "cluster_XX"]

      # Step 3 & 4: Square and multiply by first_obs_by_clust_XX
      df[, .clust_var_sq := get(clust_sum_col)^2 * first_obs_by_clust_XX]

      # Step 5: Sum total
      var_sq_sum <- sum(df[[".clust_var_sq"]], na.rm = TRUE)

      if (getOption("DID_DEBUG_VARIANCE", FALSE) && i == 1) {
        cat("\nEffect", i, ":\n")
        cat("  Non-zero var_weighted:", sum(df[[".var_weighted"]] != 0 & !is.na(df[[".var_weighted"]])), "\n")
        cat("  Sum of var_weighted:", sum(df[[".var_weighted"]], na.rm = TRUE), "\n")
        cat("  Non-zero clust_var_sq:", sum(df[[".clust_var_sq"]] != 0 & !is.na(df[[".clust_var_sq"]])), "\n")
        cat("  var_sq_sum:", var_sq_sum, "\n")
        cat("  var_sq_sum / G_XX^2:", var_sq_sum / G_XX^2, "\n")
        cat("  SE:", sqrt(var_sq_sum / G_XX^2), "\n")
      }

      agg_df[[paste0("var_sq_sum_", i)]] <- var_sq_sum

      # Replace U_Gg_var_glob with cluster-summed values for later covariance computation
      df[, (col_var_glob) := get(clust_sum_col)]

      # Clean up temp columns
      df[, c(".var_weighted", ".clust_var_sq") := NULL]
    }
  }

  # Step 4: Process results for each effect
  for (i in 1:l_XX) {
    N1_i <- N1_vec[i]
    N0_i <- N0_vec[i]
    total_N <- N1_i + N0_i

    col_plus <- paste0("U_Gg", i, "_plus_XX")
    col_minus <- paste0("U_Gg", i, "_minus_XX")

    # Get aggregated values
    sum_plus <- if (paste0("sum_plus_", i) %chin% names(agg_df)) agg_df[[paste0("sum_plus_", i)]] else 0
    sum_minus <- if (paste0("sum_minus_", i) %chin% names(agg_df)) agg_df[[paste0("sum_minus_", i)]] else 0
    var_sq_sum <- agg_df[[paste0("var_sq_sum_", i)]]
    N_effects_i <- agg_df[[paste0("count_effects_", i)]]
    count_global_dw <- agg_df[[paste0("count_dw_", i)]]

    if (is.na(sum_plus)) sum_plus <- 0
    if (is.na(sum_minus)) sum_minus <- 0
    if (is.na(var_sq_sum)) var_sq_sum <- 0
    if (is.na(N_effects_i)) N_effects_i <- 0
    if (is.na(count_global_dw)) count_global_dw <- 0

    # Compute DID estimate
    # CRAN: DID = sum(U_Gg_global * first_obs) / G_XX
    if (total_N > 0) {
      DID_i <- (N1_i * sum_plus / G_XX + N0_i * sum_minus / G_XX) / total_N
    } else {
      DID_i <- NA
    }

    # Compute SE
    if (total_N > 0 && var_sq_sum > 0) {
      SE_i <- sqrt(var_sq_sum) / G_var
    } else {
      SE_i <- NA
    }

    N_switchers_i <- N1_i + N0_i

    # Handle normalization
    if (normalized == TRUE && total_N > 0) {
      delta_in <- if (exists(paste0("delta_D_", i, "_in_XX"))) get(paste0("delta_D_", i, "_in_XX")) else 0
      delta_out <- if (exists(paste0("delta_D_", i, "_out_XX"))) get(paste0("delta_D_", i, "_out_XX")) else 0
      delta_D_global <- (N1_i / total_N) * delta_in + (N0_i / total_N) * delta_out

      if (delta_D_global != 0 && !is.na(delta_D_global)) {
        DID_i <- DID_i / delta_D_global
        SE_i <- SE_i / delta_D_global
        assign(paste0("delta_D_", i, "_global_XX"), delta_D_global)
      }
    }

    # Check if effect can be estimated
    if ((switchers == "" && N1_i == 0 && N0_i == 0) ||
        (switchers == "out" && N0_i == 0) ||
        (switchers == "in" && N1_i == 0)) {
      DID_i <- NA
    }

    # Store results
    assign(paste0("DID_", i, "_XX"), DID_i)
    assign(paste0("Effect_", i), DID_i)
    assign(paste0("se_", i, "_XX"), SE_i)
    assign(paste0("se_effect_", i), SE_i)
    assign(paste0("N_switchers_effect_", i, "_XX"), N_switchers_i)
    assign(paste0("N_switchers_effect_", i), N_switchers_i)
    assign(paste0("N_effect_", i, "_XX"), N_effects_i)
    assign(paste0("N_effect_", i), N_effects_i)

    # Get N_dw values
    N1_dw <- if (exists(paste0("N1_dw_", i, "_XX"))) get(paste0("N1_dw_", i, "_XX")) else 0
    N0_dw <- if (exists(paste0("N0_dw_", i, "_XX"))) get(paste0("N0_dw_", i, "_XX")) else 0
    assign(paste0("N_switchers_effect_", i, "_dwXX"), N1_dw + N0_dw)
    assign(paste0("N_effect_", i, "_dwXX"), count_global_dw)

    # Store in matrix
    mat_res_XX[i, 1] <- DID_i
    mat_res_XX[i, 2] <- SE_i
    mat_res_XX[i, 3] <- if (!is.na(DID_i) && !is.na(SE_i)) DID_i - z_level * SE_i else NA
    mat_res_XX[i, 4] <- if (!is.na(DID_i) && !is.na(SE_i)) DID_i + z_level * SE_i else NA
    mat_res_XX[i, 5] <- count_global_dw
    mat_res_XX[i, 6] <- N1_dw + N0_dw
    mat_res_XX[i, 7] <- N_effects_i
    mat_res_XX[i, 8] <- N_switchers_i
    mat_res_XX[i, 9] <- i

    rownames <- append(rownames, paste0("Effect_", i, strrep(" ", (12 - nchar(paste0("Effect_", i))))))

    # Error message if DID_l cannot be estimated
    if (N_switchers_i == 0 || N_effects_i == 0) {
      message(paste0("Effect_", i, " cannot be estimated. There is no switcher or no control for this effect."))
    }
  }

  # Add count_dw columns to df for later use
  lapply(1:l_XX, function(i) {
    count_global_col <- paste0("count", i, "_global_XX")
    df[, (paste0("count", i, "_global_dwXX")) := data.table::fifelse(!is.na(get(count_global_col)) & get(count_global_col) > 0, 1, 0)]
  })

  ###### Computing the average total effect
  U_Gg_den_plus_XX <- 0
  U_Gg_den_minus_XX <- 0
  if ("U_Gg_den_plus_XX" %chin% names(df)) {
    result <- mean(df[["U_Gg_den_plus_XX"]], na.rm = TRUE)
    U_Gg_den_plus_XX <- if (is.na(result) || is.null(result)) 0 else result
  }
  if ("U_Gg_den_minus_XX" %chin% names(df)) {
    result <- mean(df[["U_Gg_den_minus_XX"]], na.rm = TRUE)
    U_Gg_den_minus_XX <- if (is.na(result) || is.null(result)) 0 else result
  }

  #### The average effect cannot be estimated when the trends_lin option is specified
  if (isFALSE(trends_lin)) {
    ## Computing the weight w_+
    sum_N1_l_XX <- sum(sapply(1:l_XX, function(i) get(paste0("N1_", i, "_XX_new"))))
    sum_N0_l_XX <- sum(sapply(1:l_XX, function(i) get(paste0("N0_", i, "_XX_new"))))

    if (switchers == "") {
      denom <- U_Gg_den_plus_XX * sum_N1_l_XX + U_Gg_den_minus_XX * sum_N0_l_XX
      w_plus_XX <- if (denom > 0) U_Gg_den_plus_XX * sum_N1_l_XX / denom else 0.5
    } else if (switchers == "out") {
      w_plus_XX <- 0
    } else if (switchers == "in") {
      w_plus_XX <- 1
    }

    # Compute average effect
    plus_vals <- if ("U_Gg_plus_XX" %chin% names(df)) data.table::nafill(df[["U_Gg_plus_XX"]], fill = 0) else rep(0, nrow(df))
    minus_vals <- if ("U_Gg_minus_XX" %chin% names(df)) data.table::nafill(df[["U_Gg_minus_XX"]], fill = 0) else rep(0, nrow(df))

    df[, U_Gg_global_XX := w_plus_XX * plus_vals + (1 - w_plus_XX) * minus_vals]

    # Sum for average effect
    sum_avg <- dt_sum_first_obs(df, "U_Gg_global_XX")
    delta_XX <- sum_avg / G_XX

    # Variance for average effect
    var_plus_vals <- if ("U_Gg_var_plus_XX" %chin% names(df)) data.table::nafill(df[["U_Gg_var_plus_XX"]], fill = 0) else rep(0, nrow(df))
    var_minus_vals <- if ("U_Gg_var_minus_XX" %chin% names(df)) data.table::nafill(df[["U_Gg_var_minus_XX"]], fill = 0) else rep(0, nrow(df))

    df[, U_Gg_var_global_XX := w_plus_XX * var_plus_vals + (1 - w_plus_XX) * var_minus_vals]

    var_sum_avg <- dt_compute_variance_sum(df, "U_Gg_var_global_XX", clustered)
    se_XX <- sqrt(var_sum_avg) / G_var

    assign("Av_tot_effect", delta_XX)
    assign("se_avg_total_effect", se_XX)

    # Store average effect in matrix
    mat_res_XX[l_XX + 1, 1] <- delta_XX
    mat_res_XX[l_XX + 1, 2] <- se_XX
    mat_res_XX[l_XX + 1, 3] <- delta_XX - z_level * se_XX
    mat_res_XX[l_XX + 1, 4] <- delta_XX + z_level * se_XX

    # Count switchers
    N_switchers_effect_XX <- sum(sapply(1:l_XX, function(i) get(paste0("N_switchers_effect_", i, "_XX"))))
    N_switchers_effect_dwXX <- sum(sapply(1:l_XX, function(i) get(paste0("N_switchers_effect_", i, "_dwXX"))))
    mat_res_XX[l_XX + 1, 8] <- N_switchers_effect_XX
    mat_res_XX[l_XX + 1, 6] <- N_switchers_effect_dwXX
    mat_res_XX[l_XX + 1, 9] <- 0
    assign("N_switchers_effect_average", N_switchers_effect_XX)

    # Build count_global using max across all effect counts
    count_cols_list <- lapply(1:l_XX, function(i) {
      v <- df[[paste0("count", i, "_global_XX")]]
      v[is.na(v)] <- 0
      v
    })
    df[, count_global_XX := do.call(pmax, count_cols_list)]

    # Count observations
    N_effect_XX <- dt_sum_col(df, "count_global_XX")
    N_effect_dwXX <- sum(!is.na(df[["count_global_XX"]]) & df[["count_global_XX"]] > 0)

    if (getOption("DID_DEBUG_COUNT", FALSE)) {
      cat("\n=== DEBUG: Average effect N computation ===\n")
      cat("count_global_XX exists:", "count_global_XX" %chin% names(df), "\n")
      if ("count_global_XX" %chin% names(df)) {
        cat("count_global_XX sample:\n")
        print(utils::head(df[, .(count_global_XX, first_obs_by_gp_XX)], 20))
        cat("N_effect_XX:", N_effect_XX, "\n")
      }
    }

    if (is.null(N_effect_XX) || is.na(N_effect_XX)) N_effect_XX <- 0
    if (is.null(N_effect_dwXX) || is.na(N_effect_dwXX)) N_effect_dwXX <- 0

    mat_res_XX[l_XX + 1, 7] <- N_effect_XX
    mat_res_XX[l_XX + 1, 5] <- N_effect_dwXX
    assign("N_avg_total_effect", N_effect_XX)
  }
  rownames <- append(rownames, paste0("Av_tot_eff", strrep(" ", (12 - nchar("Av_tot_eff")))))
  mat_res_XX[l_XX + 1, 9] <- 0

  #### Computing the placebo estimators
  if (l_placebo_XX != 0) {
    for (i in 1:l_placebo_XX) {
      N1_pl_i <- if (exists(paste0("N1_placebo_", i, "_XX_new"))) get(paste0("N1_placebo_", i, "_XX_new")) else 0
      N0_pl_i <- if (exists(paste0("N0_placebo_", i, "_XX_new"))) get(paste0("N0_placebo_", i, "_XX_new")) else 0
      total_N_pl <- N1_pl_i + N0_pl_i

      # Column names
      col_plus <- paste0("U_Gg_pl_", i, "_plus_XX")
      col_minus <- paste0("U_Gg_pl_", i, "_minus_XX")
      col_var_in <- paste0("U_Gg_var_pl_", i, "_in_XX")
      col_var_out <- paste0("U_Gg_var_pl_", i, "_out_XX")
      col_count_plus <- paste0("count", i, "_pl_plus_XX")
      col_count_minus <- paste0("count", i, "_pl_minus_XX")

      # Compute weighted placebo DID estimate
      sum_plus <- dt_sum_first_obs(df, col_plus)
      sum_minus <- dt_sum_first_obs(df, col_minus)

      if (total_N_pl > 0) {
        DID_pl_i <- (N1_pl_i * sum_plus / G_XX + N0_pl_i * sum_minus / G_XX) / total_N_pl
      } else {
        DID_pl_i <- NA
      }

      # Compute variance
      if (total_N_pl > 0) {
        w_in <- N1_pl_i / total_N_pl
        w_out <- N0_pl_i / total_N_pl

        var_in_vals <- if (col_var_in %chin% names(df)) data.table::nafill(df[[col_var_in]], fill = 0) else rep(0, nrow(df))
        var_out_vals <- if (col_var_out %chin% names(df)) data.table::nafill(df[[col_var_out]], fill = 0) else rep(0, nrow(df))

        df[, (paste0("U_Gg_var_glob_pl_", i, "_XX")) := w_in * var_in_vals + w_out * var_out_vals]

        var_sum_pl <- dt_compute_variance_sum(df, paste0("U_Gg_var_glob_pl_", i, "_XX"), clustered)
        SE_pl_i <- sqrt(var_sum_pl) / G_var

        # For clustered SE, update the df column with cluster-summed values for covariance computation
        if (clustered) {
          pl_col_name <- paste0("U_Gg_var_glob_pl_", i, "_XX")
          df[, .var_weighted_pl := get(pl_col_name) * first_obs_by_gp_XX]
          df[, .clust_var_sum_pl := sum(.var_weighted_pl, na.rm = TRUE), by = "cluster_XX"]
          df[, (pl_col_name) := .clust_var_sum_pl]
          df[, c(".var_weighted_pl", ".clust_var_sum_pl") := NULL]
        }
      } else {
        SE_pl_i <- NA
        df[, (paste0("U_Gg_var_glob_pl_", i, "_XX")) := 0]
      }

      # Count switchers and effects for placebo
      N_switchers_pl_i <- N1_pl_i + N0_pl_i

      count_plus_vals <- if (col_count_plus %chin% names(df)) df[[col_count_plus]] else rep(NA_real_, nrow(df))
      count_minus_vals <- if (col_count_minus %chin% names(df)) df[[col_count_minus]] else rep(NA_real_, nrow(df))
      df[, (paste0("count", i, "_pl_global_XX")) := data.table::fifelse(!is.na(count_plus_vals), count_plus_vals, count_minus_vals)]
      N_effects_pl_i <- dt_sum_col(df, paste0("count", i, "_pl_global_XX"))

      if (getOption("DID_DEBUG_COUNT", FALSE) && i == 1) {
        cat("\n=== DEBUG: Placebo 1 N computation ===\n")
        pl_col <- paste0("count", i, "_pl_global_XX")
        cat("Column", pl_col, "exists:", pl_col %chin% names(df), "\n")
        if (pl_col %chin% names(df)) {
          cat("Sample values:\n")
          print(utils::head(df[, .SD, .SDcols = c(pl_col, "first_obs_by_gp_XX")], 20))
          cat("N_effects_pl_i:", N_effects_pl_i, "\n")
        }
      }

      # Handle normalization
      if (normalized == TRUE && total_N_pl > 0) {
        delta_in <- if (exists(paste0("delta_D_pl_", i, "_in_XX"))) get(paste0("delta_D_pl_", i, "_in_XX")) else 0
        delta_out <- if (exists(paste0("delta_D_pl_", i, "_out_XX"))) get(paste0("delta_D_pl_", i, "_out_XX")) else 0
        delta_D_pl <- (N1_pl_i / total_N_pl) * delta_in + (N0_pl_i / total_N_pl) * delta_out

        if (delta_D_pl != 0 && !is.na(delta_D_pl)) {
          DID_pl_i <- DID_pl_i / delta_D_pl
          SE_pl_i <- SE_pl_i / delta_D_pl
          assign(paste0("delta_D_pl_", i, "_global_XX"), delta_D_pl)
        }
      }

      # Check if placebo can be estimated
      if ((switchers == "" && N1_pl_i == 0 && N0_pl_i == 0) ||
          (switchers == "out" && N0_pl_i == 0) ||
          (switchers == "in" && N1_pl_i == 0)) {
        DID_pl_i <- NA
      }

      # Store placebo results
      assign(paste0("DID_placebo_", i, "_XX"), DID_pl_i)
      assign(paste0("Placebo_", i), DID_pl_i)
      assign(paste0("se_placebo_", i, "_XX"), SE_pl_i)
      assign(paste0("se_placebo_", i), SE_pl_i)
      assign(paste0("N_switchers_placebo_", i, "_XX"), N_switchers_pl_i)
      assign(paste0("N_switchers_placebo_", i), N_switchers_pl_i)
      assign(paste0("N_placebo_", i, "_XX"), N_effects_pl_i)
      assign(paste0("N_placebo_", i), N_effects_pl_i)

      # Get N_dw values
      N1_dw_pl <- if (exists(paste0("N1_dw_placebo_", i, "_XX"))) get(paste0("N1_dw_placebo_", i, "_XX")) else 0
      N0_dw_pl <- if (exists(paste0("N0_dw_placebo_", i, "_XX"))) get(paste0("N0_dw_placebo_", i, "_XX")) else 0
      assign(paste0("N_switchers_placebo_", i, "_dwXX"), N1_dw_pl + N0_dw_pl)

      # Store in matrix
      mat_res_XX[l_XX + 1 + i, 1] <- DID_pl_i
      mat_res_XX[l_XX + 1 + i, 2] <- SE_pl_i
      mat_res_XX[l_XX + 1 + i, 3] <- if (!is.na(DID_pl_i) && !is.na(SE_pl_i)) DID_pl_i - z_level * SE_pl_i else NA
      mat_res_XX[l_XX + 1 + i, 4] <- if (!is.na(DID_pl_i) && !is.na(SE_pl_i)) DID_pl_i + z_level * SE_pl_i else NA

      # Count dw for placebo
      pl_col_name <- paste0("count", i, "_pl_global_XX")
      count_dw_pl <- sum(!is.na(df[[pl_col_name]]) & df[[pl_col_name]] > 0)
      if (is.na(count_dw_pl)) count_dw_pl <- 0

      mat_res_XX[l_XX + 1 + i, 5] <- count_dw_pl
      mat_res_XX[l_XX + 1 + i, 6] <- N1_dw_pl + N0_dw_pl
      mat_res_XX[l_XX + 1 + i, 7] <- N_effects_pl_i
      mat_res_XX[l_XX + 1 + i, 8] <- N_switchers_pl_i
      mat_res_XX[l_XX + 1 + i, 9] <- -i

      rownames <- append(rownames, paste0("Placebo_", i, strrep(" ", (12 - nchar(paste0("Placebo_", i))))))

      if (N_switchers_pl_i == 0 || N_effects_pl_i == 0) {
        message(paste0("Placebo_", i, " cannot be estimated. There is no switcher or no control for this placebo."))
      }
    }
  }

  # df is already a data.table - no conversion needed

  ## Average number of cumulated effects
  df[, paste0("delta_D_g_", 1:l_XX, "_XX") := NULL]
  df[, M_g_XX := data.table::fifelse(l_XX <= T_g_XX - F_g_XX + 1, as.numeric(l_XX), T_g_XX - F_g_XX + 1)]

  #### Calling variables delta_D_g_`i'_XX here like that does not work because switcher in/out are run one after another!!!

  ## second sum over g: total ... if F_g_XX<=T_g_XX
  ## actually I think it can be done in one total as we sum over the periods within groups and then across groups which are all different cells
  ## generate one variable that stores all the different delta_D_g_`i'_XX

  df[, delta_D_g_XX := 0]
  for (j in 1:l_XX) {
    col_plus <- paste0("delta_D_g_",j,"_plus_XX")
    col_minus <- paste0("delta_D_g_",j,"_minus_XX")
    df[, delta_D_g_XX_temp := data.table::fifelse(get(col_plus) != 0, get(col_plus), get(col_minus))]
    df[, delta_D_g_XX_temp := data.table::fifelse(delta_D_g_XX_temp == 0, NA_real_, delta_D_g_XX_temp)]
    df[, delta_D_g_XX := data.table::fifelse(switchers_tag_XX == j, delta_D_g_XX + delta_D_g_XX_temp, delta_D_g_XX)]
  }
  df[["delta_D_g_num_XX"]] <- df[["delta_D_g_XX"]] * (df[["M_g_XX"]] - (df[["switchers_tag_XX"]] - 1))
  delta_D_num_total <- sum(df[["delta_D_g_num_XX"]], na.rm = TRUE)
  delta_D_denom_total <- sum(df[["delta_D_g_XX"]], na.rm = TRUE)
  delta_D_avg_total <- delta_D_num_total / delta_D_denom_total

  ###### Fast path for bootstrap: return point estimates only
  if (isTRUE(point_estimates_only)) {
    rownames(mat_res_XX) <- rownames
    colnames(mat_res_XX) <- c("Estimate", "SE", "LB CI", "UB CI", "N", "Switchers", "N.w", "Switchers.w", "Time")

    Effect_mat <- matrix(mat_res_XX[1:l_XX, 1:(ncol(mat_res_XX) - 1)],
                         ncol = ncol(mat_res_XX) - 1, nrow = l_XX)
    rownames(Effect_mat) <- rownames[1:l_XX]
    colnames(Effect_mat) <- c("Estimate", "SE", "LB CI", "UB CI", "N", "Switchers", "N.w", "Switchers.w")

    ATE_mat <- matrix(mat_res_XX[l_XX + 1, 1:(ncol(mat_res_XX) - 1)],
                      ncol = ncol(mat_res_XX) - 1, nrow = 1)
    rownames(ATE_mat) <- rownames[l_XX + 1]
    colnames(ATE_mat) <- c("Estimate", "SE", "LB CI", "UB CI", "N", "Switchers", "N.w", "Switchers.w")

    did_multiplegt_dyn <- list(
      N_Effects = l_XX,
      N_Placebos = l_placebo_XX,
      Effects = Effect_mat,
      ATE = ATE_mat,
      delta_D_avg_total = delta_D_avg_total,
      max_pl = max_pl_XX,
      max_pl_gap = max_pl_gap_XX
    )

    if (l_placebo_XX > 0) {
      Placebo_mat <- matrix(mat_res_XX[(l_XX + 2):nrow(mat_res_XX), 1:(ncol(mat_res_XX) - 1)],
                            ncol = ncol(mat_res_XX) - 1, nrow = l_placebo_XX)
      rownames(Placebo_mat) <- rownames[(l_XX + 2):nrow(mat_res_XX)]
      colnames(Placebo_mat) <- c("Estimate", "SE", "LB CI", "UB CI", "N", "Switchers", "N.w", "Switchers.w")
      did_multiplegt_dyn$Placebos <- Placebo_mat
    }

    ret <- list(
      df = df,
      did_multiplegt_dyn = did_multiplegt_dyn,
      delta = list(),
      l_XX = l_XX,
      T_max_XX = T_max_XX,
      mat_res_XX = mat_res_XX
    )
    if (placebo != 0) ret$l_placebo_XX <- l_placebo_XX
    ret$coef <- list(b = mat_res_XX[-(l_XX + 1), 1], vcov = NULL)
    return(ret)
  }

  ###### 6. Computing p-values from the tests

  # If the option cluster is specified, we have previously replaced U_Gg_var_glob_pl_`i'_XX by clust_U_Gg_var_glob_pl_`i'_XX, and U_Gg_var_glob_`i'_XX by clust_U_Gg_var_glob_`i'_XX.
  # Now, we must also replace first_obs_by_gp_XX by first_obs_by_clust_XX
  if (!is.null(cluster)) {
    df[["first_obs_by_gp_XX"]] <- df[["first_obs_by_clust_XX"]]
  }

  ###### Performing a test to see whether all effects are jointly equal to 0
  all_Ns_not_zero <- NA
  all_delta_not_zero <- NA
  p_jointeffects <- NULL
  ## Test can only be run when at least two effects requested:
  if (l_XX != 0 && l_XX > 1) {
    ## If test is feasible, initalize scalar at 0
    all_Ns_not_zero <- 0
    all_delta_not_zero <- 0

    ## Count the number of estimated effects included in the test
    all_Ns_not_zero <- sum(vapply(1:l_XX, function(i) {
      (switchers == "" && (get(paste0("N1_",i,"_XX_new")) != 0 || get(paste0("N0_",i,"_XX_new")) != 0)) ||
      (switchers == "out" && get(paste0("N0_",i,"_XX_new")) != 0) ||
      (switchers == "in" && get(paste0("N1_",i,"_XX_new")) != 0)
    }, logical(1)))

    all_delta_not_zero <- if (isTRUE(normalized)) {
      sum(vapply(1:l_XX, function(i) {
        val <- get(paste0("delta_D_",i,"_global_XX"))
        val != 0 && !is.na(val)
      }, logical(1)))
    } else { 0 }

    ## Test can only be run when all requested effects could be computed:
    if ((all_Ns_not_zero == l_XX && isFALSE(normalized)) || (isTRUE(normalized) && all_Ns_not_zero == l_XX && all_delta_not_zero == l_XX)) {

      ## Creating a vector with all dynamic effect estimates
      didmgt_Effects <- matrix(0, nrow = l_XX, ncol = 1)

      ## Creating a matrix where the variances and the covariances of the effects will be stored.
      didmgt_Var_Effects <- matrix(0, nrow = l_XX, ncol = l_XX)

      ## Fill those matrices
      for (i in 1:l_XX) {
        didmgt_Effects[i,1] <- get(paste0("DID_",i,"_XX"))
        didmgt_Var_Effects[i,i] <- get(paste0("se_",i,"_XX"))^2

        if (i < l_XX) {
          for (j in (i+1):l_XX) {
            ## Create variables necessary to compute the covariances
            if (normalized == FALSE) {
              df[[paste0("U_Gg_var_",i,"_",j,"_XX")]] <- df[[paste0("U_Gg_var_glob_",i,"_XX")]] +  df[[paste0("U_Gg_var_glob_",j,"_XX")]]
            } else {
              df[[paste0("U_Gg_var_",i,"_",j,"_XX")]] <- df[[paste0("U_Gg_var_glob_",i,"_XX")]] / get(paste0("delta_D_",i,"_global_XX")) +  df[[paste0("U_Gg_var_glob_",j,"_XX")]] / get(paste0("delta_D_",j,"_global_XX"))
            }

            ## Estimate the covariances
            df[[paste0("U_Gg_var_",i,"_",j,"_2_XX")]] <- df[[paste0("U_Gg_var_",i,"_",j,"_XX")]]^2 * df[["first_obs_by_gp_XX"]]
            assign(paste0("var_sum_",i,"_",j,"_XX"), sum( df[[paste0("U_Gg_var_",i,"_",j,"_2_XX")]], na.rm = TRUE) / G_XX^2)
            assign(paste0("cov_",i,"_",j,"_XX"), (get(paste0("var_sum_",i,"_",j,"_XX")) - get(paste0("se_",i,"_XX"))^2 - get(paste0("se_",j,"_XX"))^2)/2)

            ## Store the results
            didmgt_Var_Effects[i,j] <- get(paste0("cov_",i,"_",j,"_XX"))
            didmgt_Var_Effects[j,i] <- get(paste0("cov_",i,"_",j,"_XX"))
          }
        }
      }

      ## Compute P-value for the F-test on joint nullity of all effects
      ## Check if variance matrix is invertible
      eig_effects <- eigen(didmgt_Var_Effects, only.values = TRUE)$values
      eig_effects_real <- Re(eig_effects[abs(Im(eig_effects)) < 1e-10])
      eig_effects_pos <- eig_effects_real[eig_effects_real > 1e-10]

      if (length(eig_effects_pos) < l_XX) {
        ## Matrix is singular/not invertible
        p_jointeffects <- NA
        warn_msg <- "The F-test that all effects are equal to zero is not computed because the variance of effects is not invertible. This can for instance happen if you cluster standard errors and you have more effect estimators than clusters."
        vcov_warnings <- c(vcov_warnings, warn_msg)
        warning(warn_msg)
      } else {
        warning_eff_ratio <- max(eig_effects_pos) / min(eig_effects_pos)
        if (warning_eff_ratio >= 1000) {
          warn_msg <- "The F-test that all effects are equal to zero may not be reliable, because the variance of the effects is close to not being invertible (the ratio of its largest and smallest eigenvalues is larger than 1000). This can for instance happen when you compute many effects estimators, or when your effects are very strongly correlated."
          vcov_warnings <- c(vcov_warnings, warn_msg)
          warning(warn_msg)
        }
        didmgt_Var_Effects_inv <- MASS::ginv(didmgt_Var_Effects)
        didmgt_chi2effects <- t(didmgt_Effects) %*% didmgt_Var_Effects_inv  %*% didmgt_Effects
        p_jointeffects <- 1 - stats::pchisq(didmgt_chi2effects[1,1], df = l_XX)
      }
    } else {
      p_jointeffects <- NA
      ## Error message if not all of the specified effects could be estimated
      message("Some effects could not be estimated. Therefore, the test of joint nullity of the effects could not be computed.")
    }
  }


  ###### Performing a test to see whether all placebos are jointly equal to 0
  all_Ns_pl_not_zero <- NA
  all_delta_pl_not_zero <- NA
  ## Test can only be run when at least two placebos requested:
  if (l_placebo_XX != 0 && l_placebo_XX > 1) {
    ## If test is feasible, initalize scalar at 0
    all_Ns_pl_not_zero <- 0
    all_delta_pl_not_zero <- 0

    ## Count the number of estimated placebos included in the test
    all_Ns_pl_not_zero <- sum(vapply(1:l_placebo_XX, function(i) {
      (switchers == "" && (get(paste0("N1_placebo_",i,"_XX_new")) != 0 || get(paste0("N0_placebo_",i,"_XX_new")) != 0)) ||
      (switchers == "out" && get(paste0("N0_placebo_",i,"_XX_new")) != 0) ||
      (switchers == "in" && get(paste0("N1_placebo_",i,"_XX_new")) != 0)
    }, logical(1)))

    all_delta_pl_not_zero <- if (isTRUE(normalized)) {
      sum(vapply(1:l_placebo_XX, function(i) {
        val <- get(paste0("delta_D_pl_",i,"_global_XX"))
        val != 0 && !is.na(val)
      }, logical(1)))
    } else { 0 }

    ## Test can only be run when all requested placebos could be computed:
    if ((all_Ns_pl_not_zero == l_placebo_XX && isFALSE(normalized)) || (isTRUE(normalized) && all_Ns_pl_not_zero == l_placebo_XX && all_delta_pl_not_zero == l_placebo_XX)) {

      ## Creating a vector with all placebo estimates
      didmgt_Placebo <- matrix(0, nrow = l_placebo_XX, ncol = 1)

      ## Creating a matrix where the variances and the covariances of the placebos will be stored.
      didmgt_Var_Placebo <- matrix(0, nrow = l_placebo_XX, ncol = l_placebo_XX)

      ## Fill those matrices
      for (i in 1:l_placebo_XX) {
        didmgt_Placebo[i,1] <- get(paste0("DID_placebo_",i,"_XX"))
        didmgt_Var_Placebo[i,i] <- get(paste0("se_placebo_",i,"_XX"))^2

        if (i < l_placebo_XX) {
          for (j in (i+1):l_placebo_XX) {
            ## Create variables necessary to compute the covariances
            if (normalized == FALSE) {
              df[[paste0("U_Gg_var_pl_",i,"_",j,"_XX")]] <- df[[paste0("U_Gg_var_glob_pl_",i,"_XX")]] +  df[[paste0("U_Gg_var_glob_pl_",j,"_XX")]]
            } else {
              df[[paste0("U_Gg_var_pl_",i,"_",j,"_XX")]] <- df[[paste0("U_Gg_var_glob_pl_",i,"_XX")]] / get(paste0("delta_D_pl_",i,"_global_XX")) +  df[[paste0("U_Gg_var_glob_pl_",j,"_XX")]] / get(paste0("delta_D_pl_",j,"_global_XX"))
            }

            ## Estimate the covariances
            df[[paste0("U_Gg_var_pl_",i,"_",j,"_2_XX")]] <- df[[paste0("U_Gg_var_pl_",i,"_",j,"_XX")]]^2 * df[["first_obs_by_gp_XX"]]
            assign(paste0("var_sum_pl_",i,"_",j,"_XX"), sum( df[[paste0("U_Gg_var_pl_",i,"_",j,"_2_XX")]], na.rm = TRUE) / G_XX^2)
            assign(paste0("cov_pl_",i,"_",j,"_XX"), (get(paste0("var_sum_pl_",i,"_",j,"_XX")) - get(paste0("se_placebo_",i,"_XX"))^2 - get(paste0("se_placebo_",j,"_XX"))^2)/2)

            ## Store the results
            didmgt_Var_Placebo[i,j] <- get(paste0("cov_pl_",i,"_",j,"_XX"))
            didmgt_Var_Placebo[j,i] <- get(paste0("cov_pl_",i,"_",j,"_XX"))
          }
        }
      }

      ## Compute P-value for the F-test on joint nullity of all placebos
      ## Check if variance matrix is invertible
      eig_placebo <- eigen(didmgt_Var_Placebo, only.values = TRUE)$values
      eig_placebo_real <- Re(eig_placebo[abs(Im(eig_placebo)) < 1e-10])
      eig_placebo_pos <- eig_placebo_real[eig_placebo_real > 1e-10]

      if (length(eig_placebo_pos) < l_placebo_XX) {
        ## Matrix is singular/not invertible
        p_jointplacebo <- NA
        warn_msg <- "The F-test that all placebos are equal to zero is not computed because the variance of placebos is not invertible. This can for instance happen if you cluster standard errors and you have more placebo estimators than clusters."
        vcov_warnings <- c(vcov_warnings, warn_msg)
        warning(warn_msg)
      } else {
        warning_pl_ratio <- max(eig_placebo_pos) / min(eig_placebo_pos)
        if (warning_pl_ratio >= 1000) {
          warn_msg <- "The F-test that all placebos are equal to zero may not be reliable, because the variance of the placebos is close to not being invertible (the ratio of its largest and smallest eigenvalues is larger than 1000). This can for instance happen when you compute many placebo estimators, or when your placebos are very strongly correlated."
          vcov_warnings <- c(vcov_warnings, warn_msg)
          warning(warn_msg)
        }
        didmgt_Var_Placebo_inv <- MASS::ginv(didmgt_Var_Placebo)
        didmgt_chi2placebo <- t(didmgt_Placebo) %*% didmgt_Var_Placebo_inv  %*% didmgt_Placebo
        p_jointplacebo <- 1 - stats::pchisq(didmgt_chi2placebo[1,1], df = l_placebo_XX)
      }
    } else {
      p_jointplacebo <- NA
      ## Error message if not all of the specified placebos could be estimated
      message("Some placebos could not be estimated. Therefore, the test of joint nullity of the placebos could not be computed.")
    }
  }

  ###### Testing for effect heterogeneity
  if (!is.null(predict_het)) {
    ## Define number of effects we want to calculate
    if (length(predict_het_good) > 0L) {
      if (-1 %in% het_effects) {
        het_effects <- 1:l_XX
      }
      all_effects_XX <- c(1:l_XX)[het_effects]
      if (NA %in% all_effects_XX) {
        ## error if specified effects not matching with those actually calculated
        stop("Error in predict_het second argument: please specify only numbers that are smaller or equal to the number you request in effects()")
      }

      # Preliminaries: Yg Fg1
      df[, Yg_Fg_min1_XX := data.table::fifelse(time_XX == F_g_XX - 1, outcome_non_diff_XX, NA_real_)]
      df[, Yg_Fg_min1_XX := mean(Yg_Fg_min1_XX, na.rm = TRUE), by = "group_XX"]
      df[["feasible_het_XX"]] <- !is.na(df[["Yg_Fg_min1_XX"]])
      if (!is.null(trends_lin)) {
        df[, Yg_Fg_min2_XX := data.table::fifelse(time_XX == F_g_XX - 2, outcome_non_diff_XX, NA_real_)]
        df[, Yg_Fg_min2_XX := mean(Yg_Fg_min2_XX, na.rm = TRUE), by = "group_XX"]
        df[, Yg_Fg_min2_XX := data.table::fifelse(is.nan(Yg_Fg_min2_XX), NA_real_, Yg_Fg_min2_XX)]

        df[["feasible_het_XX"]] <- df[["feasible_het_XX"]] & !is.na(df[["Yg_Fg_min2_XX"]])
      }
      data.table::setorder(df, group_XX, time_XX)
      df[, gr_id := seq_len(.N), by = "group_XX"]

      lhyp <- paste0(predict_het_good, "=0")

      het_res <- data.frame()
      ## Loop the procedure over all requested effects for which potential heterogeneity should be predicted
      for (i in all_effects_XX) {
        # Generation of factor dummies for regression
        het_sample <- df[F_g_XX - 1 + i <= T_g_XX & feasible_het_XX == TRUE,
                         .SD, .SDcols = c("F_g_XX", "d_sq_XX", "S_g_XX", trends_nonparam)]
        het_interact <- ""
        for (v in c("F_g_XX", "d_sq_XX", "S_g_XX", trends_nonparam)) {
          if (length(levels(as.factor(het_sample[[v]]))) > 1) {
            df[[paste0(v,"_h")]] <- factor(df[[v]])
            for (l in levels(df[[paste0(v,"_h")]])) {
              df[[paste0(v,"_h",l)]] <- as.numeric(df[[v]] == l)
            }
            het_interact <- paste0(het_interact,":",v,"_h")
          }
        }
        het_interact <- substr(het_interact,2,nchar(het_interact))
        het_sample <- NULL

        # Yg,Fg-1 + l
        df[, paste0("Yg_Fg_", i, "_XX") := data.table::fifelse(time_XX == F_g_XX - 1 + i, outcome_non_diff_XX, NA_real_)]
        df[, paste0("Yg_Fg_",i,"_XX") := mean(get(paste0("Yg_Fg_",i,"_XX")), na.rm = TRUE), by = "group_XX"]

        df[["diff_het_XX"]] <- df[[paste0("Yg_Fg_",i,"_XX")]] - df[["Yg_Fg_min1_XX"]]
        if (isTRUE(trends_lin)) {
          df[["diff_het_XX"]] <- df[["diff_het_XX"]] - i * (df[["Yg_Fg_min1_XX"]] - df[["Yg_Fg_min2_XX"]])
        }

        df[[paste0("prod_het_",i,"_XX")]] <- df[["S_g_het_XX"]] * df[["diff_het_XX"]]
        df[["diff_het_XX"]] <- NULL

        # keep one observation by group to not artificially increase sample
        col_prod <- paste0("prod_het_",i,"_XX")
        df[, (col_prod) := data.table::fifelse(gr_id == 1, get(col_prod), NA_real_)]

        # In order to perform the test with coeftest, we need a vector of non missing regression coefficients. To avoid collinearity, we run the regression two times: the first time with the full set of regressors (F_g_XX_h#d_sq_XX_h#S_g_XX_h), then with just the non-collinear variables.
        het_reg <- paste0("prod_het_",i,"_XX ~ ", paste(predict_het_good, collapse = " + "), " + ", het_interact)
        het_sample <- df[F_g_XX - 1 + i <= T_g_XX]
        model <- stats::lm(stats::as.formula(het_reg), data = het_sample, weights = het_sample[["weight_XX"]])
        het_reg <- gsub(het_interact, "", het_reg)
        keep_k <- names(model$coefficients)[!names(model$coefficients) %chin% c("(Intercept)", predict_het_good) & !is.na(model$coefficients)]
        if (length(keep_k) > 0L) het_reg <- paste0(het_reg, " + ", paste(keep_k, collapse = " + "))
        model <- stats::lm(stats::as.formula(het_reg), data = het_sample, weights = het_sample[["weight_XX"]])
        ## Use HC2 with dfadjust when predict_het_hc2bm is TRUE
        if (isTRUE(predict_het_hc2bm)) {
          ## Determine cluster variable for HC2 BM
          if (!is.null(cluster)) {
            cluster_het <- het_sample[[cluster]]
          } else {
            cluster_het <- het_sample[["group_XX"]]
          }
          het_vcov <- sandwich::vcovCL(model, cluster = cluster_het, type = "HC2", cadjust = TRUE)
        } else {
          het_vcov <- sandwich::vcovHC(model, type = "HC2")
        }
        model_r <- matrix(lmtest::coeftest(model, vcov. = het_vcov)[2:(length(predict_het_good)+1), 1:3], ncol = 3)
        f_stat <- car::linearHypothesis(model, lhyp, vcov = het_vcov)[["Pr(>F)"]][2]
        t_stat <- stats::qt(0.975, stats::df.residual(model))
        het_sample <- NULL

        ## Output Part of the predict_het option
        het_res <- rbind(het_res, data.frame(
          effect = matrix(i, nrow = length(predict_het_good)),
          covariate = predict_het_good,
          Estimate = model_r[1:nrow(model_r),1],
          SE = model_r[1:nrow(model_r),2],
          t = model_r[1:nrow(model_r),3],
          LB = model_r[1:nrow(model_r),1] - t_stat * model_r[1:nrow(model_r),2],
          UB = model_r[1:nrow(model_r),1] + t_stat * model_r[1:nrow(model_r),2],
          N = matrix(stats::nobs(model), nrow = length(predict_het_good)),
          pF = matrix(f_stat, nrow = length(predict_het_good))
        ))
      }
      df[, paste0(c("F_g_XX", "d_sq_XX", "S_g_XX", trends_nonparam), "_h") := NULL]
      het_res <- het_res[order(het_res$covariate, het_res$effect), ]
    }

    if (l_placebo_XX > 0) {
      if (-1 %in% predict_het[2]) {
        all_effects_pl_XX <- 1:l_placebo_XX
      } else {
        if (max(het_effects) > l_placebo_XX) {
          stop("You specified some numbers in predict_het that exceed the number of placebos possible to estimate! Please specify only numbers that are smaller or equal to the number of placebos you requested.")
        } else {
          all_effects_pl_XX <- het_effects
        }
      }

      for (i in all_effects_pl_XX) {
        # Generation of factor dummies for regression
        het_sample <- df[F_g_XX - 1 + i <= T_g_XX & feasible_het_XX == TRUE,
                         .SD, .SDcols = c("F_g_XX", "d_sq_XX", "S_g_XX", trends_nonparam)]
        het_interact <- ""
        for (v in c("F_g_XX", "d_sq_XX", "S_g_XX", trends_nonparam)) {
          if (length(levels(as.factor(het_sample[[v]]))) > 1) {
            df[[paste0(v,"_h")]] <- factor(df[[v]])
            for (l in levels(df[[paste0(v,"_h")]])) {
              df[[paste0(v,"_h",l)]] <- as.numeric(df[[v]] == l)
            }
            het_interact <- paste0(het_interact,":",v,"_h")
          }
        }
        het_interact <- substr(het_interact,2,nchar(het_interact))
        het_sample <- NULL

        # Yg,Fg-1 + l
        df[, paste0("Yg_Fg_pl_", i, "_XX") := data.table::fifelse(time_XX == F_g_XX - 1 - i, outcome_non_diff_XX, NA_real_)]
        df[, paste0("Yg_Fg_pl_",i,"_XX") := mean(get(paste0("Yg_Fg_pl_",i,"_XX")), na.rm = TRUE), by = "group_XX"]

        df[["diff_het_pl_XX"]] <- df[[paste0("Yg_Fg_pl_",i,"_XX")]] - df[["Yg_Fg_min1_XX"]]
        if (isTRUE(trends_lin)) {
          df[["diff_het_pl_XX"]] <- df[["diff_het_pl_XX"]] - i * (df[["Yg_Fg_min1_XX"]] - df[["Yg_Fg_min2_XX"]])
        }

        # Now we can generate
        df[[paste0("prod_het_pl_",i,"_XX")]] <- df[["S_g_het_XX"]] * df[["diff_het_pl_XX"]]
        df[["diff_het_pl_XX"]] <- NULL

        # keep one observation by group to not artificially increase sample
        col_prod_pl <- paste0("prod_het_pl_",i,"_XX")
        df[, (col_prod_pl) := data.table::fifelse(gr_id == 1, get(col_prod_pl), NA_real_)]

        # In order to perform the test with coeftest, we need a vector of non missing regression coefficients. To avoid collinearity, we run the regression two times: the first time with the full set of regressors (F_g_XX_h#d_sq_XX_h#S_g_XX_h), then with just the non-collinear variables.
        het_reg <- paste0("prod_het_pl_",i,"_XX ~ ", paste(predict_het_good, collapse = " + "), " + ", het_interact)
        het_sample <- df[F_g_XX - 1 + i <= T_g_XX]
        model <- stats::lm(stats::as.formula(het_reg), data = het_sample, weights = het_sample[["weight_XX"]])
        het_reg <- gsub(het_interact, "", het_reg)
        keep_k <- names(model$coefficients)[!names(model$coefficients) %chin% c("(Intercept)", predict_het_good) & !is.na(model$coefficients)]
        if (length(keep_k) > 0L) het_reg <- paste0(het_reg, " + ", paste(keep_k, collapse = " + "))
        model <- stats::lm(stats::as.formula(het_reg), data = het_sample, weights = het_sample[["weight_XX"]])
        ## Use HC2 with dfadjust when predict_het_hc2bm is TRUE (placebo section)
        if (isTRUE(predict_het_hc2bm)) {
          ## Determine cluster variable for HC2 BM
          if (!is.null(cluster)) {
            cluster_het <- het_sample[[cluster]]
          } else {
            cluster_het <- het_sample[["group_XX"]]
          }
          het_vcov <- sandwich::vcovCL(model, cluster = cluster_het, type = "HC2", cadjust = TRUE)
        } else {
          het_vcov <- sandwich::vcovHC(model, type = "HC2")
        }
        model_r <- matrix(lmtest::coeftest(model, vcov. = het_vcov)[2:(length(predict_het_good)+1), 1:3], ncol = 3)
        f_stat <- car::linearHypothesis(model, lhyp, vcov = het_vcov)[["Pr(>F)"]][2]
        t_stat <- stats::qt(0.975, stats::df.residual(model))
        het_sample <- NULL

        ## Output Part of the predict_het option (placebos)
        het_res <- rbind(het_res, data.frame(
          effect = matrix(-i, nrow = length(predict_het_good)),
          covariate = predict_het_good,
          Estimate = model_r[1:nrow(model_r),1],
          SE = model_r[1:nrow(model_r),2],
          t = model_r[1:nrow(model_r),3],
          LB = model_r[1:nrow(model_r),1] - t_stat * model_r[1:nrow(model_r),2],
          UB = model_r[1:nrow(model_r),1] + t_stat * model_r[1:nrow(model_r),2],
          N = matrix(stats::nobs(model), nrow = length(predict_het_good)),
          pF = matrix(f_stat, nrow = length(predict_het_good))
        ))
      }
      df[, paste0(c("F_g_XX", "d_sq_XX", "S_g_XX", trends_nonparam), "_h") := NULL]
      het_res <- het_res[order(het_res$covariate, het_res$effect), ]
    }
  }

  ###### Performing a test that all DID_\ell effects are equal (similar structure as test on placebos, not commented, except for the small differences with placebos)
  if (effects_equal == TRUE && l_XX > 1) {
    ## Determine bounds for the test (default is all effects)
    ee_lb <- if (is.null(effects_equal_lb)) 1 else effects_equal_lb
    ee_ub <- if (is.null(effects_equal_ub)) l_XX else effects_equal_ub

    ## Validate bounds
    if (ee_ub > l_XX) {
      message(sprintf("Upper bound %d exceeds number of effects %d. Using %d as upper bound.", ee_ub, l_XX, l_XX))
      ee_ub <- l_XX
    }

    ee_length <- ee_ub - ee_lb + 1

    all_Ns_not_zero <- sum(vapply(ee_lb:ee_ub, function(i) {
      (switchers == "" && (get(paste0("N1_",i,"_XX_new")) != 0 || get(paste0("N0_",i,"_XX_new")) != 0)) ||
      (switchers == "out" && get(paste0("N0_",i,"_XX_new")) != 0) ||
      (switchers == "in" && get(paste0("N1_",i,"_XX_new")) != 0)
    }, logical(1)))
    if (all_Ns_not_zero == ee_length) {
      didmgt_Effects <- mat_res_XX[ee_lb:ee_ub, 1]
      didmgt_Var_Effects <- matrix(0, nrow = ee_length, ncol = ee_length)
      didmgt_identity <- matrix(0, nrow = ee_length - 1, ncol = ee_length)

      for (i in ee_lb:ee_ub) {
        ## Index in the submatrix (1-based within range)
        ii <- i - ee_lb + 1
        if (((switchers == "" && (get(paste0("N1_",i,"_XX_new")) != 0 || get(paste0("N0_",i,"_XX_new")) != 0)) ||
            (switchers == "out" && get(paste0("N0_",i,"_XX_new")) != 0) ||
            (switchers == "in" && get(paste0("N1_",i,"_XX_new")) != 0))) {

          didmgt_Var_Effects[ii,ii] <- get(paste0("se_", i, "_XX")) ^ 2
          if (ii < ee_length) {
            didmgt_identity[ii,ii] <- 1
          }

          if (i < ee_ub) {
            for (j in (i + 1):ee_ub) {
              ## Index in the submatrix for j
              jj <- j - ee_lb + 1
              if (normalized == FALSE) {
                df[[paste0("U_Gg_var_", i, "_", j,"_XX")]] <- df[[paste0("U_Gg_var_glob_", i, "_XX")]] +  df[[paste0("U_Gg_var_glob_", j, "_XX")]]
              } else {
                df[[paste0("U_Gg_var_", i, "_", j,"_XX")]] <-
                  (df[[paste0("U_Gg_var_glob_", i, "_XX")]] / get(paste0("delta_D_",i,"_global_XX"))) +
                  (df[[paste0("U_Gg_var_glob_", j, "_XX")]] / get(paste0("delta_D_",j,"_global_XX")))
              }

              df[[paste0("U_Gg_var_", i, "_", j, "_2_XX")]] <- df[[paste0("U_Gg_var_", i, "_", j, "_XX")]]^2 * df[["first_obs_by_gp_XX"]]
              assign(paste0("var_sum_",i,"_",j,"_XX"),
                    sum(df[[paste0("U_Gg_var_", i, "_", j, "_2_XX")]], na.rm = TRUE)/
                      G_XX^2)
              assign(paste0("cov_",i,"_",j,"_XX"),
                    (get(paste0("var_sum_",i,"_",j,"_XX")) - get(paste0("se_",i,"_XX"))^2 - get(paste0("se_",j,"_XX"))^2)/2)

              didmgt_Var_Effects[ii,jj] <- get(paste0("cov_",i,"_",j,"_XX"))
              didmgt_Var_Effects[jj,ii] <- get(paste0("cov_",i,"_",j,"_XX"))
            }
          }

        }
      }

      ## Creating a matrix of demeaned effects: null being tested = joint equality, not jointly 0
      didmgt_D <- didmgt_identity - matrix(1/ee_length, nrow = ee_length - 1, ncol = ee_length)
      didmgt_test_effects <- didmgt_D %*% didmgt_Effects
      didmgt_test_var <- didmgt_D %*% didmgt_Var_Effects %*% t(didmgt_D)
      # Enforcing symmetry
      didmgt_test_var <- (didmgt_test_var + t(didmgt_test_var)) / 2

      ## Check if variance matrix is invertible for equality test
      eig_equality <- eigen(didmgt_test_var, only.values = TRUE)$values
      eig_equality_real <- Re(eig_equality[abs(Im(eig_equality)) < 1e-10])
      eig_equality_pos <- eig_equality_real[eig_equality_real > 1e-10]

      if (length(eig_equality_pos) < (ee_length - 1)) {
        p_equality_effects <- NA
        if (ee_lb == 1 && ee_ub == l_XX) {
          warn_msg <- "The F-test that all effects are equal is not computed because the variance of effects is not invertible. This may be due to perfect multicollinearity among the effects. Consider reducing the number of effects estimated."
        } else {
          warn_msg <- sprintf("The F-test that effects %d to %d are equal is not computed because the variance of effects is not invertible. This may be due to perfect multicollinearity among the effects.", ee_lb, ee_ub)
        }
        vcov_warnings <- c(vcov_warnings, warn_msg)
        warning(warn_msg)
        assign("p_equality_effects", p_equality_effects, inherits = TRUE)
      } else {
        warning_eq_ratio <- max(eig_equality_pos) / min(eig_equality_pos)
        if (warning_eq_ratio >= 1000) {
          if (ee_lb == 1 && ee_ub == l_XX) {
            warn_msg <- "The F-test that all effects are equal may not be reliable, because the variance of the effects is close to not being invertible (the ratio of its largest and smallest eigenvalues is larger than 1000). This may be due to strong multicollinearity among the effects. Consider reducing the number of effects estimated."
          } else {
            warn_msg <- sprintf("The F-test that effects %d to %d are equal may not be reliable, because the variance of the effects is close to not being invertible (the ratio of its largest and smallest eigenvalues is larger than 1000).", ee_lb, ee_ub)
          }
          vcov_warnings <- c(vcov_warnings, warn_msg)
          warning(warn_msg)
        }
        didmgt_chi2_equal_ef <- t(didmgt_test_effects) %*% MASS::ginv(didmgt_test_var) %*% didmgt_test_effects
        p_equality_effects <-
          1 - stats::pchisq(didmgt_chi2_equal_ef[1,1], df = ee_length - 1)
        assign("p_equality_effects", p_equality_effects, inherits = TRUE)
      }
    } else {
      if (ee_lb == 1 && ee_ub == l_XX) {
        message("Some effects could not be estimated. Therefore, the test of equality of effects could not be computed.")
      } else {
        message(sprintf("Some effects in range %d to %d could not be estimated. Therefore, the test of equality of effects could not be computed.", ee_lb, ee_ub))
      }
    }
  }

  ###### Storing coefficients, variances and covariances of the estimators
  l_tot_XX <- l_XX + l_placebo_XX
  didmgt_vcov <- matrix(NA, nrow = l_tot_XX, ncol = l_tot_XX)
  mat_names <-
    colnames(didmgt_vcov) <- rownames(didmgt_vcov) <- sapply(1:l_tot_XX, function(x) ifelse(x <= l_XX, paste0("Effect_",x), paste0("Placebo_",x-l_XX)))
  lapply(1:l_XX, function(i) {
    src <- paste0("U_Gg_var_glob_", i, "_XX")
    tgt <- paste0("U_Gg_var_comb_", i, "_XX")
    if (src %in% names(df)) {
      if (isFALSE(normalized)) {
        df[, (tgt) := get(src)]
      } else {
        denom <- get(paste0("delta_D_", i, "_global_XX"))
        df[, (tgt) := get(src) / denom]
      }
    } else {
      df[, (tgt) := NA_real_]
    }
  })
  if (l_placebo_XX != 0) {
    lapply(1:l_placebo_XX, function(i) {
      src <- paste0("U_Gg_var_glob_pl_", i, "_XX")
      tgt <- paste0("U_Gg_var_comb_", l_XX + i, "_XX")
      if (src %in% names(df)) {
        if (isFALSE(normalized)) {
          df[, (tgt) := get(src)]
        } else {
          denom <- get(paste0("delta_D_pl_", i, "_global_XX"))
          df[, (tgt) := get(src) / denom]
        }
      } else {
        df[, (tgt) := NA_real_]
      }
    })
  }

  for (i in 1:l_tot_XX) {
    didmgt_vcov[i,i] <- mat_res_XX[i + (i>l_XX),2]^2
    j <- 1
    while (j < i) {
      df[[paste0("U_Gg_var_comb_",i,"_",j,"_2_XX")]] <- (df[[paste0("U_Gg_var_comb_",i,"_XX")]] + df[[paste0("U_Gg_var_comb_",j,"_XX")]])^2 * df[["first_obs_by_gp_XX"]]
      var_temp <- sum(df[[paste0("U_Gg_var_comb_",i,"_",j,"_2_XX")]], na.rm = TRUE)/G_XX^2
      didmgt_vcov[i,j] <- didmgt_vcov[j,i] <- (var_temp - mat_res_XX[i + (i>l_XX),2]^2 - mat_res_XX[j + (j>l_XX),2]^2)/2
      df[[paste0("U_Gg_var_comb_",i,"_",j,"_2_XX")]] <- var_temp <- NULL
      j <- j + 1
    }
  }

  ###### Returning the results of the estimation

  ## All the results from the estimations and tests are attached to the did_multiplegt_dyn object as its "results" branch (or as the "_by_level_n$results" for n in 1:length(levels(by)) with the by option)
  ## The whole estimation dataset plus some scalars are by default stored and passed to other functions for post-estimation features.

  mat_res_XX[,1:4] <- mat_res_XX[,1:4]
  mat_res_XX[,5:8] <- mat_res_XX[,5:8]
  rownames(mat_res_XX) <- rownames
  colnames(mat_res_XX) <- c("Estimate", "SE", "LB CI", "UB CI", "N", "Switchers", "N.w", "Switchers.w", "Time")

  # Saving the results if requested
  if (!is.null(save_results)) {
    utils::write.csv(mat_res_XX, save_results, row.names = TRUE, col.names = TRUE)
  }

  Effect_mat <- matrix(mat_res_XX[1:l_XX, 1:(ncol(mat_res_XX) -1)], ncol = ncol(mat_res_XX)-1, nrow = l_XX)
  rownames(Effect_mat) <- rownames[1:l_XX]
  colnames(Effect_mat) <- c("Estimate", "SE", "LB CI", "UB CI", "N", "Switchers", "N.w", "Switchers.w")

  ATE_mat <- matrix(mat_res_XX[l_XX + 1, 1:(ncol(mat_res_XX) -1)], ncol = ncol(mat_res_XX)-1, nrow = 1)
  rownames(ATE_mat) <- rownames[l_XX+1]
  colnames(ATE_mat) <- c("Estimate", "SE", "LB CI", "UB CI", "N", "Switchers", "N.w", "Switchers.w")

  out_names <- c("N_Effects", "N_Placebos", "Effects", "ATE", "delta_D_avg_total", "max_pl", "max_pl_gap")
  did_multiplegt_dyn <- list(
    l_XX,
    l_placebo_XX,
    Effect_mat,
    ATE_mat,
    delta_D_avg_total,
    max_pl_XX,
    max_pl_gap_XX
  )
  if (!is.null(p_jointeffects)) {
    did_multiplegt_dyn <- append(did_multiplegt_dyn, p_jointeffects)
    out_names <- c(out_names, "p_jointeffects")
  }
  if (isTRUE(effects_equal)) {
    did_multiplegt_dyn <- append(did_multiplegt_dyn, p_equality_effects)
    out_names <- c(out_names, "p_equality_effects")
  }
  if (l_placebo_XX > 0) {
    Placebo_mat <- matrix(mat_res_XX[(l_XX+2):nrow(mat_res_XX), 1:(ncol(mat_res_XX) -1)], ncol = ncol(mat_res_XX) -1, nrow = l_placebo_XX)
    rownames(Placebo_mat) <- rownames[(l_XX+2):nrow(mat_res_XX)]
    colnames(Placebo_mat) <- c("Estimate", "SE", "LB CI", "UB CI", "N", "Switchers", "N.w", "Switchers.w")


    did_multiplegt_dyn <- append(did_multiplegt_dyn, list(Placebo_mat))
    out_names <- c(out_names, "Placebos")
    if (placebo > 1) {
      if (l_placebo_XX > 1) {
        did_multiplegt_dyn <- append(did_multiplegt_dyn, p_jointplacebo)
        out_names <- c(out_names, "p_jointplacebo")
      }
    }
  }
  if (!is.null(predict_het)) {
    if (length(predict_het_good) > 0L) {
      did_multiplegt_dyn <- append(did_multiplegt_dyn, list(het_res))
      out_names <- c(out_names, "predict_het")
    }
  }

  # Add vcov warnings if any were collected
  if (length(vcov_warnings) > 0L) {
    did_multiplegt_dyn <- append(did_multiplegt_dyn, list(vcov_warnings))
    out_names <- c(out_names, "vcov_warnings")
  }

  # Uncomment for debugging #
  #did_multiplegt_dyn <- append(did_multiplegt_dyn, list(df))
  #out_names <- c(out_names, "debug")

  names(did_multiplegt_dyn) <- out_names

  delta <- if (isTRUE(normalized)) {
    stats::setNames(
      lapply(1:l_XX, function(i) get(paste0("delta_D_", i, "_global_XX"))),
      paste0("delta_D_", 1:l_XX, "_global_XX")
    )
  } else {
    list()
  }

  coef <- list(b = mat_res_XX[-(l_XX+1), 1], vcov = didmgt_vcov)

  ret <- list(
    df,
    did_multiplegt_dyn,
    delta,
    l_XX,
    T_max_XX,
    mat_res_XX
  )
  ret_names <- c("df", "did_multiplegt_dyn", "delta", "l_XX", "T_max_XX", "mat_res_XX")
  if (placebo!= 0) {
    ret <- append(ret, l_placebo_XX)
    ret_names <- c(ret_names, "l_placebo_XX")
  }
  ret <- append(ret, list(coef))
  ret_names <- c(ret_names, "coef")

  # Add controls_globals (includes didmgt_XX, didmgt_Xy, coefs_sq, inv_Denom)
  if (!is.null(controls_globals)) {
    ret <- append(ret, list(controls_globals))
    ret_names <- c(ret_names, "controls_globals")
  }

  names(ret) <- ret_names
  return(ret)
  })
  }
