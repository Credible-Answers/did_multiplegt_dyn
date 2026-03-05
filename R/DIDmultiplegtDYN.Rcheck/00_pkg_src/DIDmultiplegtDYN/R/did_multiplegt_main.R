# Global option to control backend: FALSE = pure data.table (fastest), TRUE = use Polars
# Set options(DID_USE_POLARS = FALSE) to use pure data.table backend
.DID_USE_POLARS <- function() {
  getOption("DID_USE_POLARS", default = TRUE)
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
#' @note polars is suggested for better performance
#' @import data.table
#' @importFrom dplyr lag
#' @importFrom matlib Ginv
#' @importFrom utils write.csv
#' @importFrom stats pchisq qnorm sd weighted.mean as.formula df.residual lm nobs qt relevel
#' @importFrom stats na.omit predict setNames
#' @importFrom MASS ginv
#' @importFrom fixest feols
#' @import lmtest
#' @import sandwich
#' @importFrom car linearHypothesis
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
  data_only = FALSE
  ) {

# Check polars availability
if (!.polars_available()) {
  stop(
    "The 'polars' package is required but not installed.\n",
    "Please install it from r-universe with:\n",
    "  install.packages('polars', repos = 'https://rpolars.r-universe.dev')\n",
    call. = FALSE
  )
}

# Load polars namespace
pl <- .get_pl()
as_polars_df <- function(x) .as_polars_df(x)

suppressWarnings({

  ###### Initialize warnings collector for vcov invertibility issues
  vcov_warnings <- c()

  ###### 0. Pre-allocate variables that are generated via polars (to satisfy CRAN requirements)
  gr_id <- NULL
  weight_XX <- NULL
  F_g_XX <- NULL
  F_g_trunc_XX <- NULL
  N_gt_XX <- NULL
  T_g_XX <- NULL
  U_Gg_var_global_XX <- NULL
  Yg_Fg_min1_XX <- NULL
  Yg_Fg_min2_XX <- NULL
  avg_diff_temp_XX <- NULL
  avg_post_switch_treat_XX   <- NULL
  avg_post_switch_treat_XX_temp <- NULL
  clust_U_Gg_var_global_XX <- NULL
  cluster_XX <- NULL
  cluster_var_g_XX <- NULL
  controls_time_XX <- NULL
  count_time_post_switch_XX<- NULL
  count_time_post_switch_XX_temp <- NULL
  counter <- NULL
  counter_temp <- NULL
  d_F_g_XX <- NULL
  d_F_g_temp_XX <- NULL
  d_fg_XX <- NULL
  d_sq_XX <- NULL
  d_sq_int_XX <- NULL
  d_sq_temp_XX <- NULL
  diff_y_XX <- NULL
  ever_change_d_XX <- NULL
  fd_X_all_non_missing_XX <- NULL
  first_obs_by_clust_XX <- NULL
  first_obs_by_gp_XX <- NULL
  group_XX <- NULL
  last_obs_D_bef_switch_XX <- NULL
  last_obs_D_bef_switch_t_XX <- NULL
  max_time_d_nonmiss_XX <- NULL
  mean_D <- NULL
  mean_Y <- NULL
  min_time_d_miss_aft_ynm_XX <- NULL
  min_time_d_nonmiss_XX <- NULL
  min_time_y_nonmiss_XX <- NULL
  never_change_d_XX <- NULL
  sd_het <- NULL
  sum_weights_control_XX<- NULL
  temp_F_g_XX <- NULL
  time_XX <- NULL
  time_d_miss_XX <- NULL
  time_d_nonmiss_XX<- NULL
  time_y_nonmiss_XX <- NULL
  treatment_XX_v1 <- NULL
  var_F_g_XX<- NULL


  ######## 1. Checking that syntax correctly specified
  #### Add a stop message: same_switchers_pl only works when same_switchers is specified.
  if (same_switchers == FALSE & same_switchers_pl == TRUE) {
    stop("The same_switchers_pl option only works if same_switchers is specified as well!")
  }


  #### Continous option: checking that polynomial order specified, and putting it into degree_pol scalar.
  if (!is.null(continuous)) {
    degree_pol <- continuous
  }

  ######## 2. Data preparation steps
  #### Renaming the variables in the dataset
  original_names <- c(c(outcome, group, time, treatment), trends_nonparam, weight, controls, cluster, unlist(predict_het[1]))
  df <- subset(df, select = original_names)
  names(df)[names(df) == outcome] <- "outcome"
  names(df)[names(df) == group] <- "group"
  names(df)[names(df) == time] <- "time"
  names(df)[names(df) == treatment] <- "treatment"
  df <- as_polars_df(df)

  #### Grouping together trends_nonparam variables
  #if (!is.null(trends_nonparam)) {
  #  df$trends_nonparam_XX <- df[trends_nonparam]
  #}

  #### Patching the cluster variable: by default, the command clusters at group level. If the user specifies clustering by group, the clustering option goes to NULL.
  if (!is.null(cluster)) {
    if (paste0(cluster) == paste0(group)) {
      cluster <- NULL
    } else{
      df <- df$with_columns(pl$col(cluster)$alias("cluster_XX"))
    }
  }

  #### Selecting the sample
  ## Dropping observations with missing group or time
  df <- df$filter(pl$col("group")$is_not_null() & pl$col("time")$is_not_null())
  ## Dropping observations with missing controls
  if (!is.null(controls) && length(controls) > 0) {
    # Build filter expression to drop nulls in any control variable
    for (ctrl in controls) {
      df <- df$filter(pl$col(ctrl)$is_not_null())
    }
  }

  #### Further sample selection steps
  ## Dropping observations with a missing clustering variable
  if (!is.null(cluster)) {
    df <- df$filter(pl$col("cluster_XX")$is_not_null())
  }

  ## Dropping groups with always missing treatment or outcomes
  ## Note: Must check both is_not_null AND is_not_nan to match R's !is.na() behavior
  df <- df$with_columns(
    pl$col("treatment")$mean()$over("group")$alias("mean_D"),
    pl$col("outcome")$mean()$over("group")$alias("mean_Y")
  )
  df <- df$filter(
    pl$col("mean_Y")$is_not_null() & pl$col("mean_Y")$is_not_nan() &
    pl$col("mean_D")$is_not_null() & pl$col("mean_D")$is_not_nan()
  )
  df <- df$drop(c("mean_Y", "mean_D"))

  #### Predict_het option for heterogeneous treatment effects analysis
  predict_het_good <- c()
  if (!is.null(predict_het)) {
    if (length(predict_het) != 2 & inherits(predict_het, "list")) {
      stop("Syntax error in predict_hat option: list with 2 elements required. Set the second element to -1 to include all the effects.")
    }
    ## Checks if predict_het and normalized are both specified
    if (isTRUE(normalized)) {
      message("The options normalized and predict_het cannot be specified together. The option predict_het will be ignored.")
    } else {
      pred_het <- unlist(predict_het[1])
      het_effects <- unlist(predict_het[2])
      ## Checks if only time-invariant variables are specified in predict_het
      for (v in pred_het) {
        df <- df$with_columns(
          pl$when(pl$col(v)$std()$over("group")$is_null())$then(0)$otherwise(pl$col(v)$std()$over("group"))$alias("sd_het")
        )
        # Use native Polars mean instead of converting to data.frame
        sd_het_mean <- as.data.frame(df$select(pl$col("sd_het")$mean()))[[1,1]]
        if (is.na(sd_het_mean) || sd_het_mean == 0) {
          predict_het_good <- c(predict_het_good, v)
        } else {
          message(sprintf("The variable %s specified in the option predict_het is time-varying, the command will therefore ignore it.", v))
        }
        df <- df$drop("sd_het")
      }
    }
  }

  #### Collapse and weight
  ## Creating the weight variable
  if (is.null(weight)) {
    df <- df$with_columns(pl$lit(1)$alias("weight_XX"))
  } else{
    df <- df$with_columns(pl$col(weight)$alias("weight_XX"))
  }
  df <- df$with_columns(
    pl$when(pl$col("weight_XX")$is_null())$then(0)$otherwise(pl$col("weight_XX"))$alias("weight_XX")
  )

  ## Checking if the data has to be collapsed
  # Use native Polars to check max count per group-time
  max_group_time_count <- as.data.frame(df$group_by(c("group", "time"))$agg(pl$len()$alias("count"))$select(pl$col("count")$max()))[[1,1]]
  aggregated_data <- max_group_time_count == 1

  ## Collapsing the data if necessary
  if (aggregated_data != 1) {
    df <- df$with_columns(
      pl$when(pl$col("treatment")$is_null())$then(0)$otherwise(pl$col("weight_XX"))$alias("weight_XX")
    )
    if (is.null(cluster)) {
      df <- df$with_columns(pl$lit(1)$alias("cluster_XX"))
    }

    # Use native polars group_by/agg for weighted mean calculation
    # weighted_mean(col, w) = sum(col * w) / sum(w)
    agg_cols <- c("treatment", "outcome", trends_nonparam, weight, controls, predict_het_good, "cluster_XX", cluster)
    agg_cols <- unique(agg_cols[agg_cols %in% df$columns])

    # Build weighted mean expressions for each column
    weighted_mean_exprs <- lapply(agg_cols, function(col) {
      ((pl$col(col) * pl$col("weight_XX"))$sum() / pl$col("weight_XX")$sum())$alias(col)
    })
    # Add weight_XX sum
    all_exprs <- c(weighted_mean_exprs, list(pl$col("weight_XX")$sum()$alias("weight_XX")))

    # Perform group_by aggregation using do.call for proper list expansion
    df <- do.call(function(...) df$group_by(c("group", "time"))$agg(...), all_exprs)

    if (is.null(cluster)) {
      df <- df$drop("cluster_XX")
    }
  }

  ## --- Generate factorized versions of Y, G, T and D ---
  outcome <- "outcome"
  group <- "group"
  time <- "time"
  treatment <- "treatment"

  # outcome_XX = outcome
  df <- df$with_columns(pl$col("outcome")$alias("outcome_XX"))

  # sort by time
  df <- df$sort("time")

  # group_XX and time_XX as "factorized" (1,2,3,...) in order of appearance
  df <- df$with_columns(
    pl$col("group")$rank("dense")$alias("group_XX"),
    pl$col("time")$rank("dense")$alias("time_XX"),
    pl$col("treatment")$alias("treatment_XX")
  )

  # first/last date where D not missing
  df <- df$with_columns(
    pl$when(pl$col("treatment_XX")$is_not_null())$then(pl$col("time_XX"))$otherwise(pl$lit(NA))$alias("time_d_nonmiss_XX"),
    pl$when(pl$col("outcome_XX")$is_not_null())$then(pl$col("time_XX"))$otherwise(pl$lit(NA))$alias("time_y_nonmiss_XX")
  )

  # per-group mins & max
  df <- df$with_columns(
    pl$col("time_d_nonmiss_XX")$min()$over("group_XX")$alias("min_time_d_nonmiss_XX"),
    pl$col("time_d_nonmiss_XX")$max()$over("group_XX")$alias("max_time_d_nonmiss_XX"),
    pl$col("time_y_nonmiss_XX")$min()$over("group_XX")$alias("min_time_y_nonmiss_XX")
  )

  # first date D missing *after* Y seen
  df <- df$with_columns(
    pl$when(pl$col("treatment_XX")$is_null() & (pl$col("time_XX") >= pl$col("min_time_y_nonmiss_XX")))$then(pl$col("time_XX"))$otherwise(pl$lit(NA))$alias("time_d_miss_XX")
  )

  # per-group min of time_d_miss_XX
  df <- df$with_columns(
    pl$col("time_d_miss_XX")$min()$over("group_XX")$alias("min_time_d_miss_aft_ynm_XX")
  )

  # drop intermediate cols
  df <- df$drop(c("time_d_nonmiss_XX", "time_y_nonmiss_XX", "time_d_miss_XX"))

  ## --- Baseline treatment D_{g,1} ---

  # d_sq_temp_XX = treatment_XX at min_time_d_nonmiss_XX
  df <- df$with_columns(
    pl$when(pl$col("time_XX") == pl$col("min_time_d_nonmiss_XX"))$then(pl$col("treatment_XX"))$otherwise(pl$lit(NA))$alias("d_sq_temp_XX")
  )

  # d_sq_XX = group mean of that (only one non-NA per group, so it's the baseline)
  df <- df$with_columns(
    pl$col("d_sq_temp_XX")$mean()$over("group_XX")$alias("d_sq_XX")
  )

  # drop temp
  df <- df$drop("d_sq_temp_XX")

  ## --- Enforce "Design Restriction 2" ---

  df <- df$with_columns(
    (pl$col("treatment_XX") - pl$col("d_sq_XX"))$alias("diff_from_sq_XX")
  )

  # sort by group_XX, time_XX
  df <- df$sort(c("group_XX", "time_XX"))

  # T_XX = int(df['time_XX'].max()) - use native Polars
  T_XX <- as.integer(as.data.frame(df$select(pl$col("time_XX")$max()))[[1,1]])


  if (!(dont_drop_larger_lower == TRUE)) {
    # Sort by group_XX and time_XX
    df <- df$sort(c("group_XX", "time_XX"))

    # 2. strict increase: ever_strict_increase_XX is 1 if it ever happens within group_XX
    df <- df$with_columns(
      ((pl$col("diff_from_sq_XX") > 0) & pl$col("treatment_XX")$is_not_null())$cast(pl$Int32)$cum_sum()$over("group_XX")$clip(0, 1)$alias("ever_strict_increase_XX")
    )

    # 3. strict decrease: ever_strict_decrease_XX is 1 if it ever happens within group_XX
    df <- df$with_columns(
      ((pl$col("diff_from_sq_XX") < 0) & pl$col("treatment_XX")$is_not_null())$cast(pl$Int32)$cum_sum()$over("group_XX")$clip(0, 1)$alias("ever_strict_decrease_XX")
    )

    # 4. drop rows where both == 1
    df <- df$filter(!(pl$col("ever_strict_increase_XX") == 1 & pl$col("ever_strict_decrease_XX") == 1))
  }

  #### Counting number of groups
  # G_XX <- max(df$group_XX, na.rm = TRUE)

  #### Ever changed treatment
  df <- df$with_columns(
    ((pl$col("diff_from_sq_XX")$abs() > 0) & pl$col("treatment_XX")$is_not_null())$cast(pl$Int32)$alias("ever_change_d_XX")
  )
  # Use cummax over group
  df <- df$sort(c("group_XX", "time_XX"))
  df <- df$with_columns(
    pl$col("ever_change_d_XX")$cum_max()$over("group_XX")$alias("ever_change_d_XX")
  )

  #### Creating date of the first treatment change
  df <- df$with_columns(
    pl$when(
      (pl$col("ever_change_d_XX") == 1) & (pl$col("ever_change_d_XX")$shift(1)$over("group_XX") == 0)
    )$then(pl$col("time_XX"))$otherwise(pl$lit(0))$alias("temp_F_g_XX")
  )
  df <- df$with_columns(
    pl$col("temp_F_g_XX")$max()$over("group_XX")$alias("F_g_XX")
  )
  df <- df$drop("temp_F_g_XX")

  #### If continuous option specified, generating polynomials of D_{g,1},
  #### storing D_{g,1} somewhere, and replacing it by 0.
  if (!is.null(continuous)) {
    for (pol_level in 1:degree_pol) {
      col_name <- paste0("d_sq_", pol_level, "_XX")
      df <- df$with_columns(
        (pl$col("d_sq_XX")$pow(pol_level))$alias(col_name)
      )
    }
    df <- df$with_columns(pl$col("d_sq_XX")$alias("d_sq_XX_orig"))
    df <- df$with_columns(pl$lit(0)$alias("d_sq_XX"))
  }

  ## Creating a new value with integer levels of d_sq_XX
  # Use native Polars cast to convert rank result to Float64 directly
  df <- df$with_columns(
    pl$col("d_sq_XX")$rank("dense")$cast(pl$Float64)$alias("d_sq_int_XX")
  )

  #### Dropping values of baseline treatment such that there is no variance in F_g within
  by_cols_var <- c("d_sq_XX", trends_nonparam)
  by_cols_var <- by_cols_var[by_cols_var != "" & !is.na(by_cols_var) & nchar(by_cols_var) > 0]

  df <- df$with_columns(
    pl_over_cols(pl$col("F_g_XX")$std(), by_cols_var)$alias("var_F_g_XX")
  )
  df <- df$filter(pl$col("var_F_g_XX") > 0)
  df <- df$drop("var_F_g_XX")

  #### Counting number of groups - use native Polars to_list
  G_XX <- as.data.frame(df$select(pl$col("group_XX")$n_unique()))[[1,1]]


  if (nrow(df) == 0) {
    stop("No treatment effect can be estimated.\n  This is because Design Restriction 1 in de Chaisemartin & D'Haultfoeuille (2024) is not satisfied in the data, given the options requested.\n  This may be due to the fact that groups' period-one treatment is continuous, or takes a large number of values, and you have not specified the continuous option.\n  If so, you can try to specify this option.\n  If the issue persists even with this option, this means that all groups experience their first treatment change at the same date.\n  In this situation, estimators of de Chaisemartin & D'Haultfoeuille (2024) cannot be used.")
  }

  #### For each value of d_sq_XX, we drop time periods such that we do not have any control with the same baseline treatment afterwards
  #### This means the panel is no longer balanced, though it is balanced within values of the baseline treatment
  df <- df$with_columns(
    (pl$lit(1) - pl$col("ever_change_d_XX"))$alias("never_change_d_XX")
  )
  by_cols_ctrl <- c("time_XX", "d_sq_XX", trends_nonparam)
  by_cols_ctrl <- by_cols_ctrl[by_cols_ctrl != "" & !is.na(by_cols_ctrl) & nchar(by_cols_ctrl) > 0]
  # Use pure Polars window function for max by group
  df <- df$with_columns(
    pl_over_cols(pl$col("never_change_d_XX")$max(), by_cols_ctrl)$alias("controls_time_XX")
  )
  df <- df$filter(pl$col("controls_time_XX") > 0)

  #### Computing t_min, T_max and adjusting F_g by last period plus one for those that never change treatment
  # Use native Polars for min/max calculations
  t_min_XX <- as.data.frame(df$select(pl$col("time_XX")$min()))[[1,1]]
  T_max_XX <- as.data.frame(df$select(pl$col("time_XX")$max()))[[1,1]]
  df <- df$with_columns(
    pl$when(pl$col("F_g_XX") == 0)$then(pl$lit(T_max_XX + 1))$otherwise(pl$col("F_g_XX"))$alias("F_g_XX")
  )



  ######## Dealing with missing treatments: most conservative option
  #### Let FMD_g denote the first date when g's treatment is missing while y has been not missing at least once, so that we know for sure that g already exists. 
  #### If that date is before the first period when g's treatment changes, we do not know when g's treatment has changed for the first time. Then, a conservative option is to drop all of g's outcomes starting at FMD_g.

  if (drop_if_d_miss_before_first_switch == TRUE) {
    df <- df$with_columns(
      pl$when(
        pl$col("min_time_d_miss_aft_ynm_XX")$is_not_null() &
        (pl$col("min_time_d_miss_aft_ynm_XX") < pl$col("F_g_XX")) &
        (pl$col("time_XX") >= pl$col("min_time_d_miss_aft_ynm_XX"))
      )$then(pl$lit(NA_real_))$otherwise(pl$col("outcome_XX"))$alias("outcome_XX")
    )
  }

  ######## Dealing with missing treatments: most liberal option
  df <- df$with_columns(
    pl$when((pl$col("time_XX") < pl$col("F_g_XX")) & pl$col("treatment_XX")$is_not_null())
      $then(pl$col("time_XX"))$otherwise(pl$lit(NA_real_))$alias("last_obs_D_bef_switch_t_XX")
  )
  df <- df$with_columns(
    pl$col("last_obs_D_bef_switch_t_XX")$max()$over("group_XX")$alias("last_obs_D_bef_switch_XX")
  )

  #### For t<FD_g, outcome set to NA
  df <- df$with_columns(
    pl$when(pl$col("time_XX") < pl$col("min_time_d_nonmiss_XX"))
      $then(pl$lit(NA_real_))$otherwise(pl$col("outcome_XX"))$alias("outcome_XX")
  )

  #### Replace missing treatment by status-quo
  df <- df$with_columns(
    pl$when(
      (pl$col("F_g_XX") < pl$lit(T_max_XX + 1)) &
      pl$col("treatment_XX")$is_null() &
      (pl$col("time_XX") < pl$col("last_obs_D_bef_switch_XX")) &
      (pl$col("time_XX") > pl$col("min_time_d_nonmiss_XX"))
    )$then(pl$col("d_sq_XX"))$otherwise(pl$col("treatment_XX"))$alias("treatment_XX")
  )

  #### Set outcomes to NA for uncertain switch dates
  df <- df$with_columns(
    pl$when(
      (pl$col("F_g_XX") < pl$lit(T_max_XX + 1)) &
      (pl$col("time_XX") > pl$col("last_obs_D_bef_switch_XX")) &
      (pl$col("last_obs_D_bef_switch_XX") < (pl$col("F_g_XX") - 1))
    )$then(pl$lit(NA_real_))$otherwise(pl$col("outcome_XX"))$alias("outcome_XX")
  )
  df <- df$with_columns(
    pl$when(
      (pl$col("F_g_XX") < pl$lit(T_max_XX + 1)) &
      (pl$col("last_obs_D_bef_switch_XX") < (pl$col("F_g_XX") - 1))
    )$then(pl$col("last_obs_D_bef_switch_XX") + 1)$otherwise(pl$lit(NA_real_))$alias("trunc_control_XX")
  )
  df <- df$with_columns(
    pl$when(
      (pl$col("F_g_XX") < pl$lit(T_max_XX + 1)) &
      (pl$col("last_obs_D_bef_switch_XX") < (pl$col("F_g_XX") - 1))
    )$then(pl$lit(T_max_XX + 1))$otherwise(pl$col("F_g_XX"))$alias("F_g_XX")
  )

  #### Replace missing treatment after F_g by D(g,F_g)
  df <- df$with_columns(
    pl$when(pl$col("time_XX") == pl$col("F_g_XX"))
      $then(pl$col("treatment_XX"))$otherwise(pl$lit(NA_real_))$alias("d_F_g_temp_XX")
  )
  df <- df$with_columns(
    pl$col("d_F_g_temp_XX")$mean()$over("group_XX")$alias("d_F_g_XX")
  )
  df <- df$with_columns(
    pl$when(
      (pl$col("F_g_XX") < pl$lit(T_max_XX + 1)) &
      pl$col("treatment_XX")$is_null() &
      (pl$col("time_XX") > pl$col("F_g_XX")) &
      (pl$col("last_obs_D_bef_switch_XX") == (pl$col("F_g_XX") - 1))
    )$then(pl$col("d_F_g_XX"))$otherwise(pl$col("treatment_XX"))$alias("treatment_XX")
  )

  #### For never-switchers, replace missing treatment by D_g1
  df <- df$with_columns(
    pl$when(
      (pl$col("F_g_XX") == pl$lit(T_max_XX + 1)) &
      pl$col("treatment_XX")$is_null() &
      (pl$col("time_XX") > pl$col("min_time_d_nonmiss_XX")) &
      (pl$col("time_XX") < pl$col("max_time_d_nonmiss_XX"))
    )$then(pl$col("d_sq_XX"))$otherwise(pl$col("treatment_XX"))$alias("treatment_XX")
  )

  #### For never-switchers, outcomes missing at t>LD_g
  df <- df$with_columns(
    pl$when(
      (pl$col("F_g_XX") == pl$lit(T_max_XX + 1)) &
      (pl$col("time_XX") > pl$col("max_time_d_nonmiss_XX"))
    )$then(pl$lit(NA_real_))$otherwise(pl$col("outcome_XX"))$alias("outcome_XX")
  )
  df <- df$with_columns(
    pl$when(pl$col("F_g_XX") == pl$lit(T_max_XX + 1))
      $then(pl$col("max_time_d_nonmiss_XX") + 1)$otherwise(pl$col("trunc_control_XX"))$alias("trunc_control_XX")
  )

  #### Store the outcome in levels for predict_het
  if (!is.null(predict_het)) {
    if (length(predict_het_good) > 0) {
      df <- df$with_columns(pl$col("outcome_XX")$alias("outcome_non_diff_XX"))
    }
  }

  #### When trends_lin specified, first difference outcome and controls
  if (isTRUE(trends_lin)) {
    df <- df$filter(pl$col("F_g_XX") != 2)
    df <- df$sort(c("group_XX", "time_XX"))

    df <- df$with_columns(
      (pl$col("outcome_XX") - pl$col("outcome_XX")$shift(1)$over("group_XX"))$alias("outcome_XX")
    )
    if (!is.null(controls) && length(controls) > 0) {
      for (v in controls) {
        df <- df$with_columns(
          (pl$col(v) - pl$col(v)$shift(1)$over("group_XX"))$alias(v)
        )
      }
    }
    df <- df$filter(pl$col("time_XX") != 1)
    # Use native Polars for min calculation
    t_min_XX <- as.data.frame(df$select(pl$col("time_XX")$min()))[[1,1]]
  }

  #### Balancing the panel using polars cross join
  unique_groups <- df$select(pl$col("group_XX")$unique())
  unique_times <- df$select(pl$col("time_XX")$unique())
  grid <- unique_groups$join(unique_times, how = "cross")
  df <- grid$join(df, on = c("group_XX", "time_XX"), how = "left")

  df <- df$with_columns(
    pl$col("d_sq_XX")$mean()$over("group_XX")$alias("d_sq_XX"),
    pl$col("d_sq_int_XX")$mean()$over("group_XX")$alias("d_sq_int_XX"),
    pl$col("F_g_XX")$mean()$over("group_XX")$alias("F_g_XX")
  )

  #### Defining N_gt
  df <- df$with_columns(pl$lit(1)$alias("N_gt_XX"))
  df <- df$with_columns(
    pl$when(pl$col("outcome_XX")$is_null() | pl$col("treatment_XX")$is_null())
      $then(pl$lit(0))$otherwise(pl$col("weight_XX") * pl$col("N_gt_XX"))$alias("N_gt_XX")
  )

  #### Determining last period where g still has a control group
  df <- df$with_columns(
    pl$when(pl$col("F_g_XX") < pl$col("trunc_control_XX"))
      $then(pl$col("F_g_XX"))$otherwise(pl$col("trunc_control_XX"))$alias("F_g_trunc_XX")
  )
  df <- df$with_columns(
    pl$when(pl$col("trunc_control_XX")$is_null())
      $then(pl$col("F_g_XX"))$otherwise(pl$col("F_g_trunc_XX"))$alias("F_g_trunc_XX")
  )
  df <- df$with_columns(
    pl$when(pl$col("F_g_XX")$is_null())
      $then(pl$col("trunc_control_XX"))$otherwise(pl$col("F_g_trunc_XX"))$alias("F_g_trunc_XX")
  )

  by_cols_tg <- c("d_sq_XX", trends_nonparam)
  by_cols_tg <- by_cols_tg[by_cols_tg != "" & !is.na(by_cols_tg) & nchar(by_cols_tg) > 0]
  df <- df$with_columns(
    pl_over_cols(pl$col("F_g_trunc_XX")$max(), by_cols_tg)$alias("T_g_XX")
  )
  df <- df$with_columns((pl$col("T_g_XX") - 1)$alias("T_g_XX"))

  #### Defining S_g: 
  #### an indicator variable for groups whose average post switch 
  #### treatment value is larger than their initial treatment D_{g,1}. 
  #### They will be considered switchers in. If S_g==0, the group is a switcher out. 
  #### For never-switchers, S_g is undefined.
  #### Definition of S_g matches that in paper, unless dont_drop_larger_lower specified.

  # treatment_XX_v1: treatment in post-switch period only
  df <- df$with_columns(
    pl$when((pl$col("time_XX") >= pl$col("F_g_XX")) & (pl$col("time_XX") <= pl$col("T_g_XX")))
      $then(pl$col("treatment_XX"))$otherwise(pl$lit(NA_real_))$alias("treatment_XX_v1")
  )

  # avg_post_switch_treat_XX_temp: treatment value in post-switch period
  df <- df$with_columns(
    pl$when((pl$col("time_XX") >= pl$col("F_g_XX")) & (pl$col("time_XX") <= pl$col("T_g_XX")))
      $then(pl$col("treatment_XX"))$otherwise(pl$lit(NA_real_))$alias("avg_post_switch_treat_XX_temp")
  )

  # Count of non-missing treatment observations in the post-switch period
  df <- df$with_columns(
    pl$when(
      (pl$col("time_XX") >= pl$col("F_g_XX")) &
      (pl$col("time_XX") <= pl$col("T_g_XX")) &
      pl$col("treatment_XX")$is_not_null()
    )$then(pl$lit(1L))$otherwise(pl$lit(0L))$alias("count_time_post_switch_XX_temp")
  )

  # Sum within group
  df <- df$with_columns(
    pl$col("avg_post_switch_treat_XX_temp")$sum()$over("group_XX")$alias("avg_post_switch_treat_XX_temp"),
    pl$col("count_time_post_switch_XX_temp")$sum()$over("group_XX")$alias("count_time_post_switch_XX")
  )

  # Divide sum by count to get the group-specific average
  df <- df$with_columns(
    (pl$col("avg_post_switch_treat_XX_temp") / pl$col("count_time_post_switch_XX"))$alias("avg_post_switch_treat_XX_temp")
  )

  # Get the mean of that average across group
  df <- df$with_columns(
    pl$col("avg_post_switch_treat_XX_temp")$mean()$over("group_XX")$alias("avg_post_switch_treat_XX")
  )


  # Drop temporary columns
  df <- df$drop(c("treatment_XX_v1", "avg_post_switch_treat_XX_temp", "count_time_post_switch_XX_temp"))

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
    df <- df$filter(
      !(
        (pl$col("avg_post_switch_treat_XX") == pl$col("d_sq_XX")) &
        pl$col("avg_post_switch_treat_XX")$is_not_null() &
        (pl$col("F_g_XX") != (pl$col("T_g_XX") + 1)) &
        pl$col("F_g_XX")$is_not_null() &
        pl$col("T_g_XX")$is_not_null()
      )
    )
    # S_g_XX: 1 if switcher in, 0 if switcher out
    df <- df$with_columns(
      (pl$col("avg_post_switch_treat_XX") > pl$col("d_sq_XX"))$cast(pl$Float64)$alias("S_g_XX")
    )
    df <- df$with_columns(
      pl$when(pl$col("F_g_XX") != pl$lit(T_max_XX + 1))
        $then(pl$col("S_g_XX"))$otherwise(pl$lit(NA_real_))$alias("S_g_XX")
    )
  } else {
    # Filter using d_sq_XX_orig for continuous case
    df <- df$filter(
      !(
        (pl$col("avg_post_switch_treat_XX") == pl$col("d_sq_XX_orig")) &
        pl$col("avg_post_switch_treat_XX")$is_not_null() &
        (pl$col("F_g_XX") != (pl$col("T_g_XX") + 1)) &
        pl$col("F_g_XX")$is_not_null() &
        pl$col("T_g_XX")$is_not_null()
      )
    )
    df <- df$with_columns(
      (pl$col("avg_post_switch_treat_XX") > pl$col("d_sq_XX_orig"))$cast(pl$Float64)$alias("S_g_XX")
    )
    df <- df$with_columns(
      pl$when(pl$col("F_g_XX") != pl$lit(T_max_XX + 1))
        $then(pl$col("S_g_XX"))$otherwise(pl$lit(NA_real_))$alias("S_g_XX")
    )
  }

  #### Define another version where S_g=-1 for switchers out, which we need
  #### when predict_het or continuous specified.
  if (length(predict_het) > 0 | !is.null(continuous)) {
    df <- df$with_columns(
      pl$when(pl$col("S_g_XX") == 0)$then(pl$lit(-1))$otherwise(pl$col("S_g_XX"))$alias("S_g_het_XX")
    )
  }

  #### If continuous option specified: binarizing and staggerizing treatment,
  #### and adding time_FEs interacted with D_{g,1} as controls
  if (!is.null(continuous)) {
    ## Binarizing and staggerizing treatment
    df <- df$with_columns(
      pl$when(pl$col("S_g_het_XX")$is_not_null())
        $then((pl$col("F_g_XX") <= pl$col("time_XX"))$cast(pl$Float64) * pl$col("S_g_het_XX"))
        $otherwise(pl$lit(NA_real_))$alias("treatment_temp_XX")
    )
    df <- df$with_columns(pl$col("treatment_XX")$alias("treatment_XX_orig"))
    df <- df$with_columns(pl$col("treatment_temp_XX")$alias("treatment_XX"))
    ## Enriching controls - use native Polars for unique time values
    time_fe_XX <- sort(as.data.frame(df$select(pl$col("time_XX")$unique()))[[1]])
    for (j in 2:length(time_fe_XX)) {
      for (k in 1:degree_pol) {
        col_name <- paste0("time_fe_XX_", j, "_bt", k, "_XX")
        d_sq_col <- paste0("d_sq_", k, "_XX")
        # Use native Polars to create the column
        df <- df$with_columns(
          ((pl$col("time_XX") >= j)$cast(pl$Float64) * pl$col(d_sq_col))$alias(col_name)
        )
        controls <- c(controls, col_name)
      }
    }
  }

  #### Creating treatment at F_g: D_{g,F_g}
  df <- df$with_columns(
    pl$when(pl$col("time_XX") == pl$col("F_g_XX"))
      $then(pl$col("treatment_XX"))$otherwise(pl$lit(NA_real_))$alias("d_fg_XX")
  )
  df <- df$with_columns(
    pl$col("d_fg_XX")$mean()$over("group_XX")$alias("d_fg_XX")
  )
  df <- df$with_columns(
    pl$when(pl$col("d_fg_XX")$is_null() & (pl$col("F_g_XX") == pl$lit(T_max_XX + 1)))
      $then(pl$col("d_sq_XX"))$otherwise(pl$col("d_fg_XX"))$alias("d_fg_XX")
  )

  #### Creating the variable L_g_XX = T_g_XX - F_g_XX so that we can compute L_u or L_a afterwards
  df <- df$with_columns(
    (pl$col("T_g_XX") - pl$col("F_g_XX") + 1)$alias("L_g_XX")
  )

  #### Creating the equivalent variable L_g_placebo_XX for placebos
  if (placebo > 0) {
    df <- df$with_columns(
      pl$when(pl$col("F_g_XX") >= 3)
        $then(
          pl$when(pl$col("L_g_XX") > (pl$col("F_g_XX") - 2))
            $then(pl$col("F_g_XX") - 2)$otherwise(pl$col("L_g_XX"))
        )$otherwise(pl$lit(NA_real_))$alias("L_g_placebo_XX")
    )
    df <- df$with_columns(
      pl$when(pl$col("L_g_placebo_XX") == pl$lit(Inf))
        $then(pl$lit(NA_real_))$otherwise(pl$col("L_g_placebo_XX"))$alias("L_g_placebo_XX")
    )
  }

  #### Tagging first observation of each group_XX
  df <- df$sort(c("group_XX", "time_XX"))
  df <- df$with_columns(
    (pl$col("time_XX") == pl$col("time_XX")$min()$over("group_XX"))$cast(pl$Float64)$alias("first_obs_by_gp_XX")
  )

  #### If cluster option if specified, flagging first obs in cluster and checking if the cluster variable is weakly coarser than the group one.
  if (!is.null(cluster)) {
    ## complete missing clusters based on the min
    df <- df$with_columns(pl$col("cluster_XX")$cast(pl$Float64)$alias("cluster_XX"))
    df <- df$with_columns(
      pl$col("cluster_XX")$min()$over("group_XX")$alias("cluster_group_XX")
    )
    df <- df$with_columns(
      pl$when(pl$col("cluster_XX")$is_null())
        $then(pl$col("cluster_group_XX"))$otherwise(pl$col("cluster_XX"))$alias("cluster_XX")
    )

    df <- df$sort(c("cluster_XX", "group_XX", "time_XX"))
    df <- df$with_columns(
      # Row number within cluster using rank with ordinal method
      (pl$struct(c("group_XX", "time_XX"))$rank("ordinal")$over("cluster_XX") == 1L)$cast(pl$Float64)$alias("first_obs_by_clust_XX")
    )

    df <- df$with_columns(
      pl$col("cluster_XX")$std()$over("group_XX")$alias("cluster_var_g_XX")
    )
    ## Error message for clustering: non-nested case - use native Polars max
    max_cluster_var <- as.data.frame(df$select(pl$col("cluster_var_g_XX")$max()))[[1,1]]
    if (!is.na(max_cluster_var) && max_cluster_var > 0) {
      stop("The group variable should be nested within the clustering variable.")
    }
  }

  #### Declaring the data as panel after the changes above
  # Compute first differences using polars shift
  df <- df$sort(c("group_XX", "time_XX"))
  df <- df$with_columns(
    (pl$col("outcome_XX") - pl$col("outcome_XX")$shift(1)$over("group_XX"))$alias("diff_y_XX"),
    (pl$col("treatment_XX") - pl$col("treatment_XX")$shift(1)$over("group_XX"))$alias("diff_d_XX")
  )
  ######## 3. Necessary pre-estimation steps when the controls option is specified
  ######### Pure Polars implementation with optimized matrix operations

  if (!is.null(controls) && length(controls) > 0L) {
    # Sort for shift operations
    df <- df$sort(c("group_XX", "time_XX"))

    ## 1) First differences of each control + missing flag using Polars
    count_controls <- 0L
    df <- df$with_columns(pl$lit(1L)$alias("fd_X_all_non_missing_XX"))

    for (var in controls) {
      count_controls <- count_controls + 1L
      diff_col <- sprintf("diff_X%d_XX", count_controls)

      # Group-wise first difference using Polars shift
      df <- df$with_columns(
        (pl$col(var) - pl$col(var)$shift(1)$over("group_XX"))$alias(diff_col)
      )

      # If diff is NA, mark as missing
      df <- df$with_columns(
        pl$when(pl$col(diff_col)$is_null())
          $then(pl$lit(0L))
          $otherwise(pl$col("fd_X_all_non_missing_XX"))
          $alias("fd_X_all_non_missing_XX")
      )
    }

    ## 2) Residualization prep using Polars
    count_controls <- 0L
    mycontrols_XX <- character(0L)
    grp_cols <- c("time_XX", "d_sq_XX")
    if (!is.null(trends_nonparam)) grp_cols <- c(grp_cols, trends_nonparam)

    for (var in controls) {
      count_controls <- count_controls + 1L
      diff_col <- sprintf("diff_X%d_XX", count_controls)
      avg_col <- sprintf("avg_diff_X%d_XX", count_controls)
      resid_col <- sprintf("resid_X%d_time_FE_XX", count_controls)
      prod_col <- sprintf("prod_X%d_Ngt_XX", count_controls)

      # Create mask condition: control obs (never switched, non-missing y, controls non-missing)
      mask_cond <- (pl$col("ever_change_d_XX") == 0) &
                   pl$col("diff_y_XX")$is_not_null() &
                   (pl$col("fd_X_all_non_missing_XX") == 1L)

      # Sum of N_gt for controls within mask (using masked value)
      df <- df$with_columns(
        pl$when(mask_cond)$then(pl$col("N_gt_XX"))$otherwise(pl$lit(0))$alias("_N_for_ctrl_")
      )
      df <- df$with_columns(
        pl_over_cols(pl$col("_N_for_ctrl_")$sum(), grp_cols)$alias("sum_weights_control_XX")
      )
      df <- df$with_columns(
        pl$when(mask_cond)$then(pl$col("sum_weights_control_XX"))$otherwise(pl$lit(NA_real_))$alias("sum_weights_control_XX")
      )

      # Weighted sum of first-diffs (masked)
      df <- df$with_columns(
        (pl$col("N_gt_XX") * pl$col(diff_col))$alias("avg_diff_temp_XX")
      )
      df <- df$with_columns(
        pl$when(mask_cond)$then(pl$col("avg_diff_temp_XX"))$otherwise(pl$lit(0))$alias("_avg_diff_temp_masked_")
      )
      df <- df$with_columns(
        pl_over_cols(pl$col("_avg_diff_temp_masked_")$sum(), grp_cols)$alias(avg_col)
      )
      df <- df$with_columns(
        pl$when(mask_cond)$then(pl$col(avg_col))$otherwise(pl$lit(NA_real_))$alias(avg_col)
      )
      df <- df$with_columns(
        (pl$col(avg_col) / pl$col("sum_weights_control_XX"))$alias(avg_col)
      )

      # Residual
      df <- df$with_columns(
        (pl$col("N_gt_XX")$sqrt() * (pl$col(diff_col) - pl$col(avg_col)))$alias(resid_col)
      )
      df <- df$with_columns(
        pl$col(resid_col)$fill_null(0)$alias(resid_col)
      )
      mycontrols_XX <- c(mycontrols_XX, resid_col)

      # Prepare product with deltaY
      df <- df$with_columns(
        (pl$col("N_gt_XX")$sqrt() * pl$col("diff_y_XX"))$alias("diff_y_wXX")
      )
      df <- df$with_columns(
        (pl$col("N_gt_XX")$sqrt() * pl$col(resid_col))$fill_null(0)$alias(prod_col)
      )
    }

    # Clean up temp columns
    df <- df$drop(c("_N_for_ctrl_", "_avg_diff_temp_masked_", "avg_diff_temp_XX"))

    ## Dictionaries / storage - need to convert to R for matrix operations
    levels_d_sq_XX <- pl_unique(df, "d_sq_int_XX")
    levels_d_sq_XX <- sort(levels_d_sq_XX[!is.na(levels_d_sq_XX)])
    store_singular <- setNames(rep(FALSE, length(levels_d_sq_XX)), as.character(levels_d_sq_XX))
    store_noresidualization_XX <- integer(0L)
    levels_d_sq_XX_final <- integer(0L)

    ## Loop over each baseline-treatment level - matrix operations in R
    for (l in levels_d_sq_XX) {
      # Count unique F_g values for this level
      df_l <- df$filter(pl$col("d_sq_int_XX") == l)
      useful <- pl_n_unique(df_l, "F_g_XX")
      assign(paste0("useful_res_", l, "_XX"), useful)

      if (useful > 1L) {
        # Filter for control observations
        data_pl <- df$filter(
          (pl$col("ever_change_d_XX") == 0) &
          pl$col("diff_y_XX")$is_not_null() &
          (pl$col("fd_X_all_non_missing_XX") == 1L) &
          (pl$col("d_sq_int_XX") == l)
        )

        if (data_pl$height == 0L) {
          store_singular[as.character(l)] <- TRUE
          store_noresidualization_XX <- c(store_noresidualization_XX, l)
          assign(paste0("useful_res_", l, "_XX"), 1L)
          next
        }

        # Extract vectors for matrix operations
        data_XX <- as.data.frame(data_pl)
        Y_vec <- data_XX$diff_y_wXX
        X_mat <- as.matrix(data_XX[, mycontrols_XX, drop = FALSE])
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

          theta_d <- ginv(M) %*% v
          assign(paste0("coefs_sq_", l, "_XX"), theta_d)
          levels_d_sq_XX_final <- c(levels_d_sq_XX_final, l)

          if (abs(det(M)) <= 1e-16) {
            store_singular[as.character(l)] <- TRUE
          }

          # Stata computes rsum on control sample only (after keep if ever_change_d_XX==0&...)
          # Filter to control sample first, then apply time conditions
          control_sample <- df$filter(
            (pl$col("ever_change_d_XX") == 0) &
            pl$col("diff_y_XX")$is_not_null() &
            (pl$col("fd_X_all_non_missing_XX") == 1) &
            (pl$col("d_sq_int_XX") == l)
          )
          rmax <- pl_max(control_sample, "F_g_XX")
          rsum_df <- control_sample$filter(
            (pl$col("time_XX") >= 2) &
            (pl$col("time_XX") <= (rmax - 1)) &
            (pl$col("time_XX") < pl$col("F_g_XX")) &
            pl$col("diff_y_XX")$is_not_null()
          )
          rsum <- pl_sum(rsum_df, "N_gt_XX")

          assign(paste0("inv_Denom_", l, "_XX"), ginv(M) * rsum * G_XX)
        }
      }
    }

    ## Handle singular levels warnings
    levels_d_sq_bis_XX <- pl_unique(df, "d_sq_XX")
    levels_d_sq_bis_XX <- sort(levels_d_sq_bis_XX[!is.na(levels_d_sq_bis_XX)])
    singular_levels <- integer(0L)
    for (l in levels_d_sq_bis_XX) {
      key <- as.character(l)
      if (!is.null(store_singular[key]) && isTRUE(store_singular[key])) {
        singular_levels <- c(singular_levels, l)
      }
    }

    if (length(singular_levels) > 0L) {
      store_singular_XX <- paste(singular_levels, collapse = " ")
      warning("Some control variables are not taken into account for groups with baseline treatment equal to:", store_singular_XX)
      warning("1. For these groups, the regression of Y evolution X evolution and time-FE had fewer observations than regressors.")
      warning("2. For these groups, one or more controls were perfectly collinear (no time variation).")
    }

    ## Drop levels where residualization failed
    if (length(store_noresidualization_XX) > 0L) {
      filter_expr <- pl$lit(TRUE)
      for (lvl in store_noresidualization_XX) {
        filter_expr <- filter_expr & (pl$col("d_sq_int_XX") != lvl)
      }
      df <- df$filter(filter_expr)
    }

    ## 3. FE regression using feols (optimized C code) - need data.frame for feols
    df <- df$with_columns(pl$col("time_XX")$cast(pl$Int32)$alias("time_FE_XX"))

    # Add row_id for reliable joining (like CRAN implementation)
    df <- df$with_row_index("row_id_XX")

    ## 4. Loop over baseline-treatment levels - use feols (optimized C code)
    for (l in levels_d_sq_XX_final) {
      outcol <- sprintf("E_y_hat_gt_int_%d_XX", l)

      # Filter data for this level
      data_reg_pl <- df$filter(
        (pl$col("d_sq_int_XX") == l) & (pl$col("F_g_XX") > pl$col("time_XX"))
      )

      if (data_reg_pl$height == 0L) {
        df <- df$with_columns(pl$lit(NA_real_)$alias(outcol))
        next
      }

      # Convert to data.frame for feols
      data_reg <- as.data.frame(data_reg_pl)

      # Use feols for FE regression (highly optimized)
      fe_terms <- sprintf("diff_X%d_XX", seq_len(count_controls))
      formula_str <- paste("diff_y_XX ~", paste(fe_terms, collapse = " + "), "- 1 | time_FE_XX")
      form <- as.formula(formula_str)

      model <- feols(form, data = data_reg, weights = data_reg$weight_XX)
      data_reg$y_hat <- predict(model, newdata = data_reg)

      # Create lookup table with row_id for reliable joining
      # Cast row_id_XX back to UInt32 to match original type (R converts to Float64)
      lookup_pl <- as_polars_df(data_reg[, c("row_id_XX", "y_hat")])
      lookup_pl <- lookup_pl$select(
        pl$col("row_id_XX")$cast(pl$UInt32),
        pl$col("y_hat")$alias(outcol)
      )

      # Join back to main df using row_id
      df <- df$join(lookup_pl, on = "row_id_XX", how = "left")
    }

    ## Clean up temporary columns
    df <- df$drop(c("time_FE_XX", "row_id_XX"))
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

  ## Pure Polars path for computing L_u/L_a
  ## For switchers in - use native Polars filtering
  if (switchers == "" | switchers == "in") {
    switchers_in_df <- df$filter(pl$col("S_g_XX") == 1)
    n_switchers_in <- switchers_in_df$height
    if (n_switchers_in == 0) {
      L_u_XX <- 0
    } else {
      L_u_XX <- pl_max(switchers_in_df, "L_g_XX")
      if (is.na(L_u_XX) || is.infinite(L_u_XX)) L_u_XX <- 0
    }
    ## For placebos
    if (placebo != 0 && n_switchers_in > 0) {
      L_placebo_u_XX <- pl_max(switchers_in_df, "L_g_placebo_XX")
      L_placebo_u_XX <- ifelse(is.na(L_placebo_u_XX) || L_placebo_u_XX < 0, 0, L_placebo_u_XX)
      ## If the trends_lin option was specified, L_placebo_u_XX should be decreased by 1
      ## because data starts at period 2 instead of 1.
      if (isTRUE(trends_lin)) {
        L_placebo_u_XX <- L_placebo_u_XX - 1
      }
    }
  }

  ## For switchers out - use native Polars filtering
  if (switchers == "" | switchers == "out") {
    switchers_out_df <- df$filter(pl$col("S_g_XX") == 0)
    n_switchers_out <- switchers_out_df$height
    if (n_switchers_out == 0) {
      L_a_XX <- 0
    } else {
      L_a_XX <- pl_max(switchers_out_df, "L_g_XX")
      if (is.na(L_a_XX) || is.infinite(L_a_XX)) L_a_XX <- 0
    }
    if (placebo != 0 && n_switchers_out > 0) {
      L_placebo_a_XX <- pl_max(switchers_out_df, "L_g_placebo_XX")
      L_placebo_a_XX <- ifelse(is.na(L_placebo_a_XX) || L_placebo_a_XX < 0, 0, L_placebo_a_XX)
      if (isTRUE(trends_lin)) {
        L_placebo_a_XX <- L_placebo_a_XX - 1
      }
    }
  }

  # Keep working with polars DataFrame throughout
  # df is already a polars DataFrame from earlier processing

  ## Error message if Design restriction 1 is not met
  if (
    (switchers == "in" & (is.na(L_u_XX) | L_u_XX == 0)) | 
    (switchers == "out" & (is.na(L_a_XX) | L_a_XX == 0)) | 
    (switchers == "" &  ((is.na(L_u_XX) | L_u_XX == 0) & (is.na(L_a_XX) | L_a_XX == 0)))
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
    if (l_placebo_XX < placebo & effects >= placebo) {
      message(sprintf("The number of placebos which can be estimated is at most %.0f.The command will therefore try to estimate %.0f placebo(s).", l_placebo_XX, l_placebo_XX))
    }
    if (effects < placebo) {
      message(sprintf("The number of placebo requested cannot be larger than the number of effects requested. The command cannot compute more than %.0f placebo(s).", l_placebo_XX))
    }
  }

  ## Adjustment to add more placebos (did_multiplegt_dyn_all_pl)
  max_pl_u_XX <- max_pl_a_XX <- max_pl_gap_u_XX <- max_pl_gap_a_XX <- 0
  df <- df$with_columns(
    pl$when(pl$col("S_g_XX")$is_not_null())$
      then(pl$col("F_g_XX") - 2 - pl$col("L_g_XX"))$
      otherwise(pl$lit(NA_real_))$alias("pl_gap_XX")
  )
  if (switchers == "" | switchers == "in") {
    max_pl_u_XX <- pl_max(df$filter(pl$col("S_g_XX") == 1), "F_g_XX") - 2
    max_pl_gap_u_XX <- pl_max(df$filter(pl$col("S_g_XX") == 1), "pl_gap_XX")
    if (is.na(max_pl_u_XX)) max_pl_u_XX <- 0
    if (is.na(max_pl_gap_u_XX)) max_pl_gap_u_XX <- 0
  }
  if (switchers == "" | switchers == "out") {
    max_pl_a_XX <- pl_max(df$filter(pl$col("S_g_XX") == 0), "F_g_XX") - 2
    max_pl_gap_a_XX <- pl_max(df$filter(pl$col("S_g_XX") == 0), "pl_gap_XX")
    if (is.na(max_pl_a_XX)) max_pl_a_XX <- 0
    if (is.na(max_pl_gap_a_XX)) max_pl_gap_a_XX <- 0
  }
  max_pl_XX <- max(max_pl_u_XX, max_pl_a_XX)
  max_pl_gap_XX <- max(max_pl_gap_u_XX, max_pl_gap_a_XX)
  max_pl_u_XX <- max_pl_a_XX <- max_pl_gap_u_XX <- max_pl_gap_a_XX <- NULL
  df <- df$drop("pl_gap_XX")

  ## Generating default values for the variables which will be aggregated
  ## after Program 2 below has been run for switchers in and for switchers out.

  inh_obj <- c()
  # Initialize effect columns using polars
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
  df <- pl_init_zero_cols(df, effect_cols)
  assign("sum_for_var_in_XX", 0)
  assign("sum_for_var_out_XX", 0)
  inh_obj <- c(inh_obj, "sum_for_var_in_XX", "sum_for_var_out_XX")
  if (placebo != 0) {
    # Initialize placebo columns using polars
    placebo_cols <- c(
      paste0("U_Gg_pl_", 1:l_XX, "_plus_XX"),
      paste0("U_Gg_pl_", 1:l_XX, "_minus_XX"),
      paste0("count", 1:l_XX, "_pl_plus_XX"),
      paste0("count", 1:l_XX, "_pl_minus_XX"),
      paste0("U_Gg_var_pl_", 1:l_XX, "_in_XX"),
      paste0("U_Gg_var_pl_", 1:l_XX, "_out_XX")
    )
    df <- pl_init_zero_cols(df, placebo_cols)
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
  for (v in base_vars) assign(v, 0)
  inh_obj <- c(inh_obj, base_vars)

  if (normalized == TRUE) {
    norm_vars <- c(paste0("delta_D_", 1:l_XX, "_in_XX"), paste0("delta_D_", 1:l_XX, "_out_XX"))
    for (v in norm_vars) assign(v, 0)
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
    for (v in placebo_vars) assign(v, 0)
    inh_obj <- c(inh_obj, placebo_vars)

    if (normalized == TRUE) {
      norm_pl_vars <- c(paste0("delta_D_pl_", 1:l_XX, "_in_XX"), paste0("delta_D_pl_", 1:l_XX, "_out_XX"))
      for (v in norm_pl_vars) assign(v, 0)
      inh_obj <- c(inh_obj, norm_pl_vars)
    }
  }

  df <- pl_init_zero_cols(df, c("U_Gg_plus_XX", "U_Gg_minus_XX", "U_Gg_var_plus_XX", "U_Gg_var_minus_XX"))
  assign("U_Gg_den_plus_XX", 0)
  assign("U_Gg_den_minus_XX", 0)
  assign("sum_N1_l_XX", 0)
  assign("sum_N0_l_XX", 0)
  inh_obj <- c(inh_obj,"U_Gg_den_plus_XX", "U_Gg_den_minus_XX", "sum_N1_l_XX", "sum_N0_l_XX")

  # Scalars previously passed as inherited objects
  # Their values will be changes through the next routines
  const <- NULL
  for (v in inh_obj) {
    const[[v]] <- get(v)
  }

  # Saving useful scalars to the Global Environment
  # Their values will be not changes through the next routines
  gs <- c("L_u_XX", "L_a_XX", "l_XX", "t_min_XX", "T_max_XX", "G_XX")
  if (placebo != 0) {
    gs <- c(gs, "L_placebo_u_XX", "L_placebo_a_XX")
  }
  # Add inheritance of controls #
  globals <- NULL
  for (v in gs) {
    globals[[v]] <- get(v)
  }

  controls_globals <- NULL
  if (!is.null(controls)) {
    controls_globals <- list()
    for (l in levels_d_sq_XX) {
      controls_globals <- append(controls_globals, get(paste0("useful_res_", l, "_XX")))
      names(controls_globals)[length(controls_globals)] <- paste0("useful_res_", l, "_XX")
      controls_globals <- append(controls_globals, list(get(paste0("coefs_sq_", l, "_XX"))))
      names(controls_globals)[length(controls_globals)] <- paste0("coefs_sq_", l, "_XX")
      controls_globals <- append(controls_globals, list(get(paste0("inv_Denom_",l,"_XX"))))
      names(controls_globals)[length(controls_globals)] <- paste0("inv_Denom_", l, "_XX")
    }
  }

  ## Initialize variable to earmark switchers by the number of the event-study effect
  df <- df$with_columns(pl$lit(NA_real_)$alias("switchers_tag_XX"))

  ## Store the data prior to estimation if requested
  if (isTRUE(data_only)) {
    data <- list(df, l_XX, T_max_XX)
    names(data) <- c("df", "l_XX", "T_max_XX")
    return(data)
  }

  ## Perform the estimation: call the program did_multiplegt_dyn_core,
  ## for switchers in and for switchers out, and store the results.
  ## df is now a polars DataFrame passed directly to core function

  if (switchers == "" | switchers == "in") {
    if (!is.na(L_u_XX) & L_u_XX != 0) {

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
        for (e in names(data$const)) {
          const[[e]] <- data$const[[e]]
          assign(e, const[[e]])
        }

        # Store the number of the event-study effect for switchers-in
        for (k in 1:l_XX) {
          dist_col <- paste0("distance_to_switch_", k, "_XX")
          df <- df$with_columns(
            pl$when(pl$col(dist_col) == 1)$then(pl$lit(as.numeric(k)))$otherwise(pl$col("switchers_tag_XX"))$alias("switchers_tag_XX")
          )
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
          for (e in names(data$const)) {
            const[[e]] <- data$const[[e]]
            assign(e, const[[e]])
          }

          ## Store the number of the event-study effect for switchers-in
          col_dist_i <- sprintf("distance_to_switch_%d_XX", i)
          df <- df$with_columns(
            pl$when(pl$col(col_dist_i) == 1)$then(pl$lit(as.numeric(i)))$otherwise(pl$col("switchers_tag_XX"))$alias("switchers_tag_XX")
          )
        }

        ## Store variables necessary for computation of effects.
        ## N.B.: in the case of unbalanced panels, it can happen that the U_Gg`i'_XX are not computed by program 2 (for example when y is missing). Consequently, for the command not to display an error message and continue running, we need to verify the variable is created, which is conditional on  N1_`i'_XX!=0.

        if (get(paste0("N1_",i,"_XX")) != 0) {
          src_col <- paste0("U_Gg", i, "_XX")
          dst_col <- paste0("U_Gg", i, "_plus_XX")
          df <- df$with_columns(pl$col(src_col)$alias(dst_col))

          src_col <- paste0("count", i, "_core_XX")
          dst_col <- paste0("count", i, "_plus_XX")
          df <- df$with_columns(pl$col(src_col)$alias(dst_col))

          src_col <- paste0("U_Gg", i, "_var_XX")
          dst_col <- paste0("U_Gg_var_", i, "_in_XX")
          df <- df$with_columns(pl$col(src_col)$alias(dst_col))

          assign(paste0("N1_",i,"_XX_new"), get(paste0("N1_",i,"_XX")))
          const[[paste0("N1_",i,"_XX_new")]] <- get(paste0("N1_",i,"_XX_new"))

          if (normalized == TRUE) {
            assign(paste0("delta_D_",i,"_in_XX"), get(paste0("delta_norm_",i,"_XX")))
            const[[paste0("delta_D_",i,"_in_XX")]] <- get(paste0("delta_D_",i,"_in_XX"))
          }

          if (isFALSE(trends_lin)) {
            src_col <- paste0("delta_D_g_", i, "_XX")
            dst_col <- paste0("delta_D_g_", i, "_plus_XX")
            df <- df$with_columns(pl$col(src_col)$alias(dst_col))
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
            for (e in names(data$const)) {
              const[[e]] <- data$const[[e]]
              assign(e, const[[e]])
            }

            col_dist_i <- sprintf("distance_to_switch_%d_XX", i)
            df <- df$with_columns(
              pl$when(pl$col(col_dist_i) == 1)$then(pl$lit(as.numeric(i)))$otherwise(pl$col("switchers_tag_XX"))$alias("switchers_tag_XX")
            )

          }

          if (get(paste0("N1_placebo_",i,"_XX")) != 0) {
            df <- df$with_columns(pl$col(paste0("U_Gg_placebo_",i,"_XX"))$alias(paste0("U_Gg_pl_",i,"_plus_XX")))
            df <- df$with_columns(pl$col(paste0("count",i,"_pl_core_XX"))$alias(paste0("count",i,"_pl_plus_XX")))
            df <- df$with_columns(pl$col(paste0("U_Gg_pl_",i,"_var_XX"))$alias(paste0("U_Gg_var_pl_",i,"_in_XX")))
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
          df <- df$with_columns(pl$col("U_Gg_XX")$alias("U_Gg_plus_XX"))
          df <- df$with_columns(pl$col("U_Gg_den_XX")$alias("U_Gg_den_plus_XX"))
          df <- df$with_columns(pl$col("U_Gg_var_XX")$alias("U_Gg_var_plus_XX"))
        }
      }
    }
  }



  ######################## Puedes Volver aqui en cualquier momento ###############



  ## Same thing as above, for switchers out
  if (switchers == "" | switchers == "out") {
    if (!is.na(L_a_XX) & L_a_XX != 0) {

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
        for (e in names(data$const)) {
          const[[e]] <- data$const[[e]]
          assign(e, const[[e]])
        }

        for (k in 1:l_XX) {
          ## Store the number of the event-study effect for switchers-out
          dist_col <- paste0("distance_to_switch_", k, "_XX")
          df <- df$with_columns(
            pl$when(pl$col(dist_col) == 1)$then(pl$lit(as.numeric(k)))$otherwise(pl$col("switchers_tag_XX"))$alias("switchers_tag_XX")
          )
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
          for (e in names(data$const)) {
            const[[e]] <- data$const[[e]]
            assign(e, const[[e]])
          }

          ## Store the number of the event-study effect for switchers-out
          dist_col <- paste0("distance_to_switch_", i, "_XX")
          df <- df$with_columns(
            pl$when(pl$col(dist_col) == 1)$then(pl$lit(as.numeric(i)))$otherwise(pl$col("switchers_tag_XX"))$alias("switchers_tag_XX")
          )
        }

        if (get(paste0("N0_",i,"_XX")) != 0) {
          # Negate and copy columns using polars
          df <- df$with_columns((-pl$col(paste0("U_Gg",i,"_XX")))$alias(paste0("U_Gg",i,"_minus_XX")))
          df <- df$with_columns(pl$col(paste0("count",i,"_core_XX"))$alias(paste0("count",i,"_minus_XX")))
          df <- df$with_columns((-pl$col(paste0("U_Gg",i,"_var_XX")))$alias(paste0("U_Gg_var_",i,"_out_XX")))
          assign(paste0("N0_",i,"_XX_new"), get(paste0("N0_",i,"_XX")))
          const[[paste0("N0_",i,"_XX_new")]] <- get(paste0("N0_",i,"_XX_new"))

          if (normalized == TRUE) {
            assign(paste0("delta_D_",i,"_out_XX"), get(paste0("delta_norm_",i,"_XX")))
            const[[paste0("delta_D_",i,"_out_XX")]] <- get(paste0("delta_D_",i,"_out_XX"))
          }

          if (isFALSE(trends_lin)) {
            df <- df$with_columns(pl$col(paste0("delta_D_g_",i,"_XX"))$alias(paste0("delta_D_g_",i,"_minus_XX")))
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
            for (e in names(data$const)) {
              const[[e]] <- data$const[[e]]
              assign(e, const[[e]])
            }
            dist_col <- paste0("distance_to_switch_", i, "_XX")
            df <- df$with_columns(
              pl$when(pl$col(dist_col) == 1)$then(pl$lit(as.numeric(i)))$otherwise(pl$col("switchers_tag_XX"))$alias("switchers_tag_XX")
            )
          }

          if (get(paste0("N0_placebo_",i,"_XX")) != 0) {
            df <- df$with_columns((-pl$col(paste0("U_Gg_placebo_",i,"_XX")))$alias(paste0("U_Gg_pl_",i,"_minus_XX")))
            df <- df$with_columns(pl$col(paste0("count",i,"_pl_core_XX"))$alias(paste0("count",i,"_pl_minus_XX")))
            df <- df$with_columns((-pl$col(paste0("U_Gg_pl_",i,"_var_XX")))$alias(paste0("U_Gg_var_pl_",i,"_out_XX")))
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
          df <- df$with_columns((-pl$col("U_Gg_XX"))$alias("U_Gg_minus_XX"))
          df <- df$with_columns(pl$col("U_Gg_den_XX")$alias("U_Gg_den_minus_XX"))
          df <- df$with_columns((-pl$col("U_Gg_var_XX"))$alias("U_Gg_var_minus_XX"))
        }
      }
    }
  }
  rownames <- c()

  ###### 5. Computing the estimators and their variances (PURE POLARS LAZY EVALUATION)

  # Helper function to check if column exists in polars dataframe
  pl_has_col <- function(df, col_name) {
    col_name %in% df$columns
  }

  # Helper to extract scalar value from polars result
  pl_scalar <- function(pl_result) {
    as.data.frame(pl_result)[[1]]
  }

  # Helper to get scalar sum from polars column with first_obs filter
  pl_sum_first_obs <- function(df, col_name, first_obs_col = "first_obs_by_gp_XX") {
    if (!pl_has_col(df, col_name)) return(0)
    result <- df$lazy()$filter(pl$col(first_obs_col) == 1)$select(pl$col(col_name)$sum())$collect()
    val <- pl_scalar(result)
    if (is.null(val) || length(val) == 0 || is.na(val)) return(0)
    return(val)
  }

  # Helper to get scalar sum of squared values with first_obs filter
  pl_sum_sq_first_obs <- function(df, col_name, first_obs_col = "first_obs_by_gp_XX") {
    if (!pl_has_col(df, col_name)) return(0)
    result <- df$lazy()$filter(pl$col(first_obs_col) == 1)$select((pl$col(col_name)$pow(2))$sum())$collect()
    val <- pl_scalar(result)
    if (is.null(val) || length(val) == 0 || is.na(val)) return(0)
    return(val)
  }

  # Helper to count non-NA values with first_obs filter
  pl_count_first_obs <- function(df, col_name, first_obs_col = "first_obs_by_gp_XX") {
    if (!pl_has_col(df, col_name)) return(0)
    result <- df$lazy()$filter(pl$col(first_obs_col) == 1)$select(pl$col(col_name)$is_not_null()$sum())$collect()
    val <- pl_scalar(result)
    if (is.null(val) || length(val) == 0 || is.na(val)) return(0)
    return(val)
  }

  # Helper to sum all values of a column (like CRAN's sum(col, na.rm=TRUE))
  pl_sum_col <- function(df, col_name) {
    if (!pl_has_col(df, col_name)) return(0)
    result <- df$lazy()$select(pl$col(col_name)$sum())$collect()
    val <- pl_scalar(result)
    if (is.null(val) || length(val) == 0 || is.na(val)) return(0)
    return(val)
  }

  # Helper for variance computation that handles clustering correctly
  # CRAN approach for clustered variance:
  #   1. Multiply var column by first_obs_by_gp_XX
  #   2. Sum by cluster_XX
  #   3. Square the cluster sums
  #   4. Multiply by first_obs_by_clust_XX
  #   5. Sum total
  # For non-clustered: just square and sum with first_obs_by_gp_XX filter
  pl_compute_variance_sum <- function(df, col_name, clustered) {
    if (!pl_has_col(df, col_name)) return(0)

    if (!clustered) {
      # Non-clustered: just square and sum
      result <- df$lazy()$filter(pl$col("first_obs_by_gp_XX") == 1)$select((pl$col(col_name)$pow(2))$sum())$collect()
      val <- pl_scalar(result)
      if (is.null(val) || length(val) == 0 || is.na(val)) return(0)
      return(val)
    } else {
      # Clustered: use data.table for exact match with CRAN
      # Convert to data.table for the variance computation
      dt <- as.data.table(as.data.frame(df$select(c("cluster_XX", "first_obs_by_gp_XX", "first_obs_by_clust_XX", col_name))))

      # Step 1: Multiply by first_obs_by_gp_XX
      dt[, var_weighted := get(col_name) * first_obs_by_gp_XX]

      # Step 2: Sum by cluster_XX
      dt[, clust_var_sum := sum(var_weighted, na.rm = TRUE), by = cluster_XX]

      # Step 3 & 4: Square and multiply by first_obs_by_clust_XX
      dt[, clust_var_sq := clust_var_sum^2 * first_obs_by_clust_XX]

      # Step 5: Sum total
      var_sq_sum <- sum(dt$clust_var_sq, na.rm = TRUE)

      return(var_sq_sum)
    }
  }

  # Creation of the matrix which stores all the estimators (DID_l, DID_pl, delta, etc.), their sd and the CIs
  mat_res_XX <- matrix(NA, nrow = l_XX + l_placebo_XX + 1, ncol = 9)

  # CI level
  ci_level <- ci_level / 100
  z_level <- qnorm(ci_level + (1 - ci_level)/2)

  # Handle clustering for variance computation
  clustered <- !is.null(cluster)
  first_obs_col <- if (clustered) "first_obs_by_clust_XX" else "first_obs_by_gp_XX"
  # CRAN uses G_XX for both clustered and non-clustered variance computation
  G_var <- G_XX

  # BATCHED APPROACH: Build all columns first, then aggregate once
  # Step 1: Build weight vectors
  N1_vec <- numeric(l_XX)
  N0_vec <- numeric(l_XX)
  for (i in 1:l_XX) {
    N1_vec[i] <- get(paste0("N1_", i, "_XX_new"))
    N0_vec[i] <- get(paste0("N0_", i, "_XX_new"))
  }

  if (getOption("DID_DEBUG_VARIANCE", FALSE)) {
    cat("\n=== DEBUG: Weights for variance ===\n")
    cat("N1_vec:", N1_vec, "\n")
    cat("N0_vec:", N0_vec, "\n")
    cat("w_in (N1/(N1+N0)):", N1_vec / (N1_vec + N0_vec), "\n")
    cat("w_out (N0/(N1+N0)):", N0_vec / (N1_vec + N0_vec), "\n")
  }

  # Step 2: Add all global columns in batched with_columns calls
  col_exprs <- list()
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
      var_in_expr <- if (pl_has_col(df, col_var_in)) pl$col(col_var_in)$fill_null(0) else pl$lit(0)
      var_out_expr <- if (pl_has_col(df, col_var_out)) pl$col(col_var_out)$fill_null(0) else pl$lit(0)
      col_exprs[[length(col_exprs) + 1]] <- (pl$lit(w_in) * var_in_expr + pl$lit(w_out) * var_out_expr)$alias(paste0("U_Gg_var_glob_", i, "_XX"))
    } else {
      col_exprs[[length(col_exprs) + 1]] <- pl$lit(0)$alias(paste0("U_Gg_var_glob_", i, "_XX"))
    }

    # Global U_Gg column
    plus_expr <- if (pl_has_col(df, col_plus)) pl$col(col_plus)$fill_null(0) else pl$lit(0)
    minus_expr <- if (pl_has_col(df, col_minus)) pl$col(col_minus)$fill_null(0) else pl$lit(0)
    if (total_N > 0) {
      col_exprs[[length(col_exprs) + 1]] <- (pl$lit(N1_i / total_N) * plus_expr + pl$lit(N0_i / total_N) * minus_expr)$alias(paste0("U_Gg", i, "_global_XX"))
    } else {
      col_exprs[[length(col_exprs) + 1]] <- pl$lit(0)$alias(paste0("U_Gg", i, "_global_XX"))
    }

    # Global count column - CRAN takes MAX of plus and minus, handling NAs
    # Logic: if both non-NA, take max; if one is NA, take the other
    count_plus_expr <- if (pl_has_col(df, col_count_plus)) pl$col(col_count_plus) else pl$lit(NA_real_)
    count_minus_expr <- if (pl_has_col(df, col_count_minus)) pl$col(col_count_minus) else pl$lit(NA_real_)
    col_exprs[[length(col_exprs) + 1]] <- pl$when(count_plus_expr$is_null())$then(count_minus_expr)$when(count_minus_expr$is_null())$then(count_plus_expr)$when(count_plus_expr > count_minus_expr)$then(count_plus_expr)$otherwise(count_minus_expr)$alias(paste0("count", i, "_global_XX"))
  }

  # Apply all column expressions at once
  df <- do.call(function(...) df$with_columns(...), col_exprs)

  # Step 3: Batch all aggregations in a single collect
  agg_exprs <- list()
  for (i in 1:l_XX) {
    col_plus <- paste0("U_Gg", i, "_plus_XX")
    col_minus <- paste0("U_Gg", i, "_minus_XX")
    col_var_glob <- paste0("U_Gg_var_glob_", i, "_XX")
    col_count_global <- paste0("count", i, "_global_XX")

    # Sum of plus column (first_obs filtered)
    if (pl_has_col(df, col_plus)) {
      agg_exprs[[length(agg_exprs) + 1]] <- (pl$col(col_plus) * pl$col("first_obs_by_gp_XX"))$sum()$alias(paste0("sum_plus_", i))
    }
    # Sum of minus column
    if (pl_has_col(df, col_minus)) {
      agg_exprs[[length(agg_exprs) + 1]] <- (pl$col(col_minus) * pl$col("first_obs_by_gp_XX"))$sum()$alias(paste0("sum_minus_", i))
    }
    # Sum of squared variance - NON-CLUSTERED case only
    # For clustered case, we handle separately below
    if (!clustered) {
      agg_exprs[[length(agg_exprs) + 1]] <- ((pl$col(col_var_glob)$pow(2)) * pl$col("first_obs_by_gp_XX"))$sum()$alias(paste0("var_sq_sum_", i))
    }
    # Sum of count_global (N_effect) - CRAN sums all values, not just first_obs filtered
    agg_exprs[[length(agg_exprs) + 1]] <- pl$col(col_count_global)$sum()$alias(paste0("count_effects_", i))
    # Count dw (non-null and > 0)
    agg_exprs[[length(agg_exprs) + 1]] <- ((pl$col(col_count_global)$is_not_null() & (pl$col(col_count_global) > 0))$cast(pl$Int32))$sum()$alias(paste0("count_dw_", i))
  }

  # Execute all aggregations in one collect
  agg_result <- do.call(function(...) df$lazy()$select(...)$collect(), agg_exprs)
  agg_df <- as.data.frame(agg_result)

  # Handle clustered variance computation separately
  # CRAN approach:
  #   1. Multiply U_Gg_var_glob_XX by first_obs_by_gp_XX
  #   2. Sum by cluster_XX
  #   3. Square the cluster sums
  #   4. Multiply by first_obs_by_clust_XX
  #   5. Sum total
  if (clustered) {
    # Use data.table for exact match with CRAN cluster variance computation
    dt <- as.data.table(as.data.frame(df$select(c(
      "cluster_XX", "first_obs_by_gp_XX", "first_obs_by_clust_XX",
      paste0("U_Gg_var_glob_", 1:l_XX, "_XX")
    ))))

    # Debug: Check counts
    if (getOption("DID_DEBUG_VARIANCE", FALSE)) {
      cat("\n=== DEBUG: Clustered variance computation ===\n")
      cat("Rows in dt:", nrow(dt), "\n")
      cat("Sum of first_obs_by_gp_XX:", sum(dt$first_obs_by_gp_XX, na.rm = TRUE), "\n")
      cat("Sum of first_obs_by_clust_XX:", sum(dt$first_obs_by_clust_XX, na.rm = TRUE), "\n")
      cat("Unique clusters:", uniqueN(dt$cluster_XX), "\n")
      cat("G_XX used:", G_XX, "\n")

      # Check U_Gg_var_glob values directly
      col1 <- "U_Gg_var_glob_1_XX"
      if (col1 %in% names(dt)) {
        cat("\nU_Gg_var_glob_1_XX stats:\n")
        cat("  Min:", min(dt[[col1]], na.rm = TRUE), "\n")
        cat("  Max:", max(dt[[col1]], na.rm = TRUE), "\n")
        cat("  Mean:", mean(dt[[col1]], na.rm = TRUE), "\n")
        cat("  SD:", sd(dt[[col1]], na.rm = TRUE), "\n")
        cat("  Sum:", sum(dt[[col1]], na.rm = TRUE), "\n")
        cat("  Sum of squares:", sum(dt[[col1]]^2, na.rm = TRUE), "\n")
        cat("  Non-NA count:", sum(!is.na(dt[[col1]])), "\n")
        cat("  Non-zero count:", sum(dt[[col1]] != 0, na.rm = TRUE), "\n")

        # Check first_obs filtered values
        first_obs_vals <- dt[[col1]] * dt$first_obs_by_gp_XX
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
      dt[, var_weighted := get(col_var_glob) * first_obs_by_gp_XX]

      # Step 2: Sum by cluster_XX
      dt[, (clust_sum_col) := sum(var_weighted, na.rm = TRUE), by = cluster_XX]

      # Step 3 & 4: Square and multiply by first_obs_by_clust_XX
      dt[, clust_var_sq := get(clust_sum_col)^2 * first_obs_by_clust_XX]

      # Step 5: Sum total
      var_sq_sum <- sum(dt$clust_var_sq, na.rm = TRUE)

      if (getOption("DID_DEBUG_VARIANCE", FALSE) && i == 1) {
        cat("\nEffect", i, ":\n")
        cat("  Non-zero var_weighted:", sum(dt$var_weighted != 0 & !is.na(dt$var_weighted)), "\n")
        cat("  Sum of var_weighted:", sum(dt$var_weighted, na.rm = TRUE), "\n")
        cat("  Non-zero clust_var_sq:", sum(dt$clust_var_sq != 0 & !is.na(dt$clust_var_sq)), "\n")
        cat("  var_sq_sum:", var_sq_sum, "\n")
        cat("  var_sq_sum / G_XX^2:", var_sq_sum / G_XX^2, "\n")
        cat("  SE:", sqrt(var_sq_sum / G_XX^2), "\n")
      }

      agg_df[[paste0("var_sq_sum_", i)]] <- var_sq_sum

      # CRAN replaces U_Gg_var_glob with cluster-summed values for later covariance computation
      # We need to do the same: replace U_Gg_var_glob_i_XX with clust_U_Gg_var_glob_i_XX
      dt[, (col_var_glob) := get(clust_sum_col)]

      # Clean up temp columns
      dt[, c("var_weighted", "clust_var_sq") := NULL]
    }

    # Update the polars df with the cluster-summed U_Gg_var_glob values
    # This is needed for the covariance computation in section 6
    for (i in 1:l_XX) {
      col_var_glob <- paste0("U_Gg_var_glob_", i, "_XX")
      clust_sum_col <- paste0("clust_U_Gg_var_glob_", i, "_XX")
      # Create a polars series from the data.table column and update df
      clust_vals <- dt[[clust_sum_col]]
      df <- df$with_columns(pl$lit(clust_vals)$alias(col_var_glob))
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
    sum_plus <- if (paste0("sum_plus_", i) %in% names(agg_df)) agg_df[[paste0("sum_plus_", i)]] else 0
    sum_minus <- if (paste0("sum_minus_", i) %in% names(agg_df)) agg_df[[paste0("sum_minus_", i)]] else 0
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
  for (i in 1:l_XX) {
    count_global_col <- paste0("count", i, "_global_XX")
    df <- df$with_columns(
      pl$when(pl$col(count_global_col)$is_not_null() & (pl$col(count_global_col) > 0))$then(1)$otherwise(0)$alias(paste0("count", i, "_global_dwXX"))
    )
  }

  ###### Computing the average total effect
  U_Gg_den_plus_XX <- 0
  U_Gg_den_minus_XX <- 0
  if (pl_has_col(df, "U_Gg_den_plus_XX")) {
    result <- pl_scalar(df$lazy()$select(pl$col("U_Gg_den_plus_XX")$mean())$collect())
    U_Gg_den_plus_XX <- if (is.na(result) || is.null(result)) 0 else result
  }
  if (pl_has_col(df, "U_Gg_den_minus_XX")) {
    result <- pl_scalar(df$lazy()$select(pl$col("U_Gg_den_minus_XX")$mean())$collect())
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

    # Compute average effect using polars
    plus_expr <- if (pl_has_col(df, "U_Gg_plus_XX")) pl$col("U_Gg_plus_XX")$fill_null(0) else pl$lit(0)
    minus_expr <- if (pl_has_col(df, "U_Gg_minus_XX")) pl$col("U_Gg_minus_XX")$fill_null(0) else pl$lit(0)

    df <- df$with_columns(
      (pl$lit(w_plus_XX) * plus_expr + pl$lit(1 - w_plus_XX) * minus_expr)$alias("U_Gg_global_XX")
    )

    # Sum for average effect
    sum_avg <- pl_sum_first_obs(df, "U_Gg_global_XX")
    delta_XX <- sum_avg / G_XX

    # Variance for average effect
    var_plus_expr <- if (pl_has_col(df, "U_Gg_var_plus_XX")) pl$col("U_Gg_var_plus_XX")$fill_null(0) else pl$lit(0)
    var_minus_expr <- if (pl_has_col(df, "U_Gg_var_minus_XX")) pl$col("U_Gg_var_minus_XX")$fill_null(0) else pl$lit(0)

    df <- df$with_columns(
      (pl$lit(w_plus_XX) * var_plus_expr + pl$lit(1 - w_plus_XX) * var_minus_expr)$alias("U_Gg_var_global_XX")
    )

    var_sum_avg <- pl_compute_variance_sum(df, "U_Gg_var_global_XX", clustered)
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

    # Build count_global using max across all effect counts (like CRAN's fifelse loop)
    # CRAN takes max of count{i}_global_XX across all i, with NA treated as 0
    count_exprs <- lapply(1:l_XX, function(i) pl$col(paste0("count", i, "_global_XX"))$fill_null(0))
    df <- df$with_columns(
      do.call(pl$max_horizontal, count_exprs)$alias("count_global_XX")
    )

    # Count observations - CRAN sums all values without first_obs filter
    N_effect_XX <- pl_sum_col(df, "count_global_XX")
    # N_effect_dwXX: CRAN creates indicator (count > 0) and sums all rows, not just first_obs
    # count_global_dwXX = as.numeric(!is.na(count_global_XX) & count_global_XX > 0)
    # N_effect_dwXX = sum(count_global_dwXX)
    N_effect_dwXX <- pl_scalar(df$lazy()$select(
      ((pl$col("count_global_XX")$is_not_null()) & (pl$col("count_global_XX") > 0))$cast(pl$Int32)$sum()
    )$collect())

    if (getOption("DID_DEBUG_COUNT", FALSE)) {
      cat("\n=== DEBUG: Average effect N computation ===\n")
      cat("count_global_XX exists:", pl_has_col(df, "count_global_XX"), "\n")
      if (pl_has_col(df, "count_global_XX")) {
        cat("count_global_XX sample:\n")
        print(head(as.data.frame(df$select(c("count_global_XX", "first_obs_by_gp_XX"))), 20))
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

  #### Computing the placebo estimators using pure polars
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
      sum_plus <- pl_sum_first_obs(df, col_plus)
      sum_minus <- pl_sum_first_obs(df, col_minus)

      if (total_N_pl > 0) {
        DID_pl_i <- (N1_pl_i * sum_plus / G_XX + N0_pl_i * sum_minus / G_XX) / total_N_pl
      } else {
        DID_pl_i <- NA
      }

      # Compute variance
      if (total_N_pl > 0) {
        w_in <- N1_pl_i / total_N_pl
        w_out <- N0_pl_i / total_N_pl

        var_in_expr <- if (pl_has_col(df, col_var_in)) pl$col(col_var_in)$fill_null(0) else pl$lit(0)
        var_out_expr <- if (pl_has_col(df, col_var_out)) pl$col(col_var_out)$fill_null(0) else pl$lit(0)

        df <- df$with_columns(
          (pl$lit(w_in) * var_in_expr + pl$lit(w_out) * var_out_expr)$alias(paste0("U_Gg_var_glob_pl_", i, "_XX"))
        )

        var_sum_pl <- pl_compute_variance_sum(df, paste0("U_Gg_var_glob_pl_", i, "_XX"), clustered)
        SE_pl_i <- sqrt(var_sum_pl) / G_var

        # For clustered SE, update the df column with cluster-summed values for covariance computation
        # This mirrors what CRAN does: replaces U_Gg_var_glob_pl_i_XX with clust_U_Gg_var_glob_pl_i_XX
        if (clustered) {
          pl_col_name <- paste0("U_Gg_var_glob_pl_", i, "_XX")
          dt_pl <- as.data.table(as.data.frame(df$select(c("cluster_XX", "first_obs_by_gp_XX", pl_col_name))))
          dt_pl[, var_weighted := get(pl_col_name) * first_obs_by_gp_XX]
          dt_pl[, clust_var_sum := sum(var_weighted, na.rm = TRUE), by = cluster_XX]

          # Update df with cluster-summed values
          df <- df$with_columns(pl$lit(dt_pl$clust_var_sum)$alias(pl_col_name))
        }
      } else {
        SE_pl_i <- NA
        df <- df$with_columns(pl$lit(0)$alias(paste0("U_Gg_var_glob_pl_", i, "_XX")))
      }

      # Count switchers and effects for placebo
      N_switchers_pl_i <- N1_pl_i + N0_pl_i

      count_plus_expr <- if (pl_has_col(df, col_count_plus)) pl$col(col_count_plus) else pl$lit(NA_real_)
      count_minus_expr <- if (pl_has_col(df, col_count_minus)) pl$col(col_count_minus) else pl$lit(NA_real_)
      df <- df$with_columns(
        pl$coalesce(count_plus_expr, count_minus_expr)$alias(paste0("count", i, "_pl_global_XX"))
      )
      # CRAN sums all values of count column for N, not just first_obs filtered
      N_effects_pl_i <- pl_sum_col(df, paste0("count", i, "_pl_global_XX"))

      if (getOption("DID_DEBUG_COUNT", FALSE) && i == 1) {
        cat("\n=== DEBUG: Placebo 1 N computation ===\n")
        pl_col <- paste0("count", i, "_pl_global_XX")
        cat("Column", pl_col, "exists:", pl_has_col(df, pl_col), "\n")
        if (pl_has_col(df, pl_col)) {
          cat("Sample values:\n")
          print(head(as.data.frame(df$select(c(pl_col, "first_obs_by_gp_XX"))), 20))
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

      # Count dw for placebo - CRAN counts all rows where count is non-null and > 0
      # Not just first_obs groups, but ALL rows
      pl_col_name <- paste0("count", i, "_pl_global_XX")
      count_dw_pl <- pl_scalar(df$lazy()$select(
        ((pl$col(pl_col_name)$is_not_null()) & (pl$col(pl_col_name) > 0))$cast(pl$Int32)$sum()
      )$collect())
      if (is.null(count_dw_pl) || is.na(count_dw_pl)) count_dw_pl <- 0

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

  # Convert df from polars to data.table for section 6 (p-values and joint tests)
  df <- data.table::as.data.table(as.data.frame(df))

  ## Average number of cumulated effects
  for (i in 1:l_XX) {
    df[[paste0("delta_D_g_",i,"_XX")]] <- NULL
  }
  df[, M_g_XX := ifelse(l_XX <= T_g_XX - F_g_XX + 1, as.numeric(l_XX), T_g_XX - F_g_XX + 1)]

  #### Calling variables delta_D_g_`i'_XX here like that does not work because switcher in/out are run one after another!!!

  ## second sum over g: total ... if F_g_XX<=T_g_XX
  ## actually I think it can be done in one total as we sum over the periods within groups and then across groups which are all different cells
  ## generate one variable that stores all the different delta_D_g_`i'_XX

  df[, delta_D_g_XX := 0]
  for (j in 1:l_XX) {
    col_plus <- paste0("delta_D_g_",j,"_plus_XX")
    col_minus <- paste0("delta_D_g_",j,"_minus_XX")
    df[, delta_D_g_XX_temp := ifelse(get(col_plus) != 0, get(col_plus), get(col_minus))]
    df[, delta_D_g_XX_temp := ifelse(delta_D_g_XX_temp == 0, NA_real_, delta_D_g_XX_temp)]
    df[, delta_D_g_XX := ifelse(switchers_tag_XX == j, delta_D_g_XX + delta_D_g_XX_temp, delta_D_g_XX)]
  }
  df$delta_D_g_num_XX <- df$delta_D_g_XX * (df$M_g_XX - (df$switchers_tag_XX - 1))
  delta_D_num_total <- sum(df$delta_D_g_num_XX, na.rm = TRUE)
  delta_D_denom_total <- sum(df$delta_D_g_XX, na.rm = TRUE)
  delta_D_avg_total <- delta_D_num_total / delta_D_denom_total
  ###### 6. Computing p-values from the tests

  # If the option cluster is specified, we have previously replaced U_Gg_var_glob_pl_`i'_XX by clust_U_Gg_var_glob_pl_`i'_XX, and U_Gg_var_glob_`i'_XX by clust_U_Gg_var_glob_`i'_XX.
  # Now, we must also replace first_obs_by_gp_XX by first_obs_by_clust_XX
  if (!is.null(cluster)) {
    df$first_obs_by_gp_XX <- df$first_obs_by_clust_XX
  }

  ###### Performing a test to see whether all effects are jointly equal to 0
  all_Ns_not_zero <- NA
  all_delta_not_zero <- NA
  p_jointeffects <- NULL
  ## Test can only be run when at least two effects requested:
  if (l_XX != 0 & l_XX > 1) {
    ## If test is feasible, initalize scalar at 0
    all_Ns_not_zero <- 0
    all_delta_not_zero <- 0

    ## Count the number of estimated effects included in the test
    for (i in 1:l_XX) {
      if ( (switchers == "" & (get(paste0("N1_",i,"_XX_new"))!= 0 | get(paste0("N0_",i,"_XX_new"))!= 0 )) | (switchers == "out" & get(paste0("N0_",i,"_XX_new")) != 0) | (switchers == "in" & get(paste0("N1_",i,"_XX_new")) != 0) ) {
        all_Ns_not_zero <- all_Ns_not_zero + 1
      }

      if (isTRUE(normalized)) {
        if (get(paste0("delta_D_",i,"_global_XX")) != 0 & !is.na(get(paste0("delta_D_",i,"_global_XX")))) {
          all_delta_not_zero <- all_delta_not_zero + 1
        }
      }
    }

    ## Test can only be run when all requested effects could be computed:
    if ((all_Ns_not_zero == l_XX & isFALSE(normalized)) | (isTRUE(normalized) & all_Ns_not_zero == l_XX & all_delta_not_zero == l_XX)) {

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
            df[[paste0("U_Gg_var_",i,"_",j,"_2_XX")]] <- df[[paste0("U_Gg_var_",i,"_",j,"_XX")]]^2 * df$first_obs_by_gp_XX
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
        p_jointeffects <- 1 - pchisq(didmgt_chi2effects[1,1], df = l_XX)
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
  if (l_placebo_XX != 0 & l_placebo_XX > 1) {
    ## If test is feasible, initalize scalar at 0
    all_Ns_pl_not_zero <- 0
    all_delta_pl_not_zero <- 0

    ## Count the number of estimated placebos included in the test
    for (i in 1:l_placebo_XX) {
      if ( (switchers == "" & (get(paste0("N1_placebo_",i,"_XX_new"))!= 0 | get(paste0("N0_placebo_",i,"_XX_new"))!= 0 )) | (switchers == "out" & get(paste0("N0_placebo_",i,"_XX_new")) != 0) | (switchers == "in" & get(paste0("N1_placebo_",i,"_XX_new")) != 0) ) {
        all_Ns_pl_not_zero <- all_Ns_pl_not_zero + 1
      }

      if (isTRUE(normalized)) {
        if (get(paste0("delta_D_pl_",i,"_global_XX")) != 0 & !is.na(get(paste0("delta_D_pl_",i,"_global_XX")))) {
          all_delta_pl_not_zero <- all_delta_pl_not_zero + 1
        }
      }
    }

    ## Test can only be run when all requested placebos could be computed:
    if ((all_Ns_pl_not_zero == l_placebo_XX & isFALSE(normalized)) | (isTRUE(normalized) & all_Ns_pl_not_zero == l_placebo_XX & all_delta_pl_not_zero == l_placebo_XX)) {

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
            df[[paste0("U_Gg_var_pl_",i,"_",j,"_2_XX")]] <- df[[paste0("U_Gg_var_pl_",i,"_",j,"_XX")]]^2 * df$first_obs_by_gp_XX
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
        p_jointplacebo <- 1 - pchisq(didmgt_chi2placebo[1,1], df = l_placebo_XX)
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
    if (length(predict_het_good) > 0) {
      if (-1 %in% het_effects) {
        het_effects <- 1:l_XX
      }
      all_effects_XX <- c(1:l_XX)[het_effects]
      if (NA %in% all_effects_XX) {
        ## error if specified effects not matching with those actually calculated
        stop("Error in predict_het second argument: please specify only numbers that are smaller or equal to the number you request in effects()")
      }

      # Preliminaries: Yg Fg1
      df[, Yg_Fg_min1_XX := ifelse(time_XX == F_g_XX - 1, outcome_non_diff_XX, NA_real_)]
      df[, Yg_Fg_min1_XX := mean(Yg_Fg_min1_XX, na.rm = TRUE), by = group_XX]
      df$feasible_het_XX <- !is.na(df$Yg_Fg_min1_XX)
      if (!is.null(trends_lin)) {
        df[, Yg_Fg_min2_XX := ifelse(time_XX == F_g_XX - 2, outcome_non_diff_XX, NA_real_)]
        df[, Yg_Fg_min2_XX := mean(Yg_Fg_min2_XX, na.rm = TRUE), by = group_XX]
        df[, Yg_Fg_min2_XX := ifelse(is.nan(Yg_Fg_min2_XX), NA_real_, Yg_Fg_min2_XX)]

        df$feasible_het_XX <- df$feasible_het_XX & !is.na(df$Yg_Fg_min2_XX)
      }
      data.table::setorder(df, group_XX, time_XX)
      df[, gr_id := seq_len(.N), by = group_XX]

      lhyp <- c()
      for (v in predict_het_good) {
        lhyp <- c(lhyp, paste0(v, "=0"))
      }

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
        df[, paste0("Yg_Fg_", i, "_XX") := ifelse(time_XX == F_g_XX - 1 + i, outcome_non_diff_XX, NA_real_)]
        df[, paste0("Yg_Fg_",i,"_XX") := mean(get(paste0("Yg_Fg_",i,"_XX")), na.rm = TRUE), by = group_XX]

        df$diff_het_XX <- df[[paste0("Yg_Fg_",i,"_XX")]] - df$Yg_Fg_min1_XX
        if (isTRUE(trends_lin)) {
          df$diff_het_XX <- df$diff_het_XX - i * (df$Yg_Fg_min1_XX - df$Yg_Fg_min2_XX)
        }

        df[[paste0("prod_het_",i,"_XX")]] <- df$S_g_het_XX * df$diff_het_XX
        df$diff_het_XX <- NULL

        # keep one observation by group to not artificially increase sample
        col_prod <- paste0("prod_het_",i,"_XX")
        df[, (col_prod) := ifelse(gr_id == 1, get(col_prod), NA_real_)]

        # In order to perform the test with coeftest, we need a vector of non missing regression coefficients. To avoid collinearity, we run the regression two times: the first time with the full set of regressors (F_g_XX_h#d_sq_XX_h#S_g_XX_h), then with just the non-collinear variables.
        het_reg <- paste0("prod_het_",i,"_XX ~ ")
        for (v in predict_het_good) {
          het_reg <- paste0(het_reg,v," + ")
        }
        het_reg <- paste0(het_reg, het_interact)
        het_sample <- df[F_g_XX - 1 + i <= T_g_XX]
        model <- lm(as.formula(het_reg), data = het_sample, weights = het_sample$weight_XX)
        het_reg <- gsub(het_interact, "", het_reg)
        for (k in names(model$coefficients)) {
          if (!(k %in% c("(Intercept)", predict_het_good))) {
            if (!is.na(model$coefficients[[k]])) {
              het_reg <- paste0(het_reg, " + ", k)
            }
          }
        }
        model <- lm(as.formula(het_reg), data = het_sample, weights = het_sample$weight_XX)
        ## Use HC2 with dfadjust when predict_het_hc2bm is TRUE
        if (isTRUE(predict_het_hc2bm)) {
          ## Determine cluster variable for HC2 BM
          if (!is.null(cluster)) {
            cluster_het <- het_sample[[cluster]]
          } else {
            cluster_het <- het_sample$group_XX
          }
          het_vcov <- vcovCL(model, cluster = cluster_het, type = "HC2", cadjust = TRUE)
        } else {
          het_vcov <- vcovHC(model, type = "HC2")
        }
        model_r <- matrix(coeftest(model, vcov. = het_vcov)[2:(length(predict_het_good)+1), 1:3], ncol = 3)
        f_stat <- linearHypothesis(model, lhyp, vcov = het_vcov)[["Pr(>F)"]][2]
        t_stat <- qt(0.975, df.residual(model))
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
          N = matrix(nobs(model), nrow = length(predict_het_good)),
          pF = matrix(f_stat, nrow = length(predict_het_good))
        ))
      }
      for (v in c("F_g_XX", "d_sq_XX", "S_g_XX", trends_nonparam)) {
        df[[paste0(v,"_h")]] <- NULL
      }
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
        df[, paste0("Yg_Fg_pl_", i, "_XX") := ifelse(time_XX == F_g_XX - 1 - i, outcome_non_diff_XX, NA_real_)]
        df[, paste0("Yg_Fg_pl_",i,"_XX") := mean(get(paste0("Yg_Fg_pl_",i,"_XX")), na.rm = TRUE), by = group_XX]

        df$diff_het_pl_XX <- df[[paste0("Yg_Fg_pl_",i,"_XX")]] - df$Yg_Fg_min1_XX
        if (isTRUE(trends_lin)) {
          df$diff_het_pl_XX <- df$diff_het_pl_XX - i * (df$Yg_Fg_min1_XX - df$Yg_Fg_min2_XX)
        }

        # Now we can generate
        df[[paste0("prod_het_pl_",i,"_XX")]] <- df$S_g_het_XX * df$diff_het_pl_XX
        df$diff_het_pl_XX <- NULL

        # keep one observation by group to not artificially increase sample
        col_prod_pl <- paste0("prod_het_pl_",i,"_XX")
        df[, (col_prod_pl) := ifelse(gr_id == 1, get(col_prod_pl), NA_real_)]

        # In order to perform the test with coeftest, we need a vector of non missing regression coefficients. To avoid collinearity, we run the regression two times: the first time with the full set of regressors (F_g_XX_h#d_sq_XX_h#S_g_XX_h), then with just the non-collinear variables.
        het_reg <- paste0("prod_het_pl_",i,"_XX ~ ")
        for (v in predict_het_good) {
          het_reg <- paste0(het_reg,v," + ")
        }
        het_reg <- paste0(het_reg, het_interact)
        het_sample <- df[F_g_XX - 1 + i <= T_g_XX]
        model <- lm(as.formula(het_reg), data = het_sample, weights = het_sample$weight_XX)
        het_reg <- gsub(het_interact, "", het_reg)
        for (k in names(model$coefficients)) {
          if (!(k %in% c("(Intercept)", predict_het_good))) {
            if (!is.na(model$coefficients[[k]])) {
              het_reg <- paste0(het_reg, " + ", k)
            }
          }
        }
        model <- lm(as.formula(het_reg), data = het_sample, weights = het_sample$weight_XX)
        ## Use HC2 with dfadjust when predict_het_hc2bm is TRUE (placebo section)
        if (isTRUE(predict_het_hc2bm)) {
          ## Determine cluster variable for HC2 BM
          if (!is.null(cluster)) {
            cluster_het <- het_sample[[cluster]]
          } else {
            cluster_het <- het_sample$group_XX
          }
          het_vcov <- vcovCL(model, cluster = cluster_het, type = "HC2", cadjust = TRUE)
        } else {
          het_vcov <- vcovHC(model, type = "HC2")
        }
        model_r <- matrix(coeftest(model, vcov. = het_vcov)[2:(length(predict_het_good)+1), 1:3], ncol = 3)
        f_stat <- linearHypothesis(model, lhyp, vcov = het_vcov)[["Pr(>F)"]][2]
        t_stat <- qt(0.975, df.residual(model))
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
          N = matrix(nobs(model), nrow = length(predict_het_good)),
          pF = matrix(f_stat, nrow = length(predict_het_good))
        ))
      }
      for (v in c("F_g_XX", "d_sq_XX", "S_g_XX", trends_nonparam)) {
        df[[paste0(v,"_h")]] <- NULL
      }
      het_res <- het_res[order(het_res$covariate, het_res$effect), ]
    }
  }

  ###### Performing a test that all DID_\ell effects are equal (similar structure as test on placebos, not commented, except for the small differences with placebos)
  if (effects_equal == TRUE & l_XX > 1) {
    ## Determine bounds for the test (default is all effects)
    ee_lb <- ifelse(is.null(effects_equal_lb), 1, effects_equal_lb)
    ee_ub <- ifelse(is.null(effects_equal_ub), l_XX, effects_equal_ub)

    ## Validate bounds
    if (ee_ub > l_XX) {
      message(sprintf("Upper bound %d exceeds number of effects %d. Using %d as upper bound.", ee_ub, l_XX, l_XX))
      ee_ub <- l_XX
    }

    ee_length <- ee_ub - ee_lb + 1

    all_Ns_not_zero <- 0
    for (i in ee_lb:ee_ub) {
      if (((switchers == "" & (get(paste0("N1_",i,"_XX_new")) != 0 | get(paste0("N0_",i,"_XX_new")) != 0)) |
          (switchers == "out" & get(paste0("N0_",i,"_XX_new")) != 0) |
          (switchers == "in" & get(paste0("N1_",i,"_XX_new")) != 0))) {
        all_Ns_not_zero <- all_Ns_not_zero + 1
      }
    }
    if (all_Ns_not_zero == ee_length) {
      didmgt_Effects <- mat_res_XX[ee_lb:ee_ub, 1]
      didmgt_Var_Effects <- matrix(0, nrow = ee_length, ncol = ee_length)
      didmgt_identity <- matrix(0, nrow = ee_length - 1, ncol = ee_length)

      for (i in ee_lb:ee_ub) {
        ## Index in the submatrix (1-based within range)
        ii <- i - ee_lb + 1
        if (((switchers == "" & (get(paste0("N1_",i,"_XX_new")) != 0 | get(paste0("N0_",i,"_XX_new")) != 0)) |
            (switchers == "out" & get(paste0("N0_",i,"_XX_new")) != 0) |
            (switchers == "in" & get(paste0("N1_",i,"_XX_new")) != 0))) {

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

              df[[paste0("U_Gg_var_", i, "_", j, "_2_XX")]] <- df[[paste0("U_Gg_var_", i, "_", j, "_XX")]]^2 * df$first_obs_by_gp_XX
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
        if (ee_lb == 1 & ee_ub == l_XX) {
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
          if (ee_lb == 1 & ee_ub == l_XX) {
            warn_msg <- "The F-test that all effects are equal may not be reliable, because the variance of the effects is close to not being invertible (the ratio of its largest and smallest eigenvalues is larger than 1000). This may be due to strong multicollinearity among the effects. Consider reducing the number of effects estimated."
          } else {
            warn_msg <- sprintf("The F-test that effects %d to %d are equal may not be reliable, because the variance of the effects is close to not being invertible (the ratio of its largest and smallest eigenvalues is larger than 1000).", ee_lb, ee_ub)
          }
          vcov_warnings <- c(vcov_warnings, warn_msg)
          warning(warn_msg)
        }
        didmgt_chi2_equal_ef <- t(didmgt_test_effects) %*% MASS::ginv(didmgt_test_var) %*% didmgt_test_effects
        p_equality_effects <-
          1 - pchisq(didmgt_chi2_equal_ef[1,1], df = ee_length - 1)
        assign("p_equality_effects", p_equality_effects, inherits = TRUE)
      }
    } else {
      if (ee_lb == 1 & ee_ub == l_XX) {
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
  for (i in 1:l_XX) {
    if (isFALSE(normalized)) {
      df[[paste0("U_Gg_var_comb_",i,"_XX")]] <- ifelse(is.null(df[[paste0("U_Gg_var_glob_",i,"_XX")]]), NA, df[[paste0("U_Gg_var_glob_",i,"_XX")]])
    } else {
      df[[paste0("U_Gg_var_comb_",i,"_XX")]] <-  ifelse(is.null(df[[paste0("U_Gg_var_glob_",i,"_XX")]]), NA, df[[paste0("U_Gg_var_glob_",i,"_XX")]]/ get(paste0("delta_D_",i,"_global_XX")))
    }
  }
  if (l_placebo_XX != 0) {
    for (i in 1:l_placebo_XX) {
      if (isFALSE(normalized)) {
        df[[paste0("U_Gg_var_comb_",l_XX + i,"_XX")]] <- ifelse(is.null(df[[paste0("U_Gg_var_glob_pl_",i,"_XX")]]), NA, df[[paste0("U_Gg_var_glob_pl_",i,"_XX")]])
      } else {
        df[[paste0("U_Gg_var_comb_",l_XX + i,"_XX")]] <- ifelse(is.null(df[[paste0("U_Gg_var_glob_pl_",i,"_XX")]]), NA, df[[paste0("U_Gg_var_glob_pl_",i,"_XX")]]/get(paste0("delta_D_pl_",i,"_global_XX")))
      }
    }
  }

  for (i in 1:l_tot_XX) {
    didmgt_vcov[i,i] <- mat_res_XX[i + (i>l_XX),2]^2
    j <- 1
    while (j < i) {
      df[[paste0("U_Gg_var_comb_",i,"_",j,"_2_XX")]] <- (df[[paste0("U_Gg_var_comb_",i,"_XX")]] + df[[paste0("U_Gg_var_comb_",j,"_XX")]])^2 * df$first_obs_by_gp_XX
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
    write.csv(mat_res_XX, save_results, row.names = TRUE, col.names = TRUE)
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
    if (length(predict_het_good) > 0) {
      did_multiplegt_dyn <- append(did_multiplegt_dyn, list(het_res))
      out_names <- c(out_names, "predict_het")
    }
  }

  # Add vcov warnings if any were collected
  if (length(vcov_warnings) > 0) {
    did_multiplegt_dyn <- append(did_multiplegt_dyn, list(vcov_warnings))
    out_names <- c(out_names, "vcov_warnings")
  }

  # Uncomment for debugging #
  #did_multiplegt_dyn <- append(did_multiplegt_dyn, list(df))
  #out_names <- c(out_names, "debug")

  names(did_multiplegt_dyn) <- out_names

  delta <- list()
  if (isTRUE(normalized)) {
    for (i in 1:l_XX) {
      delta[[paste0("delta_D_",i,"_global_XX")]] <-
        get(paste0("delta_D_", i, "_global_XX"))
    }
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

  names(ret) <- ret_names
  ret
  })
  }
