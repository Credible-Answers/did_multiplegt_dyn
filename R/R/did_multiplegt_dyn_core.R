# Global option to enable Rcpp optimization
# Set options(DID_USE_RCPP = TRUE) to enable Rcpp-optimized computations
.DID_USE_RCPP <- function() {
  getOption("DID_USE_RCPP", default = FALSE)
}

#' Internal function of did_multiplegt_dyn that computes U_Gg_plus_XX, U_Gg_minus_XX, U_Gg_var_plus_XX, and U_Gg_var_minus_XX.
#' These are essential variables for the computation of the DID_\ell estimators and their variances.
#' POLARS-OPTIMIZED VERSION - No data.table or data.frame operations
#' @param df polars DataFrame
#' @param outcome outcome
#' @param group group
#' @param time time
#' @param treatment treatment
#' @param effects effects
#' @param placebo placebo
#' @param cluster cluster
#' @param switchers_core switchers_core
#' @param trends_nonparam trends_nonparam
#' @param controls controls
#' @param same_switchers same_switchers
#' @param same_switchers_pl same_switchers_pl
#' @param only_never_switchers only_never_switchers
#' @param normalized normalized
#' @param globals globals
#' @param const constants
#' @param trends_lin trends_lin
#' @param controls_globals controls_globals
#' @param less_conservative_se less_conservative_se
#' @param continuous continuous
#' @note polars is suggested for better performance
#' @importFrom stats na.omit predict setNames
#' @importFrom MASS ginv
#' @importFrom fixest feols
#' @returns A list containing df (polars DataFrame) and const (constants).
#' @noRd
did_multiplegt_dyn_core <- function(
    df,
    outcome,
    group,
    time,
    treatment,
    effects,
    placebo,
    cluster,
    switchers_core = NULL,
    trends_nonparam,
    controls,
    same_switchers,
    same_switchers_pl,
    only_never_switchers,
    normalized,
    globals,
    const,
    trends_lin,
    controls_globals,
    less_conservative_se,
    continuous
) {

  # Inherited Globals
  L_u_XX <- globals$L_u_XX
  L_placebo_u_XX <- globals$L_placebo_u_XX
  L_placebo_a_XX <- globals$L_placebo_a_XX
  L_a_XX <- globals$L_a_XX
  t_min_XX <- globals$t_min_XX
  T_max_XX <- globals$T_max_XX
  G_XX <- globals$G_XX

  for (e in names(const)) {
    assign(e, const[[e]])
  }

  if (!is.null(controls)) {
    for (e in names(controls_globals)) {
      assign(e, controls_globals[[e]])
    }
  }

  # Get pl from polars namespace (polars availability already checked by caller)
  pl <- .get_pl()

  # Ensure df is a polars DataFrame
  is_polars <- inherits(df, "polars_data_frame") || inherits(df, "RPolarsDataFrame")
  if (!is_polars) {
    if (inherits(df, "data.table") || inherits(df, "data.frame")) {
      df <- .as_polars_df(as.data.frame(df))
    } else {
      stop("df must be a polars DataFrame, data.table, or data.frame")
    }
  }

  suppressWarnings({

    ####### 1. Scalars initialization
    if (switchers_core == "in") {
      l_u_a_XX <- min(L_u_XX, effects, na.rm = TRUE)
      if (placebo != 0) {
        l_placebo_u_a_XX <- min(placebo, L_placebo_u_XX)
      }
      increase_XX <- 1
    }

    if (switchers_core == "out") {
      l_u_a_XX <- min(L_a_XX, effects, na.rm = TRUE)
      if (placebo != 0) {
        l_placebo_u_a_XX <- min(placebo, L_placebo_a_XX)
      }
      increase_XX <- 0
    }

    # Get levels of baseline treatment
    levels_d_sq_XX <- pl_factor_levels(df, "d_sq_int_XX")

    # Drop columns if they exist
    cols_to_drop <- c("num_g_paths_0_XX", "cohort_fullpath_0_XX")
    df <- pl_batch_drop_cols(df, cols_to_drop)

    # Sort by group and time once (polars is efficient with sorted data)
    df <- df$sort(c("group_XX", "time_XX"))

    ####### 2. Data preparation - Loop over effects
    for (i in 1:l_u_a_XX) {

      # Build list of columns to drop for this iteration
      cols_to_drop_i <- c(
        paste0("distance_to_switch_", i, "_XX"),
        paste0("never_change_d_", i, "_XX"),
        paste0("N", increase_XX, "_t_", i, "_XX"),
        paste0("N", increase_XX, "_t_", i, "_g_XX"),
        paste0("N_gt_control_", i, "_XX"),
        paste0("diff_y_", i, "_XX"),
        paste0("diff_y_", i, "_XX_temp"),
        paste0("dummy_U_Gg", i, "_XX"),
        paste0("U_Gg", i, "_temp_XX"),
        paste0("U_Gg", i, "_XX"),
        paste0("count", i, "_core_XX"),
        paste0("mean_diff_y_", i, "_nd_sq_t_XX"),
        paste0("mean_diff_y_", i, "_d_sq_t_XX"),
        paste0("U_Gg", i, "_temp_var_XX"),
        paste0("U_Gg", i, "_var_XX"),
        paste0("U_Gg", i, "_var_2_XX"),
        paste0("count_var_", i, "_ntreat_XX_temp"),
        paste0("count_var_", i, "_ntreat_XX"),
        paste0("count_var_", i, "_treat_XX_temp"),
        paste0("count_var_", i, "_treat_XX"),
        paste0("avg_diff_y_", i, "_tnp_XX"),
        paste0("count_diff_y_", i, "_nd_sq_t_XX"),
        paste0("count_diff_y_", i, "_d_sq_t_XX"),
        paste0("never_change_d_", i, "_wXX"),
        paste0("distance_to_switch_", i, "_wXX"),
        paste0("dof_cohort_", i, "_ns_t_XX"),
        paste0("dof_cohort_", i, "_s_t_XX"),
        paste0("dof_cohort_", i, "_s0_t_XX"),
        paste0("dof_cohort_", i, "_s1_t_XX"),
        paste0("dof_cohort_", i, "_s2_t_XX"),
        paste0("count_cohort_", i, "_ns_t_XX"),
        paste0("count_cohort_", i, "_s_t_XX"),
        paste0("count_cohort_", i, "_s0_t_XX"),
        paste0("count_cohort_", i, "_s1_t_XX"),
        paste0("count_cohort_", i, "_s2_t_XX"),
        paste0("total_cohort_", i, "_ns_t_XX"),
        paste0("total_cohort_", i, "_s_t_XX"),
        paste0("total_cohort_", i, "_s0_t_XX"),
        paste0("total_cohort_", i, "_s1_t_XX"),
        paste0("total_cohort_", i, "_s2_t_XX"),
        paste0("mean_cohort_", i, "_ns_t_XX"),
        paste0("mean_cohort_", i, "_s_t_XX"),
        paste0("mean_cohort_", i, "_s0_t_XX"),
        paste0("mean_cohort_", i, "_s1_t_XX"),
        paste0("mean_cohort_", i, "_s2_t_XX")
      )
      df <- pl_batch_drop_cols(df, cols_to_drop_i)

      # Create long difference of outcome using polars window function
      diff_y_col <- paste0("diff_y_", i, "_XX")
      df <- df$with_columns(
        (pl$col("outcome_XX") - pl$col("outcome_XX")$shift(i)$over("group_XX"))$alias(diff_y_col)
      )

      # Creating treatment paths if less_conservative_se option specified
      if (isTRUE(less_conservative_se)) {
        # d_fg_XX_temp: treatment at F_g-1+i
        df <- df$with_columns(
          pl$when(pl$col("time_XX") == pl$col("F_g_XX") + i - 1)$
            then(pl$col("treatment_XX"))$
            otherwise(pl$lit(NA_real_))$
            alias("d_fg_XX_temp")
        )

        # Group mean of d_fg_XX_temp
        d_fg_col <- paste0("d_fg", i, "_XX")
        df <- pl_mean_over(df, "d_fg_XX_temp", "group_XX", d_fg_col)

        if (i == 1) {
          df <- df$with_columns(pl$col("d_sq_XX")$alias("d_fg0_XX"))
          df <- pl_grp_id(df, c("d_fg0_XX", "F_g_XX"), "path_0_XX")
        }

        # Fill missing d_fg with previous value
        prev_d_fg <- paste0("d_fg", i - 1, "_XX")
        df <- df$with_columns(
          pl$when(pl$col(d_fg_col)$is_null())$
            then(pl$col(prev_d_fg))$
            otherwise(pl$col(d_fg_col))$
            alias(d_fg_col)
        )

        # Create path variable
        prev_path <- paste0("path_", i - 1, "_XX")
        path_col <- paste0("path_", i, "_XX")
        df <- pl_grp_id(df, c(prev_path, d_fg_col), path_col)

        df <- pl_batch_drop_cols(df, "d_fg_XX_temp")

        # Count groups per path
        if (i == 1) {
          df <- pl_uniqueN_over(df, "group_XX", "path_0_XX", "num_g_paths_0_XX")
          df <- df$with_columns(
            (pl$col("num_g_paths_0_XX") > 1)$cast(pl$Float64)$alias("cohort_fullpath_0_XX")
          )
        }

        num_g_col <- paste0("num_g_paths_", i, "_XX")
        cohort_col <- paste0("cohort_fullpath_", i, "_XX")
        df <- pl_uniqueN_over(df, "group_XX", path_col, num_g_col)
        df <- df$with_columns(
          (pl$col(num_g_col) > 1)$cast(pl$Float64)$alias(cohort_col)
        )
      }

      # Identifying control (g,t)s
      never_col <- paste0("never_change_d_", i, "_XX")
      df <- df$with_columns(
        (pl$col("F_g_XX") > pl$col("time_XX"))$cast(pl$Float64)$alias(never_col)
      )

      # Set to NA where diff_y is NA
      df <- df$with_columns(
        pl$when(pl$col(diff_y_col)$is_null())$
          then(pl$lit(NA_real_))$
          otherwise(pl$col(never_col))$
          alias(never_col)
      )

      if (isTRUE(only_never_switchers)) {
        df <- df$with_columns(
          pl$when(
            (pl$col("F_g_XX") > pl$col("time_XX")) &
            (pl$col("F_g_XX") < T_max_XX + 1) &
            pl$col(diff_y_col)$is_not_null()
          )$then(pl$lit(0.0))$otherwise(pl$col(never_col))$alias(never_col)
        )
      }

      # Creating N_gt_control: weighted sum of never_change
      never_w_col <- paste0("never_change_d_", i, "_wXX")
      df <- df$with_columns(
        (pl$col(never_col) * pl$col("N_gt_XX"))$alias(never_w_col)
      )

      by_cols <- c("time_XX", "d_sq_XX")
      if (!is.null(trends_nonparam) && length(trends_nonparam) > 0) {
        by_cols <- c(by_cols, trends_nonparam)
      }

      N_gt_ctrl_col <- paste0("N_gt_control_", i, "_XX")
      df <- pl_sum_over(df, never_w_col, by_cols, N_gt_ctrl_col)

      # Same switchers logic
      if (same_switchers == TRUE) {
        df <- df$with_columns(pl$lit(0.0)$alias("N_g_control_check_XX"))

        for (q in 1:effects) {
          # Compute diff_y_last
          df <- df$with_columns(
            (pl$col("outcome_XX") - pl$col("outcome_XX")$shift(q)$over("group_XX"))$alias("diff_y_last_XX")
          )

          # never_change_d_last
          df <- df$with_columns(
            pl$when(
              pl$col("diff_y_last_XX")$is_not_null() & (pl$col("F_g_XX") > pl$col("time_XX"))
            )$then(pl$lit(1.0))$otherwise(pl$lit(NA_real_))$alias("never_change_d_last_XX")
          )

          if (isTRUE(only_never_switchers)) {
            df <- df$with_columns(
              pl$when(
                (pl$col("F_g_XX") > pl$col("time_XX")) &
                (pl$col("F_g_XX") < T_max_XX + 1) &
                pl$col("diff_y_last_XX")$is_not_null()
              )$then(pl$lit(0.0))$otherwise(pl$col("never_change_d_last_XX"))$alias("never_change_d_last_XX")
            )
          }

          # N_gt_control_last
          df <- df$with_columns(
            (pl$col("never_change_d_last_XX") * pl$col("N_gt_XX"))$alias("__temp_weighted__")
          )
          df <- pl_sum_over(df, "__temp_weighted__", by_cols, "N_gt_control_last_XX")
          df <- pl_batch_drop_cols(df, "__temp_weighted__")

          # N_g_control_last_m: mean at F_g - 1 + q
          df <- df$with_columns(
            pl$when(pl$col("time_XX") == pl$col("F_g_XX") - 1 + q)$
              then(pl$col("N_gt_control_last_XX"))$
              otherwise(pl$lit(NA_real_))$
              alias("__temp_ctrl__")
          )
          df <- pl_mean_over(df, "__temp_ctrl__", "group_XX", "N_g_control_last_m_XX")
          df <- pl_batch_drop_cols(df, "__temp_ctrl__")

          # diff_y_relev
          df <- df$with_columns(
            pl$when(pl$col("time_XX") == pl$col("F_g_XX") - 1 + q)$
              then(pl$col("diff_y_last_XX"))$
              otherwise(pl$lit(NA_real_))$
              alias("__temp_diff__")
          )
          df <- pl_mean_over(df, "__temp_diff__", "group_XX", "diff_y_relev_XX")
          df <- pl_batch_drop_cols(df, "__temp_diff__")

          # Update N_g_control_check
          df <- df$with_columns(
            (pl$col("N_g_control_check_XX") +
              ((pl$col("N_g_control_last_m_XX") > 0) & pl$col("diff_y_relev_XX")$is_not_null())$cast(pl$Float64)
            )$alias("N_g_control_check_XX")
          )
        }

        # same_switchers_pl logic
        if (same_switchers_pl == TRUE) {
          df <- df$with_columns(pl$lit(0.0)$alias("N_g_control_check_pl_XX"))

          for (q in 1:placebo) {
            df <- df$with_columns(
              (pl$col("outcome_XX") - pl$col("outcome_XX")$shift(-q)$over("group_XX"))$alias("diff_y_last_XX")
            )

            df <- df$with_columns(
              pl$when(
                pl$col("diff_y_last_XX")$is_not_null() & (pl$col("F_g_XX") > pl$col("time_XX"))
              )$then(pl$lit(1.0))$otherwise(pl$lit(NA_real_))$alias("never_change_d_last_XX")
            )

            if (isTRUE(only_never_switchers)) {
              df <- df$with_columns(
                pl$when(
                  (pl$col("F_g_XX") > pl$col("time_XX")) &
                  (pl$col("F_g_XX") < T_max_XX + 1) &
                  pl$col("diff_y_last_XX")$is_not_null()
                )$then(pl$lit(0.0))$otherwise(pl$col("never_change_d_last_XX"))$alias("never_change_d_last_XX")
              )
            }

            df <- df$with_columns(
              (pl$col("never_change_d_last_XX") * pl$col("N_gt_XX"))$alias("__temp_weighted__")
            )
            df <- pl_sum_over(df, "__temp_weighted__", by_cols, "N_gt_control_last_XX")
            df <- pl_batch_drop_cols(df, "__temp_weighted__")

            df <- df$with_columns(
              pl$when(pl$col("time_XX") == pl$col("F_g_XX") - 1 - q)$
                then(pl$col("N_gt_control_last_XX"))$
                otherwise(pl$lit(NA_real_))$
                alias("__temp_ctrl__")
            )
            df <- pl_mean_over(df, "__temp_ctrl__", "group_XX", "N_g_control_last_m_XX")
            df <- pl_batch_drop_cols(df, "__temp_ctrl__")

            df <- df$with_columns(
              pl$when(pl$col("time_XX") == pl$col("F_g_XX") - 1 - q)$
                then(pl$col("diff_y_last_XX"))$
                otherwise(pl$lit(NA_real_))$
                alias("__temp_diff__")
            )
            df <- pl_mean_over(df, "__temp_diff__", "group_XX", "diff_y_relev_XX")
            df <- pl_batch_drop_cols(df, "__temp_diff__")

            df <- df$with_columns(
              (pl$col("N_g_control_check_pl_XX") +
                ((pl$col("N_g_control_last_m_XX") > 0) & pl$col("diff_y_relev_XX")$is_not_null())$cast(pl$Float64)
              )$alias("N_g_control_check_pl_XX")
            )
          }

          df <- df$with_columns(
            (pl$col("N_g_control_check_pl_XX") == placebo)$alias("fillin_g_pl_XX")
          )

          # still_switcher column
          still_sw_col <- paste0("still_switcher_", i, "_XX")
          df <- df$with_columns(
            ((pl$col("F_g_XX") - 1 + effects <= pl$col("T_g_XX")) &
              (pl$col("N_g_control_check_XX") == effects))$alias(still_sw_col)
          )

          # distance_to_switch
          dist_col <- paste0("distance_to_switch_", i, "_XX")
          df <- df$with_columns(
            pl$when(pl$col(diff_y_col)$is_not_null())$
              then(
                (pl$col(still_sw_col) == TRUE) &
                (pl$col("time_XX") == pl$col("F_g_XX") - 1 + i) &
                (pl$lit(i) <= pl$col("L_g_XX")) &
                (pl$col("S_g_XX") == increase_XX) &
                (pl$col(N_gt_ctrl_col) > 0) &
                pl$col(N_gt_ctrl_col)$is_not_null()
              )$otherwise(pl$lit(NA))$
              cast(pl$Float64)$
              alias(dist_col)
          )
        } else {
          # Without same_switchers_pl
          still_sw_col <- paste0("still_switcher_", i, "_XX")
          df <- df$with_columns(
            ((pl$col("F_g_XX") - 1 + effects <= pl$col("T_g_XX")) &
              (pl$col("N_g_control_check_XX") == effects))$alias(still_sw_col)
          )

          dist_col <- paste0("distance_to_switch_", i, "_XX")
          df <- df$with_columns(
            pl$when(pl$col(diff_y_col)$is_not_null())$
              then(
                (pl$col(still_sw_col) == TRUE) &
                (pl$col("time_XX") == pl$col("F_g_XX") - 1 + i) &
                (pl$lit(i) <= pl$col("L_g_XX")) &
                (pl$col("S_g_XX") == increase_XX) &
                (pl$col(N_gt_ctrl_col) > 0) &
                pl$col(N_gt_ctrl_col)$is_not_null()
              )$otherwise(pl$lit(NA))$
              cast(pl$Float64)$
              alias(dist_col)
          )
        }
      } else {
        # Without same_switchers option
        dist_col <- paste0("distance_to_switch_", i, "_XX")
        df <- df$with_columns(pl$lit(NA_real_)$alias(dist_col))

        df <- df$with_columns(
          pl$when(pl$col(diff_y_col)$is_not_null())$
            then(
              ((pl$col("time_XX") == pl$col("F_g_XX") - 1 + i) &
               (pl$lit(i) <= pl$col("L_g_XX")) &
               (pl$col("S_g_XX") == increase_XX) &
               (pl$col(N_gt_ctrl_col) > 0) &
               pl$col(N_gt_ctrl_col)$is_not_null())$cast(pl$Float64)
            )$otherwise(pl$lit(NA_real_))$
            alias(dist_col)
        )
      }

      # distance_to_switch weighted
      dist_w_col <- paste0("distance_to_switch_", i, "_wXX")
      df <- df$with_columns(
        (pl$col(dist_col) * pl$col("N_gt_XX"))$alias(dist_w_col)
      )

      # N_t columns
      N_t_col <- paste0("N", increase_XX, "_t_", i, "_XX")
      N_dw_t_col <- paste0("N_dw", increase_XX, "_t_", i, "_XX")
      df <- pl_sum_over(df, dist_w_col, "time_XX", N_t_col)
      df <- pl_sum_over(df, dist_col, "time_XX", N_dw_t_col)

      # Compute N_increase_i scalar
      filter_expr <- (pl$col("time_XX") >= t_min_XX) & (pl$col("time_XX") <= T_max_XX)
      N_val <- pl_scalar_mean_sum(df, N_t_col, filter_expr, "time_XX")
      assign(paste0("N", increase_XX, "_", i, "_XX"), N_val)

      N_dw_val <- pl_scalar_mean_sum(df, N_dw_t_col, filter_expr, "time_XX")
      assign(paste0("N", increase_XX, "_dw_", i, "_XX"), N_dw_val)

      # N_t_g: by time, d_sq, trends_nonparam
      N_t_g_col <- paste0("N", increase_XX, "_t_", i, "_g_XX")
      df <- pl_sum_over(df, dist_w_col, by_cols, N_t_g_col)

      # Controls adjustment
      if (!is.null(controls)) {
        part2_col <- paste0("part2_switch", increase_XX, "_", i, "_XX")
        df <- df$with_columns(pl$lit(0.0)$alias(part2_col))

        # T_d_XX: max F_g by d_sq_int_XX
        df <- pl_expr_over(df, "T_d_XX", pl$col("F_g_XX")$max(), "d_sq_int_XX")
        df <- df$with_columns((pl$col("T_d_XX") - 1)$alias("T_d_XX"))

        count_controls <- 0
        for (var in controls) {
          count_controls <- count_controls + 1

          # diff_X
          diff_X_col <- paste0("diff_X", count_controls, "_", i, "_XX")
          df <- df$with_columns(
            (pl$col(var) - pl$col(var)$shift(i)$over("group_XX"))$alias(diff_X_col)
          )

          # diff_X * N_gt
          diff_X_N_col <- paste0("diff_X", count_controls, "_", i, "_N_XX")
          df <- df$with_columns(
            (pl$col("N_gt_XX") * pl$col(diff_X_col))$alias(diff_X_N_col)
          )

          for (l in levels_d_sq_XX) {
            l_num <- as.numeric(l)

            # m_increase_g column
            m_g_col <- paste0("m", increase_XX, "_g_", l, "_", count_controls, "_", i, "_XX")
            N_inc_val <- get(paste0("N", increase_XX, "_", i, "_XX"))

            # Safe division to avoid 0/0 = NaN
            safe_ratio_m <- pl$when(pl$col(N_gt_ctrl_col) == 0)$
              then(pl$lit(0))$
              otherwise(pl$col(N_t_g_col) / pl$col(N_gt_ctrl_col))
            df <- df$with_columns(
              (
                ((pl$lit(i) <= pl$col("T_g_XX") - 2) & (pl$col("d_sq_int_XX") == l_num))$cast(pl$Float64) *
                (G_XX / N_inc_val) *
                (pl$col(dist_col) - safe_ratio_m * pl$col(never_col)) *
                ((pl$col("time_XX") >= i + 1) & (pl$col("time_XX") <= pl$col("T_g_XX")))$cast(pl$Float64) *
                pl$col(diff_X_N_col)
              )$alias(m_g_col)
            )

            # Sum by group
            m_col <- paste0("m", increase_XX, "_", l, "_", count_controls, "_", i, "_XX")
            df <- pl_sum_over(df, m_g_col, "group_XX", m_col)

            # Set NA where not first_obs
            df <- df$with_columns(
              pl$when(pl$col("first_obs_by_gp_XX") == 1)$
                then(pl$col(m_col))$
                otherwise(pl$lit(NA_real_))$
                alias(m_col)
            )

            # M (scalar mean)
            M_col <- paste0("M", increase_XX, "_", l, "_", count_controls, "_", i, "_XX")
            M_val <- pl_scalar_sum(df, m_col) / G_XX
            df <- df$with_columns(pl$lit(M_val)$alias(M_col))

            # E_hat_denom - only count rows where diff_y_XX is not missing (matches Stata)
            df <- df$with_columns(
              pl$when(
                (pl$col("F_g_XX") > pl$col("time_XX")) &
                (pl$col("d_sq_int_XX") == l_num) &
                pl$col("diff_y_XX")$is_not_null()
              )$
                then(pl$lit(1.0))$
                otherwise(pl$lit(0.0))$
                alias("dummy_XX")
            )

            E_hat_denom_col <- paste0("E_hat_denom_", count_controls, "_", l, "_XX")
            df <- pl_sum_over(df, "dummy_XX", c("time_XX", "d_sq_int_XX"), E_hat_denom_col)

            df <- df$with_columns(
              pl$when(pl$col("d_sq_int_XX") == l_num)$
                then(pl$col(E_hat_denom_col))$
                otherwise(pl$lit(NA_real_))$
                alias(E_hat_denom_col)
            )

            # E_y_hat_gt
            E_y_hat_col <- paste0("E_y_hat_gt_", l, "_XX")
            E_y_hat_int_col <- paste0("E_y_hat_gt_int_", l, "_XX")
            df <- df$with_columns(
              (pl$col(E_y_hat_int_col) * (pl$col(E_hat_denom_col) >= 2)$cast(pl$Float64))$alias(E_y_hat_col)
            )

            # N_c columns and in_sum
            N_c_temp_col <- paste0("N_c_", l, "_temp_XX")
            N_c_col <- paste0("N_c_", l, "_XX")

            df <- df$with_columns(
              (pl$col("N_gt_XX") *
                ((pl$col("d_sq_int_XX") == l_num) &
                 (pl$col("time_XX") >= 2) &
                 (pl$col("time_XX") <= pl$col("T_d_XX")) &
                 (pl$col("time_XX") < pl$col("F_g_XX")) &
                 pl$col("diff_y_XX")$is_not_null())$cast(pl$Float64)
              )$alias(N_c_temp_col)
            )

            N_c_val <- pl_scalar_sum(df, N_c_temp_col)
            df <- df$with_columns(pl$lit(N_c_val)$alias(N_c_col))

            # in_sum_temp
            prod_X_col <- paste0("prod_X", count_controls, "_Ngt_XX")
            in_sum_temp_col <- paste0("in_sum_temp_", count_controls, "_", l, "_XX")

            # DOF adjustment for in_sum_temp - avoid NaN by only computing sqrt when E_hat_denom > 1
            # Stata: in_sum_temp_adj = 0 when E_hat_denom <= 1, else sqrt(E_hat_denom/(E_hat_denom-1))-1
            in_sum_adj_col <- paste0("in_sum_temp_adj_", count_controls, "_", l, "_XX")
            df <- df$with_columns(
              pl$when(pl$col(E_hat_denom_col) > 1)$
                then((pl$col(E_hat_denom_col) / (pl$col(E_hat_denom_col) - 1))$sqrt() - 1)$
                otherwise(pl$lit(0.0))$
                alias(in_sum_adj_col)
            )

            df <- df$with_columns(
              (
                pl$col(prod_X_col) *
                (pl$lit(1.0) +
                  (pl$col(E_hat_denom_col) >= 2)$cast(pl$Float64) *
                  pl$col(in_sum_adj_col)
                ) *
                (pl$col("diff_y_XX") - pl$col(E_y_hat_col)) *
                ((pl$col("time_XX") >= 2) & (pl$col("time_XX") <= pl$col("F_g_XX") - 1))$cast(pl$Float64) /
                pl$col(N_c_col)
              )$alias(in_sum_temp_col)
            )

            # in_sum by group
            in_sum_col <- paste0("in_sum_", count_controls, "_", l, "_XX")
            df <- pl_sum_over(df, in_sum_temp_col, "group_XX", in_sum_col)

            # Residualize outcome if useful_res > 1
            useful_res_val <- get(paste0("useful_res_", l, "_XX"))
            if (!is.null(useful_res_val) && useful_res_val > 1) {
              coefs_val <- get(paste0("coefs_sq_", l, "_XX"))[count_controls, 1]

              df <- df$with_columns(
                pl$when(pl$col("d_sq_int_XX") == l_num)$
                  then(pl$col(diff_y_col) - coefs_val * pl$col(diff_X_col))$
                  otherwise(pl$col(diff_y_col))$
                  alias(diff_y_col)
              )

              in_brackets_col <- paste0("in_brackets_", l, "_", count_controls, "_XX")
              df <- df$with_columns(pl$lit(0.0)$alias(in_brackets_col))
            }
          }
        }
      }

      # DOF and mean computations for variance
      diff_y_N_col <- paste0("diff_y_", i, "_N_gt_XX")
      dof_ns_col <- paste0("dof_ns_", i, "_XX")
      dof_s_col <- paste0("dof_s_", i, "_XX")
      N_t_col <- paste0("N", increase_XX, "_t_", i, "_XX")

      df <- df$with_columns(
        (pl$col("N_gt_XX") * pl$col(diff_y_col))$alias(diff_y_N_col)
      )

      # dof_ns: indicator for controls
      df <- df$with_columns(
        (
          pl$col("N_gt_XX")$is_not_null() & (pl$col("N_gt_XX") != 0) &
          pl$col(diff_y_col)$is_not_null() &
          pl$col(never_col)$is_not_null() & (pl$col(never_col) == 1) &
          pl$col(N_t_col)$is_not_null() & (pl$col(N_t_col) > 0)
        )$cast(pl$Float64)$alias(dof_ns_col)
      )

      # dof_s: indicator for switchers
      df <- df$with_columns(
        (
          pl$col("N_gt_XX")$is_not_null() & (pl$col("N_gt_XX") != 0) &
          pl$col(dist_col)$is_not_null() & (pl$col(dist_col) == 1)
        )$cast(pl$Float64)$alias(dof_s_col)
      )

      # Grouped totals for controls
      ns_by_cols <- c("d_sq_XX")
      if (!is.null(trends_nonparam) && length(trends_nonparam) > 0) {
        ns_by_cols <- c(ns_by_cols, trends_nonparam)
      }
      ns_by_cols <- c(ns_by_cols, "time_XX")

      count_ns_col <- paste0("count_cohort_", i, "_ns_t_XX")
      total_ns_col <- paste0("total_cohort_", i, "_ns_t_XX")
      mean_ns_col <- paste0("mean_cohort_", i, "_ns_t_XX")
      dof_coh_ns_col <- paste0("dof_cohort_", i, "_ns_t_XX")

      # Filtered aggregations
      df <- pl_filtered_agg_over(df, "N_gt_XX", pl$col(dof_ns_col) == 1, ns_by_cols, count_ns_col, "sum")
      df <- pl_filtered_agg_over(df, diff_y_N_col, pl$col(dof_ns_col) == 1, ns_by_cols, total_ns_col, "sum")
      df <- df$with_columns((pl$col(total_ns_col) / pl$col(count_ns_col))$alias(mean_ns_col))

      # DOF counting
      if (is.null(cluster) || cluster == "" || is.na(cluster)) {
        df <- pl_filtered_agg_over(df, dof_ns_col, pl$col(dof_ns_col) == 1, ns_by_cols, dof_coh_ns_col, "sum", filter_result = TRUE)
      } else {
        cluster_dof_col <- paste0("cluster_dof_", i, "_ns_XX")
        df <- df$with_columns(
          pl$when(pl$col(dof_ns_col) == 1)$
            then(pl$col(cluster))$
            otherwise(pl$lit(NA))$
            alias(cluster_dof_col)
        )
        df <- pl_uniqueN_over(df, cluster_dof_col, ns_by_cols, dof_coh_ns_col, pl$col(cluster_dof_col)$is_not_null())
      }

      # Diff_y * N_gt and dof indicator
      dof_y_N_col <- paste0("dof_y_", i, "_N_gt_XX")
      df <- df$with_columns(
        ((pl$col("N_gt_XX") != 0) & pl$col(diff_y_col)$is_not_null())$cast(pl$Float64)$alias(dof_y_N_col)
      )

      # Switchers cohort demeaning
      if (isFALSE(less_conservative_se)) {
        sw_by_cols <- c("d_sq_XX", "F_g_XX", "d_fg_XX")
        if (!is.null(trends_nonparam) && length(trends_nonparam) > 0) {
          sw_by_cols <- c(sw_by_cols, trends_nonparam)
        }

        count_s_col <- paste0("count_cohort_", i, "_s_t_XX")
        total_s_col <- paste0("total_cohort_", i, "_s_t_XX")
        mean_s_col <- paste0("mean_cohort_", i, "_s_t_XX")
        dof_coh_s_col <- paste0("dof_cohort_", i, "_s_t_XX")

        df <- pl_filtered_agg_over(df, "N_gt_XX", pl$col(dof_s_col) == 1, sw_by_cols, count_s_col, "sum")
        df <- pl_filtered_agg_over(df, diff_y_N_col, pl$col(dof_s_col) == 1, sw_by_cols, total_s_col, "sum")
        df <- df$with_columns((pl$col(total_s_col) / pl$col(count_s_col))$alias(mean_s_col))

        if (is.null(cluster) || cluster == "" || is.na(cluster)) {
          df <- pl_filtered_agg_over(df, dof_s_col, pl$col(dof_s_col) == 1, sw_by_cols, dof_coh_s_col, "sum", filter_result = TRUE)
        } else {
          cluster_dof_s_col <- paste0("cluster_dof_", i, "_s_XX")
          df <- df$with_columns(
            pl$when(pl$col(dof_s_col) == 1)$
              then(pl$col(cluster))$
              otherwise(pl$lit(NA))$
              alias(cluster_dof_s_col)
          )
          df <- pl_uniqueN_over(df, cluster_dof_s_col, sw_by_cols, dof_coh_s_col, pl$col(cluster_dof_s_col)$is_not_null())
        }
      } else {
        # less_conservative_se path columns
        path_0_col <- "path_0_XX"
        path_1_col <- "path_1_XX"
        path_i_col <- paste0("path_", i, "_XX")

        for (suffix in c("s0", "s1", "s2")) {
          path_col <- switch(suffix,
            "s0" = path_0_col,
            "s1" = path_1_col,
            "s2" = path_i_col
          )

          by_cols_path <- c(path_col)
          if (!is.null(trends_nonparam) && length(trends_nonparam) > 0) {
            by_cols_path <- c(by_cols_path, trends_nonparam)
          }

          count_col <- paste0("count_cohort_", i, "_", suffix, "_t_XX")
          total_col <- paste0("total_cohort_", i, "_", suffix, "_t_XX")
          dof_col <- paste0("dof_cohort_", i, "_", suffix, "_t_XX")

          df <- pl_filtered_agg_over(df, "N_gt_XX", pl$col(dist_col) == 1, by_cols_path, count_col, "sum")
          df <- df$with_columns(
            pl$when(pl$col(dist_col) == 1)$
              then(pl$col(count_col))$
              otherwise(pl$lit(NA_real_))$
              alias(count_col)
          )

          df <- pl_filtered_agg_over(df, diff_y_N_col, pl$col(dist_col) == 1, by_cols_path, total_col, "sum")
          df <- df$with_columns(
            pl$when(pl$col(dist_col) == 1)$
              then(pl$col(total_col))$
              otherwise(pl$lit(NA_real_))$
              alias(total_col)
          )

          df <- pl_filtered_agg_over(df, dof_y_N_col, pl$col(dist_col) == 1, by_cols_path, dof_col, "sum")
          df <- df$with_columns(
            pl$when(pl$col(dist_col) == 1)$
              then(pl$col(dof_col))$
              otherwise(pl$lit(NA_real_))$
              alias(dof_col)
          )
        }

        # Compute mean based on cohort hierarchy
        mean_s_col <- paste0("mean_cohort_", i, "_s_t_XX")
        dof_coh_s_col <- paste0("dof_cohort_", i, "_s_t_XX")
        cohort_i_col <- paste0("cohort_fullpath_", i, "_XX")

        count_s0 <- paste0("count_cohort_", i, "_s0_t_XX")
        count_s1 <- paste0("count_cohort_", i, "_s1_t_XX")
        count_s2 <- paste0("count_cohort_", i, "_s2_t_XX")
        total_s0 <- paste0("total_cohort_", i, "_s0_t_XX")
        total_s1 <- paste0("total_cohort_", i, "_s1_t_XX")
        total_s2 <- paste0("total_cohort_", i, "_s2_t_XX")
        dof_s0 <- paste0("dof_cohort_", i, "_s0_t_XX")
        dof_s1 <- paste0("dof_cohort_", i, "_s1_t_XX")
        dof_s2 <- paste0("dof_cohort_", i, "_s2_t_XX")

        df <- df$with_columns(
          pl$when(pl$col(cohort_i_col) == 1)$
            then(pl$col(total_s2) / pl$col(count_s2))$
            otherwise(pl$lit(NA_real_))$
            alias(mean_s_col)
        )

        df <- df$with_columns(
          pl$when((pl$col(cohort_i_col) == 0) & (pl$col("cohort_fullpath_1_XX") == 1))$
            then(pl$col(total_s1) / pl$col(count_s1))$
            otherwise(pl$col(mean_s_col))$
            alias(mean_s_col)
        )

        df <- df$with_columns(
          pl$when(pl$col("cohort_fullpath_1_XX") == 0)$
            then(pl$col(total_s0) / pl$col(count_s0))$
            otherwise(pl$col(mean_s_col))$
            alias(mean_s_col)
        )

        # DOF cohort
        df <- df$with_columns(
          pl$when(pl$col(cohort_i_col) == 1)$
            then(pl$col(dof_s2))$
            otherwise(pl$lit(NA_real_))$
            alias(dof_coh_s_col)
        )

        df <- df$with_columns(
          pl$when((pl$col(cohort_i_col) == 0) & (pl$col("cohort_fullpath_1_XX") == 1))$
            then(pl$col(dof_s1))$
            otherwise(pl$col(dof_coh_s_col))$
            alias(dof_coh_s_col)
        )

        df <- df$with_columns(
          pl$when(pl$col("cohort_fullpath_1_XX") == 0)$
            then(pl$col(dof_s0))$
            otherwise(pl$col(dof_coh_s_col))$
            alias(dof_coh_s_col)
        )
      }

      # Union of switchers and not-yet switchers (ns_s)
      dof_ns_s_col <- paste0("dof_ns_s_", i, "_XX")
      count_ns_s_col <- paste0("count_cohort_", i, "_ns_s_t_XX")
      total_ns_s_col <- paste0("total_cohort_", i, "_ns_s_t_XX")
      mean_ns_s_col <- paste0("mean_cohort_", i, "_ns_s_t_XX")
      dof_coh_ns_s_col <- paste0("dof_cohort_", i, "_ns_s_t_XX")

      df <- df$with_columns(
        ((pl$col(dof_s_col) == 1) | (pl$col(dof_ns_col) == 1))$cast(pl$Float64)$alias(dof_ns_s_col)
      )

      df <- pl_filtered_agg_over(df, "N_gt_XX", pl$col(dof_ns_s_col) == 1, ns_by_cols, count_ns_s_col, "sum")
      df <- pl_filtered_agg_over(df, diff_y_N_col, pl$col(dof_ns_s_col) == 1, ns_by_cols, total_ns_s_col, "sum")
      df <- df$with_columns((pl$col(total_ns_s_col) / pl$col(count_ns_s_col))$alias(mean_ns_s_col))

      if (is.null(cluster) || cluster == "" || is.na(cluster)) {
        df <- pl_filtered_agg_over(df, dof_ns_s_col, pl$col(dof_ns_s_col) == 1, ns_by_cols, dof_coh_ns_s_col, "sum", filter_result = TRUE)
      } else {
        cluster_dof_ns_s_col <- paste0("cluster_dof_", i, "_ns_s_XX")
        df <- df$with_columns(
          pl$when(pl$col(dof_ns_s_col) == 1)$
            then(pl$col(cluster))$
            otherwise(pl$lit(NA))$
            alias(cluster_dof_ns_s_col)
        )
        df <- pl_uniqueN_over(df, cluster_dof_ns_s_col, ns_by_cols, dof_coh_ns_s_col, pl$col(cluster_dof_ns_s_col)$is_not_null())
      }

      # Compute E_hat_gt and DOF_gt
      df <- compute_E_hat_gt_polars(df, i, "effect")
      df <- compute_DOF_gt_polars(df, i, "effect")

      ###### 3. Computing U_Gg_l variables
      N_inc_val <- get(paste0("N", increase_XX, "_", i, "_XX"))

      if (!is.null(N_inc_val) && N_inc_val != 0) {
        dummy_U_col <- paste0("dummy_U_Gg", i, "_XX")
        df <- df$with_columns(
          (pl$lit(i) <= pl$col("T_g_XX") - 1)$cast(pl$Float64)$alias(dummy_U_col)
        )

        # U_Gg_temp
        # Note: Use safe division to avoid 0/0 = NaN when N_gt_ctrl is 0
        U_Gg_temp_col <- paste0("U_Gg", i, "_temp_XX")
        safe_ratio <- pl$when(pl$col(N_gt_ctrl_col) == 0)$
          then(pl$lit(0))$
          otherwise(pl$col(N_t_g_col) / pl$col(N_gt_ctrl_col))
        df <- df$with_columns(
          (
            pl$col(dummy_U_col) * (G_XX / N_inc_val) *
            ((pl$col("time_XX") >= i + 1) & (pl$col("time_XX") <= pl$col("T_g_XX")))$cast(pl$Float64) *
            pl$col("N_gt_XX") *
            (pl$col(dist_col) - safe_ratio * pl$col(never_col)) *
            pl$col(diff_y_col)
          )$alias(U_Gg_temp_col)
        )

        # U_Gg: sum by group
        U_Gg_col <- paste0("U_Gg", i, "_XX")
        df <- pl_sum_over(df, U_Gg_temp_col, "group_XX", U_Gg_col)
        df <- df$with_columns(
          (pl$col(U_Gg_col) * pl$col("first_obs_by_gp_XX"))$alias(U_Gg_col)
        )

        # count_core
        count_core_col <- paste0("count", i, "_core_XX")
        df <- df$with_columns(
          pl$when(
            (pl$col(U_Gg_temp_col)$is_not_null() & (pl$col(U_Gg_temp_col) != 0)) |
            ((pl$col(U_Gg_temp_col) == 0) & (pl$col(diff_y_col) == 0) &
              ((pl$col(dist_col) != 0) | ((pl$col(N_t_g_col) != 0) & (pl$col(never_col) != 0))))
          )$then(pl$col("N_gt_XX"))$otherwise(pl$lit(0.0))$alias(count_core_col)
        )

        # U_Gg_temp_var
        U_Gg_temp_var_col <- paste0("U_Gg", i, "_temp_var_XX")
        E_hat_col <- paste0("E_hat_gt_", i, "_XX")
        DOF_col <- paste0("DOF_gt_", i, "_XX")

        # Re-use safe_ratio from U_Gg_temp computation
        df <- df$with_columns(
          (
            pl$col(dummy_U_col) * (G_XX / N_inc_val) *
            (pl$col(dist_col) - safe_ratio * pl$col(never_col)) *
            ((pl$col("time_XX") >= i + 1) & (pl$col("time_XX") <= pl$col("T_g_XX")))$cast(pl$Float64) *
            pl$col("N_gt_XX") * pl$col(DOF_col) *
            (pl$col(diff_y_col) - pl$col(E_hat_col))
          )$alias(U_Gg_temp_var_col)
        )

        # Controls adjustment for variance
        if (!is.null(controls)) {
          for (l in levels_d_sq_XX) {
            # Only include cohort in variance adjustment if useful_res > 1 (matches Stata)
            useful_res_val <- get(paste0("useful_res_", l, "_XX"))
            if (is.null(useful_res_val) || useful_res_val <= 1) next

            l_num <- as.numeric(l)
            comb_col <- paste0("combined", increase_XX, "_temp_", l, "_", i, "_XX")
            df <- df$with_columns(pl$lit(0.0)$alias(comb_col))

            for (j in 1:count_controls) {
              in_brackets_col <- paste0("in_brackets_", l, "_", j, "_XX")

              for (k in 1:count_controls) {
                in_sum_col <- paste0("in_sum_", k, "_", l, "_XX")
                inv_val <- get(paste0("inv_Denom_", l, "_XX"))[j, k]

                df <- df$with_columns(
                  (pl$col(in_brackets_col) +
                    inv_val * pl$col(in_sum_col) *
                    ((pl$col("d_sq_int_XX") == l_num) & (pl$col("F_g_XX") >= 3))$cast(pl$Float64)
                  )$alias(in_brackets_col)
                )
              }

              coef_val <- get(paste0("coefs_sq_", l, "_XX"))[j, 1]
              df <- df$with_columns(
                (pl$col(in_brackets_col) - coef_val)$alias(in_brackets_col)
              )

              M_col <- paste0("M", increase_XX, "_", l, "_", j, "_", i, "_XX")
              df <- df$with_columns(
                (pl$col(comb_col) + pl$col(M_col) * pl$col(in_brackets_col))$alias(comb_col)
              )
            }

            part2_col <- paste0("part2_switch", increase_XX, "_", i, "_XX")
            df <- df$with_columns(
              (pl$col(part2_col) + pl$col(comb_col))$alias(part2_col)
            )
          }
        }

        # Sum U_Gg_var by group
        U_Gg_var_col <- paste0("U_Gg", i, "_var_XX")
        df <- pl_sum_over(df, U_Gg_temp_var_col, "group_XX", U_Gg_var_col)

        if (!is.null(controls)) {
          part2_col <- paste0("part2_switch", increase_XX, "_", i, "_XX")
          df <- df$with_columns(
            (pl$col(U_Gg_var_col) - pl$col(part2_col))$alias(U_Gg_var_col)
          )
        }
      }

      ###### 4. Normalized option
      if (normalized == TRUE) {
        if (is.null(continuous)) {
          df <- df$with_columns(
            pl$when(
              (pl$col("time_XX") >= pl$col("F_g_XX")) &
              (pl$col("time_XX") <= pl$col("F_g_XX") - 1 + i) &
              (pl$col("S_g_XX") == increase_XX)
            )$then(pl$col("treatment_XX") - pl$col("d_sq_XX"))$
              otherwise(pl$lit(NA_real_))$
              alias("sum_temp_XX")
          )
        } else {
          df <- df$with_columns(
            pl$when(
              (pl$col("time_XX") >= pl$col("F_g_XX")) &
              (pl$col("time_XX") <= pl$col("F_g_XX") - 1 + i) &
              (pl$col("S_g_XX") == increase_XX)
            )$then(pl$col("treatment_XX_orig") - pl$col("d_sq_XX_orig"))$
              otherwise(pl$lit(NA_real_))$
              alias("sum_temp_XX")
          )
        }

        sum_treat_col <- paste0("sum_treat_until_", i, "_XX")
        df <- pl_sum_over(df, "sum_temp_XX", "group_XX", sum_treat_col)
        df <- pl_batch_drop_cols(df, "sum_temp_XX")

        delta_temp_col <- paste0("delta_D_", i, "_cum_temp_XX")
        N_inc_val <- get(paste0("N", increase_XX, "_", i, "_XX"))

        df <- df$with_columns(
          pl$when(pl$col(dist_col) == 1)$
            then(
              (pl$col("N_gt_XX") / N_inc_val) * (
                pl$col("S_g_XX") * pl$col(sum_treat_col) +
                (1 - pl$col("S_g_XX")) * (-pl$col(sum_treat_col))
              )
            )$otherwise(pl$lit(NA_real_))$
            alias(delta_temp_col)
        )

        delta_val <- pl_scalar_sum(df, delta_temp_col)
        assign(paste0("delta_norm_", i, "_XX"), delta_val)
      }
    }

    # Trends_lin option
    Ntrendslin <- 1
    for (i in 1:l_u_a_XX) {
      Ntrendslin <- min(Ntrendslin, get(paste0("N", increase_XX, "_", i, "_XX")), na.rm = TRUE)
    }

    if (isTRUE(trends_lin) && Ntrendslin != 0) {
      lu <- as.integer(l_u_a_XX)

      col_TL <- sprintf("U_Gg%d_TL", lu)
      col_var_TL <- sprintf("U_Gg%d_var_TL", lu)
      col_XX <- sprintf("U_Gg%d_XX", lu)
      col_var_XX <- sprintf("U_Gg%d_var_XX", lu)

      df <- pl_batch_drop_cols(df, c(col_TL, col_var_TL))
      df <- df$with_columns(
        pl$lit(0.0)$alias(col_TL),
        pl$lit(0.0)$alias(col_var_TL)
      )

      for (i in seq_len(lu)) {
        U_i <- sprintf("U_Gg%d_XX", i)
        U_var_i <- sprintf("U_Gg%d_var_XX", i)

        df <- df$with_columns(
          (pl$col(col_TL) + pl$col(U_i))$alias(col_TL),
          (pl$col(col_var_TL) + pl$col(U_var_i))$alias(col_var_TL)
        )
      }

      df <- df$with_columns(
        pl$col(col_TL)$alias(col_XX),
        pl$col(col_var_TL)$alias(col_var_XX)
      )
    }

    ###### 5. Placebo effects
    if (placebo != 0 && exists("l_placebo_u_a_XX") && l_placebo_u_a_XX >= 1) {
      for (i in 1:l_placebo_u_a_XX) {
        df <- compute_placebo_effects_polars(
          df, i, increase_XX, G_XX, t_min_XX, T_max_XX,
          trends_nonparam, cluster, controls, levels_d_sq_XX,
          same_switchers_pl, normalized, continuous, controls_globals
        )

        if (normalized == TRUE) {
          N_pl_val <- get(paste0("N", increase_XX, "_placebo_", i, "_XX"))
          if (!is.null(N_pl_val) && N_pl_val != 0) {
            delta_pl_col <- paste0("delta_D_pl_", i, "_cum_temp_XX")
            delta_val <- pl_scalar_sum(df, delta_pl_col)
            assign(paste0("delta_norm_pl_", i, "_XX"), delta_val)
          }
        }
      }

      # Trends_lin for placebos
      if (isTRUE(trends_lin)) {
        Ntrendslin_pl <- 1
        for (i in 1:l_placebo_u_a_XX) {
          N_pl_val <- get(paste0("N", increase_XX, "_placebo_", i, "_XX"))
          if (!is.null(N_pl_val)) {
            Ntrendslin_pl <- min(Ntrendslin_pl, N_pl_val, na.rm = TRUE)
          }
        }

        if (Ntrendslin_pl != 0) {
          lp <- as.integer(l_placebo_u_a_XX)

          col_TL <- sprintf("U_Gg_pl_%d_TL", lp)
          col_var_TL <- sprintf("U_Gg_pl_%d_var_TL", lp)
          col_placebo <- sprintf("U_Gg_placebo_%d_XX", lp)
          col_pl_var <- sprintf("U_Gg_pl_%d_var_XX", lp)

          df <- pl_batch_drop_cols(df, c(col_TL, col_var_TL))
          df <- df$with_columns(
            pl$lit(0.0)$alias(col_TL),
            pl$lit(0.0)$alias(col_var_TL)
          )

          for (i in seq_len(lp)) {
            U_i <- sprintf("U_Gg_placebo_%d_XX", i)
            U_var_i <- sprintf("U_Gg_pl_%d_var_XX", i)

            df <- df$with_columns(
              (pl$col(col_TL) + pl$col(U_i))$alias(col_TL),
              (pl$col(col_var_TL) + pl$col(U_var_i))$alias(col_var_TL)
            )
          }

          df <- df$with_columns(
            pl$col(col_TL)$alias(col_placebo),
            pl$col(col_var_TL)$alias(col_pl_var)
          )
        }
      }
    }

    ###### 8. Average Total Effect
    if (!trends_lin) {
      total_key <- sprintf("sum_N%s_l_XX", increase_XX)

      sum_N <- sum(vapply(
        seq_len(as.integer(l_u_a_XX)),
        function(j) get(sprintf("N%s_%s_XX", increase_XX, j)),
        numeric(1)
      ))
      assign(total_key, sum_N)

      init_cols <- c("U_Gg_XX", "U_Gg_num_XX", "U_Gg_den_XX", "U_Gg_num_var_XX", "U_Gg_var_XX")
      df <- pl_init_zero_cols(df, init_cols)

      for (i in seq_len(as.integer(l_u_a_XX))) {
        N_name <- sprintf("N%s_%s_XX", increase_XX, i)
        N_increase <- get(N_name)
        sum_N_increase <- get(total_key)

        if (!is.null(N_increase) && N_increase != 0) {
          w_i <- N_increase / sum_N_increase
          assign(sprintf("w_%s_XX", i), w_i)

          delta_temp <- sprintf("delta_D_%s_temp_XX", i)
          delta_col <- sprintf("delta_D_%s_XX", i)
          delta_g <- sprintf("delta_D_g_%s_XX", i)
          dist_col <- sprintf("distance_to_switch_%s_XX", i)

          if (is.null(continuous)) {
            df <- df$with_columns(pl$lit(0.0)$alias(delta_temp))
            df <- df$with_columns(
              pl$when(pl$col(dist_col) == 1)$
                then(
                  (pl$col("N_gt_XX") / N_increase) *
                  ((pl$col("treatment_XX") - pl$col("d_sq_XX")) * pl$col("S_g_XX") +
                    (1 - pl$col("S_g_XX")) * (pl$col("d_sq_XX") - pl$col("treatment_XX")))
                )$otherwise(pl$lit(0.0))$
                alias(delta_temp)
            )
          } else {
            df <- df$with_columns(pl$lit(0.0)$alias(delta_temp))
            df <- df$with_columns(
              pl$when(pl$col(dist_col) == 1)$
                then(
                  (pl$col("N_gt_XX") / N_increase) *
                  ((pl$col("treatment_XX_orig") - pl$col("d_sq_XX_orig")) * pl$col("S_g_XX") +
                    (1 - pl$col("S_g_XX")) * (pl$col("d_sq_XX_orig") - pl$col("treatment_XX_orig")))
                )$otherwise(pl$lit(0.0))$
                alias(delta_temp)
            )
          }

          total_delta <- pl_scalar_sum(df, delta_temp)
          df <- df$with_columns(pl$lit(total_delta)$alias(delta_col))

          df <- df$with_columns(
            (pl$col(delta_temp) * (N_increase / pl$col("N_gt_XX")))$alias(delta_g)
          )

          df <- pl_batch_drop_cols(df, delta_temp)

          U_col_i <- sprintf("U_Gg%s_XX", i)
          U_var_col_i <- sprintf("U_Gg%s_var_XX", i)

          df <- df$with_columns(
            (pl$col("U_Gg_num_XX") + w_i * pl$col(U_col_i))$alias("U_Gg_num_XX"),
            (pl$col("U_Gg_num_var_XX") + w_i * pl$col(U_var_col_i))$alias("U_Gg_num_var_XX"),
            (pl$col("U_Gg_den_XX") + w_i * pl$col(delta_col))$alias("U_Gg_den_XX")
          )
        }
      }

      df <- df$with_columns(
        (pl$col("U_Gg_num_XX") / pl$col("U_Gg_den_XX"))$alias("U_Gg_XX"),
        (pl$col("U_Gg_num_var_XX") / pl$col("U_Gg_den_XX"))$alias("U_Gg_var_XX")
      )
    }

    # Update constants
    for (e in names(const)) {
      if (exists(e)) {
        const[[e]] <- get(e)
      }
    }

    sum_N_key <- paste0("sum_N", increase_XX, "_l_XX")
    if (exists(sum_N_key)) {
      const[[sum_N_key]] <- get(sum_N_key)
    }

    for (i in 1:l_u_a_XX) {
      if (isTRUE(normalized)) {
        delta_key <- paste0("delta_norm_", i, "_XX")
        if (exists(delta_key)) {
          const[[delta_key]] <- get(delta_key)
        }
      }
    }

    if (placebo != 0 && exists("l_placebo_u_a_XX") && l_placebo_u_a_XX >= 1) {
      for (i in 1:l_placebo_u_a_XX) {
        if (isTRUE(normalized)) {
          delta_pl_key <- paste0("delta_norm_pl_", i, "_XX")
          if (exists(delta_pl_key)) {
            const[[delta_pl_key]] <- get(delta_pl_key)
          }
        }
      }
    }
  })

  data <- list(df = df, const = const)
  return(data)
}

#' Compute E_hat_gt with NaN handling (polars version)
#' @param df polars DataFrame
#' @param i effect number
#' @param type_sect "effect" or "placebo"
#' @return polars DataFrame
#' @noRd
compute_E_hat_gt_polars <- function(df, i, type_sect = "effect") {
  if (type_sect == "effect") {
    E_hat <- sprintf("E_hat_gt_%s_XX", i)
    mean_ns <- sprintf("mean_cohort_%s_ns_t_XX", i)
    mean_s <- sprintf("mean_cohort_%s_s_t_XX", i)
    mean_nss <- sprintf("mean_cohort_%s_ns_s_t_XX", i)
    dof_ns <- sprintf("dof_cohort_%s_ns_t_XX", i)
    dof_s <- sprintf("dof_cohort_%s_s_t_XX", i)
    dof_nss <- sprintf("dof_cohort_%s_ns_s_t_XX", i)
  } else {
    E_hat <- sprintf("E_hat_gt_pl_%s_XX", i)
    mean_ns <- sprintf("mean_cohort_pl_%s_ns_t_XX", i)
    mean_s <- sprintf("mean_cohort_pl_%s_s_t_XX", i)
    mean_nss <- sprintf("mean_cohort_pl_%s_ns_s_t_XX", i)
    dof_ns <- sprintf("dof_cohort_pl_%s_ns_t_XX", i)
    dof_s <- sprintf("dof_cohort_pl_%s_s_t_XX", i)
    dof_nss <- sprintf("dof_cohort_pl_%s_ns_s_t_XX", i)
  }

  # Initialize to NA
  df <- df$with_columns(pl$lit(NA_real_)$alias(E_hat))

  # Condition A: time < Fg OR (Fg - 1 + i == time)
  cond_A <- (pl$col("time_XX") < pl$col("F_g_XX")) |
            (pl$col("F_g_XX") - 1 + i == pl$col("time_XX"))
  df <- df$with_columns(
    pl$when(cond_A)$then(pl$lit(0.0))$otherwise(pl$col(E_hat))$alias(E_hat)
  )

  # Condition B: time < Fg AND dof_ns >= 2 (Stata: dof_cohort_ns_t >= 2)
  # Note: In Stata, if dof_ns is missing, the condition is FALSE, so E_hat stays at 0
  cond_B <- (pl$col("time_XX") < pl$col("F_g_XX")) &
            pl$col(dof_ns)$is_not_null() & (pl$col(dof_ns) >= 2)
  df <- df$with_columns(
    pl$when(cond_B)$then(pl$col(mean_ns))$otherwise(pl$col(E_hat))$alias(E_hat)
  )

  # Condition C: (Fg - 1 + i == time) AND dof_s >= 2 (Stata: dof_cohort_s_t >= 2)
  cond_C <- (pl$col("F_g_XX") - 1 + i == pl$col("time_XX")) &
            pl$col(dof_s)$is_not_null() & (pl$col(dof_s) >= 2)
  df <- df$with_columns(
    pl$when(cond_C)$then(pl$col(mean_s))$otherwise(pl$col(E_hat))$alias(E_hat)
  )

  # Condition D: use mean_nss
  cond_D <- (pl$col(dof_nss)$is_not_null() & (pl$col(dof_nss) >= 2)) &
            (
              ((pl$col("F_g_XX") - 1 + i == pl$col("time_XX")) &
               pl$col(dof_s)$is_not_null() & (pl$col(dof_s) == 1)) |
              ((pl$col("time_XX") < pl$col("F_g_XX")) &
               pl$col(dof_ns)$is_not_null() & (pl$col(dof_ns) == 1))
            )
  df <- df$with_columns(
    pl$when(cond_D)$then(pl$col(mean_nss))$otherwise(pl$col(E_hat))$alias(E_hat)
  )

  return(df)
}

#' Compute DOF_gt with NaN handling (polars version)
#' @param df polars DataFrame
#' @param i effect number
#' @param type_sect "effect" or "placebo"
#' @return polars DataFrame
#' @noRd
compute_DOF_gt_polars <- function(df, i, type_sect = "effect") {
  if (type_sect == "effect") {
    DOF_col <- sprintf("DOF_gt_%s_XX", i)
    dof_s_t <- sprintf("dof_cohort_%s_s_t_XX", i)
    dof_ns_t <- sprintf("dof_cohort_%s_ns_t_XX", i)
    dof_ns_s_t <- sprintf("dof_cohort_%s_ns_s_t_XX", i)
  } else {
    DOF_col <- sprintf("DOF_gt_pl_%s_XX", i)
    dof_s_t <- sprintf("dof_cohort_pl_%s_s_t_XX", i)
    dof_ns_t <- sprintf("dof_cohort_pl_%s_ns_t_XX", i)
    dof_ns_s_t <- sprintf("dof_cohort_pl_%s_ns_s_t_XX", i)
  }

  df <- pl_batch_drop_cols(df, DOF_col)
  df <- df$with_columns(pl$lit(NA_real_)$alias(DOF_col))

  # DOF = 1 if (time < Fg) OR (Fg - 1 + i == time)
  cond_1 <- (pl$col("time_XX") < pl$col("F_g_XX")) |
            (pl$col("F_g_XX") - 1 + i == pl$col("time_XX"))
  df <- df$with_columns(
    pl$when(cond_1)$then(pl$lit(1.0))$otherwise(pl$col(DOF_col))$alias(DOF_col)
  )

  # sqrt(dof_s_t / (dof_s_t - 1)) for switchers
  cond_s <- (pl$col("F_g_XX") - 1 + i == pl$col("time_XX")) & (pl$col(dof_s_t) > 1)
  df <- df$with_columns(
    pl$when(cond_s)$
      then((pl$col(dof_s_t) / (pl$col(dof_s_t) - 1))$sqrt())$
      otherwise(pl$col(DOF_col))$
      alias(DOF_col)
  )

  # sqrt(dof_ns_t / (dof_ns_t - 1)) for controls
  cond_ns <- (pl$col("time_XX") < pl$col("F_g_XX")) & (pl$col(dof_ns_t) > 1)
  df <- df$with_columns(
    pl$when(cond_ns)$
      then((pl$col(dof_ns_t) / (pl$col(dof_ns_t) - 1))$sqrt())$
      otherwise(pl$col(DOF_col))$
      alias(DOF_col)
  )

  # sqrt(dof_ns_s_t / (dof_ns_s_t - 1)) for union
  cond_union <- (pl$col(dof_ns_s_t) >= 2) &
    (
      ((pl$col("F_g_XX") - 1 + i == pl$col("time_XX")) & (pl$col(dof_s_t) == 1)) |
      ((pl$col("time_XX") < pl$col("F_g_XX")) & (pl$col(dof_ns_t) == 1))
    )
  df <- df$with_columns(
    pl$when(cond_union)$
      then((pl$col(dof_ns_s_t) / (pl$col(dof_ns_s_t) - 1))$sqrt())$
      otherwise(pl$col(DOF_col))$
      alias(DOF_col)
  )

  return(df)
}

#' Compute placebo effects (polars version)
#' @param df polars DataFrame
#' @param i placebo number
#' @param increase_XX switcher direction indicator
#' @param G_XX number of groups
#' @param t_min_XX minimum time
#' @param T_max_XX maximum time
#' @param trends_nonparam trends_nonparam columns
#' @param cluster cluster variable
#' @param controls controls variables
#' @param levels_d_sq_XX levels of baseline treatment
#' @param same_switchers_pl same_switchers_pl option
#' @param normalized normalized option
#' @param continuous continuous option
#' @return polars DataFrame
#' @noRd
compute_placebo_effects_polars <- function(
    df, i, increase_XX, G_XX, t_min_XX, T_max_XX,
    trends_nonparam, cluster, controls, levels_d_sq_XX,
    same_switchers_pl, normalized, continuous, controls_globals = NULL
) {

  # Drop existing placebo columns
  pl_cols_to_drop <- c(
    paste0("diff_y_pl_", i, "_XX"),
    paste0("U_Gg_pl_", i, "_temp_XX"),
    paste0("U_Gg_placebo_", i, "_XX"),
    paste0("U_Gg_pl_", i, "_temp_var_XX"),
    paste0("U_Gg_pl_", i, "_var_XX"),
    paste0("dist_to_switch_pl_", i, "_XX"),
    paste0("never_change_d_pl_", i, "_XX"),
    paste0("N", increase_XX, "_t_placebo_", i, "_XX"),
    paste0("N", increase_XX, "_t_placebo_", i, "_g_XX"),
    paste0("N_gt_control_placebo_", i, "_XX"),
    paste0("dummy_U_Gg_pl_", i, "_XX")
  )
  df <- pl_batch_drop_cols(df, pl_cols_to_drop)

  # Compute placebo long differences: shift(outcome, 2*i) - shift(outcome, i)
  diff_y_pl_col <- paste0("diff_y_pl_", i, "_XX")
  df <- df$with_columns(
    (pl$col("outcome_XX")$shift(2 * i)$over("group_XX") -
     pl$col("outcome_XX")$shift(i)$over("group_XX"))$alias(diff_y_pl_col)
  )

  # Residualize placebo outcome differences when controls are specified
  # This matches the CRAN package logic at lines 1165-1200 in did_multiplegt_dyn_core.R
  if (!is.null(controls) && length(controls) > 0 && !is.null(controls_globals)) {
    count_controls <- 0L
    for (var in controls) {
      count_controls <- count_controls + 1L
      diff_X_pl_col <- paste0("diff_X", count_controls, "_placebo_", i, "_XX")

      # Compute long difference of control: shift(control, 2*i) - shift(control, i)
      df <- df$with_columns(
        (pl$col(var)$shift(2 * i)$over("group_XX") -
         pl$col(var)$shift(i)$over("group_XX"))$alias(diff_X_pl_col)
      )

      # Compute diff_X_pl_N = N_gt * diff_X_placebo
      diff_X_pl_N_col <- paste0("diff_X", count_controls, "_pl_", i, "_N_XX")
      df <- df$with_columns(
        (pl$col("N_gt_XX") * pl$col(diff_X_pl_col))$alias(diff_X_pl_N_col)
      )

      # Residualize diff_y_pl for each baseline treatment level where useful_res > 1
      for (l in levels_d_sq_XX) {
        l_num <- as.numeric(l)
        useful_res_name <- paste0("useful_res_", l, "_XX")
        coefs_name <- paste0("coefs_sq_", l, "_XX")

        if (useful_res_name %in% names(controls_globals) &&
            controls_globals[[useful_res_name]] > 1 &&
            coefs_name %in% names(controls_globals)) {

          coef_val <- controls_globals[[coefs_name]][count_controls, 1]

          # Subtract control effect from placebo outcome difference for this level
          # diff_y_pl = diff_y_pl - coef * diff_X_placebo (only where d_sq_int_XX == l)
          df <- df$with_columns(
            pl$when(pl$col("d_sq_int_XX") == l_num)$
              then(pl$col(diff_y_pl_col) - coef_val * pl$col(diff_X_pl_col))$
              otherwise(pl$col(diff_y_pl_col))$
              alias(diff_y_pl_col)
          )
        }
      }
    }
  }

  # Identifying controls for placebos
  never_col <- paste0("never_change_d_", i, "_XX")
  never_pl_col <- paste0("never_change_d_pl_", i, "_XX")
  df <- df$with_columns(
    (pl$col(never_col) * pl$col(diff_y_pl_col)$is_not_null()$cast(pl$Float64))$alias(never_pl_col)
  )

  never_pl_w_col <- paste0("never_change_d_pl_", i, "_wXX")
  df <- df$with_columns(
    (pl$col(never_pl_col) * pl$col("N_gt_XX"))$alias(never_pl_w_col)
  )

  # N_gt_control_placebo
  by_cols <- c("time_XX", "d_sq_XX")
  if (!is.null(trends_nonparam) && length(trends_nonparam) > 0) {
    by_cols <- c(by_cols, trends_nonparam)
  }

  N_gt_ctrl_pl_col <- paste0("N_gt_control_placebo_", i, "_XX")
  df <- pl_sum_over(df, never_pl_w_col, by_cols, N_gt_ctrl_pl_col)

  # dist_to_switch_pl
  dist_col <- paste0("distance_to_switch_", i, "_XX")
  dist_pl_col <- paste0("dist_to_switch_pl_", i, "_XX")

  df <- df$with_columns(pl$lit(NA_real_)$alias(dist_pl_col))
  df <- df$with_columns(
    pl$when(pl$col(dist_col)$is_not_null())$
      then(
        pl$col(dist_col) *
        pl$col(diff_y_pl_col)$is_not_null()$cast(pl$Float64) *
        ((pl$col(N_gt_ctrl_pl_col) > 0) & pl$col(N_gt_ctrl_pl_col)$is_not_null())$cast(pl$Float64)
      )$otherwise(pl$lit(NA_real_))$
      alias(dist_pl_col)
  )

  if (isTRUE(same_switchers_pl)) {
    df <- df$with_columns(
      (pl$col(dist_pl_col) * pl$col("fillin_g_pl_XX")$cast(pl$Float64))$alias(dist_pl_col)
    )
  }

  dist_pl_w_col <- paste0("dist_to_switch_pl_", i, "_wXX")
  df <- df$with_columns(
    (pl$col(dist_pl_col) * pl$col("N_gt_XX"))$alias(dist_pl_w_col)
  )

  # N_t_placebo
  N_t_pl_col <- paste0("N", increase_XX, "_t_placebo_", i, "_XX")
  N_dw_t_pl_col <- paste0("N", increase_XX, "_t_placebo_", i, "_dwXX")
  df <- pl_sum_over(df, dist_pl_w_col, "time_XX", N_t_pl_col)
  df <- pl_sum_over(df, dist_pl_col, "time_XX", N_dw_t_pl_col)

  # Compute N_placebo scalar
  filter_expr <- (pl$col("time_XX") >= t_min_XX) & (pl$col("time_XX") <= T_max_XX)
  N_pl_val <- pl_scalar_mean_sum(df, N_t_pl_col, filter_expr, "time_XX")
  assign(paste0("N", increase_XX, "_placebo_", i, "_XX"), N_pl_val, envir = parent.frame())

  N_dw_pl_val <- pl_scalar_mean_sum(df, N_dw_t_pl_col, filter_expr, "time_XX")
  assign(paste0("N", increase_XX, "_dw_placebo_", i, "_XX"), N_dw_pl_val, envir = parent.frame())

  # N_t_placebo_g
  N_t_pl_g_col <- paste0("N", increase_XX, "_t_placebo_", i, "_g_XX")
  df <- pl_sum_over(df, dist_pl_w_col, by_cols, N_t_pl_g_col)

  # Compute M_pl terms for controls adjustment (after N_placebo is computed)
  if (!is.null(controls) && length(controls) > 0 && !is.null(controls_globals)) {
    # Initialize part2_pl_switch column
    part2_pl_col <- paste0("part2_pl_switch", increase_XX, "_", i, "_XX")
    df <- df$with_columns(pl$lit(0.0)$alias(part2_pl_col))

    count_controls <- length(controls)
    control_idx <- 0L
    for (var in controls) {
      control_idx <- control_idx + 1L
      diff_X_pl_N_col <- paste0("diff_X", control_idx, "_pl_", i, "_N_XX")

      for (l in levels_d_sq_XX) {
        l_num <- as.numeric(l)
        useful_res_name <- paste0("useful_res_", l, "_XX")

        # m_pl_g column
        m_pl_g_col <- paste0("m", increase_XX, "_pl_g_", l, "_", control_idx, "_", i, "_XX")

        if (N_pl_val > 0) {
          # Safe division to avoid 0/0 = NaN
          safe_ratio_m_pl <- pl$when(pl$col(N_gt_ctrl_pl_col) == 0)$
            then(pl$lit(0))$
            otherwise(pl$col(N_t_pl_g_col) / pl$col(N_gt_ctrl_pl_col))
          df <- df$with_columns(
            (
              ((pl$lit(i) <= pl$col("T_g_XX") - 2) & (pl$col("d_sq_int_XX") == l_num))$cast(pl$Float64) *
              (G_XX / N_pl_val) *
              (pl$col(dist_pl_col) - safe_ratio_m_pl * pl$col(never_pl_col)) *
              ((pl$col("time_XX") >= i + 1) & (pl$col("time_XX") <= pl$col("T_g_XX")))$cast(pl$Float64) *
              pl$col(diff_X_pl_N_col)
            )$alias(m_pl_g_col)
          )
        } else {
          df <- df$with_columns(pl$lit(0.0)$alias(m_pl_g_col))
        }

        # Sum by group
        m_pl_col <- paste0("m_pl", increase_XX, "_", l, "_", control_idx, "_", i, "_XX")
        df <- pl_sum_over(df, m_pl_g_col, "group_XX", m_pl_col)

        # Set NA where not first_obs
        df <- df$with_columns(
          pl$when(pl$col("first_obs_by_gp_XX") == 1)$
            then(pl$col(m_pl_col))$
            otherwise(pl$lit(NA_real_))$
            alias(m_pl_col)
        )

        # M_pl (scalar mean)
        M_pl_col <- paste0("M_pl", increase_XX, "_", l, "_", control_idx, "_", i, "_XX")
        M_pl_val <- pl_scalar_sum(df, m_pl_col) / G_XX
        df <- df$with_columns(pl$lit(M_pl_val)$alias(M_pl_col))

        # Initialize in_brackets_pl for later use
        if (useful_res_name %in% names(controls_globals) &&
            controls_globals[[useful_res_name]] > 1) {
          in_brackets_pl_col <- paste0("in_brackets_pl_", l, "_", control_idx, "_XX")
          df <- df$with_columns(pl$lit(0.0)$alias(in_brackets_pl_col))
        }
      }
    }
  }

  # DOF computations for placebos
  diff_y_pl_N_col <- paste0("diff_y_pl_", i, "_N_gt_XX")
  dof_ns_pl <- paste0("dof_ns_pl_", i, "_XX")
  dof_s_pl <- paste0("dof_s_pl_", i, "_XX")

  df <- df$with_columns(
    (pl$col(diff_y_pl_col) * pl$col("N_gt_XX"))$alias(diff_y_pl_N_col)
  )

  df <- df$with_columns(
    (
      (pl$col("N_gt_XX") != 0) &
      pl$col(diff_y_pl_col)$is_not_null() &
      (pl$col(never_pl_col) == 1) &
      (pl$col(N_t_pl_col) > 0) &
      pl$col(N_t_pl_col)$is_not_null()
    )$cast(pl$Float64)$alias(dof_ns_pl)
  )

  df <- df$with_columns(
    ((pl$col("N_gt_XX") != 0) & (pl$col(dist_pl_col) == 1))$cast(pl$Float64)$alias(dof_s_pl)
  )
  df <- df$with_columns(
    pl$col(dof_s_pl)$fill_null(0.0)$alias(dof_s_pl)
  )

  # Cohort means for ns
  ns_by_cols <- c("d_sq_XX", "time_XX")
  if (!is.null(trends_nonparam) && length(trends_nonparam) > 0) {
    ns_by_cols <- c(ns_by_cols, trends_nonparam)
  }

  count_ns_col <- paste0("count_cohort_pl_", i, "_ns_t_XX")
  total_ns_col <- paste0("total_cohort_pl_", i, "_ns_t_XX")
  mean_ns_col <- paste0("mean_cohort_pl_", i, "_ns_t_XX")
  dof_coh_ns_col <- paste0("dof_cohort_pl_", i, "_ns_t_XX")

  df <- pl_filtered_agg_over(df, "N_gt_XX", pl$col(dof_ns_pl) == 1, ns_by_cols, count_ns_col, "sum")
  df <- pl_filtered_agg_over(df, diff_y_pl_N_col, pl$col(dof_ns_pl) == 1, ns_by_cols, total_ns_col, "sum")
  df <- df$with_columns((pl$col(total_ns_col) / pl$col(count_ns_col))$alias(mean_ns_col))

  if (is.null(cluster) || cluster == "" || is.na(cluster)) {
    df <- pl_filtered_agg_over(df, dof_ns_pl, pl$col(dof_ns_pl) == 1, ns_by_cols, dof_coh_ns_col, "sum", filter_result = TRUE)
  } else {
    cluster_dof_ns_pl <- paste0("cluster_dof_pl_", i, "_ns_XX")
    df <- df$with_columns(
      pl$when(pl$col(dof_ns_pl) == 1)$then(pl$col(cluster))$otherwise(pl$lit(NA))$alias(cluster_dof_ns_pl)
    )
    df <- pl_uniqueN_over(df, cluster_dof_ns_pl, ns_by_cols, dof_coh_ns_col, pl$col(cluster_dof_ns_pl)$is_not_null())
  }

  # Cohort means for s
  sw_by_cols <- c("d_sq_XX", "F_g_XX", "d_fg_XX")
  if (!is.null(trends_nonparam) && length(trends_nonparam) > 0) {
    sw_by_cols <- c(sw_by_cols, trends_nonparam)
  }

  count_s_col <- paste0("count_cohort_pl_", i, "_s_t_XX")
  total_s_col <- paste0("total_cohort_pl_", i, "_s_t_XX")
  mean_s_col <- paste0("mean_cohort_pl_", i, "_s_t_XX")
  dof_coh_s_col <- paste0("dof_cohort_pl_", i, "_s_t_XX")

  df <- pl_filtered_agg_over(df, "N_gt_XX", pl$col(dof_s_pl) == 1, sw_by_cols, count_s_col, "sum")
  df <- pl_filtered_agg_over(df, diff_y_pl_N_col, pl$col(dof_s_pl) == 1, sw_by_cols, total_s_col, "sum")
  df <- df$with_columns((pl$col(total_s_col) / pl$col(count_s_col))$alias(mean_s_col))

  if (is.null(cluster) || cluster == "" || is.na(cluster)) {
    df <- pl_filtered_agg_over(df, dof_s_pl, pl$col(dof_s_pl) == 1, sw_by_cols, dof_coh_s_col, "sum", filter_result = TRUE)
  } else {
    cluster_dof_s_pl <- paste0("cluster_dof_pl_", i, "_s_XX")
    df <- df$with_columns(
      pl$when(pl$col(dof_s_pl) == 1)$then(pl$col(cluster))$otherwise(pl$lit(NA))$alias(cluster_dof_s_pl)
    )
    df <- pl_uniqueN_over(df, cluster_dof_s_pl, sw_by_cols, dof_coh_s_col, pl$col(cluster_dof_s_pl)$is_not_null())
  }

  # Union of ns and s
  dof_ns_s_pl <- paste0("dof_ns_s_pl_", i, "_XX")
  count_ns_s_col <- paste0("count_cohort_pl_", i, "_ns_s_t_XX")
  total_ns_s_col <- paste0("total_cohort_pl_", i, "_ns_s_t_XX")
  mean_ns_s_col <- paste0("mean_cohort_pl_", i, "_ns_s_t_XX")
  dof_coh_ns_s_col <- paste0("dof_cohort_pl_", i, "_ns_s_t_XX")

  df <- df$with_columns(
    ((pl$col(dof_s_pl) == 1) | (pl$col(dof_ns_pl) == 1))$cast(pl$Float64)$alias(dof_ns_s_pl)
  )
  df <- df$with_columns(
    pl$when(pl$col(dof_s_pl)$is_null() | pl$col(dof_ns_pl)$is_null())$
      then(pl$lit(NA_real_))$otherwise(pl$col(dof_ns_s_pl))$alias(dof_ns_s_pl)
  )

  df <- pl_filtered_agg_over(df, "N_gt_XX", pl$col(dof_ns_s_pl) == 1, ns_by_cols, count_ns_s_col, "sum")
  df <- pl_filtered_agg_over(df, diff_y_pl_N_col, pl$col(dof_ns_s_pl) == 1, ns_by_cols, total_ns_s_col, "sum")
  df <- df$with_columns((pl$col(total_ns_s_col) / pl$col(count_ns_s_col))$alias(mean_ns_s_col))

  if (is.null(cluster) || cluster == "" || is.na(cluster)) {
    df <- pl_filtered_agg_over(df, dof_ns_s_pl, pl$col(dof_ns_s_pl) == 1, ns_by_cols, dof_coh_ns_s_col, "sum", filter_result = TRUE)
  } else {
    cluster_dof_ns_s_pl <- paste0("cluster_dof_pl_", i, "_ns_s_XX")
    df <- df$with_columns(
      pl$when(pl$col(dof_ns_s_pl) == 1)$then(pl$col(cluster))$otherwise(pl$lit(NA))$alias(cluster_dof_ns_s_pl)
    )
    df <- pl_uniqueN_over(df, cluster_dof_ns_s_pl, ns_by_cols, dof_coh_ns_s_col, pl$col(cluster_dof_ns_s_pl)$is_not_null())
  }

  # E_hat and DOF for placebos
  df <- compute_E_hat_gt_polars(df, i, "placebo")
  df <- compute_DOF_gt_polars(df, i, "placebo")

  # U_Gg placebo computation
  dummy_U_pl_col <- paste0("dummy_U_Gg_pl_", i, "_XX")
  df <- df$with_columns(
    (pl$lit(i) <= pl$col("T_g_XX") - 1)$cast(pl$Float64)$alias(dummy_U_pl_col)
  )

  N_pl_val <- get(paste0("N", increase_XX, "_placebo_", i, "_XX"), envir = parent.frame())

  if (!is.null(N_pl_val) && N_pl_val != 0) {
    U_Gg_pl_temp_col <- paste0("U_Gg_pl_", i, "_temp_XX")
    U_Gg_pl_col <- paste0("U_Gg_placebo_", i, "_XX")

    # Safe division to avoid 0/0 = NaN
    safe_ratio_pl <- pl$when(pl$col(N_gt_ctrl_pl_col) == 0)$
      then(pl$lit(0))$
      otherwise(pl$col(N_t_pl_g_col) / pl$col(N_gt_ctrl_pl_col))
    df <- df$with_columns(
      (
        pl$col(dummy_U_pl_col) * (G_XX / N_pl_val) * pl$col("N_gt_XX") *
        (pl$col(dist_pl_col) - safe_ratio_pl * pl$col(never_pl_col)) *
        pl$col(diff_y_pl_col) *
        ((pl$col("time_XX") >= i + 1) & (pl$col("time_XX") <= pl$col("T_g_XX")))$cast(pl$Float64)
      )$alias(U_Gg_pl_temp_col)
    )

    df <- pl_sum_over(df, U_Gg_pl_temp_col, "group_XX", U_Gg_pl_col)
    df <- df$with_columns(
      (pl$col(U_Gg_pl_col) * pl$col("first_obs_by_gp_XX"))$alias(U_Gg_pl_col)
    )

    # count_pl_core
    count_pl_core_col <- paste0("count", i, "_pl_core_XX")
    df <- df$with_columns(
      pl$when(
        (pl$col(U_Gg_pl_temp_col)$is_not_null() & (pl$col(U_Gg_pl_temp_col) != 0)) |
        ((pl$col(U_Gg_pl_temp_col) == 0) & (pl$col(diff_y_pl_col) == 0) &
          ((pl$col(dist_pl_col) != 0) | ((pl$col(N_t_pl_g_col) != 0) & (pl$col(never_pl_col) != 0))))
      )$then(pl$col("N_gt_XX"))$otherwise(pl$lit(0.0))$alias(count_pl_core_col)
    )

    # U_Gg_pl_temp_var - reuse safe_ratio_pl from above
    U_Gg_pl_temp_var_col <- paste0("U_Gg_pl_", i, "_temp_var_XX")
    E_hat_pl_col <- paste0("E_hat_gt_pl_", i, "_XX")
    DOF_pl_col <- paste0("DOF_gt_pl_", i, "_XX")

    df <- df$with_columns(
      (
        pl$col(dummy_U_pl_col) * (G_XX / N_pl_val) *
        (pl$col(dist_pl_col) - safe_ratio_pl * pl$col(never_pl_col)) *
        ((pl$col("time_XX") >= i + 1) & (pl$col("time_XX") <= pl$col("T_g_XX")))$cast(pl$Float64) *
        pl$col("N_gt_XX") * pl$col(DOF_pl_col) *
        (pl$col(diff_y_pl_col) - pl$col(E_hat_pl_col))
      )$alias(U_Gg_pl_temp_var_col)
    )

    # Sum U_Gg_pl_var by group
    U_Gg_pl_var_col <- paste0("U_Gg_pl_", i, "_var_XX")
    df <- pl_sum_over(df, U_Gg_pl_temp_var_col, "group_XX", U_Gg_pl_var_col)

    # Controls adjustment for placebo variance (matches CRAN lines 1427-1448)
    if (!is.null(controls) && length(controls) > 0 && !is.null(controls_globals)) {
      part2_pl_col <- paste0("part2_pl_switch", increase_XX, "_", i, "_XX")
      # Reset part2_pl_switch to 0 before accumulating
      df <- df$with_columns(pl$lit(0.0)$alias(part2_pl_col))

      count_controls <- length(controls)
      for (l in levels_d_sq_XX) {
        # Only include cohort in variance adjustment if useful_res > 1 (matches Stata)
        useful_res_name <- paste0("useful_res_", l, "_XX")
        if (!(useful_res_name %in% names(controls_globals)) ||
            controls_globals[[useful_res_name]] <= 1) next

        l_num <- as.numeric(l)
        comb_pl_col <- paste0("combined_pl", increase_XX, "_temp_", l, "_", i, "_XX")
        df <- df$with_columns(pl$lit(0.0)$alias(comb_pl_col))

        for (j in 1:count_controls) {
          in_brackets_pl_col <- paste0("in_brackets_pl_", l, "_", j, "_XX")
          if (pl_has_col(df, in_brackets_pl_col)) {
            # Reset in_brackets_pl
            df <- df$with_columns(pl$lit(0.0)$alias(in_brackets_pl_col))

            # Add inv_Denom terms
            inv_denom_name <- paste0("inv_Denom_", l, "_XX")
            if (inv_denom_name %in% names(controls_globals)) {
              for (k in 1:count_controls) {
                in_sum_col <- paste0("in_sum_", k, "_", l, "_XX")
                if (pl_has_col(df, in_sum_col)) {
                  inv_denom_jk <- controls_globals[[inv_denom_name]][j, k]
                  df <- df$with_columns(
                    pl$when((pl$col("d_sq_int_XX") == l_num) & (pl$col("F_g_XX") >= 3))$
                      then(pl$col(in_brackets_pl_col) + inv_denom_jk * pl$col(in_sum_col))$
                      otherwise(pl$col(in_brackets_pl_col))$
                      alias(in_brackets_pl_col)
                  )
                }
              }

              # Subtract coef
              coefs_name <- paste0("coefs_sq_", l, "_XX")
              if (coefs_name %in% names(controls_globals)) {
                coef_val <- controls_globals[[coefs_name]][j, 1]
                df <- df$with_columns(
                  (pl$col(in_brackets_pl_col) - coef_val)$alias(in_brackets_pl_col)
                )
              }

              # Add to combined_pl
              M_pl_col <- paste0("M_pl", increase_XX, "_", l, "_", j, "_", i, "_XX")
              if (pl_has_col(df, M_pl_col)) {
                df <- df$with_columns(
                  (pl$col(comb_pl_col) + pl$col(M_pl_col) * pl$col(in_brackets_pl_col))$alias(comb_pl_col)
                )
              }
            }
          }
        }

        # Add to part2_pl_switch
        df <- df$with_columns(
          (pl$col(part2_pl_col) + pl$col(comb_pl_col))$alias(part2_pl_col)
        )
      }

      # Subtract part2_pl_switch from U_Gg_pl_var
      df <- df$with_columns(
        (pl$col(U_Gg_pl_var_col) - pl$col(part2_pl_col))$alias(U_Gg_pl_var_col)
      )
    }

    # Normalized delta for placebos
    if (normalized == TRUE) {
      if (is.null(continuous)) {
        df <- df$with_columns(
          pl$when(
            (pl$col("time_XX") >= pl$col("F_g_XX")) &
            (pl$col("time_XX") <= pl$col("F_g_XX") - 1 + i) &
            (pl$col("S_g_XX") == increase_XX)
          )$then(pl$col("treatment_XX") - pl$col("d_sq_XX"))$
            otherwise(pl$lit(NA_real_))$
            alias("sum_temp_pl_XX")
        )
      } else {
        df <- df$with_columns(
          pl$when(
            (pl$col("time_XX") >= pl$col("F_g_XX")) &
            (pl$col("time_XX") <= pl$col("F_g_XX") - 1 + i) &
            (pl$col("S_g_XX") == increase_XX)
          )$then(pl$col("treatment_XX_orig") - pl$col("d_sq_XX_orig"))$
            otherwise(pl$lit(NA_real_))$
            alias("sum_temp_pl_XX")
        )
      }

      sum_treat_pl_col <- paste0("sum_treat_until_", i, "_pl_XX")
      df <- pl_sum_over(df, "sum_temp_pl_XX", "group_XX", sum_treat_pl_col)
      df <- pl_batch_drop_cols(df, "sum_temp_pl_XX")

      delta_pl_temp_col <- paste0("delta_D_pl_", i, "_cum_temp_XX")
      df <- df$with_columns(
        pl$when(pl$col(dist_pl_col) == 1)$
          then(
            (pl$col("N_gt_XX") / N_pl_val) * (
              pl$col("S_g_XX") * pl$col(sum_treat_pl_col) +
              (1 - pl$col("S_g_XX")) * (-pl$col(sum_treat_pl_col))
            )
          )$otherwise(pl$lit(NA_real_))$
          alias(delta_pl_temp_col)
      )
    }
  }

  return(df)
}
