# Global option to enable Rcpp optimization
# Set options(DID_USE_RCPP = TRUE) to enable Rcpp-optimized computations
.DID_USE_RCPP <- function() {
  return(getOption("DID_USE_RCPP", default = FALSE))
}

#' Internal function of did_multiplegt_dyn that computes U_Gg_plus_XX, U_Gg_minus_XX, U_Gg_var_plus_XX, and U_Gg_var_minus_XX.
#' These are essential variables for the computation of the DID_\ell estimators and their variances.
#' DATA.TABLE VERSION
#' @param df data.table
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
#' @note Uses data.table backend

#' @returns A list containing df (data.table) and const (constants).
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

  list2env(const, envir = environment())

  if (!is.null(controls)) {
    list2env(controls_globals, envir = environment())
  }

  # Ensure df is a data.table
  if (!data.table::is.data.table(df)) {
    df <- data.table::as.data.table(df)
  }

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
  levels_d_sq_XX <- sort(as.character(stats::na.omit(unique(df[["d_sq_int_XX"]]))))

  # Drop columns if they exist
  cols_to_drop <- c("num_g_paths_0_XX", "cohort_fullpath_0_XX")
  dt_batch_drop_cols(df, cols_to_drop)

  # Sort by group and time once
  data.table::setorderv(df, c("group_XX", "time_XX"))

  # Pre-compute all column name strings for the effects loop
  col_names <- lapply(1:l_u_a_XX, function(i) {
    list(
      diff_y = paste0("diff_y_", i, "_XX"),
      never = paste0("never_change_d_", i, "_XX"),
      never_w = paste0("never_change_d_", i, "_wXX"),
      dist = paste0("distance_to_switch_", i, "_XX"),
      dist_w = paste0("distance_to_switch_", i, "_wXX"),
      N_gt_ctrl = paste0("N_gt_control_", i, "_XX"),
      N_t = paste0("N", increase_XX, "_t_", i, "_XX"),
      N_dw_t = paste0("N_dw", increase_XX, "_t_", i, "_XX"),
      N_t_g = paste0("N", increase_XX, "_t_", i, "_g_XX"),
      N_scalar = paste0("N", increase_XX, "_", i, "_XX"),
      N_dw_scalar = paste0("N", increase_XX, "_dw_", i, "_XX"),
      dummy_U = paste0("dummy_U_Gg", i, "_XX"),
      U_Gg_temp = paste0("U_Gg", i, "_temp_XX"),
      U_Gg = paste0("U_Gg", i, "_XX"),
      U_Gg_temp_var = paste0("U_Gg", i, "_temp_var_XX"),
      U_Gg_var = paste0("U_Gg", i, "_var_XX"),
      count_core = paste0("count", i, "_core_XX"),
      diff_y_N = paste0("diff_y_", i, "_N_gt_XX"),
      dof_ns = paste0("dof_ns_", i, "_XX"),
      dof_s = paste0("dof_s_", i, "_XX"),
      dof_ns_s = paste0("dof_ns_s_", i, "_XX"),
      dof_y_N = paste0("dof_y_", i, "_N_gt_XX"),
      E_hat = paste0("E_hat_gt_", i, "_XX"),
      DOF = paste0("DOF_gt_", i, "_XX"),
      count_ns = paste0("count_cohort_", i, "_ns_t_XX"),
      total_ns = paste0("total_cohort_", i, "_ns_t_XX"),
      mean_ns = paste0("mean_cohort_", i, "_ns_t_XX"),
      dof_coh_ns = paste0("dof_cohort_", i, "_ns_t_XX"),
      count_s = paste0("count_cohort_", i, "_s_t_XX"),
      total_s = paste0("total_cohort_", i, "_s_t_XX"),
      mean_s = paste0("mean_cohort_", i, "_s_t_XX"),
      dof_coh_s = paste0("dof_cohort_", i, "_s_t_XX"),
      count_ns_s = paste0("count_cohort_", i, "_ns_s_t_XX"),
      total_ns_s = paste0("total_cohort_", i, "_ns_s_t_XX"),
      mean_ns_s = paste0("mean_cohort_", i, "_ns_s_t_XX"),
      dof_coh_ns_s = paste0("dof_cohort_", i, "_ns_s_t_XX"),
      still_sw = paste0("still_switcher_", i, "_XX"),
      part2 = paste0("part2_switch", increase_XX, "_", i, "_XX"),
      sum_treat = paste0("sum_treat_until_", i, "_XX"),
      delta_temp = paste0("delta_D_", i, "_cum_temp_XX"),
      delta_norm = paste0("delta_norm_", i, "_XX"),
      d_fg = paste0("d_fg", i, "_XX"),
      path = paste0("path_", i, "_XX"),
      num_g_paths = paste0("num_g_paths_", i, "_XX"),
      cohort_fp = paste0("cohort_fullpath_", i, "_XX"),
      count_s0 = paste0("count_cohort_", i, "_s0_t_XX"),
      count_s1 = paste0("count_cohort_", i, "_s1_t_XX"),
      count_s2 = paste0("count_cohort_", i, "_s2_t_XX"),
      total_s0 = paste0("total_cohort_", i, "_s0_t_XX"),
      total_s1 = paste0("total_cohort_", i, "_s1_t_XX"),
      total_s2 = paste0("total_cohort_", i, "_s2_t_XX"),
      dof_s0 = paste0("dof_cohort_", i, "_s0_t_XX"),
      dof_s1 = paste0("dof_cohort_", i, "_s1_t_XX"),
      dof_s2 = paste0("dof_cohort_", i, "_s2_t_XX")
    )
  })

  # Pre-compute cols_to_drop for each iteration
  cols_to_drop_all <- lapply(1:l_u_a_XX, function(i) {
    cn <- col_names[[i]]
    c(
      cn$dist, cn$never, cn$N_t, cn$N_t_g, cn$N_gt_ctrl,
      cn$diff_y, paste0("diff_y_", i, "_XX_temp"),
      cn$dummy_U, cn$U_Gg_temp, cn$U_Gg, cn$count_core,
      paste0("mean_diff_y_", i, "_nd_sq_t_XX"),
      paste0("mean_diff_y_", i, "_d_sq_t_XX"),
      cn$U_Gg_temp_var, cn$U_Gg_var,
      paste0("U_Gg", i, "_var_2_XX"),
      paste0("count_var_", i, "_ntreat_XX_temp"),
      paste0("count_var_", i, "_ntreat_XX"),
      paste0("count_var_", i, "_treat_XX_temp"),
      paste0("count_var_", i, "_treat_XX"),
      paste0("avg_diff_y_", i, "_tnp_XX"),
      paste0("count_diff_y_", i, "_nd_sq_t_XX"),
      paste0("count_diff_y_", i, "_d_sq_t_XX"),
      cn$never_w, cn$dist_w,
      cn$dof_coh_ns, cn$dof_coh_s,
      cn$dof_s0, cn$dof_s1, cn$dof_s2,
      cn$count_ns, cn$count_s,
      cn$count_s0, cn$count_s1, cn$count_s2,
      cn$total_ns, cn$total_s,
      cn$total_s0, cn$total_s1, cn$total_s2,
      cn$mean_ns, cn$mean_s,
      paste0("mean_cohort_", i, "_s0_t_XX"),
      paste0("mean_cohort_", i, "_s1_t_XX"),
      paste0("mean_cohort_", i, "_s2_t_XX")
    )
  })

  # Pre-compute by_cols (constant across iterations)
  by_cols <- c("time_XX", "d_sq_XX")
  if (!is.null(trends_nonparam) && length(trends_nonparam) > 0L) {
    by_cols <- c(by_cols, trends_nonparam)
  }

  # Pre-compute same_switchers checks (independent of i)
  if (same_switchers == TRUE) {
    df[, N_g_control_check_XX := 0.0]
    for (q in 1:effects) {
      df[, diff_y_last_XX := outcome_XX - data.table::shift(outcome_XX, q), by = "group_XX"]
      df[, never_change_d_last_XX := data.table::fifelse(
        !is.na(diff_y_last_XX) & F_g_XX > time_XX, 1.0, NA_real_
      )]
      if (isTRUE(only_never_switchers)) {
        df[F_g_XX > time_XX & F_g_XX < T_max_XX + 1L & !is.na(diff_y_last_XX),
            never_change_d_last_XX := 0.0]
      }
      df[, `__temp_weighted__` := never_change_d_last_XX * N_gt_XX]
      df[, N_gt_control_last_XX := sum(get("__temp_weighted__"), na.rm = TRUE), by = by_cols]
      df[, `__temp_weighted__` := NULL]
      df[, `__temp_ctrl__` := data.table::fifelse(time_XX == F_g_XX - 1L + q, N_gt_control_last_XX, NA_real_)]
      df[, N_g_control_last_m_XX := mean(get("__temp_ctrl__"), na.rm = TRUE), by = "group_XX"]
      df[, `__temp_ctrl__` := NULL]
      df[, `__temp_diff__` := data.table::fifelse(time_XX == F_g_XX - 1L + q, diff_y_last_XX, NA_real_)]
      df[, diff_y_relev_XX := mean(get("__temp_diff__"), na.rm = TRUE), by = "group_XX"]
      df[, `__temp_diff__` := NULL]
      df[, N_g_control_check_XX := N_g_control_check_XX +
          as.numeric(N_g_control_last_m_XX > 0L & !is.na(diff_y_relev_XX))]
    }
    dt_batch_drop_cols(df, c("diff_y_last_XX", "never_change_d_last_XX",
      "N_gt_control_last_XX", "N_g_control_last_m_XX", "diff_y_relev_XX"))
    # Pre-compute still_switcher condition (same for all i)
    df[, `__still_sw_XX__` := (F_g_XX - 1L + effects <= T_g_XX) & (N_g_control_check_XX == effects)]

    if (same_switchers_pl == TRUE && placebo > 0L) {
      df[, N_g_control_check_pl_XX := 0.0]
      for (q in seq_len(placebo)) {
        df[, diff_y_last_XX := outcome_XX - data.table::shift(outcome_XX, -q), by = "group_XX"]
        df[, never_change_d_last_XX := data.table::fifelse(
          !is.na(diff_y_last_XX) & F_g_XX > time_XX, 1.0, NA_real_
        )]
        if (isTRUE(only_never_switchers)) {
          df[F_g_XX > time_XX & F_g_XX < T_max_XX + 1L & !is.na(diff_y_last_XX),
              never_change_d_last_XX := 0.0]
        }
        df[, `__temp_weighted__` := never_change_d_last_XX * N_gt_XX]
        df[, N_gt_control_last_XX := sum(get("__temp_weighted__"), na.rm = TRUE), by = by_cols]
        df[, `__temp_weighted__` := NULL]
        df[, `__temp_ctrl__` := data.table::fifelse(time_XX == F_g_XX - 1L - q, N_gt_control_last_XX, NA_real_)]
        df[, N_g_control_last_m_XX := mean(get("__temp_ctrl__"), na.rm = TRUE), by = "group_XX"]
        df[, `__temp_ctrl__` := NULL]
        df[, `__temp_diff__` := data.table::fifelse(time_XX == F_g_XX - 1L - q, diff_y_last_XX, NA_real_)]
        df[, diff_y_relev_XX := mean(get("__temp_diff__"), na.rm = TRUE), by = "group_XX"]
        df[, `__temp_diff__` := NULL]
        df[, N_g_control_check_pl_XX := N_g_control_check_pl_XX +
            as.numeric(N_g_control_last_m_XX > 0L & !is.na(diff_y_relev_XX))]
      }
      dt_batch_drop_cols(df, c("diff_y_last_XX", "never_change_d_last_XX",
        "N_gt_control_last_XX", "N_g_control_last_m_XX", "diff_y_relev_XX"))
      df[, fillin_g_pl_XX := (N_g_control_check_pl_XX == placebo)]
    }
  }

  ####### 2. Data preparation - Loop over effects
  for (i in 1:l_u_a_XX) {
    cn <- col_names[[i]]

    # Drop columns for this iteration
    cols_to_drop_i <- cols_to_drop_all[[i]]
    dt_batch_drop_cols(df, cols_to_drop_i)

    # Create long difference of outcome
    diff_y_col <- cn$diff_y
    df[, (diff_y_col) := outcome_XX - data.table::shift(outcome_XX, i), by = "group_XX"]

    # Creating treatment paths if less_conservative_se option specified
    if (isTRUE(less_conservative_se)) {
      # d_fg_XX_temp: treatment at F_g-1+i
      df[, d_fg_XX_temp := data.table::fifelse(time_XX == F_g_XX + i - 1L, treatment_XX, NA_real_)]

      # Group mean of d_fg_XX_temp
      d_fg_col <- cn$d_fg
      df[, (d_fg_col) := mean(get("d_fg_XX_temp"), na.rm = TRUE), by = "group_XX"]

      if (i == 1L) {
        df[, d_fg0_XX := d_sq_XX]
        df[, path_0_XX := .GRP, by = c("d_fg0_XX", "F_g_XX")]
      }

      # Fill missing d_fg with previous value
      prev_d_fg <- paste0("d_fg", i - 1L, "_XX")
      df[, (d_fg_col) := data.table::fifelse(is.na(get(d_fg_col)), get(prev_d_fg), get(d_fg_col))]

      # Create path variable
      prev_path <- if (i == 1L) "path_0_XX" else col_names[[i - 1L]]$path
      path_col <- cn$path
      df[, (path_col) := .GRP, by = c(prev_path, d_fg_col)]

      dt_batch_drop_cols(df, "d_fg_XX_temp")

      # Count groups per path
      if (i == 1L) {
        dt_uniqueN_over(df, "group_XX", "path_0_XX", "num_g_paths_0_XX")
        df[, cohort_fullpath_0_XX := as.numeric(num_g_paths_0_XX > 1)]
      }

      num_g_col <- cn$num_g_paths
      cohort_col <- cn$cohort_fp
      dt_uniqueN_over(df, "group_XX", path_col, num_g_col)
      df[, (cohort_col) := as.numeric(get(num_g_col) > 1L)]
    }

    # Identifying control (g,t)s
    never_col <- cn$never
    df[, (never_col) := as.numeric(F_g_XX > time_XX)]

    # Set to NA where diff_y is NA
    df[is.na(get(diff_y_col)), (never_col) := NA_real_]

    if (isTRUE(only_never_switchers)) {
      df[F_g_XX > time_XX & F_g_XX < T_max_XX + 1L & !is.na(get(diff_y_col)),
          (never_col) := 0.0]
    }

    # Creating N_gt_control: weighted sum of never_change
    never_w_col <- cn$never_w
    df[, (never_w_col) := get(never_col) * N_gt_XX]

    N_gt_ctrl_col <- cn$N_gt_ctrl
    df[, (N_gt_ctrl_col) := sum(get(never_w_col), na.rm = TRUE), by = by_cols]

    # Same switchers: use pre-computed __still_sw_XX__
    if (same_switchers == TRUE) {
      dist_col <- cn$dist
      df[, (dist_col) := data.table::fifelse(
        !is.na(get(diff_y_col)),
        as.numeric(
          `__still_sw_XX__` == TRUE &
          time_XX == F_g_XX - 1L + i &
          i <= L_g_XX &
          S_g_XX == increase_XX &
          get(N_gt_ctrl_col) > 0L &
          !is.na(get(N_gt_ctrl_col))
        ),
        NA_real_
      )]
    } else {
      # Without same_switchers option
      dist_col <- cn$dist
      df[, (dist_col) := NA_real_]

      df[, (dist_col) := data.table::fifelse(
        !is.na(get(diff_y_col)),
        as.numeric(
          time_XX == F_g_XX - 1L + i &
          i <= L_g_XX &
          S_g_XX == increase_XX &
          get(N_gt_ctrl_col) > 0L &
          !is.na(get(N_gt_ctrl_col))
        ),
        NA_real_
      )]
    }

    # distance_to_switch weighted
    dist_w_col <- cn$dist_w
    df[, (dist_w_col) := get(dist_col) * N_gt_XX]

    # N_t columns
    N_t_col <- cn$N_t
    N_dw_t_col <- cn$N_dw_t
    df[, (N_t_col) := sum(get(dist_w_col), na.rm = TRUE), by = "time_XX"]
    df[, (N_dw_t_col) := sum(get(dist_col), na.rm = TRUE), by = "time_XX"]

    # Compute N_increase_i scalar
    filter_idx <- df[["time_XX"]] >= t_min_XX & df[["time_XX"]] <= T_max_XX
    N_val <- dt_scalar_mean_sum(df, N_t_col, filter_idx, "time_XX")
    assign(cn$N_scalar, N_val)

    N_dw_val <- dt_scalar_mean_sum(df, N_dw_t_col, filter_idx, "time_XX")
    assign(cn$N_dw_scalar, N_dw_val)

    # N_t_g: by time, d_sq, trends_nonparam
    N_t_g_col <- cn$N_t_g
    df[, (N_t_g_col) := sum(get(dist_w_col), na.rm = TRUE), by = by_cols]

    # Controls adjustment
    if (!is.null(controls)) {
      part2_col <- cn$part2
      df[, (part2_col) := 0.0]

      # T_d_XX: max F_g by d_sq_int_XX
      df[, T_d_XX := max(F_g_XX, na.rm = TRUE), by = "d_sq_int_XX"]
      df[, T_d_XX := T_d_XX - 1]

      count_controls <- length(controls)
      N_inc_val <- get(cn$N_scalar)

      # --- Phase 1: Compute all diff_X and diff_X_N columns ---
      # Vectorized shift: global lag + boundary fixup (avoids C by-group dispatches)
      diff_X_cols <- paste0("diff_X", 1:count_controls, "_", i, "_XX")
      diff_X_N_cols <- paste0("diff_X", 1:count_controls, "_", i, "_N_XX")
      N_rows_core <- nrow(df)
      fix_rows_i <- dt_shift_boundary_rows(df[["group_XX"]], i, N_rows_core)
      N_gt_vec_core <- df[["N_gt_XX"]]
      for (j in seq_len(count_controls)) {
        ctrl_vec <- df[[controls[j]]]
        shifted_vec <- dt_vec_shift(ctrl_vec, i, fix_rows_i, N_rows_core)
        diff_vec <- ctrl_vec - shifted_vec
        data.table::set(df, j = diff_X_cols[j], value = diff_vec)
        data.table::set(df, j = diff_X_N_cols[j], value = N_gt_vec_core * diff_vec)
      }

      # --- Phase 2: For each baseline level, compute shared quantities once, then batch ---
      for (l in levels_d_sq_XX) {
        l_num <- as.numeric(l)

        # Shared: safe_ratio (same for all controls)
        safe_ratio <- data.table::fifelse(df[[N_gt_ctrl_col]] == 0, 0, df[[N_t_g_col]] / df[[N_gt_ctrl_col]])

        # Shared: common_m factor for m_g computation
        common_m <- as.numeric(df[["d_sq_int_XX"]] == l_num & i <= df[["T_g_XX"]] - 2) *
          (G_XX / N_inc_val) *
          (df[[dist_col]] - safe_ratio * df[[never_col]]) *
          as.numeric(df[["time_XX"]] >= i + 1L & df[["time_XX"]] <= df[["T_g_XX"]])

        # Compute all m_g columns: m_g_j = common_m * diff_X_N_j
        m_g_cols <- paste0("m", increase_XX, "_g_", l, "_", 1:count_controls, "_", i, "_XX")
        for (j in seq_len(count_controls)) {
          data.table::set(df, j = m_g_cols[j], value = common_m * df[[diff_X_N_cols[j]]])
        }

        # Batch by-group sum for all m columns at once
        m_cols <- paste0("m", increase_XX, "_", l, "_", 1:count_controls, "_", i, "_XX")
        df[, (m_cols) := lapply(.SD, sum, na.rm = TRUE), by = "group_XX", .SDcols = m_g_cols]

        # Set NA where not first_obs (batch)
        not_first_idx <- which(df[["first_obs_by_gp_XX"]] != 1L)
        if (length(not_first_idx) > 0L) {
          for (j in seq_len(count_controls)) {
            data.table::set(df, i = not_first_idx, j = m_cols[j], value = NA_real_)
          }
        }

        # Compute all M scalars
        M_cols <- paste0("M", increase_XX, "_", l, "_", 1:count_controls, "_", i, "_XX")
        for (j in seq_len(count_controls)) {
          M_val <- sum(df[[m_cols[j]]], na.rm = TRUE) / G_XX
          data.table::set(df, j = M_cols[j], value = M_val)
        }

        # Shared: E_hat_denom (same for all controls — compute once)
        df[, dummy_XX := data.table::fifelse(
          F_g_XX > time_XX & d_sq_int_XX == l_num & !is.na(diff_y_XX),
          1.0, 0.0
        )]
        E_hat_denom_shared_col <- paste0("E_hat_denom_shared_", l, "_XX")
        df[, (E_hat_denom_shared_col) := sum(get("dummy_XX"), na.rm = TRUE), by = c("time_XX", "d_sq_int_XX")]
        df[d_sq_int_XX != l_num, (E_hat_denom_shared_col) := NA_real_]

        # Also write per-control E_hat_denom columns (in case referenced by name elsewhere)
        for (j in seq_len(count_controls)) {
          E_hat_denom_col <- paste0("E_hat_denom_", j, "_", l, "_XX")
          data.table::set(df, j = E_hat_denom_col, value = df[[E_hat_denom_shared_col]])
        }

        # Shared: E_y_hat_gt
        E_y_hat_col <- paste0("E_y_hat_gt_", l, "_XX")
        E_y_hat_int_col <- paste0("E_y_hat_gt_int_", l, "_XX")
        df[, (E_y_hat_col) := get(E_y_hat_int_col) * as.numeric(get(E_hat_denom_shared_col) >= 2)]

        # Shared: N_c
        N_c_temp_col <- paste0("N_c_", l, "_temp_XX")
        N_c_col <- paste0("N_c_", l, "_XX")
        df[, (N_c_temp_col) :=
          N_gt_XX *
          as.numeric(d_sq_int_XX == l_num & time_XX >= 2L &
            time_XX <= T_d_XX & time_XX < F_g_XX & !is.na(diff_y_XX))
        ]
        N_c_val <- dt_scalar_sum(df, N_c_temp_col)
        data.table::set(df, j = N_c_col, value = N_c_val)

        # Shared: in_sum_adj and common_insumtemp factor
        E_hat_denom_vec <- df[[E_hat_denom_shared_col]]
        in_sum_adj_vec <- data.table::fifelse(
          E_hat_denom_vec > 1L,
          sqrt(E_hat_denom_vec / (E_hat_denom_vec - 1L)) - 1,
          0.0
        )
        common_insumtemp <- (1.0 + as.numeric(E_hat_denom_vec >= 2L) * in_sum_adj_vec) *
          (df[["diff_y_XX"]] - df[[E_y_hat_col]]) *
          as.numeric(df[["time_XX"]] >= 2L & df[["time_XX"]] <= df[["F_g_XX"]] - 1L) /
          N_c_val

        # Compute all in_sum_temp columns: in_sum_temp_j = prod_X_j * common_insumtemp
        in_sum_temp_cols <- paste0("in_sum_temp_", 1:count_controls, "_", l, "_XX")
        for (j in seq_len(count_controls)) {
          prod_X_col <- paste0("prod_X", j, "_Ngt_XX")
          data.table::set(df, j = in_sum_temp_cols[j], value = df[[prod_X_col]] * common_insumtemp)
        }

        # Also write per-control in_sum_adj columns (for consistency)
        for (j in seq_len(count_controls)) {
          in_sum_adj_col <- paste0("in_sum_temp_adj_", j, "_", l, "_XX")
          data.table::set(df, j = in_sum_adj_col, value = in_sum_adj_vec)
        }

        # Batch by-group sum for all in_sum columns at once
        in_sum_cols <- paste0("in_sum_", 1:count_controls, "_", l, "_XX")
        df[, (in_sum_cols) := lapply(.SD, sum, na.rm = TRUE), by = "group_XX", .SDcols = in_sum_temp_cols]

        # Residualize outcome sequentially (preserves bit-exact results)
        useful_res_val <- get(paste0("useful_res_", l, "_XX"))
        if (!is.null(useful_res_val) && useful_res_val > 1L) {
          for (j in seq_len(count_controls)) {
            coefs_val <- get(paste0("coefs_sq_", l, "_XX"))[j, 1L]

            df[, (diff_y_col) := data.table::fifelse(
              d_sq_int_XX == l_num,
              get(diff_y_col) - coefs_val * get(diff_X_cols[j]),
              get(diff_y_col)
            )]

            in_brackets_col <- paste0("in_brackets_", l, "_", j, "_XX")
            df[, (in_brackets_col) := 0.0]
          }
        }
      }

      # Clean up intermediate controls columns to prevent data.table bloat
      # Keep: M_cols, in_sum_cols, in_brackets_cols (needed for variance adjustment)
      # Drop: m_g, m, diff_X_N, in_sum_temp, E_hat_denom, in_sum_adj
      drop_cols_ctrl <- c(
        m_g_cols, m_cols, diff_X_N_cols, in_sum_temp_cols, diff_X_cols,
        paste0("E_hat_denom_shared_", levels_d_sq_XX, "_XX"),
        paste0("N_c_", levels_d_sq_XX, "_temp_XX")
      )
      for (l in levels_d_sq_XX) {
        drop_cols_ctrl <- c(drop_cols_ctrl,
          paste0("E_hat_denom_", 1:count_controls, "_", l, "_XX"),
          paste0("in_sum_temp_adj_", 1:count_controls, "_", l, "_XX"))
      }
      dt_batch_drop_cols(df, drop_cols_ctrl)
    }

    # DOF and mean computations for variance
    diff_y_N_col <- cn$diff_y_N
    dof_ns_col <- cn$dof_ns
    dof_s_col <- cn$dof_s

    df[, (diff_y_N_col) := N_gt_XX * get(diff_y_col)]

    # dof_ns: indicator for controls
    df[, (dof_ns_col) := as.numeric(
      !is.na(N_gt_XX) & (N_gt_XX != 0L) &
      !is.na(get(diff_y_col)) &
      !is.na(get(never_col)) & (get(never_col) == 1L) &
      !is.na(get(N_t_col)) & (get(N_t_col) > 0L)
    )]

    # dof_s: indicator for switchers
    df[, (dof_s_col) := as.numeric(
      !is.na(N_gt_XX) & (N_gt_XX != 0L) &
      !is.na(get(dist_col)) & (get(dist_col) == 1L)
    )]

    # Grouped totals for controls
    ns_by_cols <- c("d_sq_XX")
    if (!is.null(trends_nonparam) && length(trends_nonparam) > 0L) {
      ns_by_cols <- c(ns_by_cols, trends_nonparam)
    }
    ns_by_cols <- c(ns_by_cols, "time_XX")

    count_ns_col <- cn$count_ns
    total_ns_col <- cn$total_ns
    mean_ns_col <- cn$mean_ns
    dof_coh_ns_col <- cn$dof_coh_ns

    # Filtered aggregations
    dt_filtered_agg_over(df, "N_gt_XX", df[[dof_ns_col]] == 1L, ns_by_cols, count_ns_col, "sum")
    dt_filtered_agg_over(df, diff_y_N_col, df[[dof_ns_col]] == 1L, ns_by_cols, total_ns_col, "sum")
    df[, (mean_ns_col) := get(total_ns_col) / get(count_ns_col)]

    # DOF counting
    if (is.null(cluster) || cluster == "" || is.na(cluster)) {
      dt_filtered_agg_over(df, dof_ns_col, df[[dof_ns_col]] == 1L, ns_by_cols, dof_coh_ns_col, "sum", filter_result = TRUE)
    } else {
      cluster_dof_col <- paste0("cluster_dof_", i, "_ns_XX")
      df[, (cluster_dof_col) := data.table::fifelse(get(dof_ns_col) == 1L, get(cluster), NA)]
      dt_uniqueN_over(df, cluster_dof_col, ns_by_cols, dof_coh_ns_col, !is.na(df[[cluster_dof_col]]))
    }

    # Diff_y * N_gt and dof indicator
    dof_y_N_col <- cn$dof_y_N
    df[, (dof_y_N_col) := as.numeric((N_gt_XX != 0) & !is.na(get(diff_y_col)))]

    # Switchers cohort demeaning
    if (isFALSE(less_conservative_se)) {
      sw_by_cols <- c("d_sq_XX", "F_g_XX", "d_fg_XX")
      if (!is.null(trends_nonparam) && length(trends_nonparam) > 0L) {
        sw_by_cols <- c(sw_by_cols, trends_nonparam)
      }

      count_s_col <- cn$count_s
      total_s_col <- cn$total_s
      mean_s_col <- cn$mean_s
      dof_coh_s_col <- cn$dof_coh_s

      dt_filtered_agg_over(df, "N_gt_XX", df[[dof_s_col]] == 1L, sw_by_cols, count_s_col, "sum")
      dt_filtered_agg_over(df, diff_y_N_col, df[[dof_s_col]] == 1L, sw_by_cols, total_s_col, "sum")
      df[, (mean_s_col) := get(total_s_col) / get(count_s_col)]

      if (is.null(cluster) || cluster == "" || is.na(cluster)) {
        dt_filtered_agg_over(df, dof_s_col, df[[dof_s_col]] == 1L, sw_by_cols, dof_coh_s_col, "sum", filter_result = TRUE)
      } else {
        cluster_dof_s_col <- paste0("cluster_dof_", i, "_s_XX")
        df[, (cluster_dof_s_col) := data.table::fifelse(get(dof_s_col) == 1L, get(cluster), NA)]
        dt_uniqueN_over(df, cluster_dof_s_col, sw_by_cols, dof_coh_s_col, !is.na(df[[cluster_dof_s_col]]))
      }
    } else {
      # less_conservative_se path columns
      path_0_col <- "path_0_XX"
      path_1_col <- "path_1_XX"
      path_i_col <- cn$path

      for (suffix in c("s0", "s1", "s2")) {
        path_col <- switch(suffix,
          "s0" = path_0_col,
          "s1" = path_1_col,
          "s2" = path_i_col
        )

        by_cols_path <- c(path_col)
        if (!is.null(trends_nonparam) && length(trends_nonparam) > 0L) {
          by_cols_path <- c(by_cols_path, trends_nonparam)
        }

        count_col <- cn[[paste0("count_", suffix)]]
        total_col <- cn[[paste0("total_", suffix)]]
        dof_col <- cn[[paste0("dof_", suffix)]]

        dt_filtered_agg_over(df, "N_gt_XX", df[[dist_col]] == 1L, by_cols_path, count_col, "sum")
        df[get(dist_col) != 1L, (count_col) := NA_real_]

        dt_filtered_agg_over(df, diff_y_N_col, df[[dist_col]] == 1L, by_cols_path, total_col, "sum")
        df[get(dist_col) != 1L, (total_col) := NA_real_]

        dt_filtered_agg_over(df, dof_y_N_col, df[[dist_col]] == 1L, by_cols_path, dof_col, "sum")
        df[get(dist_col) != 1L, (dof_col) := NA_real_]
      }

      # Compute mean based on cohort hierarchy
      mean_s_col <- cn$mean_s
      dof_coh_s_col <- cn$dof_coh_s
      cohort_i_col <- cn$cohort_fp

      count_s0 <- cn$count_s0
      count_s1 <- cn$count_s1
      count_s2 <- cn$count_s2
      total_s0 <- cn$total_s0
      total_s1 <- cn$total_s1
      total_s2 <- cn$total_s2
      dof_s0 <- cn$dof_s0
      dof_s1 <- cn$dof_s1
      dof_s2 <- cn$dof_s2

      df[, (mean_s_col) := data.table::fifelse(
        get(cohort_i_col) == 1L,
        get(total_s2) / get(count_s2),
        NA_real_
      )]

      df[, (mean_s_col) := data.table::fifelse(
        (get(cohort_i_col) == 0L) & (cohort_fullpath_1_XX == 1L),
        get(total_s1) / get(count_s1),
        get(mean_s_col)
      )]

      df[, (mean_s_col) := data.table::fifelse(
        cohort_fullpath_1_XX == 0L,
        get(total_s0) / get(count_s0),
        get(mean_s_col)
      )]

      # DOF cohort
      df[, (dof_coh_s_col) := data.table::fifelse(
        get(cohort_i_col) == 1L,
        get(dof_s2),
        NA_real_
      )]

      df[, (dof_coh_s_col) := data.table::fifelse(
        (get(cohort_i_col) == 0L) & (cohort_fullpath_1_XX == 1L),
        get(dof_s1),
        get(dof_coh_s_col)
      )]

      df[, (dof_coh_s_col) := data.table::fifelse(
        cohort_fullpath_1_XX == 0L,
        get(dof_s0),
        get(dof_coh_s_col)
      )]
    }

    # Union of switchers and not-yet switchers (ns_s)
    dof_ns_s_col <- cn$dof_ns_s
    count_ns_s_col <- cn$count_ns_s
    total_ns_s_col <- cn$total_ns_s
    mean_ns_s_col <- cn$mean_ns_s
    dof_coh_ns_s_col <- cn$dof_coh_ns_s

    df[, (dof_ns_s_col) := as.numeric((get(dof_s_col) == 1) | (get(dof_ns_col) == 1))]

    dt_filtered_agg_over(df, "N_gt_XX", df[[dof_ns_s_col]] == 1L, ns_by_cols, count_ns_s_col, "sum")
    dt_filtered_agg_over(df, diff_y_N_col, df[[dof_ns_s_col]] == 1L, ns_by_cols, total_ns_s_col, "sum")
    df[, (mean_ns_s_col) := get(total_ns_s_col) / get(count_ns_s_col)]

    if (is.null(cluster) || cluster == "" || is.na(cluster)) {
      dt_filtered_agg_over(df, dof_ns_s_col, df[[dof_ns_s_col]] == 1L, ns_by_cols, dof_coh_ns_s_col, "sum", filter_result = TRUE)
    } else {
      cluster_dof_ns_s_col <- paste0("cluster_dof_", i, "_ns_s_XX")
      df[, (cluster_dof_ns_s_col) := data.table::fifelse(get(dof_ns_s_col) == 1L, get(cluster), NA)]
      dt_uniqueN_over(df, cluster_dof_ns_s_col, ns_by_cols, dof_coh_ns_s_col, !is.na(df[[cluster_dof_ns_s_col]]))
    }

    # Compute E_hat_gt and DOF_gt
    df <- compute_E_hat_gt_dt(df, i, "effect")
    df <- compute_DOF_gt_dt(df, i, "effect")

    ###### 3. Computing U_Gg_l variables
    N_inc_val <- get(cn$N_scalar)

    if (!is.null(N_inc_val) && N_inc_val != 0L) {
      dummy_U_col <- cn$dummy_U
      df[, (dummy_U_col) := as.numeric(i <= T_g_XX - 1)]

      # Extract frequently-used columns as local vectors for speed
      dist_vec <- df[[dist_col]]
      never_vec <- df[[never_col]]
      diff_y_vec <- df[[diff_y_col]]
      dummy_U_vec <- df[[dummy_U_col]]
      N_gt_ctrl_vec <- df[[N_gt_ctrl_col]]
      N_t_g_vec <- df[[N_t_g_col]]

      # U_Gg_temp
      U_Gg_temp_col <- cn$U_Gg_temp
      safe_ratio <- data.table::fifelse(N_gt_ctrl_vec == 0L, 0, N_t_g_vec / N_gt_ctrl_vec)
      U_Gg_temp_vals <- dummy_U_vec * (G_XX / N_inc_val) *
        as.numeric(df[["time_XX"]] >= i + 1L & df[["time_XX"]] <= df[["T_g_XX"]]) *
        df[["N_gt_XX"]] *
        (dist_vec - safe_ratio * never_vec) *
        diff_y_vec
      data.table::set(df, j = U_Gg_temp_col, value = U_Gg_temp_vals)

      # U_Gg: sum by group
      U_Gg_col <- cn$U_Gg
      df[, (U_Gg_col) := sum(get(U_Gg_temp_col), na.rm = TRUE), by = "group_XX"]
      df[, (U_Gg_col) := get(U_Gg_col) * first_obs_by_gp_XX]

      # count_core
      count_core_col <- cn$count_core
      df[, (count_core_col) := data.table::fifelse(
        (!is.na(U_Gg_temp_vals) & (U_Gg_temp_vals != 0L)) |
        ((U_Gg_temp_vals == 0L) & (diff_y_vec == 0L) &
          ((dist_vec != 0L) | ((N_t_g_vec != 0L) & (never_vec != 0L)))),
        N_gt_XX, 0.0
      )]

      # U_Gg_temp_var
      U_Gg_temp_var_col <- cn$U_Gg_temp_var
      E_hat_col <- cn$E_hat
      DOF_col <- cn$DOF

      # Re-use safe_ratio and local vectors from U_Gg_temp computation
      U_Gg_temp_var_vals <- dummy_U_vec * (G_XX / N_inc_val) *
        (dist_vec - safe_ratio * never_vec) *
        as.numeric(df[["time_XX"]] >= i + 1L & df[["time_XX"]] <= df[["T_g_XX"]]) *
        df[["N_gt_XX"]] * df[[DOF_col]] *
        (diff_y_vec - df[[E_hat_col]])
      data.table::set(df, j = U_Gg_temp_var_col, value = U_Gg_temp_var_vals)

      # Controls adjustment for variance (matrix-vectorized)
      if (!is.null(controls)) {
        for (l in levels_d_sq_XX) {
          useful_res_val <- get(paste0("useful_res_", l, "_XX"))
          if (is.null(useful_res_val) || useful_res_val <= 1) next

          l_num <- as.numeric(l)
          inv_Denom_l <- get(paste0("inv_Denom_", l, "_XX"))
          coefs_l <- get(paste0("coefs_sq_", l, "_XX"))[, 1]
          mask_vec <- as.numeric(df[["d_sq_int_XX"]] == l_num & df[["F_g_XX"]] >= 3)

          # Extract in_sum columns into N x C matrix
          in_sum_cols <- paste0("in_sum_", 1:count_controls, "_", l, "_XX")
          in_sum_mat <- as.matrix(df[, ..in_sum_cols])

          # in_brackets = (in_sum * mask) %*% t(inv_Denom) - coefs (broadcast)
          in_brackets_mat <- (in_sum_mat * mask_vec) %*% t(inv_Denom_l)
          in_brackets_mat <- t(t(in_brackets_mat) - coefs_l)

          # Extract M columns into N x C matrix
          M_cols <- paste0("M", increase_XX, "_", l, "_", 1:count_controls, "_", i, "_XX")
          M_mat <- as.matrix(df[, ..M_cols])

          # comb = rowSums(M * in_brackets), add to part2
          part2_col <- cn$part2
          data.table::set(df, j = part2_col,
            value = df[[part2_col]] + rowSums(M_mat * in_brackets_mat))

          # Write back in_brackets columns (needed for later covariance computations)
          for (j in 1:count_controls) {
            in_brackets_col <- paste0("in_brackets_", l, "_", j, "_XX")
            data.table::set(df, j = in_brackets_col, value = in_brackets_mat[, j])
          }
        }
      }

      # Sum U_Gg_var by group
      U_Gg_var_col <- cn$U_Gg_var
      df[, (U_Gg_var_col) := sum(get(U_Gg_temp_var_col), na.rm = TRUE), by = "group_XX"]

      if (!is.null(controls)) {
        part2_col <- cn$part2
        df[, (U_Gg_var_col) := get(U_Gg_var_col) - get(part2_col)]
      }
    }

    ###### 4. Normalized option
    if (normalized == TRUE) {
      if (is.null(continuous)) {
        df[, sum_temp_XX := data.table::fifelse(
          time_XX >= F_g_XX &
          time_XX <= F_g_XX - 1L + i &
          S_g_XX == increase_XX,
          treatment_XX - d_sq_XX,
          NA_real_
        )]
      } else {
        df[, sum_temp_XX := data.table::fifelse(
          time_XX >= F_g_XX &
          time_XX <= F_g_XX - 1L + i &
          S_g_XX == increase_XX,
          treatment_XX_orig - d_sq_XX_orig,
          NA_real_
        )]
      }

      sum_treat_col <- cn$sum_treat
      df[, (sum_treat_col) := sum(get("sum_temp_XX"), na.rm = TRUE), by = "group_XX"]
      dt_batch_drop_cols(df, "sum_temp_XX")

      delta_temp_col <- cn$delta_temp
      N_inc_val <- get(cn$N_scalar)

      df[, (delta_temp_col) := data.table::fifelse(
        get(dist_col) == 1L,
        (N_gt_XX / N_inc_val) * (
          S_g_XX * get(sum_treat_col) +
          (1 - S_g_XX) * (-get(sum_treat_col))
        ),
        NA_real_
      )]

      delta_val <- dt_scalar_sum(df, delta_temp_col)
      assign(cn$delta_norm, delta_val)
    }
  }

  # Clean up pre-computed same_switchers columns
  if (same_switchers == TRUE) {
    dt_batch_drop_cols(df, c("__still_sw_XX__", "N_g_control_check_XX"))
  }

  # Trends_lin option
  Ntrendslin <- 1L
  for (i in 1:l_u_a_XX) {
    Ntrendslin <- min(Ntrendslin, get(col_names[[i]]$N_scalar), na.rm = TRUE)
  }

  if (isTRUE(trends_lin) && Ntrendslin != 0L) {
    lu <- as.integer(l_u_a_XX)

    col_TL <- sprintf("U_Gg%d_TL", lu)
    col_var_TL <- sprintf("U_Gg%d_var_TL", lu)
    col_XX <- sprintf("U_Gg%d_XX", lu)
    col_var_XX <- sprintf("U_Gg%d_var_XX", lu)

    dt_batch_drop_cols(df, c(col_TL, col_var_TL))
    df[, (col_TL) := 0.0]
    df[, (col_var_TL) := 0.0]

    for (i in seq_len(lu)) {
      U_i <- sprintf("U_Gg%d_XX", i)
      U_var_i <- sprintf("U_Gg%d_var_XX", i)

      df[, (col_TL) := get(col_TL) + get(U_i)]
      df[, (col_var_TL) := get(col_var_TL) + get(U_var_i)]
    }

    df[, (col_XX) := get(col_TL)]
    df[, (col_var_XX) := get(col_var_TL)]
  }

  ###### 5. Placebo effects
  if (placebo != 0L && exists("l_placebo_u_a_XX") && l_placebo_u_a_XX >= 1L) {
    for (i in 1:l_placebo_u_a_XX) {
      df <- compute_placebo_effects_dt(
        df, i, increase_XX, G_XX, t_min_XX, T_max_XX,
        trends_nonparam, cluster, controls, levels_d_sq_XX,
        same_switchers_pl, normalized, continuous, controls_globals
      )

      if (normalized == TRUE) {
        N_pl_val <- get(paste0("N", increase_XX, "_placebo_", i, "_XX"))
        if (!is.null(N_pl_val) && N_pl_val != 0L) {
          delta_pl_col <- paste0("delta_D_pl_", i, "_cum_temp_XX")
          delta_val <- dt_scalar_sum(df, delta_pl_col)
          assign(paste0("delta_norm_pl_", i, "_XX"), delta_val)
        }
      }
    }

    # Trends_lin for placebos
    if (isTRUE(trends_lin)) {
      Ntrendslin_pl <- 1L
      for (i in 1:l_placebo_u_a_XX) {
        N_pl_val <- get(paste0("N", increase_XX, "_placebo_", i, "_XX"))
        if (!is.null(N_pl_val)) {
          Ntrendslin_pl <- min(Ntrendslin_pl, N_pl_val, na.rm = TRUE)
        }
      }

      if (Ntrendslin_pl != 0L) {
        lp <- as.integer(l_placebo_u_a_XX)

        col_TL <- sprintf("U_Gg_pl_%d_TL", lp)
        col_var_TL <- sprintf("U_Gg_pl_%d_var_TL", lp)
        col_placebo <- sprintf("U_Gg_placebo_%d_XX", lp)
        col_pl_var <- sprintf("U_Gg_pl_%d_var_XX", lp)

        dt_batch_drop_cols(df, c(col_TL, col_var_TL))
        df[, (col_TL) := 0.0]
        df[, (col_var_TL) := 0.0]

        for (i in seq_len(lp)) {
          U_i <- sprintf("U_Gg_placebo_%d_XX", i)
          U_var_i <- sprintf("U_Gg_pl_%d_var_XX", i)

          df[, (col_TL) := get(col_TL) + get(U_i)]
          df[, (col_var_TL) := get(col_var_TL) + get(U_var_i)]
        }

        df[, (col_placebo) := get(col_TL)]
        df[, (col_pl_var) := get(col_var_TL)]
      }
    }
  }

  ###### 8. Average Total Effect
  if (!trends_lin) {
    total_key <- sprintf("sum_N%s_l_XX", increase_XX)

    sum_N <- sum(vapply(
      seq_len(as.integer(l_u_a_XX)),
      function(j) get(col_names[[j]]$N_scalar),
      numeric(1)
    ))
    assign(total_key, sum_N)

    init_cols <- c("U_Gg_XX", "U_Gg_num_XX", "U_Gg_den_XX", "U_Gg_num_var_XX", "U_Gg_var_XX")
    df[, (init_cols) := 0.0]

    for (i in seq_len(as.integer(l_u_a_XX))) {
      cn_i <- col_names[[i]]
      N_increase <- get(cn_i$N_scalar)
      sum_N_increase <- get(total_key)

      if (!is.null(N_increase) && N_increase != 0L) {
        w_i <- N_increase / sum_N_increase
        assign(sprintf("w_%s_XX", i), w_i)

        delta_temp <- sprintf("delta_D_%s_temp_XX", i)
        delta_col <- sprintf("delta_D_%s_XX", i)
        delta_g <- sprintf("delta_D_g_%s_XX", i)
        dist_col_i <- cn_i$dist

        if (is.null(continuous)) {
          df[, (delta_temp) := 0.0]
          df[, (delta_temp) := data.table::fifelse(
            get(dist_col_i) == 1L,
            (N_gt_XX / N_increase) *
            ((treatment_XX - d_sq_XX) * S_g_XX +
              (1 - S_g_XX) * (d_sq_XX - treatment_XX)),
            0.0
          )]
        } else {
          df[, (delta_temp) := 0.0]
          df[, (delta_temp) := data.table::fifelse(
            get(dist_col_i) == 1L,
            (N_gt_XX / N_increase) *
            ((treatment_XX_orig - d_sq_XX_orig) * S_g_XX +
              (1 - S_g_XX) * (d_sq_XX_orig - treatment_XX_orig)),
            0.0
          )]
        }

        total_delta <- dt_scalar_sum(df, delta_temp)
        data.table::set(df, j = delta_col, value = total_delta)

        df[, (delta_g) := get(delta_temp) * (N_increase / N_gt_XX)]

        dt_batch_drop_cols(df, delta_temp)

        U_col_i <- cn_i$U_Gg
        U_var_col_i <- cn_i$U_Gg_var

        df[, U_Gg_num_XX := U_Gg_num_XX + w_i * get(U_col_i)]
        df[, U_Gg_num_var_XX := U_Gg_num_var_XX + w_i * get(U_var_col_i)]
        df[, U_Gg_den_XX := U_Gg_den_XX + w_i * get(delta_col)]
      }
    }

    df[, U_Gg_XX := U_Gg_num_XX / U_Gg_den_XX]
    df[, U_Gg_var_XX := U_Gg_num_var_XX / U_Gg_den_XX]
  }

  # Update constants
  const <- stats::setNames(lapply(names(const), function(e) if (exists(e)) get(e) else const[[e]]), names(const))

  sum_N_key <- paste0("sum_N", increase_XX, "_l_XX")
  if (exists(sum_N_key)) {
    const[[sum_N_key]] <- get(sum_N_key)
  }

  lapply(seq_len(as.integer(l_u_a_XX)), function(i) {
    if (isTRUE(normalized)) {
      delta_key <- col_names[[i]]$delta_norm
      if (exists(delta_key)) {
        const[[delta_key]] <<- get(delta_key)
      }
    }
  })

  if (placebo != 0L && exists("l_placebo_u_a_XX") && l_placebo_u_a_XX >= 1L) {
    lapply(seq_len(as.integer(l_placebo_u_a_XX)), function(i) {
      if (isTRUE(normalized)) {
        delta_pl_key <- paste0("delta_norm_pl_", i, "_XX")
        if (exists(delta_pl_key)) {
          const[[delta_pl_key]] <<- get(delta_pl_key)
        }
      }
    })
  }

  data <- list(df = df, const = const)
  return(data)
}

#' Compute E_hat_gt with NaN handling (data.table version)
#' @param df data.table
#' @param i effect number
#' @param type_sect "effect" or "placebo"
#' @return data.table
#' @noRd
compute_E_hat_gt_dt <- function(df, i, type_sect = "effect") {
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
  df[, (E_hat) := NA_real_]

  # Condition A: time < Fg OR (Fg - 1 + i == time)
  df[, (E_hat) := data.table::fifelse(
    (time_XX < F_g_XX) | (F_g_XX - 1 + i == time_XX),
    0.0, get(E_hat)
  )]

  # Condition B: time < Fg AND dof_ns >= 2
  df[, (E_hat) := data.table::fifelse(
    (time_XX < F_g_XX) & !is.na(get(dof_ns)) & (get(dof_ns) >= 2L),
    get(mean_ns), get(E_hat)
  )]

  # Condition C: (Fg - 1 + i == time) AND dof_s >= 2
  df[, (E_hat) := data.table::fifelse(
    (F_g_XX - 1 + i == time_XX) & !is.na(get(dof_s)) & (get(dof_s) >= 2L),
    get(mean_s), get(E_hat)
  )]

  # Condition D: use mean_nss
  df[, (E_hat) := data.table::fifelse(
    (!is.na(get(dof_nss)) & (get(dof_nss) >= 2L)) &
    (
      ((F_g_XX - 1 + i == time_XX) & !is.na(get(dof_s)) & (get(dof_s) == 1L)) |
      ((time_XX < F_g_XX) & !is.na(get(dof_ns)) & (get(dof_ns) == 1L))
    ),
    get(mean_nss), get(E_hat)
  )]

  return(df)
}

#' Compute DOF_gt with NaN handling (data.table version)
#' @param df data.table
#' @param i effect number
#' @param type_sect "effect" or "placebo"
#' @return data.table
#' @noRd
compute_DOF_gt_dt <- function(df, i, type_sect = "effect") {
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

  dt_batch_drop_cols(df, DOF_col)
  df[, (DOF_col) := NA_real_]

  # DOF = 1 if (time < Fg) OR (Fg - 1 + i == time)
  df[, (DOF_col) := data.table::fifelse(
    (time_XX < F_g_XX) | (F_g_XX - 1L + i == time_XX),
    1.0, get(DOF_col)
  )]

  # sqrt(dof_s_t / (dof_s_t - 1)) for switchers
  df[, (DOF_col) := data.table::fifelse(
    (F_g_XX - 1L + i == time_XX) & (get(dof_s_t) > 1L),
    sqrt(get(dof_s_t) / (get(dof_s_t) - 1L)),
    get(DOF_col)
  )]

  # sqrt(dof_ns_t / (dof_ns_t - 1)) for controls
  df[, (DOF_col) := data.table::fifelse(
    (time_XX < F_g_XX) & (get(dof_ns_t) > 1L),
    sqrt(get(dof_ns_t) / (get(dof_ns_t) - 1L)),
    get(DOF_col)
  )]

  # sqrt(dof_ns_s_t / (dof_ns_s_t - 1)) for union
  df[, (DOF_col) := data.table::fifelse(
    (get(dof_ns_s_t) >= 2L) &
    (
      ((F_g_XX - 1L + i == time_XX) & (get(dof_s_t) == 1L)) |
      ((time_XX < F_g_XX) & (get(dof_ns_t) == 1L))
    ),
    sqrt(get(dof_ns_s_t) / (get(dof_ns_s_t) - 1L)),
    get(DOF_col)
  )]

  return(df)
}

#' Compute placebo effects (data.table version)
#' @param df data.table
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
#' @return data.table
#' @noRd
compute_placebo_effects_dt <- function(
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
  dt_batch_drop_cols(df, pl_cols_to_drop)

  # Compute placebo long differences: shift(outcome, 2*i) - shift(outcome, i)
  # Vectorized shift for placebo outcome
  diff_y_pl_col <- paste0("diff_y_pl_", i, "_XX")
  N_rows_pl <- nrow(df)
  group_vec_pl <- df[["group_XX"]]
  fix_rows_2i <- dt_shift_boundary_rows(group_vec_pl, 2L * i, N_rows_pl)
  fix_rows_1i <- dt_shift_boundary_rows(group_vec_pl, i, N_rows_pl)
  outcome_vec <- df[["outcome_XX"]]
  data.table::set(df, j = diff_y_pl_col,
    value = dt_vec_shift(outcome_vec, 2L * i, fix_rows_2i, N_rows_pl) -
            dt_vec_shift(outcome_vec, i, fix_rows_1i, N_rows_pl))

  # Residualize placebo outcome differences when controls are specified
  if (!is.null(controls) && length(controls) > 0L && !is.null(controls_globals)) {
    count_controls <- length(controls)

    # Phase 1: Compute all placebo control differences (vectorized shift)
    diff_X_pl_cols <- paste0("diff_X", 1:count_controls, "_placebo_", i, "_XX")
    diff_X_pl_N_cols <- paste0("diff_X", 1:count_controls, "_pl_", i, "_N_XX")
    N_gt_vec_pl <- df[["N_gt_XX"]]
    for (j in seq_len(count_controls)) {
      ctrl_vec <- df[[controls[j]]]
      diff_vec <- dt_vec_shift(ctrl_vec, 2L * i, fix_rows_2i, N_rows_pl) -
                  dt_vec_shift(ctrl_vec, i, fix_rows_1i, N_rows_pl)
      data.table::set(df, j = diff_X_pl_cols[j], value = diff_vec)
      data.table::set(df, j = diff_X_pl_N_cols[j], value = N_gt_vec_pl * diff_vec)
    }

    # Phase 2: Residualize sequentially (preserves bit-exact results)
    for (l in levels_d_sq_XX) {
      l_num <- as.numeric(l)
      useful_res_name <- paste0("useful_res_", l, "_XX")
      coefs_name <- paste0("coefs_sq_", l, "_XX")

      if (useful_res_name %chin% names(controls_globals) &&
          controls_globals[[useful_res_name]] > 1L &&
          coefs_name %chin% names(controls_globals)) {

        for (j in seq_len(count_controls)) {
          coef_val <- controls_globals[[coefs_name]][j, 1L]

          df[, (diff_y_pl_col) := data.table::fifelse(
            d_sq_int_XX == l_num,
            get(diff_y_pl_col) - coef_val * get(diff_X_pl_cols[j]),
            get(diff_y_pl_col)
          )]
        }
      }
    }
  }

  # Identifying controls for placebos
  never_col <- paste0("never_change_d_", i, "_XX")
  never_pl_col <- paste0("never_change_d_pl_", i, "_XX")
  df[, (never_pl_col) := get(never_col) * as.numeric(!is.na(get(diff_y_pl_col)))]

  never_pl_w_col <- paste0("never_change_d_pl_", i, "_wXX")
  df[, (never_pl_w_col) := get(never_pl_col) * N_gt_XX]

  # N_gt_control_placebo
  by_cols <- c("time_XX", "d_sq_XX")
  if (!is.null(trends_nonparam) && length(trends_nonparam) > 0L) {
    by_cols <- c(by_cols, trends_nonparam)
  }

  N_gt_ctrl_pl_col <- paste0("N_gt_control_placebo_", i, "_XX")
  df[, (N_gt_ctrl_pl_col) := sum(get(never_pl_w_col), na.rm = TRUE), by = by_cols]

  # dist_to_switch_pl
  dist_col <- paste0("distance_to_switch_", i, "_XX")
  dist_pl_col <- paste0("dist_to_switch_pl_", i, "_XX")

  df[, (dist_pl_col) := data.table::fifelse(
    !is.na(get(dist_col)),
    get(dist_col) *
    as.numeric(!is.na(get(diff_y_pl_col))) *
    as.numeric(!is.na(get(N_gt_ctrl_pl_col)) & get(N_gt_ctrl_pl_col) > 0L),
    NA_real_
  )]

  if (isTRUE(same_switchers_pl)) {
    df[, (dist_pl_col) := get(dist_pl_col) * as.numeric(fillin_g_pl_XX)]
  }

  dist_pl_w_col <- paste0("dist_to_switch_pl_", i, "_wXX")
  df[, (dist_pl_w_col) := get(dist_pl_col) * N_gt_XX]

  # N_t_placebo
  N_t_pl_col <- paste0("N", increase_XX, "_t_placebo_", i, "_XX")
  N_dw_t_pl_col <- paste0("N", increase_XX, "_t_placebo_", i, "_dwXX")
  df[, (N_t_pl_col) := sum(get(dist_pl_w_col), na.rm = TRUE), by = "time_XX"]
  df[, (N_dw_t_pl_col) := sum(get(dist_pl_col), na.rm = TRUE), by = "time_XX"]

  # Compute N_placebo scalar
  filter_idx <- df[["time_XX"]] >= t_min_XX & df[["time_XX"]] <= T_max_XX
  N_pl_val <- dt_scalar_mean_sum(df, N_t_pl_col, filter_idx, "time_XX")
  assign(paste0("N", increase_XX, "_placebo_", i, "_XX"), N_pl_val, envir = parent.frame())

  N_dw_pl_val <- dt_scalar_mean_sum(df, N_dw_t_pl_col, filter_idx, "time_XX")
  assign(paste0("N", increase_XX, "_dw_placebo_", i, "_XX"), N_dw_pl_val, envir = parent.frame())

  # N_t_placebo_g
  N_t_pl_g_col <- paste0("N", increase_XX, "_t_placebo_", i, "_g_XX")
  df[, (N_t_pl_g_col) := sum(get(dist_pl_w_col), na.rm = TRUE), by = by_cols]

  # Compute M_pl terms for controls adjustment (after N_placebo is computed)
  if (!is.null(controls) && length(controls) > 0L && !is.null(controls_globals)) {
    # Initialize part2_pl_switch column
    part2_pl_col <- paste0("part2_pl_switch", increase_XX, "_", i, "_XX")
    df[, (part2_pl_col) := 0.0]

    count_controls <- length(controls)
    diff_X_pl_N_cols <- paste0("diff_X", 1:count_controls, "_pl_", i, "_N_XX")

    for (l in levels_d_sq_XX) {
      l_num <- as.numeric(l)
      useful_res_name <- paste0("useful_res_", l, "_XX")

      m_pl_g_cols <- paste0("m", increase_XX, "_pl_g_", l, "_", 1:count_controls, "_", i, "_XX")
      m_pl_cols <- paste0("m_pl", increase_XX, "_", l, "_", 1:count_controls, "_", i, "_XX")
      M_pl_cols <- paste0("M_pl", increase_XX, "_", l, "_", 1:count_controls, "_", i, "_XX")

      if (N_pl_val > 0L) {
        # Shared: safe_ratio and common factor (same for all controls)
        safe_ratio_m_pl <- data.table::fifelse(df[[N_gt_ctrl_pl_col]] == 0L, 0, df[[N_t_pl_g_col]] / df[[N_gt_ctrl_pl_col]])
        common_m_pl <- as.numeric(df[["d_sq_int_XX"]] == l_num & i <= df[["T_g_XX"]] - 2L) *
          (G_XX / N_pl_val) *
          (df[[dist_pl_col]] - safe_ratio_m_pl * df[[never_pl_col]]) *
          as.numeric(df[["time_XX"]] >= i + 1L & df[["time_XX"]] <= df[["T_g_XX"]])

        # Compute all m_pl_g columns: m_pl_g_j = common_m_pl * diff_X_pl_N_j
        for (j in seq_len(count_controls)) {
          data.table::set(df, j = m_pl_g_cols[j], value = common_m_pl * df[[diff_X_pl_N_cols[j]]])
        }
      } else {
        for (j in seq_len(count_controls)) {
          data.table::set(df, j = m_pl_g_cols[j], value = 0.0)
        }
      }

      # Batch by-group sum for all m_pl columns at once
      df[, (m_pl_cols) := lapply(.SD, sum, na.rm = TRUE), by = "group_XX", .SDcols = m_pl_g_cols]

      # Set NA where not first_obs (batch)
      not_first_idx <- which(df[["first_obs_by_gp_XX"]] != 1L)
      if (length(not_first_idx) > 0L) {
        for (j in seq_len(count_controls)) {
          data.table::set(df, i = not_first_idx, j = m_pl_cols[j], value = NA_real_)
        }
      }

      # Compute all M_pl scalars
      for (j in seq_len(count_controls)) {
        M_pl_val_j <- sum(df[[m_pl_cols[j]]], na.rm = TRUE) / G_XX
        data.table::set(df, j = M_pl_cols[j], value = M_pl_val_j)
      }

      # Initialize in_brackets_pl columns
      if (useful_res_name %chin% names(controls_globals) &&
          controls_globals[[useful_res_name]] > 1L) {
        for (j in seq_len(count_controls)) {
          in_brackets_pl_col <- paste0("in_brackets_pl_", l, "_", j, "_XX")
          data.table::set(df, j = in_brackets_pl_col, value = 0.0)
        }
      }
    }
  }

  # DOF computations for placebos
  diff_y_pl_N_col <- paste0("diff_y_pl_", i, "_N_gt_XX")
  dof_ns_pl <- paste0("dof_ns_pl_", i, "_XX")
  dof_s_pl <- paste0("dof_s_pl_", i, "_XX")

  df[, (diff_y_pl_N_col) := get(diff_y_pl_col) * N_gt_XX]

  df[, (dof_ns_pl) := as.numeric(
    (N_gt_XX != 0L) &
    !is.na(get(diff_y_pl_col)) &
    (get(never_pl_col) == 1L) &
    (get(N_t_pl_col) > 0L) &
    !is.na(get(N_t_pl_col))
  )]

  df[, (dof_s_pl) := as.numeric((N_gt_XX != 0L) & (get(dist_pl_col) == 1L))]
  df[is.na(get(dof_s_pl)), (dof_s_pl) := 0.0]

  # Cohort means for ns
  ns_by_cols <- c("d_sq_XX", "time_XX")
  if (!is.null(trends_nonparam) && length(trends_nonparam) > 0L) {
    ns_by_cols <- c(ns_by_cols, trends_nonparam)
  }

  count_ns_col <- paste0("count_cohort_pl_", i, "_ns_t_XX")
  total_ns_col <- paste0("total_cohort_pl_", i, "_ns_t_XX")
  mean_ns_col <- paste0("mean_cohort_pl_", i, "_ns_t_XX")
  dof_coh_ns_col <- paste0("dof_cohort_pl_", i, "_ns_t_XX")

  dt_filtered_agg_over(df, "N_gt_XX", df[[dof_ns_pl]] == 1, ns_by_cols, count_ns_col, "sum")
  dt_filtered_agg_over(df, diff_y_pl_N_col, df[[dof_ns_pl]] == 1, ns_by_cols, total_ns_col, "sum")
  df[, (mean_ns_col) := get(total_ns_col) / get(count_ns_col)]

  if (is.null(cluster) || cluster == "" || is.na(cluster)) {
    dt_filtered_agg_over(df, dof_ns_pl, df[[dof_ns_pl]] == 1, ns_by_cols, dof_coh_ns_col, "sum", filter_result = TRUE)
  } else {
    cluster_dof_ns_pl <- paste0("cluster_dof_pl_", i, "_ns_XX")
    df[, (cluster_dof_ns_pl) := data.table::fifelse(get(dof_ns_pl) == 1, get(cluster), NA)]
    dt_uniqueN_over(df, cluster_dof_ns_pl, ns_by_cols, dof_coh_ns_col, !is.na(df[[cluster_dof_ns_pl]]))
  }

  # Cohort means for s
  sw_by_cols <- c("d_sq_XX", "F_g_XX", "d_fg_XX")
  if (!is.null(trends_nonparam) && length(trends_nonparam) > 0L) {
    sw_by_cols <- c(sw_by_cols, trends_nonparam)
  }

  count_s_col <- paste0("count_cohort_pl_", i, "_s_t_XX")
  total_s_col <- paste0("total_cohort_pl_", i, "_s_t_XX")
  mean_s_col <- paste0("mean_cohort_pl_", i, "_s_t_XX")
  dof_coh_s_col <- paste0("dof_cohort_pl_", i, "_s_t_XX")

  dt_filtered_agg_over(df, "N_gt_XX", df[[dof_s_pl]] == 1L, sw_by_cols, count_s_col, "sum")
  dt_filtered_agg_over(df, diff_y_pl_N_col, df[[dof_s_pl]] == 1L, sw_by_cols, total_s_col, "sum")
  df[, (mean_s_col) := get(total_s_col) / get(count_s_col)]

  if (is.null(cluster) || cluster == "" || is.na(cluster)) {
    dt_filtered_agg_over(df, dof_s_pl, df[[dof_s_pl]] == 1L, sw_by_cols, dof_coh_s_col, "sum", filter_result = TRUE)
  } else {
    cluster_dof_s_pl <- paste0("cluster_dof_pl_", i, "_s_XX")
    df[, (cluster_dof_s_pl) := data.table::fifelse(get(dof_s_pl) == 1L, get(cluster), NA)]
    dt_uniqueN_over(df, cluster_dof_s_pl, sw_by_cols, dof_coh_s_col, !is.na(df[[cluster_dof_s_pl]]))
  }

  # Union of ns and s
  dof_ns_s_pl <- paste0("dof_ns_s_pl_", i, "_XX")
  count_ns_s_col <- paste0("count_cohort_pl_", i, "_ns_s_t_XX")
  total_ns_s_col <- paste0("total_cohort_pl_", i, "_ns_s_t_XX")
  mean_ns_s_col <- paste0("mean_cohort_pl_", i, "_ns_s_t_XX")
  dof_coh_ns_s_col <- paste0("dof_cohort_pl_", i, "_ns_s_t_XX")

  df[, (dof_ns_s_pl) := as.numeric((get(dof_s_pl) == 1L) | (get(dof_ns_pl) == 1L))]
  df[is.na(get(dof_s_pl)) | is.na(get(dof_ns_pl)), (dof_ns_s_pl) := NA_real_]

  dt_filtered_agg_over(df, "N_gt_XX", df[[dof_ns_s_pl]] == 1L, ns_by_cols, count_ns_s_col, "sum")
  dt_filtered_agg_over(df, diff_y_pl_N_col, df[[dof_ns_s_pl]] == 1L, ns_by_cols, total_ns_s_col, "sum")
  df[, (mean_ns_s_col) := get(total_ns_s_col) / get(count_ns_s_col)]

  if (is.null(cluster) || cluster == "" || is.na(cluster)) {
    dt_filtered_agg_over(df, dof_ns_s_pl, df[[dof_ns_s_pl]] == 1L, ns_by_cols, dof_coh_ns_s_col, "sum", filter_result = TRUE)
  } else {
    cluster_dof_ns_s_pl <- paste0("cluster_dof_pl_", i, "_ns_s_XX")
    df[, (cluster_dof_ns_s_pl) := data.table::fifelse(get(dof_ns_s_pl) == 1L, get(cluster), NA)]
    dt_uniqueN_over(df, cluster_dof_ns_s_pl, ns_by_cols, dof_coh_ns_s_col, !is.na(df[[cluster_dof_ns_s_pl]]))
  }

  # E_hat and DOF for placebos
  df <- compute_E_hat_gt_dt(df, i, "placebo")
  df <- compute_DOF_gt_dt(df, i, "placebo")

  # U_Gg placebo computation
  dummy_U_pl_col <- paste0("dummy_U_Gg_pl_", i, "_XX")
  df[, (dummy_U_pl_col) := as.numeric(i <= T_g_XX - 1L)]

  N_pl_val <- get(paste0("N", increase_XX, "_placebo_", i, "_XX"), envir = parent.frame())

  if (!is.null(N_pl_val) && N_pl_val != 0L) {
    U_Gg_pl_temp_col <- paste0("U_Gg_pl_", i, "_temp_XX")
    U_Gg_pl_col <- paste0("U_Gg_placebo_", i, "_XX")

    # Safe division to avoid 0/0 = NaN
    safe_ratio_pl <- data.table::fifelse(df[[N_gt_ctrl_pl_col]] == 0L, 0, df[[N_t_pl_g_col]] / df[[N_gt_ctrl_pl_col]])
    df[, (U_Gg_pl_temp_col) :=
      get(dummy_U_pl_col) * (G_XX / N_pl_val) * N_gt_XX *
      (get(dist_pl_col) - safe_ratio_pl * get(never_pl_col)) *
      get(diff_y_pl_col) *
      as.numeric(time_XX >= i + 1L & time_XX <= T_g_XX)
    ]

    df[, (U_Gg_pl_col) := sum(get(U_Gg_pl_temp_col), na.rm = TRUE), by = "group_XX"]
    df[, (U_Gg_pl_col) := get(U_Gg_pl_col) * first_obs_by_gp_XX]

    # count_pl_core
    count_pl_core_col <- paste0("count", i, "_pl_core_XX")
    df[, (count_pl_core_col) := data.table::fifelse(
      (!is.na(get(U_Gg_pl_temp_col)) & (get(U_Gg_pl_temp_col) != 0L)) |
      ((get(U_Gg_pl_temp_col) == 0L) & (get(diff_y_pl_col) == 0L) &
        ((get(dist_pl_col) != 0L) | ((get(N_t_pl_g_col) != 0L) & (get(never_pl_col) != 0L)))),
      N_gt_XX, 0.0
    )]

    # U_Gg_pl_temp_var - reuse safe_ratio_pl from above
    U_Gg_pl_temp_var_col <- paste0("U_Gg_pl_", i, "_temp_var_XX")
    E_hat_pl_col <- paste0("E_hat_gt_pl_", i, "_XX")
    DOF_pl_col <- paste0("DOF_gt_pl_", i, "_XX")

    df[, (U_Gg_pl_temp_var_col) :=
      get(dummy_U_pl_col) * (G_XX / N_pl_val) *
      (get(dist_pl_col) - safe_ratio_pl * get(never_pl_col)) *
      as.numeric(time_XX >= i + 1L & time_XX <= T_g_XX) *
      N_gt_XX * get(DOF_pl_col) *
      (get(diff_y_pl_col) - get(E_hat_pl_col))
    ]

    # Sum U_Gg_pl_var by group
    U_Gg_pl_var_col <- paste0("U_Gg_pl_", i, "_var_XX")
    df[, (U_Gg_pl_var_col) := sum(get(U_Gg_pl_temp_var_col), na.rm = TRUE), by = "group_XX"]

    # Controls adjustment for placebo variance (matrix-vectorized)
    if (!is.null(controls) && length(controls) > 0L && !is.null(controls_globals)) {
      part2_pl_col <- paste0("part2_pl_switch", increase_XX, "_", i, "_XX")
      # Reset part2_pl_switch to 0 before accumulating
      df[, (part2_pl_col) := 0.0]

      count_controls <- length(controls)
      for (l in levels_d_sq_XX) {
        useful_res_name <- paste0("useful_res_", l, "_XX")
        if (!(useful_res_name %chin% names(controls_globals)) ||
            controls_globals[[useful_res_name]] <= 1L) next

        l_num <- as.numeric(l)
        inv_denom_name <- paste0("inv_Denom_", l, "_XX")
        coefs_name <- paste0("coefs_sq_", l, "_XX")

        if (!(inv_denom_name %chin% names(controls_globals))) next
        inv_Denom_l <- controls_globals[[inv_denom_name]]
        mask_vec <- as.numeric(df[["d_sq_int_XX"]] == l_num & df[["F_g_XX"]] >= 3L)

        # Extract in_sum columns into N x C matrix
        in_sum_cols <- paste0("in_sum_", 1:count_controls, "_", l, "_XX")
        in_sum_avail <- in_sum_cols %chin% names(df)
        if (!all(in_sum_avail)) next
        in_sum_mat <- as.matrix(df[, ..in_sum_cols])

        # in_brackets = (in_sum * mask) %*% t(inv_Denom) - coefs (broadcast)
        in_brackets_mat <- (in_sum_mat * mask_vec) %*% t(inv_Denom_l)
        if (coefs_name %chin% names(controls_globals)) {
          coefs_l <- controls_globals[[coefs_name]][, 1]
          in_brackets_mat <- t(t(in_brackets_mat) - coefs_l)
        }

        # Extract M_pl columns into N x C matrix
        M_pl_cols <- paste0("M_pl", increase_XX, "_", l, "_", 1:count_controls, "_", i, "_XX")
        M_pl_avail <- M_pl_cols %chin% names(df)
        if (!all(M_pl_avail)) next
        M_pl_mat <- as.matrix(df[, ..M_pl_cols])

        # comb = rowSums(M_pl * in_brackets), add to part2_pl
        data.table::set(df, j = part2_pl_col,
          value = df[[part2_pl_col]] + rowSums(M_pl_mat * in_brackets_mat))

        # Write back in_brackets_pl columns (for consistency with downstream code)
        for (j in 1:count_controls) {
          in_brackets_pl_col <- paste0("in_brackets_pl_", l, "_", j, "_XX")
          if (in_brackets_pl_col %chin% names(df)) {
            data.table::set(df, j = in_brackets_pl_col, value = in_brackets_mat[, j])
          }
        }
      }

      # Subtract part2_pl_switch from U_Gg_pl_var
      df[, (U_Gg_pl_var_col) := get(U_Gg_pl_var_col) - get(part2_pl_col)]
    }

    # Normalized delta for placebos
    if (normalized == TRUE) {
      if (is.null(continuous)) {
        df[, sum_temp_pl_XX := data.table::fifelse(
          time_XX >= F_g_XX &
          time_XX <= F_g_XX - 1L + i &
          S_g_XX == increase_XX,
          treatment_XX - d_sq_XX,
          NA_real_
        )]
      } else {
        df[, sum_temp_pl_XX := data.table::fifelse(
          time_XX >= F_g_XX &
          time_XX <= F_g_XX - 1L + i &
          S_g_XX == increase_XX,
          treatment_XX_orig - d_sq_XX_orig,
          NA_real_
        )]
      }

      sum_treat_pl_col <- paste0("sum_treat_until_", i, "_pl_XX")
      df[, (sum_treat_pl_col) := sum(get("sum_temp_pl_XX"), na.rm = TRUE), by = "group_XX"]
      dt_batch_drop_cols(df, "sum_temp_pl_XX")

      delta_pl_temp_col <- paste0("delta_D_pl_", i, "_cum_temp_XX")
      df[, (delta_pl_temp_col) := data.table::fifelse(
        get(dist_pl_col) == 1L,
        (N_gt_XX / N_pl_val) * (
          S_g_XX * get(sum_treat_pl_col) +
          (1 - S_g_XX) * (-get(sum_treat_pl_col))
        ),
        NA_real_
      )]
    }
  }

  return(df)
}
