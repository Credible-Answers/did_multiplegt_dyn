#' @useDynLib DIDmultiplegtDYN, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom data.table ":=" .N .I .SD .GRP "%chin%"
#' @importFrom ggplot2 .data

utils::globalVariables(c(
  # data.table NSE column names used across the package
  ".", "..mycontrols_XX", "i.tmp_dof",
  "_avg_diff_temp_masked", "_N_for_ctrl",
  "avg_diff_temp_XX", "avg_post_switch_treat_XX", "avg_post_switch_treat_XX_temp",
  "cluster_group_XX", "cluster_XX", "cluster_var_g_XX",
  "clust_U_Gg_var_global_XX", "clust_var_sq", "clust_var_sum",
  "controls_time_XX", "count_time_post_switch_XX", "count_time_post_switch_XX_temp",
  "counter", "counter_temp",
  "d_F_g_XX", "d_F_g_temp_XX", "d_fg_XX",
  "d_sq_int_XX", "d_sq_temp_XX", "d_sq_XX", "d_sq_XX_orig",
  "delta_D_g_XX", "delta_D_g_XX_temp",
  "diff_from_sq_XX", "diff_y_last_XX", "diff_y_relev_XX", "diff_y_wXX", "diff_y_XX",
  "ever_change_d_XX", "ever_strict_decrease_XX", "ever_strict_increase_XX",
  "F_g_trunc_XX", "F_g_XX", "fd_X_all_non_missing_XX",
  "feasible_het_XX", "first_obs_by_clust_XX", "first_obs_by_gp_XX",
  "g_weight_XX", "gr_id", "group_XX",
  "id_XX", "in_table_XX",
  "L_g_XX", "lagout", "last_obs_D_bef_switch_XX", "last_obs_D_bef_switch_t_XX",
  "M_g_XX", "max_time_d_nonmiss_XX", "mean_D", "mean_Y",
  "min_time_d_miss_aft_ynm_XX", "min_time_d_nonmiss_XX", "min_time_y_nonmiss_XX",
  "N_g_control_check_pl_XX", "N_g_control_check_XX", "N_g_control_last_m_XX",
  "N_gt_control_last_XX", "N_gt_XX", "N_w_XX", "N_XX",
  "never_change_d_last_XX", "never_change_d_XX",
  "outcome_non_diff_XX", "outcome_XX",
  "row_id", "row_id_XX",
  "S_g_XX", "sd_by", "sd_het", "sum_weights_control_XX",
  "switcher_tag_XX", "switchers_tag_XX",
  "T_g_XX", "temp_F_g_XX", "time_d_miss_XX", "time_d_nonmiss_XX",
  "time_FE_XX", "time_l_XX", "time_XX", "time_y_nonmiss_XX",
  "treat_GRP", "treatment_XX", "treatment_XX_orig", "treatment_XX_v1",
  "U_Gg_den_XX", "U_Gg_num_var_XX", "U_Gg_num_XX", "U_Gg_var_global_XX",
  "U_Gg_var_XX", "U_Gg_XX",
  "var_F_g_XX", "var_weighted",
  "weight_XX", "y_hat", "Yg_Fg_min1_XX", "Yg_Fg_min2_XX",
  "time", "tot_s",
  # Additional NSE column names
  "__temp_weighted__", "__temp_ctrl__", "__temp_diff__", "__still_sw_XX__",
  ".mask_ctrl", ".N_for_ctrl", ".avg_diff_temp_masked", ".var_weighted",
  ".clust_var_sq", ".var_weighted_pl", ".clust_var_sum_pl",
  "baseline_XX", "cohort_fullpath_0_XX", "cohort_fullpath_1_XX",
  "count_global_XX", "cum_sum_XX", "d_fg0_XX", "d_fg_XX_temp",
  "D_fg0", "did_sample", "diff_d_XX", "dummy_XX",
  "F_g_plus_n_XX", "fillin_g_pl_XX", "group", "i.y_hat",
  "L_g_placebo_XX", "N", "neg_N_XX", "num_g_paths_0_XX",
  "path_0_XX", "path_XX", "pl_gap_XX", "S_g_het_XX", "share_XX",
  "sum_temp_XX", "sum_temp_pl_XX", "T_d_XX", "treat_key",
  "treatment_temp_XX", "trunc_control_XX",
  "U_Gg_global_XX", "U_Gg_minus_XX", "U_Gg_plus_XX",
  "U_Gg_var_minus_XX", "U_Gg_var_plus_XX",
  "yet_to_switch_XX"
))

# Compute profile name used by this package (avoids overwriting user daemons)
.mirai_compute <- "DIDmgtDYN"

# Helper to detect active mirai daemons on the package's compute profile
# Returns the number of active daemons, or 0L if mirai is unavailable
.mirai_n_daemons <- function() {
  if (!requireNamespace("mirai", quietly = TRUE)) return(0L)
  tryCatch({
    if (!mirai::daemons_set(.compute = .mirai_compute)) return(0L)
    as.integer(mirai::status(.compute = .mirai_compute)[["daemons"]])
  }, error = function(e) 0L)
}