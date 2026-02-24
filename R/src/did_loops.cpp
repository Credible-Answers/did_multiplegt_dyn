#include <Rcpp.h>
#include <unordered_map>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace Rcpp;

// [[Rcpp::export]]
IntegerVector cummax_by_group_cpp(IntegerVector x, IntegerVector group) {
  // Cumulative maximum within groups (for ever_change_d_XX propagation)
  // Equivalent to: df[, ever_change_d_XX := cummax(ever_change_d_XX), by = group_XX]
  int n = x.size();
  IntegerVector result(n);

  if (n == 0) return result;

  int current_group = group[0];
  int current_max = x[0];
  result[0] = current_max;

  for (int i = 1; i < n; i++) {
    if (group[i] != current_group) {
      current_group = group[i];
      current_max = x[i];
    } else {
      if (x[i] > current_max) {
        current_max = x[i];
      }
    }
    result[i] = current_max;
  }

  return result;
}

// [[Rcpp::export]]
NumericMatrix compute_var_covar_matrix_cpp(NumericMatrix U_Gg_vars,
                                            IntegerVector first_obs,
                                            int l_XX,
                                            double G_XX) {
  // Compute variance-covariance matrix for effects
  // U_Gg_vars: matrix where each column is U_Gg_var_glob_i_XX for i in 1:l_XX
  // first_obs: first_obs_by_gp_XX indicator
  // Returns l_XX x l_XX variance-covariance matrix

  int n = U_Gg_vars.nrow();
  NumericMatrix vcov(l_XX, l_XX);
  double G_XX_sq = G_XX * G_XX;

  // Compute variances (diagonal)
  for (int i = 0; i < l_XX; i++) {
    double sum_sq = 0.0;
    for (int j = 0; j < n; j++) {
      if (first_obs[j] == 1) {
        double val = U_Gg_vars(j, i);
        if (!NumericVector::is_na(val)) {
          sum_sq += val * val;
        }
      }
    }
    vcov(i, i) = sum_sq / G_XX_sq;
  }

  // Compute covariances (off-diagonal)
  for (int i = 0; i < l_XX - 1; i++) {
    for (int k = i + 1; k < l_XX; k++) {
      double sum_combined_sq = 0.0;
      for (int j = 0; j < n; j++) {
        if (first_obs[j] == 1) {
          double val_i = U_Gg_vars(j, i);
          double val_k = U_Gg_vars(j, k);
          if (!NumericVector::is_na(val_i) && !NumericVector::is_na(val_k)) {
            double combined = val_i + val_k;
            sum_combined_sq += combined * combined;
          }
        }
      }
      double var_sum = sum_combined_sq / G_XX_sq;
      double cov = (var_sum - vcov(i, i) - vcov(k, k)) / 2.0;
      vcov(i, k) = cov;
      vcov(k, i) = cov;
    }
  }

  return vcov;
}

// [[Rcpp::export]]
NumericVector compute_U_Gg_global_cpp(NumericVector U_Gg_plus,
                                       NumericVector U_Gg_minus,
                                       double N1_weight,
                                       double N0_weight) {
  // Compute weighted combination of U_Gg for switchers in and out
  int n = U_Gg_plus.size();
  NumericVector result(n);

  double total = N1_weight + N0_weight;
  if (total == 0) {
    std::fill(result.begin(), result.end(), NA_REAL);
    return result;
  }

  double w_plus = N1_weight / total;
  double w_minus = N0_weight / total;

  for (int i = 0; i < n; i++) {
    result[i] = w_plus * U_Gg_plus[i] + w_minus * U_Gg_minus[i];
  }

  return result;
}

// [[Rcpp::export]]
List compute_clustered_variance_cpp(NumericVector U_Gg_var,
                                     IntegerVector first_obs_gp,
                                     IntegerVector first_obs_clust,
                                     IntegerVector cluster,
                                     double G_XX) {
  // Compute clustered variance
  int n = U_Gg_var.size();

  // Step 1: Multiply by first_obs_by_gp_XX
  NumericVector U_masked(n);
  for (int i = 0; i < n; i++) {
    U_masked[i] = U_Gg_var[i] * first_obs_gp[i];
  }

  // Step 2: Sum within clusters
  std::map<int, double> cluster_sums;
  for (int i = 0; i < n; i++) {
    if (!IntegerVector::is_na(cluster[i])) {
      if (!NumericVector::is_na(U_masked[i])) {
        cluster_sums[cluster[i]] += U_masked[i];
      }
    }
  }

  // Step 3: Assign cluster sums back and compute squared sums
  NumericVector clust_sum(n);
  double sum_sq = 0.0;

  for (int i = 0; i < n; i++) {
    if (!IntegerVector::is_na(cluster[i])) {
      clust_sum[i] = cluster_sums[cluster[i]];
      if (first_obs_clust[i] == 1) {
        sum_sq += clust_sum[i] * clust_sum[i];
      }
    }
  }

  double sum_for_var = sum_sq / (G_XX * G_XX);

  return List::create(
    Named("clust_sum") = clust_sum,
    Named("sum_for_var") = sum_for_var
  );
}

// [[Rcpp::export]]
NumericVector propagate_treatment_change_cpp(NumericVector ever_change,
                                              IntegerVector group,
                                              IntegerVector time,
                                              int T_max) {
  // Propagate ever_change_d_XX forward within groups
  // This replaces the loop: for (i in 2:T_XX) { ... }

  int n = ever_change.size();
  NumericVector result = clone(ever_change);

  for (int i = 1; i < n; i++) {
    if (group[i] == group[i-1] && result[i-1] == 1 && time[i] > 1) {
      result[i] = 1;
    }
  }

  return result;
}

// [[Rcpp::export]]
NumericMatrix initialize_effect_columns_cpp(int nrow, int l_XX, bool include_placebo) {
  // Pre-allocate matrix for effect columns
  // Each column represents: U_Gg{i}_plus_XX, U_Gg{i}_minus_XX, count{i}_plus_XX, etc.
  int ncols = l_XX * 8;  // 8 columns per effect
  if (include_placebo) {
    ncols *= 2;  // Double for placebos
  }

  NumericMatrix result(nrow, ncols);
  std::fill(result.begin(), result.end(), 0.0);

  return result;
}

// [[Rcpp::export]]
double compute_weighted_sum_cpp(NumericVector x, IntegerVector mask) {
  // Compute sum of x where mask == 1, handling NAs
  double result = 0.0;
  int n = x.size();

  for (int i = 0; i < n; i++) {
    if (mask[i] == 1 && !NumericVector::is_na(x[i])) {
      result += x[i];
    }
  }

  return result;
}

// [[Rcpp::export]]
NumericVector compute_delta_D_g_cpp(NumericMatrix delta_plus,
                                     NumericMatrix delta_minus,
                                     IntegerVector switchers_tag,
                                     int l_XX) {
  // Compute delta_D_g_XX by combining plus and minus matrices
  int n = delta_plus.nrow();
  NumericVector result(n, 0.0);

  for (int i = 0; i < n; i++) {
    int tag = switchers_tag[i];
    if (!IntegerVector::is_na(tag) && tag >= 1 && tag <= l_XX) {
      int col = tag - 1;  // 0-indexed
      double val_plus = delta_plus(i, col);
      double val_minus = delta_minus(i, col);

      double val = (val_plus != 0) ? val_plus : val_minus;
      if (val != 0) {
        result[i] = val;
      }
    }
  }

  return result;
}

// [[Rcpp::export]]
List compute_full_vcov_cpp(NumericMatrix U_Gg_vars_effects,
                           NumericMatrix U_Gg_vars_placebos,
                           IntegerVector first_obs,
                           NumericVector se_effects,
                           NumericVector se_placebos,
                           double G_XX) {
  // Compute full variance-covariance matrix for effects and placebos
  int l_XX = U_Gg_vars_effects.ncol();
  int l_placebo_XX = U_Gg_vars_placebos.ncol();
  int l_tot = l_XX + l_placebo_XX;
  int n = U_Gg_vars_effects.nrow();

  NumericMatrix vcov(l_tot, l_tot);
  double G_XX_sq = G_XX * G_XX;

  // Fill diagonal with squared SEs
  for (int i = 0; i < l_XX; i++) {
    vcov(i, i) = se_effects[i] * se_effects[i];
  }
  for (int i = 0; i < l_placebo_XX; i++) {
    vcov(l_XX + i, l_XX + i) = se_placebos[i] * se_placebos[i];
  }

  // Compute covariances
  for (int i = 0; i < l_tot; i++) {
    for (int j = i + 1; j < l_tot; j++) {
      double sum_sq = 0.0;

      for (int k = 0; k < n; k++) {
        if (first_obs[k] == 1) {
          double val_i = (i < l_XX) ? U_Gg_vars_effects(k, i) :
                                       U_Gg_vars_placebos(k, i - l_XX);
          double val_j = (j < l_XX) ? U_Gg_vars_effects(k, j) :
                                       U_Gg_vars_placebos(k, j - l_XX);

          if (!NumericVector::is_na(val_i) && !NumericVector::is_na(val_j)) {
            double combined = val_i + val_j;
            sum_sq += combined * combined;
          }
        }
      }

      double var_temp = sum_sq / G_XX_sq;
      double cov = (var_temp - vcov(i, i) - vcov(j, j)) / 2.0;
      vcov(i, j) = cov;
      vcov(j, i) = cov;
    }
  }

  return List::create(Named("vcov") = vcov);
}

// ============================================================================
// HOT LOOP OPTIMIZATIONS FOR CORE FUNCTION
// ============================================================================

// [[Rcpp::export]]
NumericVector lag_diff_by_group_cpp(NumericVector x, IntegerVector group, int lag_periods) {
  // Compute x - lag(x, lag_periods) within groups
  // Data must be sorted by group, time
  int n = x.size();
  NumericVector result(n, NA_REAL);

  if (n == 0 || lag_periods <= 0) return result;

  int current_group = group[0];
  int group_start = 0;

  for (int i = 0; i < n; i++) {
    if (i > 0 && group[i] != current_group) {
      // New group starts
      current_group = group[i];
      group_start = i;
    }

    int lag_idx = i - lag_periods;
    if (lag_idx >= group_start && group[lag_idx] == current_group) {
      if (!NumericVector::is_na(x[i]) && !NumericVector::is_na(x[lag_idx])) {
        result[i] = x[i] - x[lag_idx];
      }
    }
  }

  return result;
}

// [[Rcpp::export]]
NumericVector shift_by_group_cpp(NumericVector x, IntegerVector group, int periods) {
  // Shift x by periods within groups (positive = lag, negative = lead)
  // Data must be sorted by group, time
  int n = x.size();
  NumericVector result(n, NA_REAL);

  if (n == 0) return result;

  int current_group = group[0];
  int group_start = 0;
  int group_end = 0;

  // Find group boundaries
  std::vector<int> group_starts;
  std::vector<int> group_ends;

  group_starts.push_back(0);
  for (int i = 1; i < n; i++) {
    if (group[i] != group[i-1]) {
      group_ends.push_back(i - 1);
      group_starts.push_back(i);
    }
  }
  group_ends.push_back(n - 1);

  // Process each group
  for (size_t g = 0; g < group_starts.size(); g++) {
    int start = group_starts[g];
    int end = group_ends[g];

    for (int i = start; i <= end; i++) {
      int src_idx = i - periods;  // For lag, periods > 0
      if (src_idx >= start && src_idx <= end) {
        result[i] = x[src_idx];
      }
    }
  }

  return result;
}

// [[Rcpp::export]]
NumericVector conditional_sum_by_group_cpp(NumericVector x,
                                            IntegerVector condition,
                                            IntegerVector group1,
                                            IntegerVector group2,
                                            Nullable<IntegerVector> group3_ = R_NilValue) {
  // Sum x where condition == 1, grouped by (group1, group2, [group3])
  // Returns vector with group sum for each row
  int n = x.size();
  NumericVector result(n, 0.0);

  if (n == 0) return result;

  bool has_group3 = group3_.isNotNull();
  IntegerVector group3;
  if (has_group3) {
    group3 = group3_.get();
  }

  // Use hash map for group sums
  std::unordered_map<long long, double> group_sums;

  // First pass: compute sums
  for (int i = 0; i < n; i++) {
    if (condition[i] == 1 && !NumericVector::is_na(x[i])) {
      long long key;
      if (has_group3) {
        key = ((long long)group1[i] << 40) | ((long long)group2[i] << 20) | group3[i];
      } else {
        key = ((long long)group1[i] << 32) | group2[i];
      }
      group_sums[key] += x[i];
    }
  }

  // Second pass: assign sums back
  for (int i = 0; i < n; i++) {
    long long key;
    if (has_group3) {
      key = ((long long)group1[i] << 40) | ((long long)group2[i] << 20) | group3[i];
    } else {
      key = ((long long)group1[i] << 32) | group2[i];
    }
    auto it = group_sums.find(key);
    if (it != group_sums.end()) {
      result[i] = it->second;
    }
  }

  return result;
}

// [[Rcpp::export]]
NumericVector sum_by_group_cpp(NumericVector x, IntegerVector group) {
  // Simple sum of x by single group column
  int n = x.size();
  NumericVector result(n, 0.0);

  if (n == 0) return result;

  std::unordered_map<int, double> group_sums;

  // First pass: compute sums
  for (int i = 0; i < n; i++) {
    if (!NumericVector::is_na(x[i]) && !IntegerVector::is_na(group[i])) {
      group_sums[group[i]] += x[i];
    }
  }

  // Second pass: assign sums
  for (int i = 0; i < n; i++) {
    if (!IntegerVector::is_na(group[i])) {
      result[i] = group_sums[group[i]];
    }
  }

  return result;
}

// [[Rcpp::export]]
NumericVector mean_by_group_cpp(NumericVector x, IntegerVector group) {
  // Mean of x by single group column
  int n = x.size();
  NumericVector result(n, NA_REAL);

  if (n == 0) return result;

  std::unordered_map<int, double> group_sums;
  std::unordered_map<int, int> group_counts;

  // First pass: compute sums and counts
  for (int i = 0; i < n; i++) {
    if (!NumericVector::is_na(x[i]) && !IntegerVector::is_na(group[i])) {
      group_sums[group[i]] += x[i];
      group_counts[group[i]]++;
    }
  }

  // Second pass: assign means
  for (int i = 0; i < n; i++) {
    if (!IntegerVector::is_na(group[i])) {
      auto it = group_counts.find(group[i]);
      if (it != group_counts.end() && it->second > 0) {
        result[i] = group_sums[group[i]] / it->second;
      }
    }
  }

  return result;
}

// [[Rcpp::export]]
List compute_U_Gg_core_cpp(NumericVector diff_y,
                            NumericVector distance_to_switch,
                            NumericVector N_t_g,
                            NumericVector N_gt_control,
                            NumericVector never_change,
                            NumericVector N_gt,
                            IntegerVector time_XX,
                            NumericVector T_g,
                            IntegerVector group,
                            IntegerVector first_obs,
                            double G_XX,
                            double N_inc,
                            int i,
                            int t_min,
                            int T_max) {
  // Core computation of U_Gg variables
  // This is the main hot loop in did_multiplegt_dyn_core

  int n = diff_y.size();
  NumericVector U_Gg_temp(n, 0.0);
  NumericVector U_Gg(n, NA_REAL);
  NumericVector count_core(n, 0.0);

  if (N_inc == 0) {
    return List::create(
      Named("U_Gg_temp") = U_Gg_temp,
      Named("U_Gg") = U_Gg,
      Named("count_core") = count_core
    );
  }

  double G_over_N = G_XX / N_inc;

  // Compute U_Gg_temp
  for (int j = 0; j < n; j++) {
    // Check time window: time >= i+1 and time <= T_g
    if (time_XX[j] >= i + 1 && time_XX[j] <= T_g[j]) {
      // Check dummy_U_Gg: i <= T_g - 1
      if (i <= T_g[j] - 1) {
        double dist = distance_to_switch[j];
        double n_tg = N_t_g[j];
        double n_ctrl = N_gt_control[j];
        double never = never_change[j];
        double ngt = N_gt[j];
        double dy = diff_y[j];

        if (!NumericVector::is_na(dist) && !NumericVector::is_na(n_tg) &&
            !NumericVector::is_na(n_ctrl) && n_ctrl != 0 &&
            !NumericVector::is_na(never) && !NumericVector::is_na(ngt) &&
            !NumericVector::is_na(dy)) {

          double bracket = dist - (n_tg / n_ctrl) * never;
          U_Gg_temp[j] = G_over_N * ngt * bracket * dy;
        }
      }
    }
  }

  // Sum by group
  std::unordered_map<int, double> group_sums;
  for (int j = 0; j < n; j++) {
    if (!NumericVector::is_na(U_Gg_temp[j]) && !IntegerVector::is_na(group[j])) {
      group_sums[group[j]] += U_Gg_temp[j];
    }
  }

  // Assign group sums and multiply by first_obs
  for (int j = 0; j < n; j++) {
    if (!IntegerVector::is_na(group[j])) {
      double sum_val = group_sums[group[j]];
      U_Gg[j] = sum_val * first_obs[j];
    }
  }

  // Compute count_core
  for (int j = 0; j < n; j++) {
    double temp = U_Gg_temp[j];
    double dy = diff_y[j];
    double dist = distance_to_switch[j];
    double n_tg = N_t_g[j];
    double never = never_change[j];

    bool cond1 = !NumericVector::is_na(temp) && temp != 0;
    bool cond2 = temp == 0 && !NumericVector::is_na(dy) && dy == 0 &&
                 ((!NumericVector::is_na(dist) && dist != 0) ||
                  (!NumericVector::is_na(n_tg) && n_tg != 0 &&
                   !NumericVector::is_na(never) && never != 0));

    if (cond1 || cond2) {
      count_core[j] = N_gt[j];
    }
  }

  return List::create(
    Named("U_Gg_temp") = U_Gg_temp,
    Named("U_Gg") = U_Gg,
    Named("count_core") = count_core
  );
}

// [[Rcpp::export]]
NumericVector compute_U_Gg_var_temp_cpp(NumericVector diff_y,
                                         NumericVector E_hat_gt,
                                         NumericVector DOF_gt,
                                         NumericVector distance_to_switch,
                                         NumericVector N_t_g,
                                         NumericVector N_gt_control,
                                         NumericVector never_change,
                                         NumericVector N_gt,
                                         IntegerVector time_XX,
                                         NumericVector T_g,
                                         double G_XX,
                                         double N_inc,
                                         int i) {
  // Compute U_Gg_var_temp for variance estimation

  int n = diff_y.size();
  NumericVector result(n, 0.0);

  if (N_inc == 0) return result;

  double G_over_N = G_XX / N_inc;

  for (int j = 0; j < n; j++) {
    // Check time window and dummy condition
    if (time_XX[j] >= i + 1 && time_XX[j] <= T_g[j] && i <= T_g[j] - 1) {
      double dist = distance_to_switch[j];
      double n_tg = N_t_g[j];
      double n_ctrl = N_gt_control[j];
      double never = never_change[j];
      double ngt = N_gt[j];
      double dy = diff_y[j];
      double e_hat = E_hat_gt[j];
      double dof = DOF_gt[j];

      if (!NumericVector::is_na(dist) && !NumericVector::is_na(n_tg) &&
          !NumericVector::is_na(n_ctrl) && n_ctrl != 0 &&
          !NumericVector::is_na(never) && !NumericVector::is_na(ngt) &&
          !NumericVector::is_na(dy) && !NumericVector::is_na(e_hat) &&
          !NumericVector::is_na(dof)) {

        double bracket = dist - (n_tg / n_ctrl) * never;
        result[j] = G_over_N * bracket * ngt * dof * (dy - e_hat);
      }
    }
  }

  return result;
}

// [[Rcpp::export]]
IntegerVector compute_dof_indicator_cpp(NumericVector N_gt,
                                         NumericVector diff_y,
                                         NumericVector never_change,
                                         NumericVector N_t,
                                         int indicator_type) {
  // Compute DOF indicator columns
  // indicator_type: 1 = dof_ns (non-switchers), 2 = dof_s (switchers)

  int n = N_gt.size();
  IntegerVector result(n, 0);

  for (int j = 0; j < n; j++) {
    bool ngt_valid = !NumericVector::is_na(N_gt[j]) && N_gt[j] != 0;
    bool diff_valid = !NumericVector::is_na(diff_y[j]);

    if (indicator_type == 1) {
      // dof_ns: N_gt != 0, diff_y not NA, never_change == 1, N_t > 0
      bool never_valid = !NumericVector::is_na(never_change[j]) && never_change[j] == 1;
      bool nt_valid = !NumericVector::is_na(N_t[j]) && N_t[j] > 0;

      if (ngt_valid && diff_valid && never_valid && nt_valid) {
        result[j] = 1;
      }
    } else {
      // dof_s: N_gt != 0, never_change == 1 (distance_to_switch in this case)
      bool dist_valid = !NumericVector::is_na(never_change[j]) && never_change[j] == 1;

      if (ngt_valid && dist_valid) {
        result[j] = 1;
      }
    }
  }

  return result;
}

// [[Rcpp::export]]
NumericVector compute_cohort_mean_cpp(NumericVector values,
                                       NumericVector weights,
                                       IntegerVector dof_indicator,
                                       IntegerVector group1,
                                       IntegerVector group2,
                                       Nullable<IntegerVector> group3_ = R_NilValue) {
  // Compute weighted mean within cohorts where dof_indicator == 1
  // mean = sum(values * weights) / sum(weights) by groups

  int n = values.size();
  NumericVector result(n, NA_REAL);

  if (n == 0) return result;

  bool has_group3 = group3_.isNotNull();
  IntegerVector group3;
  if (has_group3) {
    group3 = group3_.get();
  }

  std::unordered_map<long long, double> sum_values;
  std::unordered_map<long long, double> sum_weights;

  // First pass: compute sums
  for (int i = 0; i < n; i++) {
    if (dof_indicator[i] == 1) {
      long long key;
      if (has_group3) {
        key = ((long long)group1[i] << 40) | ((long long)group2[i] << 20) | group3[i];
      } else {
        key = ((long long)group1[i] << 32) | group2[i];
      }

      double val = values[i];
      double wt = weights[i];

      if (!NumericVector::is_na(val) && !NumericVector::is_na(wt)) {
        sum_values[key] += val * wt;
        sum_weights[key] += wt;
      }
    }
  }

  // Second pass: compute means
  for (int i = 0; i < n; i++) {
    long long key;
    if (has_group3) {
      key = ((long long)group1[i] << 40) | ((long long)group2[i] << 20) | group3[i];
    } else {
      key = ((long long)group1[i] << 32) | group2[i];
    }

    auto it_w = sum_weights.find(key);
    if (it_w != sum_weights.end() && it_w->second > 0) {
      result[i] = sum_values[key] / it_w->second;
    }
  }

  return result;
}

// [[Rcpp::export]]
IntegerVector count_unique_by_group_cpp(IntegerVector values,
                                         IntegerVector dof_indicator,
                                         IntegerVector group1,
                                         IntegerVector group2,
                                         Nullable<IntegerVector> group3_ = R_NilValue) {
  // Count unique values within groups where dof_indicator == 1

  int n = values.size();
  IntegerVector result(n, NA_INTEGER);

  if (n == 0) return result;

  bool has_group3 = group3_.isNotNull();
  IntegerVector group3;
  if (has_group3) {
    group3 = group3_.get();
  }

  std::unordered_map<long long, std::set<int>> unique_values;

  // First pass: collect unique values
  for (int i = 0; i < n; i++) {
    if (dof_indicator[i] == 1 && !IntegerVector::is_na(values[i])) {
      long long key;
      if (has_group3) {
        key = ((long long)group1[i] << 40) | ((long long)group2[i] << 20) | group3[i];
      } else {
        key = ((long long)group1[i] << 32) | group2[i];
      }
      unique_values[key].insert(values[i]);
    }
  }

  // Second pass: assign counts
  for (int i = 0; i < n; i++) {
    long long key;
    if (has_group3) {
      key = ((long long)group1[i] << 40) | ((long long)group2[i] << 20) | group3[i];
    } else {
      key = ((long long)group1[i] << 32) | group2[i];
    }

    auto it = unique_values.find(key);
    if (it != unique_values.end()) {
      result[i] = it->second.size();
    }
  }

  return result;
}

// ============================================================================
// SECTION 5 OPTIMIZATIONS - MAIN.R VARIANCE COMPUTATION
// ============================================================================

// [[Rcpp::export]]
List compute_all_effects_cpp(NumericMatrix U_Gg_plus,
                              NumericMatrix U_Gg_minus,
                              NumericMatrix count_plus,
                              NumericMatrix count_minus,
                              NumericVector N1_vec,
                              NumericVector N0_vec,
                              IntegerVector first_obs_by_gp,
                              double G_XX,
                              int l_XX) {
  // Compute all DID_l effects in one pass
  // U_Gg_plus/minus: n x l_XX matrices of U_Gg values for switchers in/out
  // count_plus/minus: n x l_XX matrices of count values
  // N1_vec/N0_vec: weights for each effect (length l_XX)
  // Returns: DID estimates, counts, and U_Gg_global for each effect

  int n = U_Gg_plus.nrow();
  NumericMatrix U_Gg_global(n, l_XX);
  NumericMatrix count_global(n, l_XX);
  NumericVector DID_estimates(l_XX);
  NumericVector N_switchers(l_XX);
  NumericVector N_effects(l_XX);

  for (int i = 0; i < l_XX; i++) {
    double N1 = N1_vec[i];
    double N0 = N0_vec[i];
    double total_N = N1 + N0;
    N_switchers[i] = total_N;

    if (total_N == 0) {
      DID_estimates[i] = NA_REAL;
      continue;
    }

    double w_plus = N1 / total_N;
    double w_minus = N0 / total_N;

    double sum_U_Gg = 0.0;
    double sum_count = 0.0;

    for (int j = 0; j < n; j++) {
      // Compute U_Gg_global
      double u_plus = U_Gg_plus(j, i);
      double u_minus = U_Gg_minus(j, i);
      double u_global = 0.0;

      if (!NumericVector::is_na(u_plus) && !NumericVector::is_na(u_minus)) {
        u_global = w_plus * u_plus + w_minus * u_minus;
      } else if (!NumericVector::is_na(u_plus)) {
        u_global = w_plus * u_plus;
      } else if (!NumericVector::is_na(u_minus)) {
        u_global = w_minus * u_minus;
      }

      if (first_obs_by_gp[j] == 0) {
        u_global = NA_REAL;
      }
      U_Gg_global(j, i) = u_global;

      // Sum for DID estimate
      if (!NumericVector::is_na(u_global)) {
        sum_U_Gg += u_global;
      }

      // Compute count_global (max of plus and minus)
      double c_plus = count_plus(j, i);
      double c_minus = count_minus(j, i);
      double c_global = NA_REAL;

      if (!NumericVector::is_na(c_plus) && !NumericVector::is_na(c_minus)) {
        c_global = std::max(c_plus, c_minus);
      } else if (!NumericVector::is_na(c_plus)) {
        c_global = c_plus;
      } else if (!NumericVector::is_na(c_minus)) {
        c_global = c_minus;
      }

      if (!NumericVector::is_na(c_global) && c_global > 0) {
        sum_count += c_global;
      }
      count_global(j, i) = c_global;
    }

    DID_estimates[i] = sum_U_Gg / G_XX;
    N_effects[i] = sum_count;
  }

  return List::create(
    Named("DID_estimates") = DID_estimates,
    Named("U_Gg_global") = U_Gg_global,
    Named("count_global") = count_global,
    Named("N_switchers") = N_switchers,
    Named("N_effects") = N_effects
  );
}

// [[Rcpp::export]]
List compute_all_variances_cpp(NumericMatrix U_Gg_var_in,
                                NumericMatrix U_Gg_var_out,
                                NumericVector N1_vec,
                                NumericVector N0_vec,
                                IntegerVector first_obs_by_gp,
                                IntegerVector first_obs_by_clust,
                                IntegerVector cluster_XX,
                                double G_XX,
                                int l_XX,
                                bool clustered) {
  // Compute all variances in one pass
  // Returns: SE estimates and U_Gg_var_glob for each effect

  int n = U_Gg_var_in.nrow();
  NumericMatrix U_Gg_var_glob(n, l_XX);
  NumericVector SE_estimates(l_XX);

  for (int i = 0; i < l_XX; i++) {
    double N1 = N1_vec[i];
    double N0 = N0_vec[i];
    double total_N = N1 + N0;

    if (total_N == 0) {
      SE_estimates[i] = NA_REAL;
      continue;
    }

    double w_plus = N1 / total_N;
    double w_minus = N0 / total_N;

    // Compute U_Gg_var_glob
    for (int j = 0; j < n; j++) {
      double v_in = U_Gg_var_in(j, i);
      double v_out = U_Gg_var_out(j, i);

      double v_glob = 0.0;
      if (!NumericVector::is_na(v_in)) {
        v_glob += w_plus * v_in;
      }
      if (!NumericVector::is_na(v_out)) {
        v_glob += w_minus * v_out;
      }
      U_Gg_var_glob(j, i) = v_glob;
    }

    // Compute variance
    double sum_sq = 0.0;

    if (!clustered) {
      // Non-clustered case
      for (int j = 0; j < n; j++) {
        if (first_obs_by_gp[j] == 1) {
          double val = U_Gg_var_glob(j, i);
          if (!NumericVector::is_na(val)) {
            sum_sq += val * val;
          }
        }
      }
    } else {
      // Clustered case
      // Step 1: Multiply by first_obs_by_gp
      std::vector<double> U_masked(n);
      for (int j = 0; j < n; j++) {
        U_masked[j] = U_Gg_var_glob(j, i) * first_obs_by_gp[j];
      }

      // Step 2: Sum within clusters
      std::map<int, double> cluster_sums;
      for (int j = 0; j < n; j++) {
        if (!IntegerVector::is_na(cluster_XX[j]) && !NumericVector::is_na(U_masked[j])) {
          cluster_sums[cluster_XX[j]] += U_masked[j];
        }
      }

      // Step 3: Compute sum of squared cluster sums
      std::set<int> counted_clusters;
      for (int j = 0; j < n; j++) {
        if (first_obs_by_clust[j] == 1 && !IntegerVector::is_na(cluster_XX[j])) {
          int clust = cluster_XX[j];
          if (counted_clusters.find(clust) == counted_clusters.end()) {
            double cs = cluster_sums[clust];
            sum_sq += cs * cs;
            counted_clusters.insert(clust);
          }
        }
      }

      // Update U_Gg_var_glob with cluster sums
      for (int j = 0; j < n; j++) {
        if (!IntegerVector::is_na(cluster_XX[j])) {
          U_Gg_var_glob(j, i) = cluster_sums[cluster_XX[j]];
        }
      }
    }

    double variance = sum_sq / (G_XX * G_XX);
    SE_estimates[i] = std::sqrt(variance);
  }

  return List::create(
    Named("SE_estimates") = SE_estimates,
    Named("U_Gg_var_glob") = U_Gg_var_glob
  );
}

// [[Rcpp::export]]
List compute_placebo_effects_and_variances_cpp(
    NumericMatrix U_Gg_pl_plus,
    NumericMatrix U_Gg_pl_minus,
    NumericMatrix count_pl_plus,
    NumericMatrix count_pl_minus,
    NumericMatrix U_Gg_var_pl_in,
    NumericMatrix U_Gg_var_pl_out,
    NumericVector N1_pl_vec,
    NumericVector N0_pl_vec,
    IntegerVector first_obs_by_gp,
    IntegerVector first_obs_by_clust,
    IntegerVector cluster_XX,
    double G_XX,
    int l_placebo_XX,
    bool clustered) {
  // Compute all placebo effects and variances in one pass

  int n = U_Gg_pl_plus.nrow();
  NumericVector DID_pl_estimates(l_placebo_XX);
  NumericVector SE_pl_estimates(l_placebo_XX);
  NumericVector N_switchers_pl(l_placebo_XX);
  NumericVector N_effects_pl(l_placebo_XX);
  NumericMatrix U_Gg_var_glob_pl(n, l_placebo_XX);

  for (int i = 0; i < l_placebo_XX; i++) {
    double N1 = N1_pl_vec[i];
    double N0 = N0_pl_vec[i];
    double total_N = N1 + N0;
    N_switchers_pl[i] = total_N;

    if (total_N == 0) {
      DID_pl_estimates[i] = NA_REAL;
      SE_pl_estimates[i] = NA_REAL;
      continue;
    }

    double w_plus = N1 / total_N;
    double w_minus = N0 / total_N;

    // Compute DID placebo estimate
    double sum_U_Gg = 0.0;
    double sum_count = 0.0;

    for (int j = 0; j < n; j++) {
      double u_plus = U_Gg_pl_plus(j, i);
      double u_minus = U_Gg_pl_minus(j, i);
      double u_global = 0.0;

      if (!NumericVector::is_na(u_plus) && !NumericVector::is_na(u_minus)) {
        u_global = w_plus * u_plus + w_minus * u_minus;
      } else if (!NumericVector::is_na(u_plus)) {
        u_global = w_plus * u_plus;
      } else if (!NumericVector::is_na(u_minus)) {
        u_global = w_minus * u_minus;
      }

      if (first_obs_by_gp[j] == 0) {
        u_global = NA_REAL;
      }

      if (!NumericVector::is_na(u_global)) {
        sum_U_Gg += u_global;
      }

      // Compute count
      double c_plus = count_pl_plus(j, i);
      double c_minus = count_pl_minus(j, i);
      double c_global = NA_REAL;

      if (!NumericVector::is_na(c_plus) && !NumericVector::is_na(c_minus)) {
        c_global = std::max(c_plus, c_minus);
      } else if (!NumericVector::is_na(c_plus)) {
        c_global = c_plus;
      } else if (!NumericVector::is_na(c_minus)) {
        c_global = c_minus;
      }

      if (!NumericVector::is_na(c_global) && c_global > 0) {
        sum_count += c_global;
      }
    }

    DID_pl_estimates[i] = sum_U_Gg / G_XX;
    N_effects_pl[i] = sum_count;

    // Compute variance
    for (int j = 0; j < n; j++) {
      double v_in = U_Gg_var_pl_in(j, i);
      double v_out = U_Gg_var_pl_out(j, i);

      double v_glob = 0.0;
      if (!NumericVector::is_na(v_in)) {
        v_glob += w_plus * v_in;
      }
      if (!NumericVector::is_na(v_out)) {
        v_glob += w_minus * v_out;
      }
      U_Gg_var_glob_pl(j, i) = v_glob;
    }

    double sum_sq = 0.0;

    if (!clustered) {
      for (int j = 0; j < n; j++) {
        if (first_obs_by_gp[j] == 1) {
          double val = U_Gg_var_glob_pl(j, i);
          if (!NumericVector::is_na(val)) {
            sum_sq += val * val;
          }
        }
      }
    } else {
      std::vector<double> U_masked(n);
      for (int j = 0; j < n; j++) {
        U_masked[j] = U_Gg_var_glob_pl(j, i) * first_obs_by_gp[j];
      }

      std::map<int, double> cluster_sums;
      for (int j = 0; j < n; j++) {
        if (!IntegerVector::is_na(cluster_XX[j]) && !NumericVector::is_na(U_masked[j])) {
          cluster_sums[cluster_XX[j]] += U_masked[j];
        }
      }

      std::set<int> counted_clusters;
      for (int j = 0; j < n; j++) {
        if (first_obs_by_clust[j] == 1 && !IntegerVector::is_na(cluster_XX[j])) {
          int clust = cluster_XX[j];
          if (counted_clusters.find(clust) == counted_clusters.end()) {
            double cs = cluster_sums[clust];
            sum_sq += cs * cs;
            counted_clusters.insert(clust);
          }
        }
      }

      for (int j = 0; j < n; j++) {
        if (!IntegerVector::is_na(cluster_XX[j])) {
          U_Gg_var_glob_pl(j, i) = cluster_sums[cluster_XX[j]];
        }
      }
    }

    double variance = sum_sq / (G_XX * G_XX);
    SE_pl_estimates[i] = std::sqrt(variance);
  }

  return List::create(
    Named("DID_pl_estimates") = DID_pl_estimates,
    Named("SE_pl_estimates") = SE_pl_estimates,
    Named("N_switchers_pl") = N_switchers_pl,
    Named("N_effects_pl") = N_effects_pl,
    Named("U_Gg_var_glob_pl") = U_Gg_var_glob_pl
  );
}

// [[Rcpp::export]]
NumericMatrix compute_vcov_full_cpp(NumericMatrix U_Gg_var_glob,
                                     IntegerVector first_obs,
                                     NumericVector se_vec,
                                     bool normalized,
                                     Nullable<NumericVector> delta_D_global_ = R_NilValue,
                                     double G_XX = 1.0) {
  // Compute full variance-covariance matrix efficiently
  int l_XX = U_Gg_var_glob.ncol();
  int n = U_Gg_var_glob.nrow();
  NumericMatrix vcov(l_XX, l_XX);
  double G_XX_sq = G_XX * G_XX;

  NumericVector delta_D_global;
  bool has_delta = delta_D_global_.isNotNull();
  if (has_delta) {
    delta_D_global = delta_D_global_.get();
  }

  // Fill diagonal with squared SEs
  for (int i = 0; i < l_XX; i++) {
    vcov(i, i) = se_vec[i] * se_vec[i];
  }

  // Compute off-diagonal covariances
  for (int i = 0; i < l_XX - 1; i++) {
    for (int k = i + 1; k < l_XX; k++) {
      double sum_sq = 0.0;

      for (int j = 0; j < n; j++) {
        if (first_obs[j] == 1) {
          double val_i = U_Gg_var_glob(j, i);
          double val_k = U_Gg_var_glob(j, k);

          if (normalized && has_delta) {
            if (!NumericVector::is_na(delta_D_global[i]) && delta_D_global[i] != 0) {
              val_i = val_i / delta_D_global[i];
            }
            if (!NumericVector::is_na(delta_D_global[k]) && delta_D_global[k] != 0) {
              val_k = val_k / delta_D_global[k];
            }
          }

          if (!NumericVector::is_na(val_i) && !NumericVector::is_na(val_k)) {
            double combined = val_i + val_k;
            sum_sq += combined * combined;
          }
        }
      }

      double var_sum = sum_sq / G_XX_sq;
      double cov = (var_sum - vcov(i, i) - vcov(k, k)) / 2.0;
      vcov(i, k) = cov;
      vcov(k, i) = cov;
    }
  }

  return vcov;
}

// [[Rcpp::export]]
List compute_avg_effect_cpp(NumericVector U_Gg_plus,
                             NumericVector U_Gg_minus,
                             NumericVector U_Gg_var_plus,
                             NumericVector U_Gg_var_minus,
                             IntegerVector first_obs_by_gp,
                             IntegerVector first_obs_by_clust,
                             IntegerVector cluster_XX,
                             double w_plus,
                             double G_XX,
                             bool clustered) {
  // Compute average total effect and its variance
  int n = U_Gg_plus.size();

  // Compute U_Gg_global
  NumericVector U_Gg_global(n);
  double sum_U_Gg = 0.0;

  for (int j = 0; j < n; j++) {
    double u_g = w_plus * U_Gg_plus[j] + (1.0 - w_plus) * U_Gg_minus[j];
    if (first_obs_by_gp[j] == 0) {
      u_g = NA_REAL;
    }
    U_Gg_global[j] = u_g;

    if (!NumericVector::is_na(u_g)) {
      sum_U_Gg += u_g;
    }
  }

  double delta_XX = sum_U_Gg / G_XX;

  // Compute variance
  NumericVector U_Gg_var_global(n);
  for (int j = 0; j < n; j++) {
    U_Gg_var_global[j] = w_plus * U_Gg_var_plus[j] + (1.0 - w_plus) * U_Gg_var_minus[j];
  }

  double sum_sq = 0.0;

  if (!clustered) {
    for (int j = 0; j < n; j++) {
      if (first_obs_by_gp[j] == 1) {
        double val = U_Gg_var_global[j];
        if (!NumericVector::is_na(val)) {
          sum_sq += val * val;
        }
      }
    }
  } else {
    std::vector<double> U_masked(n);
    for (int j = 0; j < n; j++) {
      U_masked[j] = U_Gg_var_global[j] * first_obs_by_gp[j];
    }

    std::map<int, double> cluster_sums;
    for (int j = 0; j < n; j++) {
      if (!IntegerVector::is_na(cluster_XX[j]) && !NumericVector::is_na(U_masked[j])) {
        cluster_sums[cluster_XX[j]] += U_masked[j];
      }
    }

    std::set<int> counted;
    for (int j = 0; j < n; j++) {
      if (first_obs_by_clust[j] == 1 && !IntegerVector::is_na(cluster_XX[j])) {
        int clust = cluster_XX[j];
        if (counted.find(clust) == counted.end()) {
          double cs = cluster_sums[clust];
          sum_sq += cs * cs;
          counted.insert(clust);
        }
      }
    }
  }

  double variance = sum_sq / (G_XX * G_XX);
  double se_XX = std::sqrt(variance);

  return List::create(
    Named("delta_XX") = delta_XX,
    Named("se_XX") = se_XX,
    Named("U_Gg_global") = U_Gg_global,
    Named("U_Gg_var_global") = U_Gg_var_global
  );
}

// [[Rcpp::export]]
List same_switchers_loop_cpp(NumericVector outcome,
                              IntegerVector group,
                              IntegerVector time,
                              NumericVector F_g,
                              NumericVector N_gt,
                              IntegerVector d_sq,
                              int effects,
                              int T_max,
                              bool only_never_switchers) {
  // Optimized same_switchers loop
  // Returns N_g_control_check_XX for each group

  int n = outcome.size();
  NumericVector N_g_control_check(n, 0.0);

  // Pre-sort indices by group for efficient lookup
  std::unordered_map<int, std::vector<int>> group_indices;
  for (int i = 0; i < n; i++) {
    group_indices[group[i]].push_back(i);
  }

  for (int q = 1; q <= effects; q++) {
    // Compute diff_y_last (lag by q)
    NumericVector diff_y_last(n, NA_REAL);
    for (auto& kv : group_indices) {
      std::vector<int>& indices = kv.second;
      for (size_t j = q; j < indices.size(); j++) {
        int idx = indices[j];
        int lag_idx = indices[j - q];
        if (!NumericVector::is_na(outcome[idx]) && !NumericVector::is_na(outcome[lag_idx])) {
          diff_y_last[idx] = outcome[idx] - outcome[lag_idx];
        }
      }
    }

    // Compute never_change_d_last
    NumericVector never_change_last(n, NA_REAL);
    for (int i = 0; i < n; i++) {
      if (!NumericVector::is_na(diff_y_last[i]) && F_g[i] > time[i]) {
        never_change_last[i] = 1.0;
      }
      if (only_never_switchers && F_g[i] > time[i] && F_g[i] < T_max + 1 && !NumericVector::is_na(diff_y_last[i])) {
        never_change_last[i] = 0.0;
      }
    }

    // Compute N_gt_control_last by (time, d_sq)
    std::map<std::pair<int, int>, double> control_sums;
    for (int i = 0; i < n; i++) {
      if (!NumericVector::is_na(never_change_last[i]) && !NumericVector::is_na(N_gt[i])) {
        auto key = std::make_pair(time[i], d_sq[i]);
        control_sums[key] += never_change_last[i] * N_gt[i];
      }
    }

    NumericVector N_gt_control_last(n, 0.0);
    for (int i = 0; i < n; i++) {
      auto key = std::make_pair(time[i], d_sq[i]);
      auto it = control_sums.find(key);
      if (it != control_sums.end()) {
        N_gt_control_last[i] = it->second;
      }
    }

    // Compute N_g_control_last_m (mean where time == F_g - 1 + q)
    std::unordered_map<int, double> group_ctrl_sum;
    std::unordered_map<int, int> group_ctrl_count;
    for (int i = 0; i < n; i++) {
      if (time[i] == F_g[i] - 1 + q) {
        group_ctrl_sum[group[i]] += N_gt_control_last[i];
        group_ctrl_count[group[i]]++;
      }
    }

    NumericVector N_g_control_last_m(n, NA_REAL);
    for (int i = 0; i < n; i++) {
      auto it = group_ctrl_count.find(group[i]);
      if (it != group_ctrl_count.end() && it->second > 0) {
        N_g_control_last_m[i] = group_ctrl_sum[group[i]] / it->second;
      }
    }

    // Compute diff_y_relev (mean where time == F_g - 1 + q)
    std::unordered_map<int, double> group_dy_sum;
    std::unordered_map<int, int> group_dy_count;
    for (int i = 0; i < n; i++) {
      if (time[i] == F_g[i] - 1 + q && !NumericVector::is_na(diff_y_last[i])) {
        group_dy_sum[group[i]] += diff_y_last[i];
        group_dy_count[group[i]]++;
      }
    }

    NumericVector diff_y_relev(n, NA_REAL);
    for (int i = 0; i < n; i++) {
      auto it = group_dy_count.find(group[i]);
      if (it != group_dy_count.end() && it->second > 0) {
        diff_y_relev[i] = group_dy_sum[group[i]] / it->second;
      }
    }

    // Update N_g_control_check
    for (int i = 0; i < n; i++) {
      if (!NumericVector::is_na(N_g_control_last_m[i]) && N_g_control_last_m[i] > 0 &&
          !NumericVector::is_na(diff_y_relev[i])) {
        N_g_control_check[i] += 1.0;
      }
    }
  }

  return List::create(Named("N_g_control_check") = N_g_control_check);
}
