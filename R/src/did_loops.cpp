#include <Rcpp.h>
#include <unordered_map>
#include <vector>
using namespace Rcpp;

// [[Rcpp::export]]
List bootstrap_prepare_groups_cpp(IntegerVector group) {
  // Pre-compute group structure for fast repeated bootstrap sampling
  // Returns: list with group_ids, group_sizes, and row_indices per group

  const int n = group.size();

  std::unordered_map<int, std::vector<int> > group_indices;
  for (int i = 0; i < n; i++) {
    if (!IntegerVector::is_na(group[i])) {
      group_indices[group[i]].push_back(i);
    }
  }

  const int n_groups = group_indices.size();
  IntegerVector group_ids(n_groups);
  IntegerVector group_sizes(n_groups);
  List row_indices(n_groups);

  int idx = 0;
  for (auto& kv : group_indices) {
    group_ids[idx] = kv.first;
    group_sizes[idx] = kv.second.size();
    row_indices[idx] = IntegerVector(kv.second.begin(), kv.second.end());
    idx++;
  }

  return List::create(
    Named("group_ids") = group_ids,
    Named("group_sizes") = group_sizes,
    Named("row_indices") = row_indices,
    Named("n_groups") = n_groups,
    Named("n_rows") = n
  );
}

// [[Rcpp::export]]
IntegerVector bootstrap_sample_indices_cpp(List group_info) {
  // Fast bootstrap sampling using pre-computed group structure
  // Samples groups with replacement and returns all row indices
  // Note: Uses R's RNG so set.seed() in R controls reproducibility

  List row_indices = group_info["row_indices"];
  const int n_groups = group_info["n_groups"];
  const int n_rows = group_info["n_rows"];

  std::vector<int> result;
  result.reserve(n_rows);

  for (int i = 0; i < n_groups; i++) {
    int sampled_group = (int)(R::runif(0.0, 1.0) * n_groups);
    if (sampled_group >= n_groups) sampled_group = n_groups - 1;

    IntegerVector indices = row_indices[sampled_group];
    for (int j = 0; j < indices.size(); j++) {
      result.push_back(indices[j]);
    }
  }

  return IntegerVector(result.begin(), result.end());
}
