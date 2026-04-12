#' @noRd
NULL

# Data.table Helper Functions for DIDmultiplegtDYN
#
# This file contains utility functions to perform common operations on data.table
# DataFrames. These are the data.table equivalents of the former polars helpers.

#' Batch drop columns that exist
#' @param df data.table
#' @param col_names vector of column names to drop
#' @return data.table (modified in place)
#' @noRd
dt_batch_drop_cols <- function(df, col_names) {
  existing <- intersect(col_names, names(df))
  if (length(existing) > 0L) {
    df[, (existing) := NULL]
  }
  invisible(df)
}

#' Compute scalar sum (optionally filtered)
#' @param df data.table
#' @param col_name column to sum
#' @param filter_idx optional logical vector for filtering
#' @return numeric scalar
#' @noRd
dt_scalar_sum <- function(df, col_name, filter_idx = NULL) {
  if (!is.null(filter_idx)) {
    result <- sum(df[[col_name]][filter_idx], na.rm = TRUE)
  } else {
    result <- sum(df[[col_name]], na.rm = TRUE)
  }
  return(if (is.na(result)) 0 else result)
}

#' Compute scalar mean-then-sum from filtered and grouped data
#' Equivalent of: filter -> group_by(by_col) -> mean(col) -> sum()
#' @param df data.table
#' @param col_name column to average
#' @param filter_idx logical vector for filter
#' @param by_col grouping column
#' @return numeric scalar
#' @noRd
dt_scalar_mean_sum <- function(df, col_name, filter_idx, by_col) {
  sub <- df[filter_idx, ]
  if (nrow(sub) == 0L) return(0)
  agg <- sub[, .(m = mean(get(col_name), na.rm = TRUE)), by = by_col]
  result <- sum(agg[["m"]], na.rm = TRUE)
  return(if (is.na(result)) 0 else result)
}

#' Conditional aggregation over groups (masked window function)
#' @param df data.table
#' @param value_col column to aggregate
#' @param filter_idx logical vector (TRUE = include, FALSE/NA = mask to NA)
#' @param by_cols grouping columns
#' @param new_col name for result
#' @param agg_func aggregation function name ("sum", "mean")
#' @param filter_result if TRUE, set result to NA where filter is FALSE
#' @return data.table (modified in place)
#' @noRd
dt_filtered_agg_over <- function(df, value_col, filter_idx, by_cols, new_col, agg_func = "sum",
                                  filter_result = FALSE) {
  # Write masked values directly to target column (avoids temp column create/delete)
  data.table::set(df, j = new_col, value = data.table::fifelse(filter_idx, df[[value_col]], NA_real_))

  if (agg_func == "sum") {
    df[, (new_col) := sum(get(new_col), na.rm = TRUE), by = by_cols]
  } else if (agg_func == "mean") {
    df[, (new_col) := mean(get(new_col), na.rm = TRUE), by = by_cols]
  } else if (agg_func == "count") {
    df[, (new_col) := sum(!is.na(get(new_col))), by = by_cols]
  } else {
    df[, (new_col) := sum(get(new_col), na.rm = TRUE), by = by_cols]
  }

  if (filter_result) {
    df[!filter_idx, (new_col) := NA_real_]
  }

  invisible(df)
}

#' Compute uniqueN over groups (count distinct non-NA values)
#' @param df data.table
#' @param count_col column to count unique values
#' @param by_cols grouping columns
#' @param new_col result column name
#' @param filter_idx optional logical vector filter
#' @return data.table (modified in place)
#' @noRd
dt_uniqueN_over <- function(df, count_col, by_cols, new_col, filter_idx = NULL) {
  if (!is.null(filter_idx)) {
    masked <- data.table::fifelse(filter_idx, df[[count_col]], NA)
    temp_col <- paste0("__masked_uniq_", new_col, "__")
    data.table::set(df, j = temp_col, value = masked)
    df[, (new_col) := data.table::uniqueN(get(temp_col), na.rm = TRUE), by = by_cols]
    df[, (temp_col) := NULL]
  } else {
    df[, (new_col) := data.table::uniqueN(get(count_col), na.rm = TRUE), by = by_cols]
  }
  invisible(df)
}

#' Compute group-boundary row indices for vectorized shift
#' For data sorted by (group_col, time_col) with contiguous groups,
#' finds the first `n` rows of each group that must be set to NA
#' after a global lag operation.
#' @param group_vec integer/numeric vector of group IDs (sorted, contiguous)
#' @param n shift amount
#' @param N_rows total number of rows
#' @return integer vector of row indices to set to NA
#' @noRd
dt_shift_boundary_rows <- function(group_vec, n, N_rows) {
  # Find where groups change
  boundary_starts <- which(c(TRUE, group_vec[-1L] != group_vec[-N_rows]))
  # For each boundary, rows boundary_start:(boundary_start + n - 1) need NA
  fix_rows <- as.integer(rep(boundary_starts, each = n) + rep(0:(n - 1L), times = length(boundary_starts)))
  fix_rows[fix_rows <= N_rows]
}

#' Vectorized group-aware lag (no data.table by-group dispatch)
#' Equivalent to shift(x, n, type = "lag") by group, but operates on
#' a pre-sorted vector with contiguous groups. Avoids per-call by-group overhead.
#' @param x numeric vector
#' @param n lag amount (positive integer)
#' @param fix_rows pre-computed boundary rows from dt_shift_boundary_rows
#' @param N_rows length of x
#' @return numeric vector: x lagged by n within groups, NA at boundaries
#' @noRd
dt_vec_shift <- function(x, n, fix_rows, N_rows) {
  shifted <- c(rep(NA_real_, n), x[1:(N_rows - n)])
  shifted[fix_rows] <- NA_real_
  shifted
}

