#' @noRd
NULL

# Polars Helper Functions for DIDmultiplegtDYN
#
# This file contains utility functions to perform common operations on polars
# DataFrames that would normally require data.frame syntax.
#
# Note: polars is a suggested package. These helpers assume polars is loaded
# by the calling function (did_multiplegt_main checks for polars availability).

# Get pl object from polars namespace (called by functions that need pl$)
.ensure_pl <- function() {
  if (!exists("pl", envir = parent.frame())) {
    if (!.polars_available()) {
      stop("polars is required. Install with: install.packages('polars', repos = 'https://rpolars.r-universe.dev')")
    }
    assign("pl", .get_pl(), envir = parent.frame())
  }
}

#' Apply over() window function with dynamic column names
#' @param expr polars expression (e.g., pl$col("x")$sum())
#' @param by_cols vector of column names to group by
#' @return polars expression with over() applied
#' @noRd
pl_over_cols <- function(expr, by_cols) {
  col_exprs <- lapply(by_cols, function(x) pl$col(x))
  do.call(expr$over, col_exprs)
}

#' Extract scalar value from polars select result
#' @param df_result result from df$select() call
#' @return scalar value
#' @noRd
pl_scalar <- function(df_result) {
  as.data.frame(df_result)[[1]][1]
}

#' Set a column to a constant value
#' @param df polars DataFrame
#' @param col_name column name
#' @param value value to set
#' @return modified polars DataFrame
#' @noRd
pl_set_col <- function(df, col_name, value) {
  df$with_columns(pl$lit(value)$alias(col_name))
}

#' Set a column using another column's values
#' @param df polars DataFrame
#' @param target_col target column name
#' @param source_col source column name
#' @return modified polars DataFrame
#' @noRd
pl_copy_col <- function(df, target_col, source_col) {
  df$with_columns(pl$col(source_col)$alias(target_col))
}

#' Get column as R vector
#' @param df polars DataFrame
#' @param col_name column name
#' @return R vector
#' @noRd
pl_get_col <- function(df, col_name) {
  as.vector(df$get_column(col_name))
}

#' Conditional update: set col to new_val where condition is TRUE
#' @param df polars DataFrame
#' @param col_name column to update
#' @param condition polars expression for condition
#' @param new_val new value when condition is TRUE
#' @return modified polars DataFrame
#' @noRd
pl_set_where <- function(df, col_name, condition, new_val) {
  df$with_columns(
    pl$when(condition)$then(pl$lit(new_val))$otherwise(pl$col(col_name))$alias(col_name)
  )
}

#' Conditional update using column expression for new value
#' @param df polars DataFrame
#' @param col_name column to update
#' @param condition polars expression for condition
#' @param new_expr polars expression for new value
#' @return modified polars DataFrame
#' @noRd
pl_set_where_expr <- function(df, col_name, condition, new_expr) {
  df$with_columns(
    pl$when(condition)$then(new_expr)$otherwise(pl$col(col_name))$alias(col_name)
  )
}

#' Set column to NA where condition is TRUE
#' @param df polars DataFrame
#' @param col_name column to update
#' @param condition polars expression for condition
#' @return modified polars DataFrame
#' @noRd
pl_set_na_where <- function(df, col_name, condition) {
  df$with_columns(
    pl$when(condition)$then(pl$lit(NA_real_))$otherwise(pl$col(col_name))$alias(col_name)
  )
}

#' Compute sum of a column
#' @param df polars DataFrame
#' @param col_name column name
#' @param na_rm remove NA values
#' @return numeric scalar
#' @noRd
pl_sum <- function(df, col_name, na_rm = TRUE) {
  if (na_rm) {
    result <- as.data.frame(df$select(pl$col(col_name)$drop_nulls()$sum()))[1, 1]
  } else {
    result <- as.data.frame(df$select(pl$col(col_name)$sum()))[1, 1]
  }
  if (is.null(result) || is.na(result)) return(0)
  return(result)
}

#' Compute mean of a column
#' @param df polars DataFrame
#' @param col_name column name
#' @param na_rm remove NA values
#' @return numeric scalar
#' @noRd
pl_mean <- function(df, col_name, na_rm = TRUE) {
  if (na_rm) {
    result <- as.data.frame(df$select(pl$col(col_name)$drop_nulls()$mean()))[1, 1]
  } else {
    result <- as.data.frame(df$select(pl$col(col_name)$mean()))[1, 1]
  }
  if (is.null(result)) return(NA_real_)
  return(result)
}

#' Compute max of a column
#' @param df polars DataFrame
#' @param col_name column name
#' @param na_rm remove NA values
#' @return numeric scalar
#' @noRd
pl_max <- function(df, col_name, na_rm = TRUE) {
  if (na_rm) {
    result <- as.data.frame(df$select(pl$col(col_name)$drop_nulls()$max()))[1, 1]
  } else {
    result <- as.data.frame(df$select(pl$col(col_name)$max()))[1, 1]
  }
  if (is.null(result)) return(NA_real_)
  return(result)
}

#' Compute min of a column
#' @param df polars DataFrame
#' @param col_name column name
#' @param na_rm remove NA values
#' @return numeric scalar
#' @noRd
pl_min <- function(df, col_name, na_rm = TRUE) {
  if (na_rm) {
    result <- as.data.frame(df$select(pl$col(col_name)$drop_nulls()$min()))[1, 1]
  } else {
    result <- as.data.frame(df$select(pl$col(col_name)$min()))[1, 1]
  }
  if (is.null(result)) return(NA_real_)
  return(result)
}

#' Count rows in dataframe
#' @param df polars DataFrame
#' @return integer count
#' @noRd
pl_nrow <- function(df) {
  df$height
}

#' Get column names
#' @param df polars DataFrame
#' @return character vector of column names
#' @noRd
pl_colnames <- function(df) {
  df$columns
}

#' Check if column exists
#' @param df polars DataFrame
#' @param col_name column name
#' @return logical
#' @noRd
pl_has_col <- function(df, col_name) {
  col_name %in% df$columns
}

#' Arithmetic operation: create new column as result of operation
#' @param df polars DataFrame
#' @param new_col name of new column
#' @param expr polars expression
#' @return modified polars DataFrame
#' @noRd
pl_mutate <- function(df, new_col, expr) {
  df$with_columns(expr$alias(new_col))
}

#' Filter rows
#' @param df polars DataFrame
#' @param condition polars expression
#' @return filtered polars DataFrame
#' @noRd
pl_filter <- function(df, condition) {
  df$filter(condition)
}

#' Sort dataframe
#' @param df polars DataFrame
#' @param by_cols columns to sort by
#' @param descending logical vector for descending order
#' @return sorted polars DataFrame
#' @noRd
pl_sort <- function(df, by_cols, descending = FALSE) {
  df$sort(by_cols, descending = descending)
}

#' Group by and aggregate
#' @param df polars DataFrame
#' @param by_cols columns to group by
#' @param agg_exprs list of aggregation expressions
#' @return aggregated polars DataFrame
#' @noRd
pl_group_agg <- function(df, by_cols, agg_exprs) {
  if (is.list(agg_exprs)) {
    do.call(function(...) df$group_by(by_cols)$agg(...), agg_exprs)
  } else {
    df$group_by(by_cols)$agg(agg_exprs)
  }
}

#' Compute window function (sum over groups) and add as new column
#' @param df polars DataFrame
#' @param col_name column to aggregate
#' @param by_cols columns to partition by
#' @param new_col name for new column
#' @param agg_func aggregation function ("sum", "mean", "max", "min", "count")
#' @return modified polars DataFrame
#' @noRd
pl_window_agg <- function(df, col_name, by_cols, new_col, agg_func = "sum") {
  # Use group_by approach to avoid duplicate column issues
  agg_expr <- switch(agg_func,
    "sum" = pl$col(col_name)$sum(),
    "mean" = pl$col(col_name)$mean(),
    "max" = pl$col(col_name)$max(),
    "min" = pl$col(col_name)$min(),
    "count" = pl$col(col_name)$count(),
    pl$col(col_name)$sum()  # default to sum
  )

  agg_df <- df$group_by(by_cols)$agg(agg_expr$alias(new_col))
  df$join(agg_df, on = by_cols, how = "left")
}

#' Compute lag within groups
#' @param df polars DataFrame
#' @param col_name column to lag
#' @param n number of periods to lag
#' @param by_col grouping column
#' @param order_col column to order by
#' @param new_col name for new column
#' @return modified polars DataFrame
#' @noRd
pl_lag <- function(df, col_name, n, by_col, order_col, new_col) {
  df <- df$sort(c(by_col, order_col))
  df$with_columns(
    pl$col(col_name)$shift(n)$over(by_col)$alias(new_col)
  )
}

#' Compute lead within groups
#' @param df polars DataFrame
#' @param col_name column to lead
#' @param n number of periods to lead
#' @param by_col grouping column
#' @param order_col column to order by
#' @param new_col name for new column
#' @return modified polars DataFrame
#' @noRd
pl_lead <- function(df, col_name, n, by_col, order_col, new_col) {
  df <- df$sort(c(by_col, order_col))
  df$with_columns(
    pl$col(col_name)$shift(-n)$over(by_col)$alias(new_col)
  )
}

#' Replace NA values with a constant
#' @param df polars DataFrame
#' @param col_name column name
#' @param replacement value to replace NA with
#' @return modified polars DataFrame
#' @noRd
pl_fill_na <- function(df, col_name, replacement) {
  df$with_columns(
    pl$col(col_name)$fill_null(replacement)$alias(col_name)
  )
}

#' Create column with ifelse logic
#' @param df polars DataFrame
#' @param new_col new column name
#' @param condition polars expression for condition
#' @param true_val value when TRUE
#' @param false_val value when FALSE
#' @return modified polars DataFrame
#' @noRd
pl_ifelse <- function(df, new_col, condition, true_val, false_val) {
  # Handle case where true_val and false_val might be expressions or literals
  true_expr <- if (inherits(true_val, "Expr")) true_val else pl$lit(true_val)
  false_expr <- if (inherits(false_val, "Expr")) false_val else pl$lit(false_val)

  df$with_columns(
    pl$when(condition)$then(true_expr)$otherwise(false_expr)$alias(new_col)
  )
}

#' Compute column as arithmetic of two columns
#' @param df polars DataFrame
#' @param new_col new column name
#' @param col1 first column name
#' @param col2 second column name (or numeric constant)
#' @param op operation: "+", "-", "*", "/"
#' @return modified polars DataFrame
#' @noRd
pl_arith <- function(df, new_col, col1, col2, op = "+") {
  col1_expr <- pl$col(col1)
  col2_expr <- if (is.character(col2)) pl$col(col2) else pl$lit(col2)

  result_expr <- switch(op,
    "+" = col1_expr + col2_expr,
    "-" = col1_expr - col2_expr,
    "*" = col1_expr * col2_expr,
    "/" = col1_expr / col2_expr,
    col1_expr + col2_expr
  )

  df$with_columns(result_expr$alias(new_col))
}

#' Add multiple columns at once
#' @param df polars DataFrame
#' @param ... named expressions: name = expression
#' @return modified polars DataFrame
#' @noRd
pl_with_cols <- function(df, ...) {
  exprs <- list(...)
  for (name in names(exprs)) {
    expr <- exprs[[name]]
    if (!inherits(expr, "Expr")) {
      expr <- pl$lit(expr)
    }
    df <- df$with_columns(expr$alias(name))
  }
  df
}

#' Count unique values
#' @param df polars DataFrame
#' @param col_name column name
#' @return integer count of unique values
#' @noRd
pl_n_unique <- function(df, col_name) {
  as.data.frame(df$select(pl$col(col_name)$n_unique()))[1, 1]
}

#' Get first row value of a column
#' @param df polars DataFrame
#' @param col_name column name
#' @return first value
#' @noRd
pl_first <- function(df, col_name) {
  as.data.frame(df$select(pl$col(col_name)$first()))[1, 1]
}

#' Get unique values of a column
#' @param df polars DataFrame
#' @param col_name column name
#' @return vector of unique values
#' @noRd
pl_unique <- function(df, col_name) {
  as.data.frame(df$select(pl$col(col_name)$unique()))[, 1]
}

#' Join two dataframes
#' @param df1 polars DataFrame (left)
#' @param df2 polars DataFrame (right)
#' @param on columns to join on
#' @param how join type: "inner", "left", "outer", "semi", "anti"
#' @return joined polars DataFrame
#' @noRd
pl_join <- function(df1, df2, on, how = "left") {
  df1$join(df2, on = on, how = how)
}

#' Select columns
#' @param df polars DataFrame
#' @param cols column names to select
#' @return polars DataFrame with selected columns
#' @noRd
pl_select <- function(df, cols) {
  df$select(cols)
}

#' Drop columns
#' @param df polars DataFrame
#' @param cols column names to drop
#' @return polars DataFrame without specified columns
#' @noRd
pl_drop <- function(df, cols) {
  df$drop(cols)
}

#' Rename columns
#' @param df polars DataFrame
#' @param ... old_name = "new_name" pairs
#' @return polars DataFrame with renamed columns
#' @noRd
pl_rename <- function(df, ...) {
  renames <- list(...)
  for (old_name in names(renames)) {
    new_name <- renames[[old_name]]
    if (old_name %in% df$columns) {
      df <- df$with_columns(pl$col(old_name)$alias(new_name))
      df <- df$drop(old_name)
    }
  }
  df
}

#' Cumulative sum within groups
#' @param df polars DataFrame
#' @param col_name column to cumsum
#' @param by_col grouping column
#' @param order_col column to order by
#' @param new_col name for new column
#' @return modified polars DataFrame
#' @noRd
pl_cumsum <- function(df, col_name, by_col, order_col, new_col) {
  df <- df$sort(c(by_col, order_col))
  df$with_columns(
    pl$col(col_name)$cum_sum()$over(by_col)$alias(new_col)
  )
}

#' Cast column to different type
#' @param df polars DataFrame
#' @param col_name column name
#' @param dtype polars data type (e.g., pl$Float64, pl$Int32)
#' @return modified polars DataFrame
#' @noRd
pl_cast <- function(df, col_name, dtype) {
  df$with_columns(
    pl$col(col_name)$cast(dtype)$alias(col_name)
  )
}

#' Aggregate and merge back (common pattern in did_multiplegt)
#' @param df polars DataFrame
#' @param by_cols columns to group by
#' @param agg_col column to aggregate
#' @param new_col name for aggregated column
#' @param agg_func aggregation function ("sum", "mean", "max", "min", "count", "n_unique")
#' @return modified polars DataFrame with aggregated column joined back
#' @noRd
pl_agg_merge <- function(df, by_cols, agg_col, new_col, agg_func = "sum") {
  # Remove existing column if present
  if (new_col %in% df$columns) {
    df <- df$drop(new_col)
  }

  # Use window function with $over() instead of group_by + join for better performance
  agg_expr <- switch(agg_func,
    "sum" = pl$col(agg_col)$sum(),
    "mean" = pl$col(agg_col)$mean(),
    "max" = pl$col(agg_col)$max(),
    "min" = pl$col(agg_col)$min(),
    "count" = pl$col(agg_col)$count(),
    "n_unique" = pl$col(agg_col)$n_unique(),
    pl$col(agg_col)$sum()
  )

  # Apply over() for window function
  over_expr <- pl_over_cols(agg_expr, by_cols)$alias(new_col)
  df$with_columns(over_expr)
}

#' Aggregate and merge with filter condition
#' @param df polars DataFrame
#' @param by_cols columns to group by
#' @param agg_col column to aggregate
#' @param new_col name for aggregated column
#' @param filter_cond polars expression for filter condition
#' @param agg_func aggregation function
#' @return modified polars DataFrame
#' @noRd
pl_agg_merge_filtered <- function(df, by_cols, agg_col, new_col, filter_cond, agg_func = "sum") {
  # Remove existing column if present
  if (new_col %in% df$columns) {
    df <- df$drop(new_col)
  }

  # Use conditional expression with window function for better performance
  # Create masked column (value where condition is true, NA otherwise)
  masked_expr <- pl$when(filter_cond)$then(pl$col(agg_col))$otherwise(pl$lit(NA_real_))

  # Apply aggregation
  agg_expr <- switch(agg_func,
    "sum" = masked_expr$sum(),
    "mean" = masked_expr$mean(),
    "max" = masked_expr$max(),
    "min" = masked_expr$min(),
    "count" = pl$when(filter_cond)$then(pl$lit(1L))$otherwise(pl$lit(NA_integer_))$count(),
    "n_unique" = masked_expr$n_unique(),
    masked_expr$sum()
  )

  # Apply over() for window function
  over_expr <- pl_over_cols(agg_expr, by_cols)$alias(new_col)
  df$with_columns(over_expr)
}

#' Get unique factor levels from a column
#' @param df polars DataFrame
#' @param col_name column name
#' @return vector of unique values
#' @noRd
pl_levels <- function(df, col_name) {
  as.data.frame(df$select(pl$col(col_name)$unique()$sort()))[, 1]
}

#' Compute group mean and merge back
#' @param df polars DataFrame
#' @param by_col grouping column
#' @param value_col column to take mean of
#' @param new_col name for mean column
#' @return modified polars DataFrame
#' @noRd
pl_group_mean_merge <- function(df, by_col, value_col, new_col) {
  if (new_col %in% df$columns) {
    df <- df$drop(new_col)
  }
  agg_df <- df$group_by(by_col)$agg(pl$col(value_col)$mean()$alias(new_col))
  df$join(agg_df, on = by_col, how = "left")
}

#' Compute lagged difference within groups
#' @param df polars DataFrame
#' @param col_name column to difference
#' @param n lag periods
#' @param by_col grouping column
#' @param order_col ordering column
#' @param new_col name for difference column
#' @return modified polars DataFrame
#' @noRd
pl_lag_diff <- function(df, col_name, n, by_col, order_col, new_col) {
  df <- df$sort(c(by_col, order_col))
  lagged_col <- paste0("__lag_temp__")
  df <- df$with_columns(
    pl$col(col_name)$shift(n)$over(by_col)$alias(lagged_col)
  )
  df <- df$with_columns(
    (pl$col(col_name) - pl$col(lagged_col))$alias(new_col)
  )
  df$drop(lagged_col)
}

#' Set column to NULL (NA for all rows)
#' @param df polars DataFrame
#' @param col_name column name
#' @return modified polars DataFrame
#' @noRd
pl_set_null <- function(df, col_name) {
  df$with_columns(pl$lit(NA_real_)$alias(col_name))
}

#' Create multiple NULL columns
#' @param df polars DataFrame
#' @param col_names vector of column names
#' @return modified polars DataFrame
#' @noRd
pl_set_nulls <- function(df, col_names) {
  # Batch all NULL column creations into a single with_columns call
  null_exprs <- lapply(col_names, function(col) pl$lit(NA_real_)$alias(col))
  do.call(df$with_columns, null_exprs)
}

#' Multiply two columns
#' @param df polars DataFrame
#' @param new_col new column name
#' @param col1 first column
#' @param col2 second column
#' @return modified polars DataFrame
#' @noRd
pl_multiply <- function(df, new_col, col1, col2) {
  df$with_columns(
    (pl$col(col1) * pl$col(col2))$alias(new_col)
  )
}

#' Subtract two columns
#' @param df polars DataFrame
#' @param new_col new column name
#' @param col1 first column (minuend)
#' @param col2 second column (subtrahend)
#' @return modified polars DataFrame
#' @noRd
pl_subtract <- function(df, new_col, col1, col2) {
  df$with_columns(
    (pl$col(col1) - pl$col(col2))$alias(new_col)
  )
}

#' Divide two columns
#' @param df polars DataFrame
#' @param new_col new column name
#' @param col1 numerator column
#' @param col2 denominator column
#' @return modified polars DataFrame
#' @noRd
pl_divide <- function(df, new_col, col1, col2) {
  df$with_columns(
    (pl$col(col1) / pl$col(col2))$alias(new_col)
  )
}

#' Add two columns
#' @param df polars DataFrame
#' @param new_col new column name
#' @param col1 first column
#' @param col2 second column
#' @return modified polars DataFrame
#' @noRd
pl_add <- function(df, new_col, col1, col2) {
  df$with_columns(
    (pl$col(col1) + pl$col(col2))$alias(new_col)
  )
}

#' Create indicator column (1/0) based on condition
#' @param df polars DataFrame
#' @param new_col new column name
#' @param condition polars expression
#' @return modified polars DataFrame
#' @noRd
pl_indicator <- function(df, new_col, condition) {
  df$with_columns(
    pl$when(condition)$then(pl$lit(1L))$otherwise(pl$lit(0L))$cast(pl$Float64)$alias(new_col)
  )
}

#' Apply expression to create new column
#' @param df polars DataFrame
#' @param new_col new column name
#' @param expr polars expression
#' @return modified polars DataFrame
#' @noRd
pl_expr <- function(df, new_col, expr) {
  df$with_columns(expr$alias(new_col))
}

#' Check if any rows match condition
#' @param df polars DataFrame
#' @param condition polars expression
#' @return logical
#' @noRd
pl_any <- function(df, condition) {
  df$filter(condition)$height > 0
}

#' Filter and count rows
#' @param df polars DataFrame
#' @param condition polars expression
#' @return integer count
#' @noRd
pl_count_where <- function(df, condition) {
  df$filter(condition)$height
}

#' Compute OLS coefficients using normal equations directly in Polars
#' β = (X'X)^(-1) X'y
#' This avoids converting to data.frame
#' @param df polars DataFrame
#' @param y_col name of dependent variable column
#' @param x_cols vector of names of independent variable columns
#' @param weight_col optional weight column name
#' @return list with coefficients and fitted values column name
#' @importFrom MASS ginv
#' @noRd
pl_ols <- function(df, y_col, x_cols, weight_col = NULL) {
  n_vars <- length(x_cols)

  # If weights provided, multiply y and X by sqrt(w)
  if (!is.null(weight_col)) {
    sqrt_w_expr <- pl$col(weight_col)$sqrt()
    y_weighted <- paste0("__y_w__")
    df <- df$with_columns((pl$col(y_col) * sqrt_w_expr)$alias(y_weighted))

    x_weighted <- paste0("__x_w_", seq_len(n_vars), "__")
    for (i in seq_len(n_vars)) {
      df <- df$with_columns((pl$col(x_cols[i]) * sqrt_w_expr)$alias(x_weighted[i]))
    }
    y_col_use <- y_weighted
    x_cols_use <- x_weighted
  } else {
    y_col_use <- y_col
    x_cols_use <- x_cols
  }

  # Compute X'X matrix elements using Polars aggregation
  XtX <- matrix(0, n_vars, n_vars)
  for (i in seq_len(n_vars)) {
    for (j in seq_len(n_vars)) {
      # Use Polars to compute sum of xi * xj
      prod_col <- paste0("__xtx_", i, "_", j, "__")
      XtX[i, j] <- as.data.frame(df$select(
        (pl$col(x_cols_use[i]) * pl$col(x_cols_use[j]))$sum()
      ))[[1, 1]]
    }
  }

  # Compute X'y vector
  Xty <- numeric(n_vars)
  for (i in seq_len(n_vars)) {
    Xty[i] <- as.data.frame(df$select(
      (pl$col(x_cols_use[i]) * pl$col(y_col_use))$sum()
    ))[[1, 1]]
  }

  # Solve for coefficients: β = (X'X)^(-1) X'y
  beta <- tryCatch({
    solve(XtX, Xty)
  }, error = function(e) {
    MASS::ginv(XtX) %*% Xty
  })

  # Clean up temporary columns
  if (!is.null(weight_col)) {
    df <- df$drop(c(y_weighted, x_weighted))
  }

  names(beta) <- x_cols
  return(list(coefficients = beta, df = df))
}

#' Compute OLS and add fitted values to dataframe
#' @param df polars DataFrame
#' @param y_col name of dependent variable column
#' @param x_cols vector of names of independent variable columns
#' @param weight_col optional weight column name
#' @param fitted_col name for fitted values column
#' @return polars DataFrame with fitted values added
#' @noRd
pl_ols_predict <- function(df, y_col, x_cols, weight_col = NULL, fitted_col = "fitted") {
  result <- pl_ols(df, y_col, x_cols, weight_col)
  beta <- result$coefficients
  df <- result$df

  # Compute fitted values: y_hat = X * β
  fitted_expr <- pl$lit(0)
  for (i in seq_along(x_cols)) {
    fitted_expr <- fitted_expr + pl$col(x_cols[i]) * beta[i]
  }

  df$with_columns(fitted_expr$alias(fitted_col))
}

#' Compute OLS residuals
#' @param df polars DataFrame
#' @param y_col name of dependent variable column
#' @param x_cols vector of names of independent variable columns
#' @param weight_col optional weight column name
#' @param resid_col name for residuals column
#' @return polars DataFrame with residuals added
#' @noRd
pl_ols_residuals <- function(df, y_col, x_cols, weight_col = NULL, resid_col = "residuals") {
  result <- pl_ols(df, y_col, x_cols, weight_col)
  beta <- result$coefficients
  df <- result$df

  # Compute fitted values
  fitted_expr <- pl$lit(0)
  for (i in seq_along(x_cols)) {
    fitted_expr <- fitted_expr + pl$col(x_cols[i]) * beta[i]
  }

  # Residuals = y - fitted
  df$with_columns(
    (pl$col(y_col) - fitted_expr)$alias(resid_col)
  )
}

#' Compute fixed effects regression by demeaning
#' @param df polars DataFrame
#' @param y_col name of dependent variable column
#' @param x_cols vector of names of independent variable columns
#' @param fe_col fixed effects column (categorical)
#' @param weight_col optional weight column name
#' @param fitted_col name for fitted values column
#' @return polars DataFrame with fitted values added
#' @noRd
pl_fe_ols <- function(df, y_col, x_cols, fe_col, weight_col = NULL, fitted_col = "fitted") {
  # Demean y and X by fixed effect groups
  y_demean <- paste0("__y_demean__")
  df <- df$with_columns(
    (pl$col(y_col) - pl$col(y_col)$mean()$over(fe_col))$alias(y_demean)
  )

  x_demean <- paste0("__x_demean_", seq_along(x_cols), "__")
  for (i in seq_along(x_cols)) {
    df <- df$with_columns(
      (pl$col(x_cols[i]) - pl$col(x_cols[i])$mean()$over(fe_col))$alias(x_demean[i])
    )
  }

  # Run OLS on demeaned data
  result <- pl_ols(df, y_demean, x_demean, weight_col)
  beta <- result$coefficients
  df <- result$df

  # Compute fitted values using original X (not demeaned) plus FE means
  fitted_expr <- pl$lit(0)
  for (i in seq_along(x_cols)) {
    fitted_expr <- fitted_expr + pl$col(x_cols[i]) * beta[i]
  }

  # Add back the mean of y by FE group
  df <- df$with_columns(
    (fitted_expr + pl$col(y_col)$mean()$over(fe_col) -
      pl$lit(0))$alias(fitted_col)  # Simplified: just the fitted from demeaned
  )

  # Actually for FE regression, fitted = X*beta (not demeaning X in prediction)
  # The FE absorbs the group means
  df <- df$with_columns(fitted_expr$alias(fitted_col))

  # Clean up
  df <- df$drop(c(y_demean, x_demean))

  df
}

#' Compute weighted group means (for demeaning)
#' @param df polars DataFrame
#' @param col_name column to demean
#' @param by_cols grouping columns
#' @param weight_col weight column
#' @param new_col name for demeaned column
#' @return polars DataFrame
#' @noRd
pl_weighted_demean <- function(df, col_name, by_cols, weight_col, new_col) {
  # Compute weighted mean: sum(w * x) / sum(w) over groups
  weighted_mean_expr <- (
    (pl$col(weight_col) * pl$col(col_name))$sum()$over(by_cols) /
    pl$col(weight_col)$sum()$over(by_cols)
  )

  # Demean: x - weighted_mean
  df$with_columns(
    (pl$col(col_name) - weighted_mean_expr)$alias(new_col)
  )
}

#' Compute cross-product matrix X'WX using Polars
#' @param df polars DataFrame
#' @param x_cols vector of X column names
#' @param weight_col optional weight column name
#' @return matrix
#' @noRd
pl_crossprod <- function(df, x_cols, weight_col = NULL) {
  n <- length(x_cols)
  result <- matrix(0, n, n)

  for (i in seq_len(n)) {
    for (j in i:n) {  # Only compute upper triangle, matrix is symmetric
      if (!is.null(weight_col)) {
        val <- as.data.frame(df$select(
          (pl$col(x_cols[i]) * pl$col(x_cols[j]) * pl$col(weight_col))$sum()
        ))[[1, 1]]
      } else {
        val <- as.data.frame(df$select(
          (pl$col(x_cols[i]) * pl$col(x_cols[j]))$sum()
        ))[[1, 1]]
      }
      result[i, j] <- val
      result[j, i] <- val  # Symmetric
    }
  }

  result
}

#' Compute X'Wy vector using Polars
#' @param df polars DataFrame
#' @param x_cols vector of X column names
#' @param y_col Y column name
#' @param weight_col optional weight column name
#' @return numeric vector
#' @noRd
pl_crossprod_y <- function(df, x_cols, y_col, weight_col = NULL) {
  n <- length(x_cols)
  result <- numeric(n)

  for (i in seq_len(n)) {
    if (!is.null(weight_col)) {
      val <- as.data.frame(df$select(
        (pl$col(x_cols[i]) * pl$col(y_col) * pl$col(weight_col))$sum()
      ))[[1, 1]]
    } else {
      val <- as.data.frame(df$select(
        (pl$col(x_cols[i]) * pl$col(y_col))$sum()
      ))[[1, 1]]
    }
    result[i] <- val
  }

  result
}

#' Fast OLS for simple regression with precomputed products
#' Uses Polars to compute sums directly
#' @param df polars DataFrame
#' @param y_col dependent variable
#' @param x_col single independent variable (for simple regression)
#' @param weight_col optional weight column
#' @return list with slope and intercept
#' @noRd
pl_simple_ols <- function(df, y_col, x_col, weight_col = NULL) {
  if (!is.null(weight_col)) {
    # Weighted regression
    stats <- df$select(
      pl$col(weight_col)$sum()$alias("sum_w"),
      (pl$col(weight_col) * pl$col(x_col))$sum()$alias("sum_wx"),
      (pl$col(weight_col) * pl$col(y_col))$sum()$alias("sum_wy"),
      (pl$col(weight_col) * pl$col(x_col) * pl$col(x_col))$sum()$alias("sum_wxx"),
      (pl$col(weight_col) * pl$col(x_col) * pl$col(y_col))$sum()$alias("sum_wxy")
    )
    s <- as.data.frame(stats)

    denom <- s$sum_w * s$sum_wxx - s$sum_wx^2
    slope <- (s$sum_w * s$sum_wxy - s$sum_wx * s$sum_wy) / denom
    intercept <- (s$sum_wy - slope * s$sum_wx) / s$sum_w
  } else {
    # Unweighted regression
    stats <- df$select(
      pl$len()$alias("n"),
      pl$col(x_col)$sum()$alias("sum_x"),
      pl$col(y_col)$sum()$alias("sum_y"),
      (pl$col(x_col) * pl$col(x_col))$sum()$alias("sum_xx"),
      (pl$col(x_col) * pl$col(y_col))$sum()$alias("sum_xy")
    )
    s <- as.data.frame(stats)

    denom <- s$n * s$sum_xx - s$sum_x^2
    slope <- (s$n * s$sum_xy - s$sum_x * s$sum_y) / denom
    intercept <- (s$sum_y - slope * s$sum_x) / s$n
  }

  list(slope = slope, intercept = intercept)
}

#' Convert Polars DataFrame to data.table (faster than data.frame)
#' Use this when data.frame operations are absolutely necessary
#' @param df polars DataFrame
#' @return data.table
#' @noRd
pl_to_dt <- function(df) {
  data.table::as.data.table(as.data.frame(df))
}

#' Convert data.table to Polars DataFrame
#' @param dt data.table
#' @return polars DataFrame
#' @noRd
dt_to_pl <- function(dt) {
  .as_polars_df(as.data.frame(dt))
}

#' Batch compute multiple column statistics
#' @param df polars DataFrame
#' @param cols columns to compute stats for
#' @param stats vector of stats: "sum", "mean", "min", "max", "std"
#' @return named list of results
#' @noRd
pl_batch_stats <- function(df, cols, stats = c("sum", "mean")) {
  exprs <- list()
  for (col in cols) {
    for (stat in stats) {
      stat_expr <- switch(stat,
        "sum" = pl$col(col)$sum(),
        "mean" = pl$col(col)$mean(),
        "min" = pl$col(col)$min(),
        "max" = pl$col(col)$max(),
        "std" = pl$col(col)$std(),
        "var" = pl$col(col)$var(),
        pl$col(col)$sum()
      )
      exprs[[paste0(col, "_", stat)]] <- stat_expr$alias(paste0(col, "_", stat))
    }
  }

  result_df <- do.call(df$select, exprs)
  as.list(as.data.frame(result_df))
}

#' Initialize multiple columns to zero
#' @param df polars DataFrame
#' @param col_names vector of column names
#' @return polars DataFrame
#' @noRd
pl_init_zeros <- function(df, col_names) {
  exprs <- lapply(col_names, function(col) pl$lit(0)$alias(col))
  do.call(df$with_columns, exprs)
}

#' Compute interaction (product) of multiple columns
#' @param df polars DataFrame
#' @param new_col name for new column
#' @param cols columns to multiply together
#' @return polars DataFrame
#' @noRd
pl_interact <- function(df, new_col, cols) {
  expr <- pl$col(cols[1])
  for (i in 2:length(cols)) {
    expr <- expr * pl$col(cols[i])
  }
  df$with_columns(expr$alias(new_col))
}

#' Conditional sum over groups - sum where condition is true
#' @param df polars DataFrame
#' @param agg_col column to aggregate
#' @param condition polars expression for filter condition
#' @param by_cols columns to group by
#' @param new_col name for new column
#' @return polars DataFrame
#' @noRd
pl_sum_where <- function(df, agg_col, condition, by_cols, new_col) {
  # Remove existing column if present
  if (new_col %in% df$columns) {
    df <- df$drop(new_col)
  }

  # Create masked expression: value where condition, NA otherwise
  masked_expr <- pl$when(condition)$then(pl$col(agg_col))$otherwise(pl$lit(NA_real_))

  # Compute sum over groups
  over_expr <- pl_over_cols(masked_expr$sum(), by_cols)$alias(new_col)
  df$with_columns(over_expr)
}

#' Create group ID column (like data.table's .GRP)
#' @param df polars DataFrame
#' @param by_cols columns to group by
#' @param new_col name for new column
#' @return polars DataFrame
#' @noRd
pl_group_id <- function(df, by_cols, new_col) {
  # Create a unique group identifier
  # Use group_by to get unique combinations, then assign row numbers
  if (new_col %in% df$columns) {
    df <- df$drop(new_col)
  }

  # Get unique combinations and assign IDs
  unique_groups <- df$select(by_cols)$unique()$with_row_index("__grp_id__")

  # Join back to original df
  df <- df$join(unique_groups, on = by_cols, how = "left")
  df <- df$with_columns(pl$col("__grp_id__")$alias(new_col))
  df$drop("__grp_id__")
}

#' Count unique values within groups (like n_distinct)
#' @param df polars DataFrame
#' @param count_col column to count unique values of
#' @param by_cols columns to group by
#' @param new_col name for new column
#' @return polars DataFrame
#' @noRd
pl_n_unique_over <- function(df, count_col, by_cols, new_col) {
  if (new_col %in% df$columns) {
    df <- df$drop(new_col)
  }

  over_expr <- pl_over_cols(pl$col(count_col)$n_unique(), by_cols)$alias(new_col)
  df$with_columns(over_expr)
}

#' Create lagged column within groups
#' @param df polars DataFrame
#' @param col_name source column
#' @param n lag periods
#' @param by_col grouping column
#' @param new_col name for new column
#' @return polars DataFrame
#' @noRd
pl_shift_over <- function(df, col_name, n, by_col, new_col) {
  df$with_columns(
    pl$col(col_name)$shift(n)$over(by_col)$alias(new_col)
  )
}

#' Create multiple lag difference columns at once (outcome - lagged outcome)
#' @param df polars DataFrame
#' @param col_name source column
#' @param lags vector of lag values
#' @param by_col grouping column
#' @param prefix prefix for new column names
#' @return polars DataFrame
#' @noRd
pl_batch_lag_diff <- function(df, col_name, lags, by_col, prefix = "diff_") {
  exprs <- lapply(lags, function(n) {
    (pl$col(col_name) - pl$col(col_name)$shift(n)$over(by_col))$alias(paste0(prefix, n))
  })
  do.call(df$with_columns, exprs)
}

#' Apply fifelse-like conditional with three-value logic (true, false, NA)
#' @param condition polars expression for condition
#' @param true_val expression or literal for TRUE case
#' @param false_val expression or literal for FALSE case
#' @param na_val expression or literal for NA case (when condition is NULL)
#' @return polars expression
#' @noRd
pl_fifelse <- function(condition, true_val, false_val, na_val = NULL) {
  true_expr <- if (inherits(true_val, "Expr")) true_val else pl$lit(true_val)
  false_expr <- if (inherits(false_val, "Expr")) false_val else pl$lit(false_val)

  if (is.null(na_val)) {
    pl$when(condition)$then(true_expr)$otherwise(false_expr)
  } else {
    na_expr <- if (inherits(na_val, "Expr")) na_val else pl$lit(na_val)
    pl$when(condition$is_null())$then(na_expr)$when(condition)$then(true_expr)$otherwise(false_expr)
  }
}

#' Conditional assignment: update column only where condition is TRUE
#' Similar to data.table syntax for conditional updates
#' @param df polars DataFrame
#' @param col_name column to update
#' @param condition polars expression for condition
#' @param value expression or literal for new value
#' @return polars DataFrame
#' @noRd
pl_update_where <- function(df, col_name, condition, value) {
  val_expr <- if (inherits(value, "Expr")) value else pl$lit(value)
  existing <- if (col_name %in% df$columns) pl$col(col_name) else pl$lit(NA_real_)

  df$with_columns(
    pl$when(condition)$then(val_expr)$otherwise(existing)$alias(col_name)
  )
}

#' Drop columns if they exist (safe drop)
#' @param df polars DataFrame
#' @param cols column names to drop
#' @return polars DataFrame
#' @noRd
pl_safe_drop <- function(df, cols) {
  existing <- intersect(cols, df$columns)
  if (length(existing) > 0) {
    df$drop(existing)
  } else {
    df
  }
}

#' Create multiple columns with constant values
#' @param df polars DataFrame
#' @param col_vals named list of column = value pairs
#' @return polars DataFrame
#' @noRd
pl_add_constants <- function(df, col_vals) {
  exprs <- lapply(names(col_vals), function(col) {
    pl$lit(col_vals[[col]])$alias(col)
  })
  do.call(df$with_columns, exprs)
}

#' Compute aggregation at time level and merge back
#' Common pattern: sum of X by time, merged back to all rows
#' @param df polars DataFrame
#' @param agg_col column to aggregate
#' @param new_col name for aggregated column
#' @param agg_func aggregation function
#' @param time_col time column name
#' @return polars DataFrame
#' @noRd
pl_time_agg <- function(df, agg_col, new_col, agg_func = "sum", time_col = "time_XX") {
  pl_agg_merge(df, time_col, agg_col, new_col, agg_func)
}

#' Compute weighted sum over groups
#' @param df polars DataFrame
#' @param value_col column with values
#' @param weight_col column with weights
#' @param by_cols grouping columns
#' @param new_col name for result column
#' @return polars DataFrame
#' @noRd
pl_weighted_sum_over <- function(df, value_col, weight_col, by_cols, new_col) {
  if (new_col %in% df$columns) {
    df <- df$drop(new_col)
  }

  weighted_sum_expr <- pl_over_cols(
    (pl$col(value_col) * pl$col(weight_col))$sum(),
    by_cols
  )$alias(new_col)

  df$with_columns(weighted_sum_expr)
}

#' Compute mean over groups where condition is met
#' @param df polars DataFrame
#' @param value_col column to average
#' @param condition filter condition
#' @param by_cols grouping columns
#' @param new_col name for result column
#' @return polars DataFrame
#' @noRd
pl_mean_where <- function(df, value_col, condition, by_cols, new_col) {
  if (new_col %in% df$columns) {
    df <- df$drop(new_col)
  }

  # Create masked expression: value where condition, NA otherwise
  masked_expr <- pl$when(condition)$then(pl$col(value_col))$otherwise(pl$lit(NA_real_))

  over_expr <- pl_over_cols(masked_expr$mean(), by_cols)$alias(new_col)
  df$with_columns(over_expr)
}

#' Count unique values in column by groups (equivalent to data.table uniqueN)
#' @param df polars DataFrame
#' @param value_col column to count unique values in
#' @param by_cols grouping columns
#' @param new_col name for result column
#' @return polars DataFrame
#' @noRd
pl_unique_n_over <- function(df, value_col, by_cols, new_col) {
  if (new_col %in% df$columns) {
    df <- df$drop(new_col)
  }
  df$with_columns(
    pl_over_cols(pl$col(value_col)$n_unique(), by_cols)$alias(new_col)
  )
}

#' Conditional sum by groups (sum values where condition is met)
#' @param df polars DataFrame
#' @param value_col column to sum
#' @param condition filter condition expression
#' @param by_cols grouping columns
#' @param new_col name for result column
#' @return polars DataFrame
#' @noRd
pl_sum_where_over <- function(df, value_col, condition, by_cols, new_col) {
  if (new_col %in% df$columns) {
    df <- df$drop(new_col)
  }

  masked_expr <- pl$when(condition)$then(pl$col(value_col))$otherwise(pl$lit(0))
  df$with_columns(
    pl_over_cols(masked_expr$sum(), by_cols)$alias(new_col)
  )
}

#' Set column value where condition is met, keep existing otherwise
#' @param df polars DataFrame
#' @param col_name column to modify
#' @param condition filter condition expression
#' @param new_value new value to set (scalar or expression)
#' @return polars DataFrame
#' @noRd
pl_set_where <- function(df, col_name, condition, new_value) {
  if (inherits(new_value, "RPolarsExpr")) {
    df$with_columns(
      pl$when(condition)$then(new_value)$otherwise(pl$col(col_name))$alias(col_name)
    )
  } else {
    df$with_columns(
      pl$when(condition)$then(pl$lit(new_value))$otherwise(pl$col(col_name))$alias(col_name)
    )
  }
}

#' Square root-based DOF correction factor
#' Computes sqrt(n / (n-1)) for DOF > 1, 1 otherwise
#' @param dof_col column name with DOF values
#' @param new_col name for result column
#' @return polars expression
#' @noRd
pl_dof_correction <- function(dof_col, new_col) {
  pl$when(pl$col(dof_col) > 1)$
    then((pl$col(dof_col) / (pl$col(dof_col) - 1))$sqrt())$
    otherwise(pl$lit(1.0))$
    alias(new_col)
}

#' Create conditional indicator column (1 where condition, 0 otherwise)
#' @param df polars DataFrame
#' @param new_col name for result column
#' @param condition polars expression
#' @return polars DataFrame
#' @noRd
pl_binary_indicator <- function(df, new_col, condition) {
  df$with_columns(
    condition$cast(pl$Float64)$alias(new_col)
  )
}

#' Chain multiple conditional updates (similar to data.table conditional assignment chain)
#' @param df polars DataFrame
#' @param col_name column to update
#' @param conditions list of conditions
#' @param values list of values (same length as conditions)
#' @param default default value if no condition matches
#' @return polars DataFrame
#' @noRd
pl_case_when <- function(df, col_name, conditions, values, default = NA_real_) {
  expr <- pl$lit(default)

  # Build nested when/then/otherwise from end to start
  for (i in rev(seq_along(conditions))) {
    if (inherits(values[[i]], "RPolarsExpr")) {
      expr <- pl$when(conditions[[i]])$then(values[[i]])$otherwise(expr)
    } else {
      expr <- pl$when(conditions[[i]])$then(pl$lit(values[[i]]))$otherwise(expr)
    }
  }

  df$with_columns(expr$alias(col_name))
}

# ============================================================================
# ADDITIONAL HELPERS FOR CORE OPTIMIZATION
# ============================================================================

#' Get factor levels from polars DataFrame column
#' @param df polars DataFrame
#' @param col_name column name
#' @return character vector of unique sorted values
#' @noRd
pl_factor_levels <- function(df, col_name) {
  result <- df$select(pl$col(col_name)$unique()$sort()$drop_nulls())
  as.character(as.data.frame(result)[[1]])
}

#' Batch create NULL columns efficiently
#' @param df polars DataFrame
#' @param col_patterns list of patterns with sprintf-style format and range
#' @param range numeric vector for the range
#' @return polars DataFrame with columns dropped
#' @noRd
pl_batch_drop_cols <- function(df, col_names) {
  existing <- intersect(col_names, df$columns)
  if (length(existing) > 0) {
    df$drop(existing)
  } else {
    df
  }
}

#' Create multiple lag difference columns efficiently
#' @param df polars DataFrame (must be sorted by group and time)
#' @param col_name source column for differences
#' @param lags vector of lag periods
#' @param by_col grouping column
#' @param prefix prefix for new column names (result: prefix1, prefix2, etc.)
#' @return polars DataFrame with new difference columns
#' @noRd
pl_multi_lag_diff <- function(df, col_name, lags, by_col, prefix = "diff_y_") {
  exprs <- lapply(lags, function(lag) {
    (pl$col(col_name) - pl$col(col_name)$shift(lag)$over(by_col))$alias(paste0(prefix, lag, "_XX"))
  })
  do.call(df$with_columns, exprs)
}

#' Compute conditional aggregation over groups using filter approach
#' @param df polars DataFrame
#' @param value_col column to aggregate
#' @param filter_expr polars expression for filter
#' @param by_cols grouping columns
#' @param new_col name for result
#' @param agg_func aggregation function name
#' @return polars DataFrame
#' @noRd
pl_filtered_agg_over <- function(df, value_col, filter_expr, by_cols, new_col, agg_func = "sum",
                                  filter_result = FALSE) {
  # Remove existing column if present
  if (new_col %in% df$columns) {
    df <- df$drop(new_col)
  }

  # Create masked column
  masked <- pl$when(filter_expr)$then(pl$col(value_col))$otherwise(pl$lit(NA_real_))

  # Apply aggregation over groups
  agg_expr <- switch(agg_func,
    "sum" = masked$sum(),
    "mean" = masked$mean(),
    "count" = pl$when(filter_expr)$then(pl$lit(1L))$otherwise(pl$lit(NA_integer_))$sum(),
    masked$sum()
  )

  over_expr <- pl_over_cols(agg_expr, by_cols)$alias(new_col)
  df <- df$with_columns(over_expr)

  # If filter_result is TRUE, only assign the aggregated value to rows where filter is TRUE
  # This matches Stata behavior where "if cond" only assigns to matching rows
  if (filter_result) {
    df <- df$with_columns(
      pl$when(filter_expr)$then(pl$col(new_col))$otherwise(pl$lit(NA_real_))$alias(new_col)
    )
  }

  df
}

#' Compute sum by multiple grouping columns
#' @param df polars DataFrame
#' @param value_col column to sum
#' @param by_cols vector of grouping column names
#' @param new_col name for result column
#' @return polars DataFrame
#' @noRd
pl_sum_over <- function(df, value_col, by_cols, new_col) {
  if (new_col %in% df$columns) {
    df <- df$drop(new_col)
  }
  over_expr <- pl_over_cols(pl$col(value_col)$sum(), by_cols)$alias(new_col)
  df$with_columns(over_expr)
}

#' Compute mean by multiple grouping columns
#' @param df polars DataFrame
#' @param value_col column to average
#' @param by_cols vector of grouping column names
#' @param new_col name for result column
#' @return polars DataFrame
#' @noRd
pl_mean_over <- function(df, value_col, by_cols, new_col) {
  if (new_col %in% df$columns) {
    df <- df$drop(new_col)
  }
  over_expr <- pl_over_cols(pl$col(value_col)$mean(), by_cols)$alias(new_col)
  df$with_columns(over_expr)
}

#' Create group ID (similar to data.table's .GRP)
#' @param df polars DataFrame
#' @param by_cols columns to group by
#' @param new_col name for new column
#' @return polars DataFrame
#' @noRd
pl_grp_id <- function(df, by_cols, new_col) {
  if (new_col %in% df$columns) {
    df <- df$drop(new_col)
  }

  # Get unique combinations with row index
  unique_groups <- df$select(by_cols)$unique()$with_row_index("__temp_grp_id__")

  # Join back
  df <- df$join(unique_groups, on = by_cols, how = "left")
  df <- df$with_columns(pl$col("__temp_grp_id__")$cast(pl$Int64)$alias(new_col))
  df$drop("__temp_grp_id__")
}

#' Apply fifelse logic (polars equivalent)
#' @param df polars DataFrame
#' @param new_col new column name
#' @param condition polars expression
#' @param true_val value when TRUE (scalar or expression)
#' @param false_val value when FALSE (scalar or expression)
#' @return polars DataFrame
#' @noRd
pl_fifelse <- function(df, new_col, condition, true_val, false_val) {
  true_expr <- if (inherits(true_val, "RPolarsExpr")) true_val else pl$lit(true_val)
  false_expr <- if (inherits(false_val, "RPolarsExpr")) false_val else pl$lit(false_val)

  df$with_columns(
    pl$when(condition)$then(true_expr)$otherwise(false_expr)$alias(new_col)
  )
}

#' Compute scalar sum from filtered DataFrame
#' @param df polars DataFrame
#' @param col_name column to sum
#' @param filter_expr optional filter expression
#' @return numeric scalar
#' @noRd
pl_scalar_sum <- function(df, col_name, filter_expr = NULL) {
  if (!is.null(filter_expr)) {
    df <- df$filter(filter_expr)
  }
  result <- as.data.frame(df$select(pl$col(col_name)$sum()))[[1]]
  if (is.null(result) || length(result) == 0 || is.na(result)) 0 else result
}

#' Compute scalar mean from filtered and grouped DataFrame
#' @param df polars DataFrame
#' @param col_name column to average
#' @param filter_expr filter expression
#' @param by_col grouping column for intermediate aggregation
#' @return numeric scalar (sum of group means)
#' @noRd
pl_scalar_mean_sum <- function(df, col_name, filter_expr, by_col) {
  result <- as.data.frame(
    df$filter(filter_expr)$
      group_by(by_col)$
      agg(pl$col(col_name)$mean()$alias("__m__"))$
      select(pl$col("__m__")$sum())
  )[[1]]
  if (is.null(result) || length(result) == 0 || is.na(result)) 0 else result
}

#' Batch initialize columns to zero
#' @param df polars DataFrame
#' @param col_names vector of column names
#' @return polars DataFrame
#' @noRd
pl_init_zero_cols <- function(df, col_names) {
  exprs <- lapply(col_names, function(col) pl$lit(0.0)$alias(col))
  do.call(df$with_columns, exprs)
}

#' Batch initialize columns to NA
#' @param df polars DataFrame
#' @param col_names vector of column names
#' @return polars DataFrame
#' @noRd
pl_init_na_cols <- function(df, col_names) {
  exprs <- lapply(col_names, function(col) pl$lit(NA_real_)$alias(col))
  do.call(df$with_columns, exprs)
}

#' Conditional update: set col to expr where condition, keep original otherwise
#' @param df polars DataFrame
#' @param col_name column to update
#' @param condition polars expression
#' @param new_value new value (scalar or expression)
#' @return polars DataFrame
#' @noRd
pl_conditional_update <- function(df, col_name, condition, new_value) {
  new_expr <- if (inherits(new_value, "RPolarsExpr")) new_value else pl$lit(new_value)
  existing <- if (col_name %in% df$columns) pl$col(col_name) else pl$lit(NA_real_)

  df$with_columns(
    pl$when(condition)$then(new_expr)$otherwise(existing)$alias(col_name)
  )
}

#' Compute uniqueN over groups (count distinct values)
#' @param df polars DataFrame
#' @param count_col column to count unique values
#' @param by_cols grouping columns
#' @param new_col result column name
#' @param filter_expr optional filter condition
#' @return polars DataFrame
#' @noRd
pl_uniqueN_over <- function(df, count_col, by_cols, new_col, filter_expr = NULL) {
  if (new_col %in% df$columns) {
    df <- df$drop(new_col)
  }

  if (!is.null(filter_expr)) {
    # Masked unique count - like CRAN's uniqueN which excludes NA
    # First mask non-matching rows to NA, then count unique
    masked <- pl$when(filter_expr)$then(pl$col(count_col))$otherwise(pl$lit(NA))
    # n_unique() counts NULL as a unique value, but CRAN's uniqueN doesn't
    # Subtract 1 if there are any nulls in the group to match CRAN behavior
    raw_count <- masked$n_unique()
    has_null <- masked$is_null()$any()$cast(pl$Int64)
    adjusted_count <- raw_count - has_null
    over_expr <- pl_over_cols(adjusted_count, by_cols)$alias(new_col)
  } else {
    # No filter - still need to exclude nulls from count
    raw_count <- pl$col(count_col)$n_unique()
    has_null <- pl$col(count_col)$is_null()$any()$cast(pl$Int64)
    adjusted_count <- raw_count - has_null
    over_expr <- pl_over_cols(adjusted_count, by_cols)$alias(new_col)
  }

  df$with_columns(over_expr)
}

#' Compute DOF correction factor: sqrt(n/(n-1)) where n > 1, else 1
#' @param df polars DataFrame
#' @param dof_col column with DOF values
#' @param new_col result column name
#' @return polars DataFrame
#' @noRd
pl_dof_correction_col <- function(df, dof_col, new_col) {
  df$with_columns(
    pl$when(pl$col(dof_col) > 1)$
      then((pl$col(dof_col) / (pl$col(dof_col) - 1))$sqrt())$
      otherwise(pl$lit(1.0))$
      alias(new_col)
  )
}

#' Create column from expression
#' @param df polars DataFrame
#' @param col_name new column name
#' @param expr polars expression
#' @return polars DataFrame
#' @noRd
pl_with_col <- function(df, col_name, expr) {
  df$with_columns(expr$alias(col_name))
}

#' Multiply column by scalar or another column
#' @param df polars DataFrame
#' @param new_col result column name
#' @param col1 first column name
#' @param col2_or_scalar second column name or numeric scalar
#' @return polars DataFrame
#' @noRd
pl_mult <- function(df, new_col, col1, col2_or_scalar) {
  if (is.numeric(col2_or_scalar)) {
    expr <- pl$col(col1) * col2_or_scalar
  } else {
    expr <- pl$col(col1) * pl$col(col2_or_scalar)
  }
  df$with_columns(expr$alias(new_col))
}

#' Divide column by another column
#' @param df polars DataFrame
#' @param new_col result column name
#' @param numerator numerator column name
#' @param denominator denominator column name
#' @return polars DataFrame
#' @noRd
pl_div <- function(df, new_col, numerator, denominator) {
  df$with_columns(
    (pl$col(numerator) / pl$col(denominator))$alias(new_col)
  )
}

#' Subtract columns: col1 - col2
#' @param df polars DataFrame
#' @param new_col result column name
#' @param col1 first column
#' @param col2 second column
#' @return polars DataFrame
#' @noRd
pl_sub <- function(df, new_col, col1, col2) {
  df$with_columns(
    (pl$col(col1) - pl$col(col2))$alias(new_col)
  )
}

#' Add columns: col1 + col2
#' @param df polars DataFrame
#' @param new_col result column name
#' @param col1 first column
#' @param col2 second column
#' @return polars DataFrame
#' @noRd
pl_add_cols <- function(df, new_col, col1, col2) {
  df$with_columns(
    (pl$col(col1) + pl$col(col2))$alias(new_col)
  )
}

#' Convert polars DataFrame column to R vector
#' @param df polars DataFrame
#' @param col_name column name
#' @return R vector
#' @noRd
pl_to_vec <- function(df, col_name) {
  as.vector(df$get_column(col_name))
}

#' Get first non-NA value from column
#' @param df polars DataFrame
#' @param col_name column name
#' @return scalar value
#' @noRd
pl_first_value <- function(df, col_name) {
  result <- as.data.frame(df$select(pl$col(col_name)$drop_nulls()$first()))[[1]]
  if (length(result) == 0) NA else result
}

#' Compute expression and add as column with over() window
#' @param df polars DataFrame
#' @param new_col new column name
#' @param expr polars expression to compute
#' @param by_cols columns for over() clause
#' @return polars DataFrame
#' @noRd
pl_expr_over <- function(df, new_col, expr, by_cols) {
  if (new_col %in% df$columns) {
    df <- df$drop(new_col)
  }
  over_expr <- pl_over_cols(expr, by_cols)$alias(new_col)
  df$with_columns(over_expr)
}

#' Create indicator column: 1 where condition TRUE, 0 otherwise (as numeric)
#' @param df polars DataFrame
#' @param new_col new column name
#' @param condition polars expression
#' @return polars DataFrame
#' @noRd
pl_indicator_num <- function(df, new_col, condition) {
  df$with_columns(
    condition$cast(pl$Float64)$alias(new_col)
  )
}

#' Create indicator column: 1 where condition TRUE, NA otherwise
#' @param df polars DataFrame
#' @param new_col new column name
#' @param condition polars expression
#' @return polars DataFrame
#' @noRd
pl_indicator_na <- function(df, new_col, condition) {
  df$with_columns(
    pl$when(condition)$then(pl$lit(1.0))$otherwise(pl$lit(NA_real_))$alias(new_col)
  )
}

#' Apply multiple with_columns at once from a list of expressions
#' @param df polars DataFrame
#' @param expr_list list of named expressions
#' @return polars DataFrame
#' @noRd
pl_with_cols_list <- function(df, expr_list) {
  if (length(expr_list) == 0) return(df)
  do.call(df$with_columns, expr_list)
}
