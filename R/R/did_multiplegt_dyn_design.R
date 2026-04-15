#' Option that shows the different treatment paths that switcher groups follow
#' @param data data
#' @param design_opt design_opt
#' @param weight weight
#' @param by by
#' @param by_index by_index
#' @param file file
#' @returns A list with the design option output.
#' @noRd
did_multiplegt_dyn_design <- function(
  data, 
  design_opt, 
  weight,
  by,
  by_index,
  file
  ) {

  # Inherited Globals #
  df <- data.table::copy(data$df)
  l_XX <- data$l_XX
  T_max_XX <- data$T_max_XX

	## Fetch the arguments 
  des_p <- as.numeric(design_opt[1L])
  des_path <- design_opt[2L]
  des_n <- l_XX
  des_per <- des_p * 100

	## keep periods up to ℓ periods after the first switch
  df[, F_g_plus_n_XX := F_g_XX + des_n - 1L]
  df <- df[time_XX >= F_g_XX - 1L & time_XX <= F_g_plus_n_XX]
  data.table::setorder(df, group_XX, time_XX)
  # Generate row number within group
  df[, time_l_XX := seq_len(.N), by = group_XX]
  df <- df[, .(group_XX, time_l_XX, weight_XX, treatment_XX, F_g_XX)]

	## Aggregate weights by group
  if (!is.null(weight)) {
    df[, g_weight_XX := sum(weight_XX, na.rm = TRUE), by = group_XX]
  } else {
    df[, g_weight_XX := 1L]
  }
  df[, weight_XX := NULL]

  max_time <- max(df$time_l_XX, na.rm = TRUE)
  treat_list <- paste0("treatment_XX", seq_len(max_time))
  treat_str <- paste(treat_list, collapse = ",")
  lapply(seq_len(max_time), function(i) {
    target_col <- treat_list[i]
    df[, (target_col) := mean(treatment_XX[time_l_XX == i], na.rm = TRUE), by = group_XX]
  })
  df[, c("time_l_XX", "treatment_XX") := NULL]
  df <- unique(df)

	## Drop missing treatments 
  for (var in treat_list) {
    df <- df[!is.na(get(var))]
  }

	## Creating variable to store number of groups per treatment path and collapsing
  df[, N_XX := 1L]
  df[, N_w_XX := (g_weight_XX * N_XX) / sum(g_weight_XX, na.rm = TRUE)]
  df[, c("group_XX", "g_weight_XX") := NULL]
  # Sum by treat_list
  df[, N_XX := sum(N_XX, na.rm = TRUE), by = treat_list]
  df[, N_w_XX := sum(N_w_XX, na.rm = TRUE), by = treat_list]
  df[, F_g_XX := NULL]
  df <- unique(df)
  tot_switch <- sum(df$N_XX, na.rm = TRUE)

	## Keep the observations amounting to p% of the detected treatment paths
  df[, neg_N_XX := -N_XX]
  # Create group rank
  df[, treat_key := do.call(paste, c(.SD, sep = "_")), .SDcols = treat_list]
  df[, treat_GRP := as.numeric(factor(treat_key))]
  df[, treat_key := NULL]
  data.table::setorder(df, neg_N_XX, treat_GRP)
  df[, c("neg_N_XX", "treat_GRP") := NULL]
  df[, cum_sum_XX := cumsum(N_w_XX)]
  df[, in_table_XX := as.numeric(cum_sum_XX <= des_p)]
  data.table::setorder(df, in_table_XX, cum_sum_XX)
  # Generate row id within in_table_XX
  df[, id_XX := seq_len(.N), by = in_table_XX]

	## Keep all observations up to the first exceeding the p%
  df <- df[in_table_XX == 1L | (in_table_XX == 0L & id_XX == 1L)]

	## Store the final % of groups included by the design option
  if (des_p < 1) {
    last_p <- 100 * min(df$cum_sum_XX[df$in_table_XX == 0L])
  } else {
    last_p <- 100
  }
  df[, neg_N_XX := -N_XX]
  df[, treat_key := do.call(paste, c(.SD, sep = "_")), .SDcols = treat_list]
  df[, treat_GRP := as.numeric(factor(treat_key))]
  df[, treat_key := NULL]
  data.table::setorder(df, neg_N_XX, treat_GRP)
  df <- df[, c("N_XX", "N_w_XX", treat_list), with = FALSE]
  df[, N_w_XX := N_w_XX * 100]

	## Prepare matrix for the output table
  coln <- c("N", "Share")
  rown <- paste0("TreatPath", seq_len(nrow(df)))
  desmat <- as.matrix(df)
  storage.mode(desmat) <- "numeric"

	## Generate the column/row names
  coln <- c(coln, paste0("\U2113", "=", seq_len(ncol(desmat) - 2L) - 1L))
  colnames(desmat) <- coln
  rownames(desmat) <- rown 
  
  desmat[, 2L] <- noquote(sprintf("%s", format(round(desmat[,2L], 2L), big.mark=",", scientific=FALSE, trim=TRUE)))
  des_const <- c(l_XX, des_per, tot_switch, last_p)
  names(des_const) <- c("effects", "coverage_opt", "switchers", "detected_coverage")

  ## Save output as xlsx
  if (des_path != "console")  {
      by_add <- ""
      if (by_index != "_no_by") {
        by_add <- paste0(", ",abbreviate(by,5L), "=", by_index)
      }
      file[[paste0("Design",by_add)]] <- as.data.frame(desmat)
  }

  design <- list(
    design_path = des_path,
    design_mat = noquote(desmat),
    design_const = des_const,
    design_file = file
  )
  return(design)
}