#' Option that produces a table showing the number of groups switching for the first time by period
#' @param data data
#' @param dfs dfs
#' @param by by
#' @param by_index by_index
#' @param file file
#' @returns A list with the date_first_switch output.
#' @noRd
did_multiplegt_dyn_dfs <- function(
  data, 
  dfs,
  by,
  by_index,
  file
  ) {

  # Inherited Globals #
  df <- data$df
  T_max_XX <- data$T_max_XX

	## Fetch the arguments 
  dfs_opt <- dfs[1]
  dfs_path <- dfs[2]

	## Error message if the arguments in the option were specified wrong
  if (dfs_opt != "" && dfs_opt != "by_baseline_treat") {
    stop("Only option by_baseline_treat allowed.")
  }

	## Drop non switchers and keep one observation per group
  df <- df[!(F_g_XX == T_max_XX + 1L | is.na(F_g_XX))]
  df <- df[time_XX == F_g_XX]
  df <- df[, .(group, time, F_g_XX, d_sq_XX)]

	## When by_baseline_treat is not specified
  if (dfs_opt == "") {
		## collapse the number of groups by time
    df[, tot_s := .N, by = time]
    df[, c("group", "F_g_XX", "d_sq_XX") := NULL]
    df <- unique(df)
    data.table::setorder(df, time)
		## generate the share of each group
    df[, share_XX := (tot_s / sum(tot_s, na.rm = TRUE)) * 100]
    df <- df[, .(tot_s, share_XX, time)]
		## make matrix with the number and share of groups by time
    dfsmat <- as.matrix(df[, .(tot_s, share_XX)])
    storage.mode(dfsmat) <- "numeric"
    rown <- as.character(df$time)
    coln <- c("N", "Share")
    rownames(dfsmat) <- rown
    colnames(dfsmat) <- coln
    dfsmat[, 2L] <- sprintf("%s", format(round(dfsmat[,2L], 2L), big.mark=",", scientific=FALSE, trim=TRUE))

		## output as excel
    if (dfs_path != "console") {
      by_add <- ""
      if (by_index != "_no_by") {
        by_add <- paste0(", ",abbreviate(by,5L), "=", by_index)
      }
      file[[paste0("Switch. Dates",by_add)]] <- as.data.frame(dfsmat)
    }
    
    res_dfs <- list(
      dfs_opt = dfs_opt,
      dfs_path = dfs_path,
      dfs_mat = noquote(dfsmat)
    )
  }

	## When by_baseline_treat is specified
  if (dfs_opt == "by_baseline_treat") {
		## collapse, but this time by time and status quo treatment
    df[, tot_s := .N, by = .(time, d_sq_XX)]
    df[, c("group", "F_g_XX") := NULL]
    df <- unique(df)
    data.table::setorder(df, d_sq_XX, time)
    levels_d_sq_XX <- levels(factor(df$d_sq_XX))

    res_dfs <- list(
      dfs_opt = dfs_opt,
      dfs_path = dfs_path,
      levels_baseline_treat = length(levels_d_sq_XX)
    )
    by_add <- ""
    if (by_index != "_no_by") {
      by_add <- paste0(", ",abbreviate(by,5L), "=", by_index)
    }
    for (l in levels_d_sq_XX) {
      df_by <- df[d_sq_XX == as.numeric(l)]
      df_by[, share_XX := (tot_s / sum(tot_s, na.rm = TRUE)) * 100]
      df_by <- df_by[, .(tot_s, share_XX, time)]
		  ## make matrix with the number and share of groups by time, one for each level of status quo treatment
      dfsmat <- as.matrix(df_by[, .(tot_s, share_XX)])
      storage.mode(dfsmat) <- "numeric"
      rownames(dfsmat) <- as.character(df_by$time)
      colnames(dfsmat) <- c("N", "Share")
      
      dfsmat[, 2L] <- sprintf("%s", format(round(dfsmat[,2L], 2L), big.mark=",", scientific=FALSE, trim=TRUE))

		  ## output as excel
      if (dfs_path != "console") {
        sheetn <- paste0("Base treat. ", l)
        file[[paste0(sheetn, by_add)]] <- as.data.frame(dfsmat)
      }
      res_dfs <- append(res_dfs, l)
      res_dfs <- append(res_dfs, list(noquote(dfsmat)))
    }
    level_names <- unlist(lapply(seq_along(levels_d_sq_XX), function(idx) c(paste0("level", idx), paste0("dfs_mat", idx))))
    names(res_dfs)[4:length(res_dfs)] <- level_names
  }

  if (length(names(file)) > 0L) {
    res_dfs$dfs_file <- file
  }
  return(res_dfs)
}