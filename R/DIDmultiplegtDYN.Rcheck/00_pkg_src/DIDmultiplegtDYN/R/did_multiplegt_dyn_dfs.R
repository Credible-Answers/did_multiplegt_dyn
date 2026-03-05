#' Option that produces a table showing the number of groups switching for the first time by period
#' @param data data
#' @param dfs dfs
#' @param by by
#' @param by_index by_index
#' @param file file
#' @note polars is suggested for better performance
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
    # Convert polars DataFrame to R data.frame for base R operations
    if (inherits(df, "polars_data_frame")) {
      df <- as.data.frame(df)
    }
    T_max_XX <- data$T_max_XX

    tot_s <- NULL
    time <- NULL

  suppressWarnings({

	## Fetch the arguments 
  dfs_opt <- dfs[1]
  dfs_path <- dfs[2]

	## Error message if the arguments in the option were specified wrong
  if (dfs_opt != "" & dfs_opt != "by_baseline_treat") {
    stop("Only option by_baseline_treat allowed.")
  }

	## Drop non switchers and keep one observation per group
  df <- subset(df, !(df$F_g_XX == T_max_XX + 1 | is.na(df$F_g_XX)))
  df <- subset(df, df$time_XX == df$F_g_XX)
  df <- subset(df, select = c("group", "time", "F_g_XX", "d_sq_XX"))

	## When by_baseline_treat is not specified
  if (dfs_opt == "") {
		## collapse the number of groups by time
    df$tot_s <- 1
    tot_agg <- aggregate(df$tot_s, by = list(time = df$time), FUN = sum, na.rm = TRUE)
    names(tot_agg)[2] <- "tot_s_sum"
    df <- merge(df, tot_agg, by = "time", all.x = TRUE)
    df$tot_s <- df$tot_s_sum
    df$tot_s_sum <- NULL
    df$group <- df$F_g_XX <- df$d_sq_XX <- NULL
    df <- unique(df)
    df <- df[order(df$time), ]
		## generate the share of each group
    df$share_XX <- (df$tot_s / sum(df$tot_s, na.rm = TRUE)) * 100
    df <- subset(df, select = c("tot_s", "share_XX", "time"))
		## make matrix with the number and share of groups by time
    dfsmat <- matrix(NA, ncol = 2, nrow = dim(df)[1])
    rown <- c()
    coln <- c("N", "Share")
    df <- data.frame(df)
    for (j in 1:2) {
      for (i in 1:dim(df)[1]) {
        if (j == 1) {
          rown <- c(rown, df$time[i])
        }
        dfsmat[i,j] <- as.numeric(df[i,j])
      }
    }
    colnames(dfsmat) <- coln
    rownames(dfsmat) <- rown 
    dfsmat[, 2] <- sprintf("%s", format(round(dfsmat[,2], 2), big.mark=",", scientific=FALSE, trim=TRUE))

		## output as excel
    if (dfs_path != "console") {
      by_add <- ""
      if (by_index != "_no_by") {
        by_add <- paste0(", ",abbreviate(by,5), "=", by_index)
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
    df$tot_s <- 1
    tot_agg2 <- aggregate(df$tot_s, by = list(time = df$time, d_sq_XX = df$d_sq_XX), FUN = sum, na.rm = TRUE)
    names(tot_agg2)[3] <- "tot_s_sum"
    df <- merge(df, tot_agg2, by = c("time", "d_sq_XX"), all.x = TRUE)
    df$tot_s <- df$tot_s_sum
    df$tot_s_sum <- NULL
    df$group <- df$F_g_XX <- NULL
    df <- unique(df)
    df <- df[order(df$d_sq_XX, df$time), ]
    levels_d_sq_XX <- levels(factor(df$d_sq_XX))

    res_dfs <- list(
      dfs_opt = dfs_opt,
      dfs_path = dfs_path,
      levels_baseline_treat = length(levels_d_sq_XX)
    )
    index <- 1
    by_add <- ""
    if (by_index != "_no_by") {
      by_add <- paste0(", ",abbreviate(by,5), "=", by_index)
    }
    for (l in levels_d_sq_XX) {
      df_by <- subset(df, df$d_sq_XX == l)
      df_by$share_XX <- (df_by$tot_s / sum(df_by$tot_s, na.rm = TRUE)) * 100
      df_by <- subset(df_by, select = c("tot_s", "share_XX", "time"))
		  ## make matrix with the number and share of groups by time, one for each level of status quo treatment
      dfsmat <- matrix(NA, ncol = 2, nrow = dim(df_by)[1])
      rown <- c()
      coln <- c("N", "Share")
      df_by <- data.frame(df_by)
      for (j in 1:2) {
        for (i in 1:dim(df_by)[1]) {
          if (j == 1) {
            rown <- c(rown, df_by$time[i])
          }
          dfsmat[i,j] <- as.numeric(df_by[i,j])
        }
      }
      colnames(dfsmat) <- coln
      rownames(dfsmat) <- rown 
      
      dfsmat[, 2] <- sprintf("%s", format(round(dfsmat[,2], 2), big.mark=",", scientific=FALSE, trim=TRUE))

      res_dfs <- append(res_dfs, l)
      res_dfs <- append(res_dfs, list(noquote(dfsmat)))
      names(res_dfs)[length(res_dfs) - 1] <- paste0("level",index)
      names(res_dfs)[length(res_dfs)] <- paste0("dfs_mat",index)

		  ## output as excel
      if (dfs_path != "console") {
        sheetn <- paste0("Base treat. ", l)
        file[[paste0(sheetn, by_add)]] <- as.data.frame(dfsmat)
      }
      index <- index + 1
    }
  }
  })

  if (length(names(file)) > 0) {
    res_dfs$dfs_file <- file
  }
  return(res_dfs)
}