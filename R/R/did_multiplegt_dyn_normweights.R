#' Option that shows a table with weights attached to each normalized effect
#' @param data data
#' @param normalized normalized
#' @param normopt normopt
#' @param same_switchers same_switchers
#' @param continuous continuous
#' @note polars is suggested for better performance
#' @returns A matrix with the normalized_weights option output.
#' @noRd
did_multiplegt_dyn_normweights <- function(
  data, 
  normalized, 
  normopt, 
  same_switchers,
  continuous
  ) {
    # Inherited Globals #
    df <- data$df
    # Convert polars DataFrame to R data.frame for base R operations
    if (inherits(df, "polars_data_frame")) {
      df <- as.data.frame(df)
    }
    l_XX <- data$l_XX
    for (v in names(data$delta)) {
      assign(v, data$delta[[v]])
    }

    group_XX <- NULL

  suppressWarnings({

	## Set up the matrix for the output table
  weight_mat <- matrix(NA, nrow = l_XX, ncol = l_XX) 
  coln <- c()
  rown <- c()
  for (i in 1:l_XX) {
    coln <- c(coln, paste0("\U2113","=",i))
    
    df[[paste0("N_gt_",i,"_temp_XX")]] <- ifelse(
      df$time_XX == df$F_g_XX - 1 + i 
      & i <= df$L_g_XX
      & !is.na(df[[paste0("N_gt_control_",i,"_XX")]])
      & df[[paste0("N_gt_control_",i,"_XX")]] >0
      & !is.na(df[[paste0("diff_y_",i,"_XX")]]), 
      df$N_gt_XX, NA)

    temp_col <- paste0("N_gt_",i,"_temp_XX")
    target_col <- paste0("N_gt_",i,"_XX")
    agg_temp <- aggregate(df[[temp_col]], by = list(group_XX = df$group_XX), FUN = mean, na.rm = TRUE)
    names(agg_temp)[2] <- target_col
    df <- merge(df, agg_temp, by = "group_XX", all.x = TRUE)
    df[[paste0("N_gt_",i,"_temp_XX")]] <- NULL
    for (k in 0:(i-1)) {

			# Visualization by k
      row <- k + 1

			## Compute the delta_l_k, if the continuous option is specified the original treatment values are used
      if (is.null(continuous)) {
        df[[paste0("delta_",i,"_",k)]] <- ifelse(df$time_XX == df$F_g_XX - 1 + i - k & df$F_g_XX - 1 + i <= df$T_g_XX, abs(df$treatment_XX - df$d_sq_XX), NA)
      } else {
        df[[paste0("delta_",i,"_",k)]] <- ifelse(df$time_XX == df$F_g_XX - 1 + i - k & df$F_g_XX - 1 + i <= df$T_g_XX, abs(df$treatment_XX_orig - df$d_sq_XX_orig), NA)
      }

      if (same_switchers == TRUE) {
        df[[paste0("delta_",i,"_",k)]] <- ifelse(df$F_g_XX - 1 + l_XX > df$T_g_XX, 0, df[[paste0("delta_",i,"_",k)]])        
      }

      df[[paste0("delta_",i,"_",k)]] <- df[[paste0("delta_",i,"_",k)]] * df[[paste0("N_gt_",i,"_XX")]]
      weight_mat[row, i] <- (sum(df[[paste0("delta_",i,"_",k)]], na.rm = TRUE) / get(paste0("delta_D_",i,"_global_XX"))) / data$mat_res_XX[i,ncol(data$mat_res_XX)-1]
    }
    df[[paste0("N_gt_",i,"_XX")]] <- NULL
  }

	## Generating the row names 
  for (j in 1:l_XX) {
    rown <- c(rown, paste0("k=",j-1))
  }     

	## Fill the values for the displayed table
  mat_total <- weight_mat
  mat_total[is.na(mat_total)] <- 0
  total <- matrix(1,nrow=1,ncol=l_XX) %*% mat_total
  weight_mat <- rbind(weight_mat, total)
  rownames(weight_mat) <- c(rown, "Total")
  colnames(weight_mat) <- coln
  weight_mat[ , ] <- sprintf("%s", format(round(weight_mat[ , ], 3), big.mark=",", scientific=FALSE, trim=TRUE))

  return(list(norm_weight_mat = noquote(weight_mat)))
  })

}