#' Option that shows a table with weights attached to each normalized effect
#' @param data data
#' @param normalized normalized
#' @param normopt normopt
#' @param same_switchers same_switchers
#' @param continuous continuous
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
  df <- data.table::copy(data$df)
  l_XX <- data$l_XX
  list2env(data$delta, envir = environment())

	## Set up the matrix for the output table
  weight_mat <- matrix(NA, nrow = l_XX, ncol = l_XX) 
  coln <- c()
  rown <- c()
  for (i in 1:l_XX) {
    coln <- c(coln, paste0("\U2113","=",i))
    
    df[, paste0("N_gt_",i,"_temp_XX") := data.table::fifelse(
      time_XX == F_g_XX - 1L + i 
      & i <= L_g_XX
      & !is.na(get(paste0("N_gt_control_",i,"_XX")))
      & get(paste0("N_gt_control_",i,"_XX")) > 0L
      & !is.na(get(paste0("diff_y_",i,"_XX"))), 
      N_gt_XX, NA_real_)]

    temp_col <- paste0("N_gt_",i,"_temp_XX")
    target_col <- paste0("N_gt_",i,"_XX")
    df[, (target_col) := mean(get(temp_col), na.rm = TRUE), by = group_XX]
    df[, (temp_col) := NULL]
    for (k in 0:(i - 1L)) {

			# Visualization by k
      row <- k + 1L

			## Compute the delta_l_k, if the continuous option is specified the original treatment values are used
      delta_col <- paste0("delta_",i,"_",k)
      if (is.null(continuous)) {
        df[, (delta_col) := data.table::fifelse(time_XX == F_g_XX - 1L + i - k & F_g_XX - 1L + i <= T_g_XX, abs(treatment_XX - d_sq_XX), NA_real_)]
      } else {
        df[, (delta_col) := data.table::fifelse(time_XX == F_g_XX - 1L + i - k & F_g_XX - 1L + i <= T_g_XX, abs(treatment_XX_orig - d_sq_XX_orig), NA_real_)]
      }

      if (same_switchers == TRUE) {
        df[, (delta_col) := data.table::fifelse(F_g_XX - 1L + l_XX > T_g_XX, 0, get(delta_col))]
      }

      df[, (delta_col) := get(delta_col) * get(target_col)]
      weight_mat[row, i] <- (sum(df[[delta_col]], na.rm = TRUE) / get(paste0("delta_D_",i,"_global_XX"))) / data$mat_res_XX[i,ncol(data$mat_res_XX)-1]
    }
    df[, (target_col) := NULL]
  }

	## Generating the row names 
  rown <- paste0("k=", 0:(l_XX - 1L))

	## Fill the values for the displayed table
  mat_total <- weight_mat
  mat_total[is.na(mat_total)] <- 0
  total <- matrix(1,nrow=1,ncol=l_XX) %*% mat_total
  weight_mat <- rbind(weight_mat, total)
  rownames(weight_mat) <- c(rown, "Total")
  colnames(weight_mat) <- coln
  weight_mat[ , ] <- sprintf("%s", format(round(weight_mat[ , ], 3L), big.mark=",", scientific=FALSE, trim=TRUE))

  return(list(norm_weight_mat = noquote(weight_mat)))
}