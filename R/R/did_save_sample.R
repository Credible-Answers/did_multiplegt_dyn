#' Option that allows to see which groups were included in the estimation and if they were switcher in/out ot control
#' @param data data
#' @param Gn Gn
#' @param Tn Tn
#' @note polars is suggested for better performance
#' @returns The input dataframe df plus two added columns.
#' @noRd
did_save_sample <- function(
    data, 
    Gn, 
    Tn
    ) {
  df <- data$df
  suppressWarnings({
	## keeping only group, time and switcher status (if not missing)
  df_save <- subset(df, !is.na(df$group) & !is.na(df$time))
  df_save <- subset(df_save, select = c("group", "time", "S_g_XX", "switchers_tag_XX"))
	## redefine S_g_XX to show if group is switcher in/out or control	
  names(df_save) <- c(Gn, Tn, "did_sample", "did_effect") 
  df_save$did_sample <- ifelse(df_save$did_sample == 0, -1, df_save$did_sample)
  df_save$did_sample <- ifelse(is.na(df_save$did_sample), 0, df_save$did_sample)
  df_save$did_sample <- factor(df_save$did_sample, levels = c(0,1,-1), labels = c("Control", "Switcher-in", "Switcher-out"))
  })
	# Return dataset to be merged to main data
  return(df_save)
}

#' Adjustment of save_sample output in case of by option
#' @param obj A did_multiplegt_dyn object
#' @returns The input dataframe df plus two added columns.
#' @noRd
adj_save_sample <- function(obj) {
  saved_sample <- obj$by_level_1$save_sample
  obj$by_level_1$save_sample <- NULL
  if (length(obj$by_levels) > 1) {
    for (j in 2:length(obj$by_levels)) {
      saved_sample <- rbind(saved_sample, obj[[paste0("by_level_", j)]]$save_sample)
      obj[[paste0("by_level_", j)]]$save_sample <- NULL
    }
  }
  saved_sample <- saved_sample[order(saved_sample[[obj$args$group]], saved_sample[[obj$args$time]]), , drop = FALSE]
  rownames(saved_sample) <- NULL
  obj <- append(obj, list(saved_sample))
  return(obj)
}


