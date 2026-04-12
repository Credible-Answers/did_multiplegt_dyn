#' Option that allows to see which groups were included in the estimation and if they were switcher in/out ot control
#' @param data data
#' @param Gn Gn
#' @param Tn Tn
#' @returns The input data.table plus two added columns.
#' @noRd
did_save_sample <- function(
    data,
    Gn,
    Tn
    ) {
  df <- data$df
  ## keeping only group, time and switcher status (if not missing)
  df_save <- df[!is.na(group) & !is.na(time), .(group, time, S_g_XX, switchers_tag_XX)]
  ## rename to user-facing names
  data.table::setnames(df_save, c("group", "time", "S_g_XX", "switchers_tag_XX"), c(Gn, Tn, "did_sample", "did_effect"))
  ## redefine did_sample to show if group is switcher in/out or control
  df_save[, did_sample := data.table::fifelse(is.na(did_sample), 0, data.table::fifelse(did_sample == 0, -1, did_sample))]
  df_save[, did_sample := factor(did_sample, levels = c(0, 1, -1), labels = c("Control", "Switcher-in", "Switcher-out"))]
  return(df_save)
}

#' Adjustment of save_sample output in case of by option
#' @param obj A did_multiplegt_dyn object
#' @returns The input data.table plus two added columns.
#' @noRd
adj_save_sample <- function(obj) {
  saved_sample <- obj$by_level_1$save_sample
  obj$by_level_1$save_sample <- NULL
  if (length(obj$by_levels) > 1L) {
    for (j in 2:length(obj$by_levels)) {
      saved_sample <- rbind(saved_sample, obj[[paste0("by_level_", j)]]$save_sample)
      obj[[paste0("by_level_", j)]]$save_sample <- NULL
    }
  }
  data.table::setorderv(saved_sample, c(obj$args$group, obj$args$time))
  obj <- append(obj, list(saved_sample))
  return(obj)
}


