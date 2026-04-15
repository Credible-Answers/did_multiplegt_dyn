#' Checks whether the variable specified in the by option is time-variant
#' The program allows only time-variant variables in the by option
#' @param df df
#' @param group group
#' @param by by
#' @returns A logical value.
#' @noRd 
did_multiplegt_dyn_by_check <- function(
    df,
    group,
    by
) {
    sd_agg <- df[, .(sd_by = stats::sd(get(by), na.rm = TRUE)), by = group]
    return(mean(sd_agg$sd_by, na.rm = TRUE) == 0)
}