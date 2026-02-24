#' Checks whether the variable specified in the by option is time-variant
#' The program allows only time-variant variables in the by option
#' @param df df
#' @param group group
#' @param by by
#' @note polars is suggested for better performance
#' @returns A logical value.
#' @noRd 
did_multiplegt_dyn_by_check <- function(
    df,
    group,
    by
) {
    sd_by <- NULL
    df <- as.data.frame(df)
    sd_agg <- aggregate(df[[by]], by = list(grp = df[[group]]), FUN = sd, na.rm = TRUE)
    names(sd_agg)[2] <- "sd_by"
    df <- merge(df, sd_agg, by.x = group, by.y = "grp", all.x = TRUE)
    return(mean(df$sd_by, na.rm = TRUE) == 0)
}