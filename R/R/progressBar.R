#' Internal function to generate a progress bar for the bootstrap option
#' @param int integer
#' @param tot tot
#' @returns No returns, just prints dots for each bootstrap replication
#' @noRd
progressBar <- function(
    int,
    tot
    ) {
    cat(".")
    if (int %% 5L == 0L) {
        cat(sprintf("%.0f", int))
    }
    if (int %% 70L == 0L) {
        cat("\n")
    }
    if (int == tot) {
        cat("\n")
    }
}