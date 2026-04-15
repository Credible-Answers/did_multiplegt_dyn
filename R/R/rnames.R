#' @title rnames method for did_multiplegt_dyn
#' @name rnames.did_multiplegt_dyn
#' @description A customized rnames method for did_multiplegt_dyn output.
#' Requires the rnames package to be installed.
#' @param obj A did_multiplegt_dyn object
#' @param ignore Sublists to be ignored
#' @param ... Undocumented
#' @returns The same output as rnames.
#' @export
rnames.did_multiplegt_dyn <- function(obj, ignore = c("plot", "args"), ...) {
    if (!requireNamespace("rnames", quietly = TRUE)) {
      stop("Package 'rnames' is required for this function. Install it with install.packages('rnames').")
    }
    class(obj) <- "list"
    return(rnames::rnames(obj = obj, ignore = c("plot", "args")))
}