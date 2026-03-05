#' @title rnames method for did_multiplegt_dyn
#' @name rnames.did_multiplegt_dyn
#' @description A customized rnames method for did_multiplegt_dyn output
#' @param obj A did_multiplegt_dyn object
#' @param ignore Sublists to be ignored
#' @param ... Undocumented
#' @import rnames
#' @returns The same output as rnames.
#' @export
rnames.did_multiplegt_dyn <- function(obj, ignore = c("plot", "args"), ...) {
    class(obj) <- "list"
    return(rnames(obj = obj, ignore = c("plot", "args")))
}