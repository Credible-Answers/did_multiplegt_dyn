#' @title A print method for did_multiplegt_dyn
#' @name print.did_multiplegt_dyn 
#' @description A customized printed display for did_multiplegt_dyn output
#' @param x A did_multiplegt_dyn object
#' @param ... Undocumented
#' @returns No return, custom print method for did_multiplegt_dyn objects. Estimation tables are fetched from the object and displayed in the same style as the Stata did_multiplegt_dyn command.
#' @export
print.did_multiplegt_dyn <- function(x, ...) {
    cat("\n")

    by_levels <- c("_no_by")
    if (!is.null(x$by_levels)) {
        by_levels <- x$by_levels
    }

    for (b in 1:length(by_levels)) {

        if (by_levels[b] == "_no_by") {
            ref <- x
        } else {
            ref <- x[[paste0("by_level_",b)]]
            if (!is.null(x$args[["by"]])) {
                section <- paste(" By",x$args$by, "=", by_levels[b], "###")
            } else if (!is.null(x$args[["by_path"]])) {
                section <- paste0(" By treatment path: (", by_levels[b],") ", "###")
            }
            cat(noquote(strrep("#", 70 - nchar(section) - 1)));cat(section);
            cat("\n");cat("\n");
        }

        cat(noquote(strrep("-", 70)));cat("\n");
        cat(strrep(" ", 7));cat("Estimation of treatment effects: Event-study effects");cat("\n");
        cat(noquote(strrep("-", 70)));cat("\n");
        mat_print(ref$results$Effects[, 1:(6 + ((!is.null(x$args$weight))*2))])
        cat("\n")
        if (!is.null(ref$results$p_jointeffects)) {
            if (is.na(ref$results$p_jointeffects)) {
                cat("Test of joint nullity of the effects : p-value = not computed (see warnings)")
            } else {
                cat(sprintf("Test of joint nullity of the effects : p-value = %.4f", ref$results$p_jointeffects))
            }
            cat("\n")
        }

        if (!is.null(ref$results$p_equality_effects)) {
            if (is.na(ref$results$p_equality_effects)) {
                cat("Test of equality of the effects : p-value = not computed (see warnings)")
            } else {
                cat(sprintf("Test of equality of the effects : p-value = %.4f", ref$results$p_equality_effects))
            }
            cat("\n");cat("\n")
        }

        if (isTRUE(x$args$trends_lin)) {
            cat(noquote(strrep("-", 70)));cat("\n");
            cat(strrep(" ", 4));cat("When the trends_lin is specified no average effects are reported");cat("\n");
            cat(noquote(strrep("-", 70)));cat("\n");

        } else {
            cat(noquote(strrep("-", 70)));cat("\n");
            cat(strrep(" ", 4));cat("Average cumulative (total) effect per treatment unit");cat("\n");
            cat(noquote(strrep("-", 70)));cat("\n");
            mat_print(ref$results$ATE[, 1:(6 + ((!is.null(x$args$weight))*2))])
            cat(sprintf("Average number of time periods over which a treatment effect is accumulated: %.4f", ref$results$delta_D_avg_total))
            cat("\n")
        }
        cat("\n")

        if (ref$results$N_Placebos != 0) {
            cat(noquote(strrep("-", 70)));cat("\n");
            cat(strrep(" ", 4));cat(" Testing the parallel trends and no anticipation assumptions");cat("\n");
            cat(noquote(strrep("-", 70)));cat("\n");
            mat_print(ref$results$Placebos[, 1:(6 + ((!is.null(x$args$weight))*2))])
            if (is.null(x$args$bootstrap)) {
                cat("\n")
                if (!is.null(ref$results$p_jointplacebo) && is.na(ref$results$p_jointplacebo)) {
                    cat("Test of joint nullity of the placebos : p-value = not computed (see warnings)")
                } else if (!is.null(ref$results$p_jointplacebo)) {
                    cat(sprintf("Test of joint nullity of the placebos : p-value = %.4f", ref$results$p_jointplacebo))
                }
                cat("\n")
            }
            cat("\n")
        }

        if (!is.null(ref$design)) {
            if (ref$design$design_path == "console") {
                cat("\n")
                cat(noquote(strrep("-", 70)));cat("\n");
                cat(strrep(" ", 4));cat(sprintf("Detection of treatment paths - %.0f periods after first switch", ref$design$design_const[1]));cat("\n");
                cat(noquote(strrep("-", 70)));cat("\n");
                print(ref$design$design_mat);cat("\n");
                cat(sprintf("Treatment paths detected in at least %.2f%% of the %.0f switching groups for which %.0f effects could be estimated", ref$design$design_const[2], ref$design$design_const[3], ref$design$design_const[1]))
                cat(sprintf(" (Total %% = %.2f%%)", ref$design$design_const[4]));cat("\n");cat("\n");
                cat("Design interpretation (first row):");cat("\n")
                n_groups <- ref$design$design_mat[1,1]
                d_start <- ref$design$design_mat[1,3]
                d_vec <- "("
                for (i in 1:ref$design$design_const[1]) {
                d_vec <- paste0(d_vec, ref$design$design_mat[1, 3 + i],",")
                }
                d_vec <- paste0(substr(d_vec,1,nchar(d_vec)-1),")")
                cat(sprintf("%s groups started with treatment %s and experienced treatment path %s", n_groups, d_start, d_vec))
                cat("\n")
            }
            else {
                cat(sprintf("Design exported to %s", ref$design$design_path)); cat("\n")
            }
        }

        if (!is.null(ref$date_first_switch)) {
            if (ref$date_first_switch$dfs_opt != "by_baseline_treat") {
                if (ref$date_first_switch$dfs_path == "console") {
                    cat("\n")
                    cat(noquote(strrep("-", 40)));cat("\n");
                    cat(strrep(" ", 7));cat("Switching dates");cat("\n");
                    cat(noquote(strrep("-", 40)));cat("\n");
                    cat("By any status quo treatment");cat("\n");
                    print(ref$date_first_switch$dfs_mat)
                    cat("\n")
                }
                else {
                    cat(sprintf("Switching dates exported to %s", ref$date_first_switch$dfs_path)); cat("\n")
                }
            }
            else {
                if (ref$date_first_switch$dfs_path == "console") {
                    cat("\n")
                    cat(noquote(strrep("-", 40)));cat("\n");
                    cat(strrep(" ", 7));cat("Switching dates");cat("\n");
                    cat(noquote(strrep("-", 40)));cat("\n");
                    for (l in 1:ref$date_first_switch$levels_baseline_treat) {
                        cat(sprintf("Status quo treatment = %s", ref$date_first_switch[[paste0("level",l)]]));cat("\n");
                        print(ref$date_first_switch[[paste0("dfs_mat",l)]])
                        cat("\n")
                    }
                }
                if (ref$date_first_switch$dfs_path != "console") {
                    cat(sprintf("Switching dates exported to %s", ref$date_first_switch$dfs_path)); cat("\n")
                }
            }
        }

        if (!is.null(ref$normalized_weights)) {
            cat("\n")
            cat(noquote(strrep("-", 60)));cat("\n");
            cat(strrep(" ", 13));cat("Weights on treatment lags");cat("\n");
            cat(noquote(strrep("-", 60)));cat("\n");
            print(ref$normalized_weights$norm_weight_mat)
            cat("\n")
        }

        if (!is.null(ref$results$predict_het)) {
            cat("\n")
            cat(noquote(strrep("-", 60)));cat("\n");
            cat(strrep(" ", 13));cat("Predicting effect heterogeneity");cat("\n");
            cat(noquote(strrep("-", 60)));cat("\n");
            het_ref <- ref$results$predict_het$effect
            for (l in levels(factor(subset(het_ref, het_ref > 0)))) {
                het_tab <- subset(ref$results$predict_het, ref$results$predict_het$effect == l)
                het_mat <- as.matrix(het_tab[,c(3,4,6,7,8)])
                rownames(het_mat) <- het_tab$covariate
                colnames(het_mat) <- c("Estimate", "SE", "LB CI", "UB CI", "N")
                pval <- mean(het_tab$pF)
                cat(sprintf("Effect %s:\n", l))
                mat_print(het_mat)
                cat(sprintf("Test of joint nullity of the estimates : p-value = %.4f\n", pval));cat("\n")
            }
            if (length(levels(factor(subset(het_ref, het_ref <  0)))) > 0) {
                het_pl_ref <- subset(ref$results$predict_het, ref$results$predict_het$effect < 0)
                het_pl_ref$effect <- - het_pl_ref$effect
                for (l in levels(factor(het_pl_ref$effect))) {
                    het_tab <- subset(het_pl_ref, het_pl_ref$effect == l)
                    het_mat <- as.matrix(het_tab[,c(3,4,6,7,8)])
                    rownames(het_mat) <- het_tab$covariate
                    colnames(het_mat) <- c("Estimate", "SE", "LB CI", "UB CI", "N")
                    pval <- mean(het_tab$pF)
                    cat(sprintf("Placebo %s:\n", l))
                    mat_print(het_mat)
                    cat(sprintf("Test of joint nullity of the estimates : p-value = %.4f\n", pval));cat("\n")
                }
            }
        }

        # Display vcov warnings if any
        if (!is.null(ref$results$vcov_warnings)) {
            cat("\n")
            cat(noquote(strrep("-", 70)));cat("\n");
            cat(strrep(" ", 4));cat("Warnings");cat("\n");
            cat(noquote(strrep("-", 70)));cat("\n");
            for (w in ref$results$vcov_warnings) {
                cat(paste0("- ", w));cat("\n")
            }
        }
    }

    cat("\n")
    cat("The development of this package was funded by the European Union.");cat("\n")
    cat("ERC REALLYCREDIBLE - GA N. 101043899");cat("\n")

}

#' @title A summary method for did_multiplegt_dyn
#' @name summary.did_multiplegt_dyn 
#' @description A customized summary display for did_multiplegt_dyn output
#' @param object A did_multiplegt_dyn object
#' @param ... Undocumented
#' @returns No return, custom summary method for did_multiplegt_dyn objects. Estimation tables are fetched from the object and displayed in the same style as the Stata did_multiplegt_dyn command.
#' @export
summary.did_multiplegt_dyn <- function(object, ...) {
    print(object)
}

#' Auxiliary function for print method
#' @param mat mat
#' @returns No return, just prints the refined input.
#' @noRd
mat_print <- function(mat) {
    if (inherits(mat,"matrix")) {
        dis <- matrix(data = 0, nrow = nrow(mat) , ncol = ncol(mat))
        dis[,1:4] <- sprintf("%s", format(round(mat[,1:4], 5), big.mark=",", scientific=FALSE, trim=TRUE))
        dis[,5:ncol(dis)] <- 
            sprintf("%s", format(round(mat[,5:ncol(dis)], 0), big.mark=",", scientific=FALSE, trim=TRUE))
        rownames(dis) <- rownames(mat)
        colnames(dis) <- colnames(mat)
        print(noquote(dis[, , drop = FALSE]))
    } else {
        dis <- vector(length = length(mat))
        dis[1:4] <- sprintf("%s", format(round(mat[1:4], 5), big.mark=",", scientific=FALSE, trim=TRUE))
        dis[5:length(mat)] <- sprintf("%s", format(round(mat[5:length(mat)], 0), big.mark=",", scientific=FALSE, trim=TRUE))
        names(dis) <- names(mat)
        print(noquote(dis[ , drop = FALSE]))
    }
}