#' Internal function of did_multiplegt_dyn by treatment path
#' @param df df
#' @param by_path by_path
#' @param outcome outcome
#' @param group group
#' @param time time
#' @param treatment treatment
#' @param effects effects
#' @param placebo placebo
#' @param ci_level ci_level
#' @param switchers switchers
#' @param trends_nonparam trends_nonparam
#' @param weight weight
#' @param controls controls
#' @param dont_drop_larger_lower dont_drop_larger_lower
#' @param drop_if_d_miss_before_first_switch drop_if_d_miss_before_first_switch
#' @param cluster cluster
#' @param same_switchers same_switchers
#' @param same_switchers_pl same_switchers_pl
#' @param only_never_switchers only_never_switchers
#' @param effects_equal effects_equal
#' @param save_results save_results
#' @param normalized normalized
#' @param predict_het predict_het
#' @param trends_lin trends_lin
#' @param less_conservative_se less_conservative_se
#' @param continuous continuous
#' @returns A dataframe with the by_path classifier
#' @noRd

did_multiplegt_by_path <- function(
    df,
    outcome, 
    group, 
    time, 
    treatment, 
    effects, 
    placebo, 
    ci_level, 
    switchers, 
    trends_nonparam, 
    weight, 
    controls, 
    dont_drop_larger_lower, 
    drop_if_d_miss_before_first_switch, 
    cluster, 
    same_switchers, 
    same_switchers_pl,
    effects_equal, 
    only_never_switchers,
    save_results, 
    normalized,
    predict_het,
    trends_lin,
    less_conservative_se,
    continuous,
    by_path
    ) {

    data <- did_multiplegt_main(df = df, outcome = outcome, group =  group, time =  time, treatment = treatment, effects = effects, placebo = placebo, ci_level = ci_level,switchers = switchers, trends_nonparam = trends_nonparam, weight = weight, controls = controls, dont_drop_larger_lower = dont_drop_larger_lower, drop_if_d_miss_before_first_switch = drop_if_d_miss_before_first_switch, cluster = cluster, same_switchers = same_switchers, same_switchers_pl = same_switchers_pl, only_never_switchers = only_never_switchers, effects_equal = effects_equal, save_results = save_results, normalized = normalized, predict_het = predict_het, trends_lin = trends_lin, less_conservative_se = less_conservative_se, continuous = continuous, data_only = TRUE)

    design_base <- did_multiplegt_dyn_design(data = data, design_opt = list(1, "console"), weight = weight, by = NULL, by_index = "_no_by", file = NULL)

    path_index <- data$df[, .(group, time, time_XX, treatment_XX, F_g_XX)]
    l_XX <- data$l_XX
    T_max_XX <- data$T_max_XX
    path_index <- path_index[!is.na(F_g_XX) & F_g_XX != T_max_XX + 1L]

    data <- NULL
    if (by_path == -1) {
        by_path <- nrow(design_base$design_mat)
    }
    design_set <- matrix(design_base$design_mat[1:min(by_path, nrow(design_base$design_mat)), ], ncol = ncol(design_base$design_mat), nrow = min(by_path, nrow(design_base$design_mat)))
    if (by_path > nrow(design_base$design_mat)) {
        message(sprintf("You requested %.0f treatment paths, but there are only %.0f paths in your data. The program will continue with the latter number of treatment paths.", by_path, nrow(design_base$design_mat)))
    }
    path <- design_set[,3L]
    for (j in 1:l_XX) {
        path <- paste0(path,",",design_set[,3L+j])
    }
    
    cols_DFg <- paste0("D_Fg", 0:l_XX)
    lapply(0:l_XX, function(i) {
        col <- cols_DFg[i + 1L]
        path_index[, (col) := data.table::fifelse(time_XX == F_g_XX - 1L + i, treatment_XX, NA_real_)]
    })
    path_index[, treatment_XX := NULL]
    lapply(0:l_XX, function(j) {
        source_col <- cols_DFg[j + 1L]
        target_col <- paste0("D_fg", j)
        path_index[, (target_col) := mean(get(source_col), na.rm = TRUE), by = "group"]
    })

    path_index[, paste0("D_Fg", 0:l_XX) := NULL]

    path_index[, path := as.character(D_fg0)]
    path_index[, D_fg0 := NULL]
    for (j in 1:l_XX) {
        col <- paste0("D_fg", j)
        path_index[, path := data.table::fifelse(!is.na(get(col)), paste0(path, ",", get(col)), path)]
        path_index[, (col) := NULL]
    }
    path_index[, yet_to_switch_XX := as.numeric(time_XX < F_g_XX)]
    path_index[, c("time_XX", "F_g_XX") := NULL]
    path_index[, baseline_XX := substr(path, 1, 1)]

    data.table::setnames(path_index, c("group", "time", "path"), c(group, time, "path_XX"))
    df <- merge(df, path_index, by = c(group, time))
    data.table::setorderv(df, c(group, time))
    data <- list(df = df, path = path)
    return(data)
}