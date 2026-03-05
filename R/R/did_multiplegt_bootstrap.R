#' Internal function of did_multiplegt_dyn - bootstrap se
#' @param df df
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
#' @param bootstrap bootstrap
#' @param bootstrap_seed bootstrap_seed
#' @param base base
#' @returns A list of the final results updated with the bootstrap standard errors
#' @noRd
did_multiplegt_bootstrap <- function(
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
  only_never_switchers,
  effects_equal,
  save_results,
  normalized,
  predict_het,
  trends_lin,
  less_conservative_se,
  continuous,
  bootstrap,
  bootstrap_seed = NULL,
  base
){

    ## Set seed if provided
    base_seed <- if (!is.null(bootstrap_seed)) bootstrap_seed else as.integer(Sys.time())
    set.seed(base_seed)

    bresults_effects <- NULL
    bresults_ATE <- NULL
    bresults_placebo <- NULL

    n_effects <- nrow(base$Effects)
    bresults_effects <- matrix(NA, nrow = bootstrap, ncol = n_effects)
    if (isFALSE(trends_lin)) {
        bresults_ATE <- matrix(NA, nrow = bootstrap, ncol = 1)
    }
    if (placebo > 0) {
        n_placebo <- nrow(base$Placebos)
        bresults_placebo <- matrix(NA, nrow = bootstrap, ncol = n_placebo)
    }

    bs_group <- if (!is.null(cluster)) cluster else group

    # Pre-compute group structure using C++ for fast repeated sampling
    group_vec <- as.integer(as.factor(df[[bs_group]]))
    group_info <- bootstrap_prepare_groups_cpp(group_vec)

    group_col <- group
    time_col <- time

    for (j in 1:bootstrap) {
        # Fast C++ bootstrap sampling (uses R's RNG, seed already set above)
        sampled_idx <- bootstrap_sample_indices_cpp(group_info)

        # Convert to 1-based R indexing
        sampled_idx <- sampled_idx + 1L
        df_boot <- df[sampled_idx, ]

        # Sort the data frame
        df_boot <- as.data.frame(df_boot)
        df_boot <- df_boot[order(df_boot[[group_col]], df_boot[[time_col]]), ]

        suppressMessages({
        df_est <- did_multiplegt_main(df = df_boot, outcome = outcome, group = group, time = time, treatment = treatment, effects = effects, placebo = placebo, ci_level = ci_level, switchers = switchers, trends_nonparam = trends_nonparam, weight = weight, controls = controls, dont_drop_larger_lower = dont_drop_larger_lower, drop_if_d_miss_before_first_switch = drop_if_d_miss_before_first_switch, cluster = cluster, same_switchers = same_switchers, same_switchers_pl = same_switchers_pl, only_never_switchers = only_never_switchers, effects_equal = effects_equal, save_results = save_results, normalized = normalized, predict_het = predict_het, trends_lin = trends_lin, less_conservative_se = less_conservative_se, continuous = continuous)})

        res <- df_est$did_multiplegt_dyn

        # Vectorized result extraction for effects
        n_res_effects <- nrow(res$Effects)
        if (n_res_effects > 0) {
            n_copy <- min(ncol(bresults_effects), n_res_effects)
            bresults_effects[j, 1:n_copy] <- res$Effects[1:n_copy, 1]
        }

        # ATE extraction
        if (!is.null(bresults_ATE) && !is.null(res$ATE[1])) {
            bresults_ATE[j, 1] <- res$ATE[1]
        }

        # Vectorized result extraction for placebos
        if (!is.null(bresults_placebo) && !is.null(res$Placebos)) {
            n_res_placebo <- nrow(res$Placebos)
            if (n_res_placebo > 0) {
                n_copy <- min(ncol(bresults_placebo), n_res_placebo)
                bresults_placebo[j, 1:n_copy] <- res$Placebos[1:n_copy, 1]
            }
        }

        rm(res, df_est, df_boot)
        progressBar(j, bootstrap)
    }

    ci_level <- ci_level / 100

    # Fast C++ SD computation for effects
    effect_sds <- bootstrap_compute_sd_cpp(bresults_effects)
    n_eff <- nrow(base$Effects)
    base$Effects[1:n_eff, 2] <- effect_sds[1:n_eff]

    # Fast C++ CI computation for effects
    ci_effects <- bootstrap_compute_ci_cpp(base$Effects[1:n_eff, 1], effect_sds[1:n_eff], ci_level)
    base$Effects[1:n_eff, 3] <- ci_effects$lb
    base$Effects[1:n_eff, 4] <- ci_effects$ub

    if (nrow(base$Effects) == 1) {
        class(base$Effects) <- "numeric"
    }

    # ATE SE computation using C++
    if (!is.null(bresults_ATE) && !is.null(base$ATE[1])) {
        ate_sd <- bootstrap_compute_sd_cpp(bresults_ATE)
        base$ATE[2] <- ate_sd[1]
        ci_ate <- bootstrap_compute_ci_cpp(base$ATE[1], ate_sd[1], ci_level)
        base$ATE[3] <- ci_ate$lb[1]
        base$ATE[4] <- ci_ate$ub[1]
    }

    # Fast C++ SD computation for placebos
    if (!is.null(bresults_placebo)) {
        placebo_sds <- bootstrap_compute_sd_cpp(bresults_placebo)
        n_pl <- nrow(base$Placebos)
        base$Placebos[1:n_pl, 2] <- placebo_sds[1:n_pl]

        # Fast C++ CI computation for placebos
        ci_placebo <- bootstrap_compute_ci_cpp(base$Placebos[1:n_pl, 1], placebo_sds[1:n_pl], ci_level)
        base$Placebos[1:n_pl, 3] <- ci_placebo$lb
        base$Placebos[1:n_pl, 4] <- ci_placebo$ub

        if (nrow(base$Placebos) == 1) {
            class(base$Placebos) <- "numeric"
        }
    }
    return(base)
}

#' Internal function to convert lists to vectors (optimized)
#' @param lis A list
#' @returns A vector
#' @noRd
list_to_vec <- function(lis) {
    unlist(lis, use.names = FALSE)
}
