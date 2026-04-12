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
        bresults_ATE <- matrix(NA, nrow = bootstrap, ncol = 1L)
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

    n_daemons <- .mirai_n_daemons()

    if (n_daemons > 0L) {
        # ---- Parallel bootstrap via mirai ----

        # Pre-generate all bootstrap sample indices (fast, preserves RNG reproducibility)
        boot_indices <- vector("list", bootstrap)
        for (j in seq_len(bootstrap)) {
            boot_indices[[j]] <- bootstrap_sample_indices_cpp(group_info) + 1L
        }

        n_workers <- min(n_daemons, bootstrap)
        chunks <- split(seq_len(bootstrap),
            rep(seq_len(n_workers), length.out = bootstrap))
        message(sprintf("  Using %d mirai workers...", n_workers))

        main_args <- list(
            outcome = outcome, group = group, time = time,
            treatment = treatment, effects = effects, placebo = placebo,
            ci_level = ci_level, switchers = switchers,
            trends_nonparam = trends_nonparam, weight = weight,
            controls = controls,
            dont_drop_larger_lower = dont_drop_larger_lower,
            drop_if_d_miss_before_first_switch = drop_if_d_miss_before_first_switch,
            cluster = cluster, same_switchers = same_switchers,
            same_switchers_pl = same_switchers_pl,
            only_never_switchers = only_never_switchers,
            effects_equal = effects_equal, save_results = save_results,
            normalized = normalized, predict_het = predict_het,
            trends_lin = trends_lin, less_conservative_se = less_conservative_se,
            continuous = continuous,
            point_estimates_only = TRUE
        )

        tasks <- vector("list", length(chunks))
        for (k in seq_along(chunks)) {
            chunk_idx <- boot_indices[chunks[[k]]]
            tasks[[k]] <- mirai::mirai(
                {
                    results <- vector("list", length(chunk_idx))
                    for (i in seq_along(chunk_idx)) {
                        df_boot <- df[chunk_idx[[i]], ]
                        data.table::setorderv(df_boot,
                            c(main_args[["group"]], main_args[["time"]]))
                        suppressMessages({
                            df_est <- do.call(
                                DIDmultiplegtDYN:::did_multiplegt_main,
                                c(list(df = df_boot), main_args)
                            )
                        })
                        res <- df_est$did_multiplegt_dyn
                        n_eff <- nrow(res$Effects)
                        results[[i]] <- list(
                            effects = if (n_eff > 0L)
                                res$Effects[1:n_eff, 1L] else numeric(0),
                            ate = res$ATE[1L],
                            placebos = if (!is.null(res$Placebos) &&
                                nrow(res$Placebos) > 0L)
                                res$Placebos[, 1L] else NULL
                        )
                    }
                    results
                },
                chunk_idx = chunk_idx,
                df = df,
                main_args = main_args,
                .compute = .mirai_compute
            )
        }

        # Collect and assemble results
        all_results <- list()
        for (k in seq_along(tasks)) {
            chunk_res <- tasks[[k]]$data
            if (!is.list(chunk_res)) {
                msg <- tryCatch(conditionMessage(chunk_res),
                    error = function(e) as.character(chunk_res))
                stop("Bootstrap worker failed: ", paste(msg, collapse = " "))
            }
            all_results <- c(all_results, chunk_res)
        }

        for (j in seq_along(all_results)) {
            r <- all_results[[j]]
            n_copy <- min(ncol(bresults_effects), length(r$effects))
            if (n_copy > 0L) bresults_effects[j, 1:n_copy] <- r$effects[1:n_copy]
            if (!is.null(bresults_ATE) && !is.null(r$ate)) {
                bresults_ATE[j, 1L] <- r$ate
            }
            if (!is.null(bresults_placebo) && !is.null(r$placebos)) {
                n_copy_pl <- min(ncol(bresults_placebo), length(r$placebos))
                if (n_copy_pl > 0L) {
                    bresults_placebo[j, 1:n_copy_pl] <- r$placebos[1:n_copy_pl]
                }
            }
        }

    } else {
        # ---- Sequential bootstrap (original) ----
        for (j in seq_len(bootstrap)) {
            # Fast C++ bootstrap sampling (uses R's RNG, seed already set above)
            sampled_idx <- bootstrap_sample_indices_cpp(group_info)

            # Convert to 1-based R indexing
            sampled_idx <- sampled_idx + 1L
            df_boot <- df[sampled_idx, ]

            # Sort the bootstrap sample
            data.table::setorderv(df_boot, c(group_col, time_col))

            suppressMessages({
            df_est <- did_multiplegt_main(df = df_boot, outcome = outcome, group = group, time = time, treatment = treatment, effects = effects, placebo = placebo, ci_level = ci_level, switchers = switchers, trends_nonparam = trends_nonparam, weight = weight, controls = controls, dont_drop_larger_lower = dont_drop_larger_lower, drop_if_d_miss_before_first_switch = drop_if_d_miss_before_first_switch, cluster = cluster, same_switchers = same_switchers, same_switchers_pl = same_switchers_pl, only_never_switchers = only_never_switchers, effects_equal = effects_equal, save_results = save_results, normalized = normalized, predict_het = predict_het, trends_lin = trends_lin, less_conservative_se = less_conservative_se, continuous = continuous, point_estimates_only = TRUE)})

            res <- df_est$did_multiplegt_dyn

            # Vectorized result extraction for effects
            n_res_effects <- nrow(res$Effects)
            if (n_res_effects > 0L) {
                n_copy <- min(ncol(bresults_effects), n_res_effects)
                bresults_effects[j, 1:n_copy] <- res$Effects[1:n_copy, 1L]
            }

            # ATE extraction
            if (!is.null(bresults_ATE) && !is.null(res$ATE[1L])) {
                bresults_ATE[j, 1L] <- res$ATE[1L]
            }

            # Vectorized result extraction for placebos
            if (!is.null(bresults_placebo) && !is.null(res$Placebos)) {
                n_res_placebo <- nrow(res$Placebos)
                if (n_res_placebo > 0L) {
                    n_copy <- min(ncol(bresults_placebo), n_res_placebo)
                    bresults_placebo[j, 1:n_copy] <- res$Placebos[1:n_copy, 1L]
                }
            }

            rm(res, df_est, df_boot)
            progressBar(j, bootstrap)
        }
    }

    ci_level <- ci_level / 100

    # SD and CI computation for effects
    effect_sds <- apply(bresults_effects, 2L, stats::sd, na.rm = TRUE)
    n_eff <- nrow(base$Effects)
    base$Effects[1:n_eff, 2L] <- effect_sds[1:n_eff]

    z <- stats::qnorm(1 - (1 - ci_level) / 2)
    base$Effects[1:n_eff, 3L] <- base$Effects[1:n_eff, 1L] - z * effect_sds[1:n_eff]
    base$Effects[1:n_eff, 4L] <- base$Effects[1:n_eff, 1L] + z * effect_sds[1:n_eff]

    if (nrow(base$Effects) == 1L) {
        class(base$Effects) <- "numeric"
    }

    # ATE SE computation
    if (!is.null(bresults_ATE) && !is.null(base$ATE[1L])) {
        ate_sd <- stats::sd(bresults_ATE[, 1L], na.rm = TRUE)
        base$ATE[2L] <- ate_sd
        base$ATE[3L] <- base$ATE[1L] - z * ate_sd
        base$ATE[4L] <- base$ATE[1L] + z * ate_sd
    }

    # SD and CI computation for placebos
    if (!is.null(bresults_placebo)) {
        placebo_sds <- apply(bresults_placebo, 2L, stats::sd, na.rm = TRUE)
        n_pl <- nrow(base$Placebos)
        base$Placebos[1:n_pl, 2] <- placebo_sds[1:n_pl]

        base$Placebos[1:n_pl, 3L] <- base$Placebos[1:n_pl, 1L] - z * placebo_sds[1:n_pl]
        base$Placebos[1:n_pl, 4L] <- base$Placebos[1:n_pl, 1L] + z * placebo_sds[1:n_pl]

        if (nrow(base$Placebos) == 1L) {
            class(base$Placebos) <- "numeric"
        }
    }
    return(base)
}
