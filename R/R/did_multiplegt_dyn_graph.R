#' Function for event study plot. the did_multiplegt_dyn command always generates a ggplot object that can be printed right after the end of the routine (graph_off = FALSE) or called afterwards from the global environment after assigning the did_multiplegt_dyn output to a variable.
#' @param data data
#' @param args args
#' @returns A ggplot object.
#' @noRd
did_multiplegt_dyn_graph <- function(data, args = list()) {  
  grmat <- rbind(cbind(data$Effects, 1:nrow(data$Effects)),cbind(data$ATE, 0))
  if (!is.null(data$Placebos)) {
    grmat <- rbind(grmat, cbind(data$Placebos, (-1) * 1:nrow(data$Placebos)))
  }
  colnames(grmat)[ncol(grmat)] <- "Time"
  grmat[nrow(data$Effects) + 1L, c(1L, 3L, 4L)] <- 0
  grmat <- data.frame(grmat[, c(1L, 3L, 4L, 9L)])
  did_multiplegt_dyn_plot <- ggplot2::ggplot(grmat, ggplot2::aes(x = .data$Time, y = .data$Estimate, group = 1L)) + 
    ggplot2::geom_line(colour = "blue") + 
    ggplot2::geom_errorbar(data = ~.x[.x$Estimate != 0, , drop = FALSE], ggplot2::aes(ymin = .data$LB.CI, ymax = .data$UB.CI), position = ggplot2::position_dodge(0.05), width = 0.2, colour = "red")  + 
    ggplot2::geom_point(colour = "blue") + 
    ggplot2::ggtitle("DID, from last period before treatment changes (t=0) to t") + 
    ggplot2::xlab("Relative time to last period before treatment changes (t=0)") +
    ggplot2::theme_minimal() +
    ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5), panel.grid.minor = ggplot2::element_blank())

  for (layer in args) {
    did_multiplegt_dyn_plot <- did_multiplegt_dyn_plot + layer
  }

  return(did_multiplegt_dyn_plot)
}


#' Internal function of did_multiplegt_dyn to overlay plots
#' @param obj A did_multiplegt_dyn object
#' @returns A ggplot object (for by) or a grid grob (for by_path).
#' @noRd
combine_plot <- function(obj) {
  if (!is.null(obj$args[["by"]])) {
    color_set <- get_colors(length(obj$by_levels))
    base_plot <- obj$by_level_1$plot
    plot <- ggplot2::ggplot(data = base_plot$data, ggplot2::aes(x = base_plot$data$Time, y = base_plot$data$Estimate))  +
              ggplot2::geom_point(colour = color_set[1L]) + ggplot2::geom_line(ggplot2::aes(colour = paste0(obj$args$by," = ", obj$by_levels[1]))) +
              ggplot2::geom_errorbar(data = base_plot$data, ggplot2::aes(ymin = base_plot$data$LB.CI, ymax = base_plot$data$UB.CI), position = ggplot2::position_dodge(0.05), width = 0.2, colour = color_set[1L])
    if (length(obj$by_levels) > 1L) {
      for (j in 2:length(obj$by_levels)) {
        add_plot <-  obj[[paste0("by_level_",j)]][["plot"]]
        plot <- plot + 
            ggplot2::geom_point(data = add_plot$data, ggplot2::aes(x = .data$Time, y = .data$Estimate), colour = color_set[j]) +
            ggplot2::geom_errorbar(data = add_plot$data, ggplot2::aes(ymin = .data$LB.CI, ymax = .data$UB.CI), position = ggplot2::position_dodge(0.05), width = 0.2, colour =  color_set[j]) +
            ggplot2::geom_line(data =  add_plot$data, ggplot2::aes(x = .data$Time, y = .data$Estimate, 
                colour = paste0(obj$args$by, " = ", obj$by_levels[j])))
      }
    }
    plot <- plot + ggplot2::ylab("Estimate") + ggplot2::ggtitle("DID, from last period before treatment changes (t=0) to t") + 
      ggplot2::xlab("Relative time to last period before treatment changes (t=0)") +
      ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5), legend.position = "bottom") +
      ggplot2::scale_colour_manual("", breaks = paste0(obj$args$by, " = ", obj$by_levels), values = color_set)
  } else if (!is.null(obj$args[["by_path"]])) {
    if (length(obj$by_levels) > 100L) {
      message("The command allows a maximum of 100 graphs to be combined in a 10 x 10 window. The resulting graph will be restricted to the first 100 treatment paths.")
    }
    n_plots <- min(length(obj$by_levels), 100L)
    sides <- ceiling(sqrt(n_plots))
    plots <- lapply(seq_len(n_plots), function(j) {
      obj[[paste0("by_level_", j)]]$plot +
        ggplot2::ggtitle(sprintf("Treatment path (%s); %.0f switchers",
          obj$by_levels[j],
          obj[[paste0("by_level_", j)]]$results$Effects[1L, 6L])) +
        ggplot2::xlab(" ")
    })
    # Pad with nullGrobs to fill the grid
    while (length(plots) < sides * sides) {
      plots <- c(plots, list(grid::nullGrob()))
    }
    grobs <- lapply(plots, function(p) {
      if (inherits(p, "ggplot")) ggplot2::ggplotGrob(p) else p
    })
    title_grob <- grid::textGrob(
      "DID from last period before treatment changes (t = 0) to t",
      gp = grid::gpar(fontsize = 14), vjust = 1
    )
    body <- do.call(gridExtra::arrangeGrob, c(grobs, ncol = sides))
    plot <- gridExtra::arrangeGrob(title_grob, body, nrow = 2L, heights = grid::unit(c(1, 12), "null"))
  }
  return(plot)
}

#' Internal function to retrieve plot colors
#' @param N Number of colors to retrieve
#' @returns A list of colors.
#' @noRd
get_colors <- function(N) {
  must_color <- c(552L, 26L, 81L, 68L, 450L, 640L, 24L, 498L) 
  # Indices of the following colors in ggplot's colors()
  # Red, blue, green, cyan, magenta, violet, black, orange
  other_color <- 1:657
  other_color <- other_color[!(other_color %in% must_color)]
  if (N > length(must_color)) {
    index_colors <- c(must_color, sample(other_color, N - length(must_color)))
  } else {
    index_colors <- must_color[1:N]
  }
  return(grDevices::colors()[index_colors])
}