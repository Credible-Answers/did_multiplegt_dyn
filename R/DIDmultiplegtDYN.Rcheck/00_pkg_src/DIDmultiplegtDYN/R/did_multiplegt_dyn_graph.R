#' Function for event study plot. the did_multiplegt_dyn command always generates a ggplot object that can be printed right after the end of the routine (graph_off = FALSE) or called afterwards from the global environment after assigning the did_multiplegt_dyn output to a variable.
#' @param data data
#' @param args args
#' @import ggplot2
#' @importFrom dplyr filter
#' @returns A ggplot object.
#' @noRd
did_multiplegt_dyn_graph <- function(data, args = list()) {  
  grmat <- rbind(cbind(data$Effects, 1:nrow(data$Effects)),cbind(data$ATE, 0))
  if (!is.null(data$Placebos)) {
    grmat <- rbind(grmat, cbind(data$Placebos, (-1) * 1:nrow(data$Placebos)))
  }
  colnames(grmat)[ncol(grmat)] <- "Time"
  grmat[nrow(data$Effects) + 1, c(1,3,4)] <- 0
  grmat <- data.frame(grmat[, c(1,3,4,9)])
  did_multiplegt_dyn_plot <- ggplot(grmat,aes(x = .data$Time, y = .data$Estimate, group = 1)) + 
    geom_line(colour = "blue") + 
    geom_errorbar(data = ~dplyr::filter(.x, grmat$Estimate != 0),aes(ymin = .data$LB.CI, ymax = .data$UB.CI), position=position_dodge(0.05), width = 0.2, colour = "red")  + 
    geom_point(colour = "blue") + 
    ggtitle("DID, from last period before treatment changes (t=0) to t") + 
    xlab("Relative time to last period before treatment changes (t=0)") +
    theme(plot.title = element_text(hjust = 0.5)) + theme_minimal_grid()

  for (layer in args) {
    did_multiplegt_dyn_plot <- did_multiplegt_dyn_plot + layer
  }

  return(did_multiplegt_dyn_plot)
}


#' Internal function of did_multiplegt_dyn to overlay plots
#' @param obj A did_multiplegt_dyn object
#' @import ggplot2
#' @import cowplot
#' @returns A ggplot object.
#' @noRd
combine_plot <- function(obj) {
  if (!is.null(obj$args[["by"]])) {
    color_set <- get_colors(length(obj$by_levels))
    base_plot <- obj$by_level_1$plot
    plot <-  ggplot(data = base_plot$data, aes(x = base_plot$data$Time, y = base_plot$data$Estimate))  +
              geom_point(colour = color_set[1]) + geom_line(aes(colour = paste0(obj$args$by," = ", obj$by_levels[1]))) +
              geom_errorbar(data = base_plot$data, aes(ymin = base_plot$data$LB.CI, ymax = base_plot$data$UB.CI), position=position_dodge(0.05), width = 0.2, colour = color_set[1])
    if (length(obj$by_levels) > 1) {
      for (j in 2:length(obj$by_levels)) {
        add_plot <-  obj[[paste0("by_level_",j)]][["plot"]]
        plot <- plot + 
            geom_point(data = add_plot$data, aes_(x =  add_plot$data$Time, y = add_plot$data$Estimate), colour = color_set[j]) +
            geom_errorbar(data = add_plot$data, aes_(ymin = add_plot$data$LB.CI, ymax = add_plot$data$UB.CI), position=position_dodge(0.05), width = 0.2, colour =  color_set[j]) +
            geom_line(data =  add_plot$data, aes_(x =  add_plot$data$Time, y =  add_plot$data$Estimate, 
                colour = paste0(obj$args$by, " = ", obj$by_levels[j])))
      }
    }
    plot <- plot + ylab("Estimate") + ggtitle("DID, from last period before treatment changes (t=0) to t") + 
      xlab("Relative time to last period before treatment changes (t=0)") +
      theme(plot.title = element_text(hjust = 0.5), legend.position = "bottom") +
      scale_colour_manual("", breaks = paste0(obj$args$by, " = ", obj$by_levels), values = color_set)
  } else if (!is.null(obj$args[["by_path"]])) {
    if (length(obj$by_levels) > 100) {
      message("The command allows a maximum of 100 graphs to be combined in a 10 x 10 window. The resulting graph will be restricted to the first 100 treatment paths.")
    }
    sides <- ceiling(sqrt(length(obj$by_levels))); len <- 1/sides;
    plot <- ggdraw()
    for (j in 1:length(obj$by_levels)) {
      plot <- plot + draw_plot(obj[[paste0("by_level_",j)]]$plot + ggtitle(sprintf("Treatment path (%s); %.0f switchers", obj$by_levels[j], obj[[paste0("by_level_",j)]]$results$Effects[1,6])) + xlab(" "), width = len, height = len, y = (sides - ceiling(j/sides)) * len, x = ((j-1) %% sides) * len)
    }
    plot <- plot + ggtitle("DID from last period before treatment changes (t = 0) to t") + theme(plot.title = element_text(hjust = 0.5))
  }
  return(plot)
}

#' Internal function to retrieve plot colors
#' @param N Number of colors to retrieve
#' @import ggplot2
#' @importFrom grDevices colors
#' @returns A list of colors.
#' @noRd
get_colors <- function(N) {
  must_color <- c(552, 26, 81, 68, 450, 640, 24, 498) 
  # Indices of the following colors in ggplot's colors()
  # Red, blue, green, cyan, magenta, violet, black, orange
  other_color <- 1:657
  other_color <- other_color[!(other_color %in% must_color)]
  if (N > length(must_color)) {
    index_colors <- c(must_color, sample(other_color, N - length(must_color)))
  } else {
    index_colors <- must_color[1:N]
  }
  return(colors()[index_colors])
}