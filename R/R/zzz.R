# Package startup and polars availability checking

# Check if polars is available
.polars_available <- function() {

  requireNamespace("polars", quietly = TRUE)
}
# Get polars namespace (only if available)
.get_polars <- function() {
  if (!.polars_available()) {
    stop(
      "The 'polars' package is required but not installed.\n",
      "Please install it from r-universe with:\n",
      "  install.packages('polars', repos = 'https://rpolars.r-universe.dev')\n",
      call. = FALSE
    )
  }
  getNamespace("polars")
}

# Safe wrapper to get pl object
.get_pl <- function() {
  polars_ns <- .get_polars()
  polars_ns$pl
}

# Safe wrapper for as_polars_df
.as_polars_df <- function(x) {
  polars_ns <- .get_polars()
  polars_ns$as_polars_df(x)
}

# Package startup message
.onAttach <- function(libname, pkgname) {
  if (.polars_available()) {
    packageStartupMessage(
      "DIDmultiplegtDYN: Using polars backend for optimized performance."
    )
  } else {
    packageStartupMessage(
      "DIDmultiplegtDYN: polars package not found.\n",
      "For this package to work, please install polars from r-universe:\n",
      "  install.packages('polars', repos = 'https://rpolars.r-universe.dev')"
    )
  }
}

# Check polars on load
.onLoad <- function(libname, pkgname) {
  # Set default option
  if (is.null(getOption("DID_USE_POLARS"))) {
    options(DID_USE_POLARS = TRUE)
  }
}
