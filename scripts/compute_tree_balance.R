#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: compute_tree_balance.R <input_csv> <output_csv>", call. = FALSE)
}

input_csv <- args[[1]]
output_csv <- args[[2]]

suppressPackageStartupMessages({
  library(ape)
  library(treebalance)
})

indices <- list(
  list(name = "sackin", fn = function(tree) sackinI(tree)),
  list(name = "average_leaf_depth", fn = function(tree) avgLeafDepI(tree)),
  list(name = "variance_leaf_depth", fn = function(tree) varLeafDepI(tree)),
  list(name = "total_cophenetic", fn = function(tree) totCophI(tree)),
  list(name = "area_per_pair", fn = function(tree) areaPerPairI(tree)),
  list(name = "b1", fn = function(tree) B1I(tree)),
  list(name = "b2", fn = function(tree) B2I(tree)),
  list(name = "cherry", fn = function(tree) cherryI(tree)),
  list(name = "colless_like_exp_mdm", fn = function(tree) collesslikeI(tree, f.size = "exp", dissim = "mdm")),
  list(name = "maximum_depth", fn = function(tree) maxDepth(tree)),
  list(name = "maximum_width", fn = function(tree) maxWidth(tree)),
  list(name = "maximum_width_over_depth", fn = function(tree) mWovermD(tree)),
  list(name = "modified_maximum_width_difference", fn = function(tree) maxDelW(tree)),
  list(name = "rooted_quartet", fn = function(tree) rQuartetI(tree)),
  list(name = "s_shape", fn = function(tree) sShapeI(tree)),
  list(name = "total_internal_path_length", fn = function(tree) totIntPathLen(tree)),
  list(name = "total_path_length", fn = function(tree) totPathLen(tree)),
  list(name = "average_vertex_depth", fn = function(tree) avgVertDep(tree))
)

safe_tree <- function(newick) {
  tryCatch(
    {
      if (is.na(newick) || !nzchar(newick)) {
        stop("Empty Newick string", call. = FALSE)
      }
      suppressWarnings(ape::read.tree(text = newick))
    },
    error = function(error) {
      structure(list(message = conditionMessage(error)), class = "tree_balance_error")
    }
  )
}

safe_index <- function(tree, index) {
  messages <- character()
  value <- tryCatch(
    withCallingHandlers(
      suppressMessages(index$fn(tree)),
      warning = function(warning) {
        messages <<- c(messages, conditionMessage(warning))
        invokeRestart("muffleWarning")
      },
      message = function(message) {
        messages <<- c(messages, conditionMessage(message))
        invokeRestart("muffleMessage")
      }
    ),
    error = function(error) {
      structure(list(message = conditionMessage(error)), class = "tree_balance_error")
    }
  )

  if (inherits(value, "tree_balance_error")) {
    return(list(value = NA_real_, status = "error", error_message = value$message))
  }

  if (!is.numeric(value) || length(value) == 0) {
    return(list(value = NA_real_, status = "error", error_message = "Index did not return a numeric value"))
  }

  numeric_value <- as.numeric(value[[1]])
  if (!is.finite(numeric_value)) {
    message_text <- if (length(messages) > 0) paste(unique(messages), collapse = " | ") else "Index returned a non-finite value"
    return(list(value = NA_real_, status = "missing", error_message = message_text))
  }

  list(
    value = numeric_value,
    status = "ok",
    error_message = if (length(messages) > 0) paste(unique(messages), collapse = " | ") else NA_character_
  )
}

input <- read.csv(input_csv, stringsAsFactors = FALSE, check.names = FALSE)
output <- data.frame(
  graph_id = integer(),
  include_punctuation = integer(),
  index_name = character(),
  value = numeric(),
  status = character(),
  error_message = character(),
  stringsAsFactors = FALSE
)

for (row_index in seq_len(nrow(input))) {
  graph_id <- input$graph_id[[row_index]]
  include_punctuation <- input$include_punctuation[[row_index]]
  tree <- safe_tree(input$newick[[row_index]])

  for (index in indices) {
    if (inherits(tree, "tree_balance_error")) {
      result <- list(value = NA_real_, status = "error", error_message = tree$message)
    } else {
      result <- safe_index(tree, index)
    }

    output[nrow(output) + 1, ] <- list(
      graph_id = graph_id,
      include_punctuation = include_punctuation,
      index_name = index$name,
      value = result$value,
      status = result$status,
      error_message = result$error_message
    )
  }
}

write.csv(output, output_csv, row.names = FALSE, na = "")
