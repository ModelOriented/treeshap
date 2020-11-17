#' Prints model_unified objects
#'
#' @param x a model_unified object
#' @param ... other arguments
#'
#' @export
#'
#'

print.model_unified <- function(x, ...){
  print(x$model)
  return(invisible(NULL))
}
