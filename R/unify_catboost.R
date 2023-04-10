#' Unify Catboost model
#'
#' Convert your Catboost model into a standarised representation.
#' The returned representation is easy to be interpreted by the user and ready to be used as an argument in \code{treeshap()} function.
#'
#' @param catboost_model An object of \code{catboost.Model} class. At the moment, models built on data with categorical features
#' are not supported - please encode them before training.
#' @param data Reference dataset. A \code{data.frame} or \code{matrix} with the same columns as in the training set of the model. Usually dataset used to train model. Note that the same order of columns is crucial for unifier to work.
#' @param recalculate logical indicating if covers should be recalculated according to the dataset given in data. Keep it \code{FALSE} if training data is used.
#'
#' @return a unified model representation - a \code{\link{model_unified.object}} object
#'
#' @export
#'
#' @seealso
#' \code{\link{lightgbm.unify}} for \code{\link[lightgbm:lightgbm]{LightGBM models}}
#'
#' \code{\link{gbm.unify}} for \code{\link[gbm:gbm]{GBM models}}
#'
#' \code{\link{xgboost.unify}} for \code{\link[xgboost:xgboost]{XGBoost models}}
#'
#' \code{\link{ranger.unify}} for \code{\link[ranger:ranger]{ranger models}}
#'
#' \code{\link{randomForest.unify}} for \code{\link[randomForest:randomForest]{randomForest models}}
#'
#' @examples
#' if(requireNamespace("catboost")){
#' library(catboost)
#' data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
#' data <- as.data.frame(lapply(data, as.numeric))
#' label <- fifa20$target
#' dt.pool <- catboost::catboost.load_pool(data = data, label = label)
#' cat_model <- catboost::catboost.train(
#'   dt.pool,
#'   params = list(loss_function = 'RMSE',
#'                 iterations = 100,
#'                 logging_level = 'Silent'))
#' um <- catboost.unify(cat_model, data)
#' shaps <- treeshap(um, data[1:2, ])
#' plot_contribution(shaps, obs = 1)
#' }
catboost.unify <- function(catboost_model, data, recalculate = FALSE) {
  if (!inherits(catboost_model,"catboost.Model")) {
    stop('Object catboost_model is not of type "catboost.Model"')
  }

  if (!any(c("data.frame", "matrix") %in% class(data))) {
    stop("Argument data has to be data.frame or matrix.")
  }

  if (!requireNamespace("catboost", quietly = TRUE)) {
    stop("Package \"catboost\" needed for this function to work. Please install it.",
         call. = FALSE)
  }

  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package \"jsonlite\" needed for this function to work. Please install it.",
         call. = FALSE)
  }



  path_to_save <- tempfile("catboost_model", fileext = ".json")
  catboost::catboost.save_model(catboost_model, path_to_save, 'json')
  json_data <- jsonlite::read_json(path_to_save)


  if (!is.null(json_data$features_info$categorical_features)) {
    stop('catboost.unify() function currently does not support models using categorical features.')
  }

  single_trees <- lapply(seq_along(json_data$oblivious_trees),
                         function(i) one_tree_transform(json_data$oblivious_trees[[i]], (i - 1)))
  united <- data.table::rbindlist(single_trees)

  stopifnot(is.numeric(united$float_feature_index)) # to delete in the future

  #stopifnot(all(sapply(json_data$features_info$float_features, function(x) x$feature_index) == 0:(length(json_data$features_info$float_features) - 1))) #assuming features are ordered

  # How are missing values treated?:
  for_missing <- sapply(json_data$features_info$float_features,
                        function(x) x$nan_value_treatment)[united$float_feature_index + 1]
  united$Missing <- for_missing
  united[!is.na(Missing) & Missing == 'AsIs', Missing := NA]
  united[!is.na(Missing) & Missing == 'AsFalse', Missing := Yes]
  united[!is.na(Missing) & Missing == 'AsTrue', Missing := No]
  united[, Missing := as.integer(Missing)]

  feature_names <- attr(catboost_model$feature_importances, "dimnames")[[1]]
  stopifnot(all(feature_names == colnames(data))) # this line can be deleted if we are sure feature names from feature_importances is correct
  united$float_feature_index <- feature_names[united$float_feature_index + 1]

  colnames(united) <- c('Split', 'Feature', 'Prediction', 'Cover', 'Yes', 'No', 'Node', 'Tree', 'Missing')

  united$Decision.type <- factor(x = rep("<=", times = nrow(united)), levels = c("<=", "<"))
  united$Decision.type[is.na(united$Feature)] <- NA

  ID <- paste0(united$Node, "-", united$Tree)
  united$Yes <- match(paste0(united$Yes, "-", united$Tree), ID)
  united$No <- match(paste0(united$No, "-", united$Tree), ID)
  united$Missing <- match(paste0(united$Missing, "-", united$Tree), ID)

  united <- united[, c('Tree', 'Node', 'Feature', 'Decision.type', 'Split', 'Yes', 'No', 'Missing', 'Prediction', 'Cover')]

  # for catboost the model prediction results are calculated as [sum(leaf_values * scale + bias)]
  # (https://catboost.ai/docs/concepts/python-reference_catboostregressor_set_scale_and_bias.html)
  # treeSHAP assumes the prediction is sum of leaf values
  # so here we adjust it
  scale <- json_data$scale_and_bias[[1]]
  bias <- json_data$scale_and_bias[[2]]
  ntrees <- sum(united$Node == 0)
  united[is.na(united$Feature), ]$Prediction <- united[is.na(united$Feature), ]$Prediction * scale + bias[[1]]/ ntrees

  ret <- list(model = as.data.frame(united), data = as.data.frame(data))
  class(ret) <- "model_unified"
  attr(ret, "missing_support") <- TRUE
  attr(ret, 'model') <- 'catboost'

  if (recalculate) {
    ret <- set_reference_dataset(ret, as.data.frame(data))
  }

  return(ret)
}


one_tree_transform <- function(oblivious_tree, tree_id) {
  #stopifnot(!is.null(oblivious_tree$splits))
  #stopifnot(!is.null(oblivious_tree$leaf_values))
  #stopifnot(!is.null(oblivious_tree$leaf_weights))
  frame <- data.table::rbindlist(lapply(oblivious_tree$splits, data.table::as.data.table))
  if (!all(frame$split_type == 'FloatFeature')) {
    stop('catboost.unify() function currently does not support models using categorical features. Please encode them before training.')
  }
  frame <- frame[nrow(frame):1, ]
  frame <- frame[, c('border', 'float_feature_index')]

  #repeat rows representing node at the kth level 2^(k-1) times:
  frame2 <- frame[rep(seq_len(nrow(frame)), times = 2**(seq_len(nrow(frame)) - 1)), ]

  #Add columns Score and Cover:
  frame2[, c('Score', 'Cover')] <- NA
  frame2$Yes <- as.integer(seq(1, nrow(frame2) * 2, 2))
  frame2$No <- as.integer(seq(2, nrow(frame2) * 2, 2))
  leaves_values <- unlist(oblivious_tree$leaf_values)
  leaves_weights <- as.numeric(unlist(oblivious_tree$leaf_weights))

  #Create the part of data frame for leaves
  leaves <- data.table::as.data.table(list(border = NA,
                                           float_feature_index = NA,
                                           Score = leaves_values,
                                           Cover = leaves_weights,
                                           Yes = NA,
                                           No = NA))
  tree_levels <- log2(length(leaves$Cover))
  #stopifnot(tree_levels == floor(tree_levels))

  internal_covers <- numeric()
  for(i in rev(seq(tree_levels))){
    internal_covers <- c(internal_covers, sapply(split(leaves$Cover, ceiling(seq_along(leaves$Cover) / (2**i))), sum))
  }
  names(internal_covers) <- NULL
  frame2$Cover <- internal_covers
  frame3 <- rbind(frame2, as.data.frame(leaves))
  #rownames(frame3) <- seq_len(nrow(frame3))
  frame3$Node <- as.integer(seq_len(nrow(frame3)) - 1)
  frame3$Tree <- as.integer(tree_id)
  return(frame3)
}
