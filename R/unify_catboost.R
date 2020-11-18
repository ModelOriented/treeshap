#' Unify Catboost model
#'
#' Convert your GBM model into a standarised data frame.
#' The returned data frame is easy to be interpreted by user and ready to be used as an argument in \code{treeshap()} function
#'
#' @param catboost_model An object of \code{catboost.Model} class. At the moment, models built on data with categorical features
#' are not supported - please encode them before training.
#' @param pool An object of \code{catboost.Pool} class used for training the model
#' @param data data.frame for which calculations should be performed.
#' @param recalculate logical indicating if covers should be recalculated according to the dataset given in data. Keep it FALSE if training data are used.
#'
#' @return Each row of a returned data frame indicates a specific node. The object has a defined structure:
#' \describe{
#'   \item{Tree}{0-indexed ID of a tree}
#'   \item{Node}{0-indexed ID of a node in a tree}
#'   \item{Feature}{In case of an internal node - name of a feature to split on. Otherwise - NA}
#'   \item{Decision.type}{A factor with two levels: "<" and "<=". In case of an internal node - predicate used for splitting observations. Otherwise - NA}
#'   \item{Split}{For internal nodes threshold used for splitting observations. All observations that satisfy the predicate Decision.type(Split) ('< Split' / '<= Split') are proceeded to the node marked as 'Yes'. Otherwise to the 'No' node. For leaves - NA}
#'   \item{Yes}{Index of a row containing a child Node. Thanks to explicit indicating the row it is much faster to move between nodes}
#'   \item{No}{Index of a row containing a child Node}
#'   \item{Missing}{Index of a row containing a child Node where are proceeded all observations with no value of the dividing feature. When the model did not meet any missing value in the feature, it is not specified (marked as NA)}
#'   \item{Prediction}{For leaves: Value of prediction in the leaf. For internal nodes: NA}
#'   \item{Cover}{Number of observations collected by the leaf or seen by the internal node}
#' }
#' @export
#'
#'
#' @seealso
#' \code{\link{xgboost.unify}} for \code{XGBoost models}
#'
#' \code{\link{lightgbm.unify}} for \code{LightGBM models}
#'
#' \code{\link{gbm.unify}} for \code{GBM models}
#'
#' @examples
#' # library(catboost)
#' # data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
#' # label <- fifa20$target
#' # dt.pool <- catboost::catboost.load_pool(data = as.data.frame(lapply(data, as.numeric)),
#' #                                        label = label)
#' # cat_model <- catboost::catboost.train(
#' #             dt.pool,
#' #             params = list(loss_function = 'RMSE',
#' #                           iterations = 100,
#' #                           metric_period = 10,
#' #                           logging_level = 'Silent'))
#' # catboost.unify(cat_model, dt.pool)
catboost.unify <- function(catboost_model, pool, data, recalculate = FALSE) {
  if(class(catboost_model) != "catboost.Model") {
    stop('Object catboost_model is not of type "catboost.Model"')
  }
  if(class(pool) != "catboost.Pool") {
    stop('Object pool is not of type "catboost.Pool"')
  }
  if (!requireNamespace("catboost", quietly = TRUE)) {
    stop("Package \"catboost\" needed for this function to work. Please install it.",
         call. = FALSE)
  }
  path_to_save <- tempfile("catboost_model", fileext = ".json")
  catboost::catboost.save_model(catboost_model, path_to_save, 'json')
  json_data <- jsonlite::read_json(path_to_save)
  if(!is.null(json_data$features_info$categorical_features)){
    stop('catboost.unify() function currently does not support models using categorical features.')
  }

  one_tree_transform <- function(oblivious_tree, tree_id){
    stopifnot(!is.null(oblivious_tree$splits))
    stopifnot(!is.null(oblivious_tree$leaf_values))
    stopifnot(!is.null(oblivious_tree$leaf_weights))
    frame <- data.table::rbindlist(lapply(oblivious_tree$splits, data.table::as.data.table))
    stopifnot(all(frame[['split_type']] == 'FloatFeature'))
    frame <- frame[,c('border', 'float_feature_index')]
    #repeat rows representing node at the kth level 2^(k-1) times:
    frame2 <- frame[rep(seq_len(nrow(frame)), times = 2**(seq_len(nrow(frame))-1)),]
    #Add columns Score and Cover:
    frame2[,c('Score', 'Cover')] <- NA
    frame2[["Yes"]] <- as.integer(seq(1, nrow(frame2)*2, 2))
    frame2[["No"]] <- as.integer(seq(2, nrow(frame2)*2, 2))
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
    stopifnot(tree_levels == floor(tree_levels))
    internal_covers <- numeric()
    for(i in rev(seq(tree_levels))){
      internal_covers <- c(internal_covers, sapply(split(leaves$Cover, ceiling(seq_along(leaves$Cover)/(2**i))), sum))
    }
    names(internal_covers) <- NULL
    frame2[['Cover']] <- internal_covers
    frame3 <- rbind(frame2, leaves)
    rownames(frame3) <- seq_len(nrow(frame3))
    frame3[['ID']] <- as.integer(seq_len(nrow(frame3))-1)
    frame3[['TreeID']] <- as.integer(tree_id)
    #frame3[['float_feature_index']] <- attr(numeric.cat, '.Dimnames')[[2]][frame3[['float_feature_index']] + 1] #potential issue
    return(frame3)
  }
  single_trees <- lapply(seq_along(json_data$oblivious_trees), function(i) one_tree_transform(json_data$oblivious_trees[[i]], (i-1)))
  united <- do.call(rbind, single_trees)
  # How are missing values treated in case of different features?:
  for_missing <- sapply(json_data$features_info$float_features,
                        function(x) x[['nan_value_treatment']])[united[['float_feature_index']]+1]
  united[['Missing']] <- for_missing
  united[(united[['Missing']] == 'AsIs') & !is.na(united[['Missing']]), 'Missing'] <- NA
  united[(united[['Missing']] == 'AsFalse') & !is.na(united[['Missing']]) , 'Missing'] <- united[(united[['Missing']] == 'AsFalse') & !is.na(united[['Missing']]) , 'Yes']
  united[(united[['Missing']] == 'AsTrue') & !is.na(united[['Missing']]) , 'Missing'] <- united[(united[['Missing']] == 'AsTrue') & !is.na(united[['Missing']]) , 'No']
  united[['Missing']] <- as.integer(united[['Missing']])
  united[['float_feature_index']] <-  attr(pool, '.Dimnames')[[2]][united[['float_feature_index']] + 1] #potential issue
  colnames(united) <- c('Split', 'Feature', 'Prediction', 'Cover', 'Yes', 'No', 'Node', 'Tree', 'Missing')
  united$Decision.type <- factor(x = rep("<=", times = nrow(united)), levels = c("<=", "<"))
  united$Decision.type[is.na(united$Feature)] <- NA

  ID <- paste0(united$Node, "-", united$Tree)
  united$Yes <- match(paste0(united$Yes, "-", united$Tree), ID)
  united$No <- match(paste0(united$No, "-", united$Tree), ID)
  united$Missing <- match(paste0(united$Missing, "-", united$Tree), ID)

  ret <- united[, c('Tree', 'Node', 'Feature', 'Decision.type', 'Split', 'Yes', 'No', 'Missing', 'Prediction', 'Cover')]

  # Here we lose "Quality" information
  united[!is.na(Feature), Prediction := NA]

  # for catboost the model prediction results are calculated as [sum(leaf_values * scale + bias)] (https://catboost.ai/docs/concepts/python-reference_catboostregressor_set_scale_and_bias.html)
  # treeSHAP assumes the prediction is sum of leaf values
  # so here we adjust it
  scale <- json_data$scale_and_bias[[1]]
  bias <- json_data$scale_and_bias[[2]]
  united[is.na(Feature), Prediction := Prediction * scale + bias]

  ret <- list(model = united, data = data)
  class(ret) <- "model_unified"
  attr(ret, "missing_support") <- TRUE
  attr(ret, 'model') <- 'catboost'

  if (recalculate) {
    ret <- set_reference_dataset(ret, data)
  }

  return(ret)
}

