#' Unify xgboost model
#'
#' Convert your xgboost model into a standarised data frame.
#' The returned data frame is easy to be interpreted by user and ready to be used as an argument in the \code{treeshap()} function.
#'
#' @param xgb_model A xgboost model - object of class \code{xgb.Booster}
#'
#' @return Each row of a returned data frame indicates a specific node. The object has a defined structure:
#' \describe{
#'   \item{Tree}{0-indexed ID of a tree}
#'   \item{Node}{0-indexed ID of a node in a tree}
#'   \item{Feature}{In case of an internal node - name of a feature to split on. Otherwise - NA.}
#'   \item{Split}{Threshold used for splitting observations.
#'   All observations with lower or equal value than it are proceeded to the node marked as 'Yes'. Otherwise to the 'No' node}
#'   \item{Yes}{Index of a row containing a child Node. Thanks to explicit indicating the row it is much faster to move between nodes.}
#'   \item{No}{Index of a row containing a child Node}
#'   \item{Missing}{Index of a row containing a child Node where are proceeded all observations with no value of the dividing feature}
#'   \item{Quality/Score}{For internal nodes - Quality: either the split gain (change in loss) or the leaf value.
#'   For leaves - Score: Value of prediction in the leaf}
#'   \item{Cover}{Number of observations seen by the internal node or collected by the leaf}
#' }
#' @export
#'
#' @seealso
#' \code{\link{lightgbm.unify}} for \code{LightGBM models}
#'
#' \code{\link{gbm.unify}} for \code{GBM models}
#'
#' \code{\link{catboost.unify}} for \code{Catboost models}
#'
#' @examples
#' library(xgboost)
#' data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
#' target <- fifa20$target
#' param <- list(objective = "reg:squarederror", max_depth = 3)
#' xgb_model <- xgboost::xgboost(as.matrix(data), params = param, label = target, nrounds = 200)
#' xgboost.unify(xgb_model)
#'
xgboost.unify <- function(xgb_model) {
  if (!requireNamespace("xgboost", quietly = TRUE)) {
    stop("Package \"xgboost\" needed for this function to work. Please install it.",
         call. = FALSE)
  }
  xgbtree <- xgboost::xgb.model.dt.tree(model = xgb_model)
  stopifnot(c("Tree", "Node", "ID", "Feature", "Split", "Yes", "No", "Missing", "Quality", "Cover") %in% colnames(xgbtree))
  xgbtree$Yes <- match(xgbtree$Yes, xgbtree$ID)
  xgbtree$No <- match(xgbtree$No, xgbtree$ID)
  xgbtree$Missing <- match(xgbtree$Missing, xgbtree$ID)
  xgbtree[xgbtree$Feature == 'Leaf', 'Feature'] <- NA
  xgbtree <- xgbtree[, c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "Quality", "Cover")]
  colnames(xgbtree) <- c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "Quality/Score", "Cover")
  attr(xgbtree, "model") <- "xgboost"
  return(xgbtree)
}


#should be preceded with lgb.model.dt.tree
#' Unify LightGBM model
#'
#' Convert your xgboost model into a standarised data frame.
#' The returned data frame is easy to be interpreted by user and ready to be used as an argument in the \code{treeshap()} function.
#'
#' @param lgb_model A lightgbm model - object of class \code{lgb.Booster}
#'
#' @return Each row of a returned data frame indicates a specific node. The object has a defined structure:
#' \describe{
#'   \item{Tree}{0-indexed ID of a tree}
#'   \item{Node}{0-indexed ID of a node in a tree}
#'   \item{Feature}{In case of an internal node - name of a feature to split on. Otherwise - NA}
#'   \item{Split}{Threshold used for splitting observations.
#'   All observations with lower or equal value than it are proceeded to the node marked as 'Yes'. Otherwise to the 'No' node}
#'   \item{Yes}{Index of a row containing a child Node. Thanks to explicit indicating the row it is much faster to move between nodes.}
#'   \item{No}{Index of a row containing a child Node}
#'   \item{Missing}{Index of a row containing a child Node where are proceeded all observations with no value of the dividing feature}
#'   \item{Quality/Score}{For internal nodes - Quality: Split gain of a node.
#'   For leaves - Score: Value of prediction in the leaf}
#'   \item{Cover}{Number of observations seen by the internal node or collected by the leaf}
#' }
#' @export
#' @import data.table
#'
#' @seealso
#' \code{\link{xgboost.unify}} for \code{XGBoost models}
#'
#' \code{\link{gbm.unify}} for \code{GBM models}
#'
#' \code{\link{catboost.unify}} for \code{Catboost models}
#'
#' @examples
#' library(lightgbm)
#' library(Matrix)
#' param_lgbm <- list(objective = "regression", max_depth = 2,  force_row_wise = TRUE)
#' data_fifa <- fifa20$data[!colnames(fifa20$data) %in%
#'              c('work_rate', 'value_eur', 'gk_diving', 'gk_handling',
#'              'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning')]
#' data <- as.matrix(na.omit(data.table::as.data.table(cbind(data_fifa, fifa20$target))))
#' sparse_data <- as(data[,ncol(data)], 'sparseMatrix')
#' x <- lightgbm::lgb.Dataset(sparse_data, label = as(data[,ncol(data)], 'sparseMatrix'))
#' lgb_data <- lightgbm::lgb.Dataset.construct(x)
#' lgb_model <- lightgbm::lightgbm(data = lgb_data, params = param_lgbm)
#' lightgbm.unify(lgb_model)
#'
lightgbm.unify <- function(lgb_model) {
  if (!requireNamespace("lightgbm", quietly = TRUE)) {
    stop("Package \"lightgbm\" needed for this function to work. Please install it.",
         call. = FALSE)
  }
  df <- lightgbm::lgb.model.dt.tree(lgb_model)
  stopifnot(c("split_index", "split_feature", "node_parent", "leaf_index", "leaf_parent", "internal_value",
              "internal_count", "leaf_value", "leaf_count")
            %in% colnames(df) )
  df <- data.table::as.data.table(df)
  #convert node_parent and leaf_parent into one parent column
  df[is.na(df$node_parent), "node_parent"] <- df[is.na(df$node_parent), "leaf_parent"]
  #convert values into one column...
  df[is.na(df$internal_value), "internal_value"] <- df[!is.na(df$leaf_value), "leaf_value"]
  #...and counts
  df[is.na(df$internal_count), "internal_count"] <- df[!is.na(df$leaf_count), "leaf_count"]
  df[["internal_count"]] <- as.numeric(df[["internal_count"]])
  #convert split_index and leaf_index into one column:
  max_split_index <- df[, max(split_index, na.rm = TRUE), tree_index]
  rep_max_split <- rep(max_split_index$V1, times = as.numeric(table(df$tree_index)))
  new_leaf_index <- rep_max_split + df[, "leaf_index"] + 1
  df[is.na(df$split_index), "split_index"] <- new_leaf_index[!is.na(new_leaf_index[["leaf_index"]]), 'leaf_index']
  df[is.na(df$split_gain), "split_gain"] <- df[is.na(df$split_gain), "leaf_value"]
  # On the basis of column 'Parent', create columns with childs: 'Yes', 'No' and 'Missing' like in the xgboost df:
  ret.first <- function(x) x[1]
  ret.second <- function(x) x[2]
  tmp <- data.table::merge.data.table(df[, .(node_parent, tree_index, split_index)], df[, .(tree_index, split_index, default_left, decision_type)],
               by.x = c("tree_index", "node_parent"), by.y = c("tree_index", "split_index"))
  y_n_m <- unique(tmp[, .(Yes = ifelse(decision_type %in% c("<=", "<"), ret.first(split_index),
                                      ifelse(decision_type %in% c(">=", ">"), ret.second(split_index), stop("Unknown decision_type"))),
                         No = ifelse(decision_type %in% c(">=", ">"), ret.first(split_index),
                                     ifelse(decision_type %in% c("<=", "<"), ret.second(split_index), stop("Unknown decision_type"))),
                         Missing = ifelse(default_left, ret.first(split_index),ret.second(split_index))),
                      .(tree_index, node_parent)])
  df <- data.table::merge.data.table(df[, c("tree_index", "depth", "split_index", "split_feature", "node_parent", "split_gain",
                    "threshold", "internal_value", "internal_count")],
              y_n_m, by.x = c("tree_index", "split_index"),
              by.y = c("tree_index", "node_parent"), all.x = TRUE)
  df <- df[, c("tree_index", "split_index", "split_feature", "threshold", "Yes", "No", "Missing", "split_gain", "internal_count")]
  colnames(df) <- c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "Quality/Score", "Cover")
  attr(df, "model") <- "LightGBM"
  attr(df, "sorted") <- NULL

  ID <- paste0(df$Node, "-", df$Tree)
  df$Yes <- match(paste0(df$Yes, "-", df$Tree), ID)
  df$No <- match(paste0(df$No, "-", df$Tree), ID)
  df$Missing <- match(paste0(df$Missing, "-", df$Tree), ID)
  return(df)
}


#' Unify GBM model
#'
#' Convert your GBM model into a standarised data frame.
#' The returned data frame is easy to be interpreted by user and ready to be used as an argument in \code{treeshap()} function
#'
#' @param gbm_model An object of \code{gbm} class. At the moment, models built on data with categorical features
#' are not supported - please encode them before training.
#'
#' @return Each row of a returned data frame indicates a specific node. The object has a defined structure:
#' \describe{
#'   \item{Tree}{0-indexed ID of a tree}
#'   \item{Node}{0-indexed ID of a node in a tree}
#'   \item{Feature}{In case of an internal node - name of a feature to split on. Otherwise - NA}
#'   \item{Split}{Threshold used for splitting observations.
#'   All observations with lower or equal value than it are proceeded to the node marked as 'Yes'. Othwerwise to the 'No' node}
#'   \item{Yes}{Index of a row containing a child Node. Thanks to explicit indicating the row it is much faster to move between nodes.}
#'   \item{No}{Index of a row containing a child Node}
#'   \item{Missing}{Index of a row containing a child Node where are proceeded all observations with no value of the dividing feature}
#'   \item{Quality/Score}{For internal nodes - Quality: the reduction in the loss function as a result of splitting this node.
#'   For leaves - Score: Value of prediction in the leaf}
#'   \item{Cover}{Number of observations seen by the internal node or collected by the leaf}
#' }
#' @export
#'
#' @seealso
#' \code{\link{xgboost.unify}} for \code{XGBoost models}
#'
#' \code{\link{lightgbm.unify}} for \code{LightGBM models}
#'
#' \code{\link{catboost.unify}} for \code{Catboost models}
#'
#' @examples
#' \dontrun{
#' library(gbm)
#' data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
#' data['value_eur'] <- fifa20$target
#' gbm_model <- gbm::gbm(
#'              formula = value_eur ~ .,
#'              data = data,
#'              distribution = "laplace",
#'              n.trees = 1000,
#'              cv.folds = 2,
#'              interaction.depth = 2,
#'              n.cores = 1)
#' gbm.unify(gbm_model)
#'}
gbm.unify <- function(gbm_model) {
  if(class(gbm_model) != 'gbm') {
    stop('Object gbm_model was not of class "gbm"')
  }
  if(any(gbm_model$var.type > 0)) {
    stop('Models built on data with categorical features are not supported - please encode them before training.')
  }
  x <- lapply(gbm_model$trees, data.table::as.data.table)
  times_vec <- sapply(x, nrow)
  y <- data.table::rbindlist(x)
  data.table::setnames(y, c("Feature", "Split", "Yes",
                            "No", "Missing", "ErrorReduction", "Cover",
                            "Prediction"))
  y[["Tree"]] <- rep(0:(length(gbm_model$trees) - 1), times = times_vec)
  y[["Node"]] <- unlist(lapply(times_vec, function(x) 0:(x - 1)))
  y <- y[, Feature:=as.character(Feature)]
  y[y$Feature<0, "Feature"]<- NA
  y[!is.na(y$Feature), "Feature"] <- attr(gbm_model$Terms, "term.labels")[as.integer(y[["Feature"]][!is.na(y$Feature)]) + 1]
  y[is.na(y$Feature), "ErrorReduction"] <- y[is.na(y$Feature), "Split"]
  y[is.na(y$Feature), "Split"] <- NA
  y[y$Yes<0, "Yes"] <- NA
  y[y$No<0, "No"] <- NA
  y[y$Missing<0, "Missing"] <- NA
  y <- y[, c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "ErrorReduction", "Cover")]
  colnames(y) <- c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "Quality/Score", "Cover")
  attr(y, "model") <- "gbm"

  ID <- paste0(y$Node, "-", y$Tree)
  y$Yes <- match(paste0(y$Yes, "-", y$Tree), ID)
  y$No <- match(paste0(y$No, "-", y$Tree), ID)
  y$Missing <- match(paste0(y$Missing, "-", y$Tree), ID)

  return(y)
}


#' Unify Catboost model
#'
#' Convert your GBM model into a standarised data frame.
#' The returned data frame is easy to be interpreted by user and ready to be used as an argument in \code{treeshap()} function
#'
#' @param catboost_model An object of \code{catboost.Model} class. At the moment, models built on data with categorical features
#' are not supported - please encode them before training.
#' @param pool An object of \code{catboost.Pool} class used for training the model
#'
#' @return Each row of a returned data frame indicates a specific node. The object has a defined structure:
#' \describe{
#'   \item{Tree}{0-indexed ID of a tree}
#'   \item{Node}{0-indexed ID of a node in a tree}
#'   \item{Feature}{In case of an internal node - name of a feature to split on. Otherwise - NA}
#'   \item{Split}{Threshold used for splitting observations.
#'   All observations with lower or equal value than it are proceeded to the node marked as 'Yes'. Othwerwise to the 'No' node}
#'   \item{Yes}{Index of a row containing a child Node. Thanks to explicit indicating the row it is much faster to move between nodes.}
#'   \item{No}{Index of a row containing a child Node}
#'   \item{Missing}{Index of a row containing a child Node where are proceeded all observations with no value of the dividing feature.
#'   When the model did not meet any missing value in the feature, it is not specified (marked as NA)}
#'   \item{Quality/Score}{For internal nodes - NA.
#'   For leaves - Score: Value of prediction in the leaf}
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
#' #library(catboost)
#' #data <- fifa20$data[colnames(fifa20$data) != 'work_rate']
#' #label <- fifa20$target
#' #dt.pool <- catboost::catboost.load_pool(data = as.data.frame(lapply(data, as.numeric)),
#'  #                                       label = label)
#' #cat_model <- catboost::catboost.train(
#'  #            dt.pool,
#'  #            params = list(loss_function = 'RMSE',
#'  #                          iterations = 100,
#'  #                          metric_period = 10,
#'  #                          logging_level = 'Info'))
#' #catboost.unify(cat_model, dt.pool)

catboost.unify <- function(catboost_model, pool) {
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
  path_to_save <- paste0(tempdir(), '/catboostmodel.json')
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
    # colnames(frame3) <- c('Threshold', 'Feature', 'split_index', 'split_type', 'Quality/Score', 'Cover', 'Yes', 'No', 'ID', 'Missing')
    # frame4 <- frame3[,c('ID', 'Feature', 'Threshold', 'Yes', 'No', 'Missing', 'Quality/Score', 'Cover')]
    # frame4
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
  colnames(united) <- c('Split', 'Feature', 'Quality/Score', 'Cover', 'Yes', 'No', 'Node', 'Tree', 'Missing')
  attr(united, 'model') <- 'catboost'

  ID <- paste0(united$Node, "-", united$Tree)
  united$Yes <- match(paste0(united$Yes, "-", united$Tree), ID)
  united$No <- match(paste0(united$No, "-", united$Tree), ID)
  united$Missing <- match(paste0(united$Missing, "-", united$Tree), ID)

  return(united[,c('Tree', 'Node', 'Feature', 'Split', 'Yes', 'No', 'Missing', 'Quality/Score', 'Cover')])
}


#' Unify randomForest model
#'
#' Convert your randomForest model into a standarised data frame.
#' The returned data frame is easy to be interpreted by user and ready to be used as an argument in \code{treeshap()} function
#'
#' @param rf_model An object of \code{randomForest} class. At the moment, models built on data with categorical features
#' are not supported - please encode them before training.
#' @param data A training frame used to fit the model.
#'
#' @return Each row of a returned data frame indicates a specific node. The object has a defined structure:
#' \describe{
#'   \item{Tree}{0-indexed ID of a tree}
#'   \item{Node}{0-indexed ID of a node in a tree}
#'   \item{Feature}{In case of an internal node - name of a feature to split on. Otherwise - NA}
#'   \item{Split}{Threshold used for splitting observations.
#'   All observations with lower or equal value than it are proceeded to the node marked as 'Yes'. Othwerwise to the 'No' node}
#'   \item{Yes}{Index of a row containing a child Node. Thanks to explicit indicating the row it is much faster to move between nodes.}
#'   \item{No}{Index of a row containing a child Node}
#'   \item{Missing}{Index of a row containing a child Node where are proceeded all observations with no value of the dividing feature}
#'   \item{Quality/Score}{For internal nodes - Quality: the reduction in the loss function as a result of splitting this node.
#'   For leaves - Score: Value of prediction in the leaf}
#'   \item{Cover}{Number of observations seen by the internal node or collected by the leaf}
#' }
#'
#' @import data.table
#'
#' @export
#'
#' @seealso
#' \code{\link{xgboost.unify}} for \code{XGBoost models}
#'
#' \code{\link{lightgbm.unify}} for \code{LightGBM models}
#'
#' \code{\link{catboost.unify}} for \code{Catboost models}
#'
#' @examples
#' \dontrun{
#' library(randomForest)
#' data_fifa <- fifa20$data[!colnames(fifa20$data) %in%
#'                            c('work_rate', 'value_eur', 'gk_diving', 'gk_handling',
#'                              'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning')]
#' data <- as.matrix(na.omit(data.table::as.data.table(cbind(data_fifa, target = fifa20$target))))
#'
#' rf <- randomForest::randomForest(target~., data = data, maxnodes = 10)
#' randomForest.unify(rf, data)
#'}
randomForest.unify <- function(rf_model, data) {
  if(!'randomForest' %in% class(rf_model)){stop('Object rf_model was not of class "randomForest"')}
  if(any(attr(rf_model$terms, "dataClasses") != "numeric")) {
    stop('Models built on data with categorical features are not supported - please encode them before training.')
  }
  n <- rf_model$ntree
  ret <- data.table()
  x <- lapply(1:n, function(tree){
    tree_data <- as.data.table(randomForest::getTree(rf_model, k = tree, labelVar = TRUE))
    tree_data[, c("left daughter", "right daughter", "split var", "split point", "prediction")]
  })
  times_vec <- sapply(x, nrow)
  y <- rbindlist(x)
  y[, Tree := rep(0:(n - 1), times = times_vec)]
  y[, Node := unlist(lapply(times_vec, function(x) 0:(x - 1)))]
  y[, Missing := NA]
  y[, Cover := 0]
  setnames(y, c("No", "Yes", "Feature", "Split",  "Quality/Score", "Tree", "Node", "Missing", "Cover"))
  setcolorder(y, c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "Quality/Score", "Cover"))
  y[, Feature := as.character(Feature)]
  y[, Yes := Yes - 1]
  y[, No := No - 1]
  y[y$Yes < 0, "Yes"] <- NA
  y[y$No < 0, "No"] <- NA
  attr(y, "model") <- "randomForest"

  ID <- paste0(y$Node, "-", y$Tree)
  y$Yes <- match(paste0(y$Yes, "-", y$Tree), ID)
  y$No <- match(paste0(y$No, "-", y$Tree), ID)
  y[, Missing := Yes]

  y <- recalculate_covers(y, data)
  return(y)
}


#' Unify ranger model
#'
#' Convert your ranger model into a standarised data frame.
#' The returned data frame is easy to be interpreted by user and ready to be used as an argument in \code{treeshap()} function
#'
#' @param rf_model An object of \code{ranger} class. At the moment, models built on data with categorical features
#' are not supported - please encode them before training.
#' @param data A training frame used to fit the model.
#'
#' @return Each row of a returned data frame indicates a specific node. The object has a defined structure:
#' \describe{
#'   \item{Tree}{0-indexed ID of a tree}
#'   \item{Node}{0-indexed ID of a node in a tree}
#'   \item{Feature}{In case of an internal node - name of a feature to split on. Otherwise - NA}
#'   \item{Split}{Threshold used for splitting observations.
#'   All observations with lower or equal value than it are proceeded to the node marked as 'Yes'. Othwerwise to the 'No' node}
#'   \item{Yes}{Index of a row containing a child Node. Thanks to explicit indicating the row it is much faster to move between nodes.}
#'   \item{No}{Index of a row containing a child Node}
#'   \item{Missing}{Index of a row containing a child Node where are proceeded all observations with no value of the dividing feature}
#'   \item{Quality/Score}{For internal nodes - Quality: the reduction in the loss function as a result of splitting this node.
#'   For leaves - Score: Value of prediction in the leaf}
#'   \item{Cover}{Number of observations seen by the internal node or collected by the leaf}
#' }
#'
#' @import data.table
#'
#' @export
#'
#' @seealso
#' \code{\link{xgboost.unify}} for \code{XGBoost models}
#'
#' \code{\link{lightgbm.unify}} for \code{LightGBM models}
#'
#' \code{\link{catboost.unify}} for \code{Catboost models}
#'
#' @examples
#' \dontrun{
#' # library(ranger)
#' # data_fifa <- fifa20$data[!colnames(fifa20$data) %in%
#' #                            c('work_rate', 'value_eur', 'gk_diving', 'gk_handling',
#' #                             'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning')]
#' # data <- na.omit(data.table::as.data.table(cbind(data_fifa, target = fifa20$target)))
#'
#' # rf <- ranger::ranger(target~., data = data, max.depth = 10)
#' # ranger.unify(rf, data)
#'}
ranger.unify <- function(rf_model, data) {
  if(!'ranger' %in% class(rf_model)) {
    stop('Object rf_model was not of class "ranger"')
  }
  n <- rf_model$num.trees
  x <- lapply(1:n, function(tree){
    tree_data <- as.data.table(ranger::treeInfo(rf_model, tree = tree))
    tree_data[, c("nodeID",  "leftChild", "rightChild", "splitvarName", "splitval", "prediction")]
  })
  times_vec <- sapply(x, nrow)
  y <- rbindlist(x)
  y[, Tree := rep(0:(n - 1), times = times_vec)]
  y[, Missing := NA]
  y[, Cover := 0]
  setnames(y, c("Node", "Yes", "No", "Feature", "Split",  "Quality/Score", "Tree", "Missing", "Cover"))
  setcolorder(y, c("Tree", "Node", "Feature", "Split", "Yes", "No", "Missing", "Quality/Score", "Cover"))
  y[, Feature := as.character(Feature)]
  y[y$Yes < 0, "Yes"] <- NA
  y[y$No < 0, "No"] <- NA
  attr(y, "model") <- "ranger"

  ID <- paste0(y$Node, "-", y$Tree)
  y$Yes <- match(paste0(y$Yes, "-", y$Tree), ID)
  y$No <- match(paste0(y$No, "-", y$Tree), ID)
  y <- recalculate_covers(y, as.data.frame(data))
  return(y)
}
