#' Attributes of all players in FIFA 20
#'
#' Dataset consists of 56 columns, 55 numeric and one of type factor \code{'work_rate'}.
#' \code{value_eur} is a potential target feature.
#'
#' @format A data frame with 18278 rows and 56 columns.
#' Most of variables representing skills are in range from 0 to 100 and will not be described here.
#' To list nonobvivous features:
#' \describe{
#' \item{overall}{Overall score of player's skills}
#' \item{potential}{Potential of a player, younger players tend to have higher level of potential}
#' \item{value_eur}{Market value of a player (in mln EUR)}
#' \item{international_reputation}{Range 1 to 5}
#' \item{weak_foot}{Range 1 to 5}
#' \item{skill_moves}{Range 1 to 5}
#' \item{work_rate}{Divided by slash levels of willingness to work in offense and defense respectively}
#' }
#'
#'@source
#'"Data has been scraped from the publicly available website https://sofifa.com."
#'\url{https://www.kaggle.com/stefanoleone992/fifa-20-complete-player-dataset}
#'
"fifa20"
