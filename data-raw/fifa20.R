options(stringsAsFactors = FALSE)
fifa20_raw <- read.csv('~/Documents/players_20.csv')
fifa20_num <- fifa20_raw[,sapply(fifa20_raw, is.numeric)]
fifa20_all <- fifa20_num[!(colnames(fifa20_num) %in% c('sofifa_id',
                        'wage_eur', 'release_clause_eur', 'team_jersey_number', 'contract_valid_until', 'nation_jersey_number'))]
fifa20_all[['work_rate']] <- as.factor(fifa20_raw[['work_rate']])
fifa_target <- fifa20_all[['value_eur']]
fifa20_data <- fifa20_all[colnames(fifa20_all) != 'value_eur']
fifa20 <- list(data = fifa20_data, target = fifa_target)


usethis::use_data(fifa20, overwrite = TRUE)
