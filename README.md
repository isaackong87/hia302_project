# hia302_project
HIA302 Group B project 

data: containing data from the original vax_malaysia.csv file pulled from gihub

data2: containing selected 14 variables of interest

1. date: yyyy-mm-dd format; data correct as of 2359hrs on that date
2. daily_partial: number of individuals who received the first dose of a two-dose protocol
3. daily_full: number of individuals who completed their original protocol (whether the 2nd dose of a two-dose protocol, or a single-dose protocol)
4. daily_booster: number of individuals who received one dose beyond the original protocol
5. daily_booster2: number of individuals who received two doses beyond the original protocol
6. daily: total doses administered
7. daily_partial_adol: subset (already included) of daily_partial, but for individuals aged 12-17 only
8. daily_full_adol: subset (already included) of daily_full, but for individuals aged 12-17 only
9. daily_booster_adol: number of individuals aged 12 - 17 who received one dose beyond the original protocol
10. daily_booster2_adol: number of individuals aged 12 - 17 who received two doses beyond the original protocol
11. daily_partial_child: subset (already included) of daily_partial, but for individuals aged 5-11 only
12. daily_full_child: subset (already included) of daily_full, but for individuals aged 5-11 only
13. daily_booster_child: number of individuals aged 5 - 11 who received one dose beyond the original protocol
14. daily_booster2_child: number of individuals aged 5 - 11 who received two doses beyond the original protocol

data3: containing the total vaccinated individuals in the population

data4: containing the total population information based on the age groups

data5: combined data3 and data4 dataframes with calculated percentage of vaccinated inndividuals in the population based on the age groups 
