# hia302_project
HIA302 Group B project 

Dr Irfan 
Vaccination

date: yyyy-mm-dd format; data correct as of 2359hrs on that date
partial_5_1l: number of children  who received 1st dose vaccine at age 5-11 only
full_5_11 : number of children who received 2nd dose vaccine at age 5-11 only
booster_5_11: number of children who received 1st booster at age 5-11 only
partial_12_17: number of children  who received 1st dose vaccine at age 12-17 only
full_12_17 : number of children who received 2nd dose vaccine at age 12-17 only
booster_12_17: number of children who received 1st booster at age 12-17 only
partial_60_69: number of elderly  who received 1st dose vaccine at age 60-69 only
full_60_69 : number of elderly who received 2nd dose vaccine at age 60-69 only
booster_60_69: number of elderly  who received 1st booster at age 60-69 only
partial_70_79: number of elderly  who received 1st dose vaccine at age 70-79 only
full_70_79 : number of elderly who received 2nd dose vaccine at age 70-79 only
booster_70_79: number of elderly  who received 1st booster at age 70-79 only
partial_80: number of elderly  who received 1st dose vaccine at age > 80 only
full_80 : number of elderly who received 2nd dose vaccine at age > 80 only
booster_80: number of elderly  who received 1st booster at age > 80 only

=======
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

data3: containing the total vaccinated individuals (overall, adolescents, and children) in the population

data4: containing the total population information (overall population, adolescents aged 12-17 only, and children aged 5-11 only)

data5: combined data3 and data4 dataframes with calculated percentage of vaccinated inndividuals in the population (overall, adolescents, and children) 

