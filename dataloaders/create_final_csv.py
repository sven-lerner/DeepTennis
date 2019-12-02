#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 20:27:19 2019

@author: kevinmonogue
"""

import pandas as pd
import numpy as np

def save_csv(years, tourney):
    
    year_num = 2011
    for year in years:
        year['Speed_MPH'] = year['Speed_MPH'].fillna(speed)
        if 'RallyCount' in year.columns:
            year['RallyCount'] = year['RallyCount'].fillna(rally)
        else:
            year['RallyCount'] = rally
        if 'P1DistanceRun' in year.columns:
            year['P1DistanceRun'] = year['P1DistanceRun'].fillna(p1distance)
        else:
            year['P1DistanceRun'] = p1distance
        if 'P2DistanceRun' in year.columns:
            year['P2DistanceRun'] = year['P2DistanceRun'].fillna(p2distance)
        else:
            year['P2DistanceRun'] = p2distance
        year = year[keepcols]

        year.to_csv('../data/final_data/' + str(year_num) + '-' + tourney + '-points-final.csv')
        year_num += 1

def populate_sets_to_win(match_points):
    
    winner = match_points.iloc[-1]['SetWinner']
    net_p1_set_wins = len(match_points.loc[match_points['SetWinner'] == 1])
    net_p2_set_wins = len(match_points.loc[match_points['SetWinner'] == 2])
    matches_to_win = max(net_p1_set_wins, net_p2_set_wins)
    p1_set_wins = 0
    p2_set_wins = 0
    for i, row in match_points.iterrows():
        if row['SetWinner'] == 1:
            p1_set_wins += 1
        elif row['SetWinner'] == 2:
            p2_set_wins += 1
        p1_games_to_win = max((matches_to_win - p1_set_wins) * 6 - row['P1GamesWon'], 0)
        p2_games_to_win = max((matches_to_win - p2_set_wins) * 6 - row['P2GamesWon'], 0)
        if row['SetWinner'] == 1:
            p1_games_to_win = (matches_to_win - p1_set_wins) * 6
        elif row['SetWinner'] == 2:
            p2_games_to_win = (matches_to_win - p2_set_wins) * 6
            
        match_points.at[i, 'p1_sets_to_win']= matches_to_win - p1_set_wins
        match_points.at[i, 'p2_sets_to_win'] = matches_to_win - p2_set_wins
        try:
            match_points.at[i, 'p1_games_to_win'] = p1_games_to_win
        except:
            match_points.at[i, 'p1_games_to_win'] = (matches_to_win - p1_set_wins) * 6 
        try:
            match_points.at[i, 'p2_games_to_win'] = p2_games_to_win
        except:
            match_points.at[i, 'p2_games_to_win'] = (matches_to_win - p2_set_wins) * 6 
        match_points.at[i, 'winner'] = winner
    return match_points

def sets_to_win_tourney(tourney):
    tourney.loc[:, 'p1_games_to_win'] = 0
    tourney.loc[:, 'p2_games_to_win'] = 0
    tourney.loc[:, 'p1_sets_to_win'] = 0
    tourney.loc[:, 'p2_sets_to_win'] = 0
    tourney.loc[:, 'winner'] = 0
    
    for match in tourney['match_id'].unique():
        tourney.loc[tourney['match_id'] == match] = populate_sets_to_win(tourney.loc[tourney['match_id'] == match])
        
    return tourney
            

def add_player_winner_sets(year, tourney):
    
    matches = pd.read_csv('../data/sackman/' + str(year) + '-' + tourney + '-matches.csv')
    points = pd.read_csv('../data/sackman/' + str(year) + '-' + tourney + '-points.csv')
    points = sets_to_win_tourney(points)
    
    matches = matches[['match_id', 'player1', 'player2']]
    result = points.merge(matches, left_on = 'match_id', right_on = 'match_id')
    return result

def merge_years(years, tourney):
    
    idv = []
    result = add_player_winner_sets(years[0], tourney)
    idv.append(result)
    for i in range(1, len(years)):
        df = add_player_winner_sets(years[i], tourney)
        result = result.append(df, sort = False)
        idv.append(df)
        
    return result, idv

years1 = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
years2 = [2011, 2012, 2013, 2014, 2015, 2016, 2017]

usopen, us_list = merge_years(years1, 'usopen')
print('us')
frenchopen, french_list = merge_years(years2, 'frenchopen')
print('french')
wimbledon, wimb_list = merge_years(years1, 'wimbledon')
print('wimb')
ausopen, aus_list = merge_years(years2, 'ausopen')
print('aus')

total = usopen.shape[0] + frenchopen.shape[0] + wimbledon.shape[0] + ausopen.shape[0]
weights = [usopen.shape[0] / total, frenchopen.shape[0] / total, wimbledon.shape[0] / total, ausopen.shape[0] / total]

speed_list = [usopen['Speed_MPH'].mean(), frenchopen['Speed_MPH'].mean(), wimbledon['Speed_MPH'].mean(), ausopen['Speed_MPH'].mean()]
rally_list = [usopen['RallyCount'].mean(), frenchopen['RallyCount'].mean(), wimbledon['RallyCount'].mean(), ausopen['RallyCount'].mean()]
p1distance_list = [usopen['P1DistanceRun'].mean(), frenchopen['P1DistanceRun'].mean(), wimbledon['P1DistanceRun'].mean(), ausopen['P1DistanceRun'].mean()]
p2distance_list = [usopen['P2DistanceRun'].mean(), frenchopen['P2DistanceRun'].mean(), wimbledon['P2DistanceRun'].mean(), ausopen['P2DistanceRun'].mean()]

speed = np.average(speed_list, weights = weights)
rally = np.average(rally_list, weights = weights)
p1distance = np.average(p1distance_list, weights = weights)
p2distance = np.average(p2distance_list, weights = weights)

keepcols = ['match_id', 'player1', 'player2', 'winner',
# 'ElapsedTime',
'SetNo',
'P1GamesWon',
'P2GamesWon',
'SetWinner',
'GameNo',
'GameWinner',
'PointNumber',
'PointWinner',
'PointServer',
# 'Speed_KMH',
# 'Rally',
'P1Score',
'P2Score',
# 'P1Momentum',
# 'P2Momentum',
'P1PointsWon',
'P2PointsWon',
'P1Ace',
'P2Ace',
'P1Winner',
'P2Winner',
'P1DoubleFault',
'P2DoubleFault',
'P1UnfErr',
'P2UnfErr',
'P1NetPoint',
'P2NetPoint',
'P1NetPointWon',
'P2NetPointWon',
'P1BreakPoint',
'P2BreakPoint',
'P1BreakPointWon',
'P2BreakPointWon',
# 'P1FirstSrvIn',
# 'P2FirstSrvIn',
# 'P1FirstSrvWon',
# 'P2FirstSrvWon',
# 'P1SecondSrvIn',
# 'P2SecondSrvIn',
# 'P1SecondSrvWon',
# 'P2SecondSrvWon',
# 'P1ForcedError',
# 'P2ForcedError',
# 'History',
 'Speed_MPH',
 'RallyCount',
    'P1DistanceRun',
    'P2DistanceRun',
# 'P1BreakPointMissed',
# 'P2BreakPointMissed',
# 'ServeIndicator',
# 'P1TurningPoint',
# 'P2TurningPoint',
    'p1_sets_to_win',
    'p2_sets_to_win',
    'p1_games_to_win',
    'p2_games_to_win',
]

save_csv(us_list, 'usopen')
save_csv(french_list, 'frenchopen')
save_csv(wimb_list, 'wimbledon')
save_csv(aus_list, 'ausopen')