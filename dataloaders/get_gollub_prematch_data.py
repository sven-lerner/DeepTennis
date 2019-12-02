#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 12:28:55 2019

Matches ATP data with Gollub pre-match prediction data, and 
assigns match ID. 

@author: kevinmonogue
"""

import pandas as pd
pd.options.display.max_columns = None

#modifier for sackman tourney names
def change_tourney_name(x):
    if x == 'wimbledon':
        return 'Wimbledon'
    elif x == 'frenchopen':
        return 'Roland Garros'
    elif x == 'ausopen':
        return 'Australian Open'
    else:
        return 'US Open'
    
#modifier for gollub player names
def drop_end_words(x):
    try:
        st = x.replace('-', ' ')
        st = st.split(' ')
        return st[0] + ' '+ st[1]
    except:
        return x

#main function to merge data
def merge_tourney(year, tourney, gollub, prob_cols):

    #load only the year and tournament we want
    sackman_df = pd.read_csv('data/'+ str(year) + '-' + tourney + '-matches.csv')
    sackman_df['slam'] = sackman_df['slam'].apply(change_tourney_name)
    sackman_df['player1'] = sackman_df['player1'].apply(drop_end_words)
    sackman_df['player2'] = sackman_df['player2'].apply(drop_end_words)
    gollub_year = gollub[gollub['match_year'] == year]

    #merge data based on if named player1 or player2
    combined_p0_df = sackman_df.merge(gollub_year, left_on = ['slam', 'player1', 'player2'], right_on = ['tny_name', 'p0_name', 'p1_name'])
    combined_p1_df = sackman_df.merge(gollub_year, left_on = ['slam', 'player1', 'player2'], right_on = ['tny_name', 'p1_name', 'p0_name'])

    #ensure probabilities from gollub correspond to player 1 from sackman
    for col in prob_cols:
        combined_p1_df[col] = 1 - combined_p1_df[col]

    #obtain the mens data
    mens_df = pd.concat([combined_p0_df, combined_p1_df], ignore_index = True)

    #add back in the womens data and fill null values with probability 0.5
    total_df = sackman_df.merge(mens_df, how = 'left', left_on = ['player1', 'player2'], right_on = ['player1', 'player2'])
    keep_cols = ['match_id_x', 'year_x', 'slam_x', 'match_num_x', 'player1', 'player2'] + prob_cols
    total_df = total_df[keep_cols].rename({'match_id_x': 'match_id', 'year_x': 'year', 'slam_x': 'slam', 'match_num_x': 'match_num'})
    total_df.fillna(0.5)
    total_df.to_csv('../data/gollubdata/gollub-prematch-' + str(year) + '-' + tourney + '.csv')
    
def merge_gollub_data():
    
    #load main gollub data
    gollub = pd.read_csv("../data/gollubdata/elo_atp_matches_all_10_29.csv")
    
    #save appropriate columns
    id_cols = ['tny_name', 'match_year', 'p0_name', 'p1_name']
    prob_cols = []
    cols = gollub.columns
    for col in cols:
        if 'prob' in col:
            id_cols.append(col)
            prob_cols.append(col)
    gollub = gollub[id_cols]
    
    years = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
    tourneys = ['usopen', 'frenchopen', 'ausopen', 'wimbledon']
    
    for year in years:
        for tourney in tourneys:
            merge_tourney(year, tourney, gollub, prob_cols)
        
    years_2 = [2018, 2019]
    tourneys_2 = ['usopen', 'wimbledon']
    
    for year in years_2:
        for tourney in tourneys_2:
            merge_tourney(year, tourney, gollub, prob_cols)
merge_gollub_data()
