import os
import pandas as pd
import random
from dataloaders.valid_data_fields import *
import numpy as np


'''
data loading utilities, for the most part these guys deal with extracting information from the csvs as well as dealing with some 
'qirks' in the data
'''

def parse_time(time_string):
    hr, m, s = [int(x) for x in time_string.split(':')]
    return 3600*hr + 60 * m + s

def extract_numpy_from_match(match_points, shuffle_players=False):
    
    match_points_copy = match_points.copy()
    if shuffle_players:
        for p1_val, p2_val in shuffle_pairs:
            tmp = match_points_copy[p1_val]
            match_points_copy[p1_val] = match_points_copy[p2_val]
            match_points_copy[p2_val] = tmp
    parsed_time = match_points_copy
    parsed_time['ElapsedTime'] = parsed_time['ElapsedTime'].map(lambda x: parse_time(x))
    parsed_scores = parsed_time.replace('AD', 55)
    
    
    scores = parsed_scores[list(valid_fields)].to_numpy(dtype=np.float)
    # scores = scores.fillna(0)
    scores = scores[~np.isnan(scores).any(axis=1)]
    assert np.sum(np.isnan(scores)) < 1, f"hit a nan {scores}"
    return scores
    
def get_match_data(match_id, match_data, point_data, soften_curve):
    shuffle_players = random.uniform(0, 1) > 0.5
    winner = match_data.loc[match_data['match_id'] == match_id].iloc[0]['winner'] - 1
    point_data = point_data.loc[point_data['match_id'] == match_id]

    if winner not in [1,2]:
        last_point = point_data.sort_values('PointNumber').iloc[-1]
        last_point_winner = last_point['PointWinner']
        last_game_winner = last_point['GameWinner']
        assert last_point_winner == last_game_winner, f'{last_point_winner} was not {last_game_winner}'
        winner = last_point_winner - 1
    
    if shuffle_players:
        winner = (winner + 1) % 2

    parsed_point_data = extract_numpy_from_match(point_data, shuffle_players)
    num_points = parsed_point_data.shape[0]

    assert num_points > 0, "dropped match due to no points played"
        
    y_gt = np.full(shape=num_points, fill_value=winner, dtype='float')

    if soften_curve:
        if winner == 1:
            for i in range(num_points):
                if i > 25:
                    y_gt[i] = y_gt[i] -0.5*(1.0-(i-25)/(num_points-25))
                else:
                    y_gt[i] = 0.5
        else:
            for i in range(num_points):
                if i > 25:
                    y_gt[i] = y_gt[i] +0.5*(1.0-(i-25)/(num_points-25))
                else:
                    y_gt[i] = 0.5
    
    return parsed_point_data, y_gt

def get_training_data_from_open(open_matches_path, open_points_path, get_match_info=False):
    matches = pd.read_csv(open_matches_path) 
    points = pd.read_csv(open_points_path) 

    # non_nan_matches = open_matches[open_matches['winner'].isin([1,2])]
    data = []
    dropped_matches = 0

    for match_id in matches['match_id']:
        try:
            t_data, label = get_match_data(match_id, matches, points, soften_curve=False)
            if get_match_info:
                p1 = matches.loc[matches['match_id'] == match_id]['player1'].iloc[0]
                p2 = matches.loc[matches['match_id'] == match_id]['player2'].iloc[0]
                winner = matches.loc[matches['match_id'] == match_id]['winner'].iloc[0]
                data.append([t_data, label, f'{p1} vs {p2} winner was {winner}'])
            else:
                data.append([t_data, label])
        except:
            dropped_matches += 1
    print(f'dropped {dropped_matches} matches')
    return data


data_base_path= 'data/'
def get_data(open_years):
    data = []
    print(open_years)
    for open_year in open_years:
        matches_path = os.path.join(data_base_path, f'{open_year}-matches.csv')
        points_path = os.path.join(data_base_path, f'{open_year}-points.csv')
        data += get_training_data_from_open(matches_path, points_path)
    return data

