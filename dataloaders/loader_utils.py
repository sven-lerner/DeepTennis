import os
import pandas as pd
import random
from dataloaders.valid_data_fields import *
import numpy as np
import logging



'''
data loading utilities, for the most part these guys deal with extracting information from the csvs as well as dealing with some 
'qirks' in the data
'''

def parse_time(time_string):
    # print(time_string)
    hr, m, s = [int(x) for x in time_string.split(':')]
    return 3600*hr + 60 * m + s

# def extract_numpy_from_match(match_points, shuffle_players):
    
#     match_points['PointServer'] = match_points['PointServer'] % 2

#     match_points_copy = match_points.copy()

#     if shuffle_players:
#         match_points['PointServer'] = (match_points['PointServer'] + 1) % 2
#         for p1_val, p2_val in shuffle_pairs:
#             tmp = match_points_copy[p1_val]
#             match_points_copy[p1_val] = match_points_copy[p2_val]
#             match_points_copy[p2_val] = tmp
#     parsed_time = match_points_copy
#     parsed_time['ElapsedTime'] = parsed_time['ElapsedTime'].map(lambda x: parse_time(x))
#     parsed_scores = parsed_time.replace('AD', 55)
    
#     scores = parsed_scores[valid_fields].to_numpy(dtype=np.float)
#     # scores = scores.fillna(0)
#     assert np.sum(np.isnan(scores)) < 1, f"hit a nan {scores}"
#     # scores = scores[~np.isnan(scores).any(axis=1)]
#     return scores


# def populate_sets_to_win(match_points, mens):
#     match_points = match_points.sort_values('PointNumber')
#     net_p1_set_wins = len(match_points.loc[match_points['SetWinner'] == 1])
#     net_p2_set_wins = len(match_points.loc[match_points['SetWinner'] == 2])
        
#     matches_to_win = max(net_p1_set_wins, net_p2_set_wins)
#     p1_set_wins = 0
#     p2_set_wins = 0
#     for i, row in match_points.iterrows():
#         if row['SetWinner'] == 1:
#             p1_set_wins += 1
#         elif row['SetWinner'] == 2:
#             p2_set_wins += 1
        
#         p1_games_to_win =  max((matches_to_win - p1_set_wins) * 6 - max(row['P1GamesWon'], 5), 0)
#         p2_games_to_win =  max((matches_to_win - p2_set_wins) * 6 - max(row['P2GamesWon'], 5), 0)

#         match_points.at[i, 'p1_sets_to_win']= matches_to_win - p1_set_wins
#         match_points.at[i, 'p2_sets_to_win'] = matches_to_win - p2_set_wins

#         match_points.at[i, 'p1_games_to_win']= p1_games_to_win
#         match_points.at[i, 'p2_games_to_win'] = p2_games_to_win

#     return match_points

    
# def get_match_data(match_id, match_data, point_data, prematch_probs, soften_curve):
#     shuffle_players = False #random.uniform(0, 1) > 0.5
#     match_info = match_data.loc[match_data['match_id'] == match_id].iloc[0]
#     winner = match_data.loc[match_data['match_id'] == match_id].iloc[0]['winner'] - 1
#     point_data = point_data.loc[point_data['match_id'] == match_id]

#     if winner not in [0,1,2]:
#         last_point = point_data.sort_values('PointNumber').iloc[-1]
#         last_point_winner = last_point['PointWinner']
#         last_game_winner = last_point['GameWinner']
#         assert last_point_winner == last_game_winner, f'{last_point_winner} was not {last_game_winner}'
#         winner = last_point_winner - 1
#     # print(match_info) 'Men\'s' in match_info['event_name']
#     point_data = populate_sets_to_win(point_data, shuffle_players)

#     if shuffle_players:
#         winner = (winner + 1) % 2
#         prematch_probs = 1 - prematch_probs

#     prematch_probs[np.isnan(prematch_probs)]=0.5

#     parsed_point_data = extract_numpy_from_match(point_data, shuffle_players)
#     num_points = parsed_point_data.shape[0]

#     assert num_points > 0, "dropped match due to no points played"
        
#     y_gt = np.full(shape=num_points, fill_value=winner, dtype='float')

#     if soften_curve:
#         if winner == 1:
#             for i in range(num_points):
#                 if i > 25:
#                     y_gt[i] = y_gt[i] -0.5*(1.0-(i-25)/(num_points-25))
#                 else:
#                     y_gt[i] = 0.5
#         else:
#             for i in range(num_points):
#                 if i > 25:
#                     y_gt[i] = y_gt[i] +0.5*(1.0-(i-25)/(num_points-25))
#                 else:
#                     y_gt[i] = 0.5
    
#     return parsed_point_data, prematch_probs, y_gt

#     def get_prematch_probs(match_id, gollub_prematch_probs):
#         probs = gollub_prematch_probs.iloc[gollub_prematch_probs['match_id'] == match_id][0]
#         return 

# def get_training_data_from_open(open_matches_path, open_points_path, gollub_prematch_path, get_match_info=False):
#     matches = pd.read_csv(open_matches_path) 
#     matches = matches.loc[matches['status'] != 'Retired']
#     # print(len(matches))   
#     points = pd.read_csv(open_points_path) 

#     golub_probs = pd.read_csv(gollub_prematch_path) 

#     # non_nan_matches = open_matches[open_matches['winner'].isin([1,2])]
#     data = []
#     dropped_matches = 0

#     for match_id in matches['match_id']:
#         try:
#             prematch_probs = golub_probs.loc[golub_probs['match_id_x'] == match_id][prematch_fields].iloc[0].to_numpy(dtype=np.float)
#             t_data, prematch_probs, label = get_match_data(match_id, matches, points, prematch_probs, soften_curve=False)
#             if get_match_info:
#                 p1 = matches.loc[matches['match_id'] == match_id]['player1'].iloc[0]
#                 p2 = matches.loc[matches['match_id'] == match_id]['player2'].iloc[0]
#                 winner = matches.loc[matches['match_id'] == match_id]['winner'].iloc[0]
#                 data.append([t_data, prematch_probs, label, f'{p1} vs {p2} winner was {winner}'])
#             else:
#                 data.append([t_data, prematch_probs, label])
#         except Exception as e:
#             # logging.exception(e)
#             dropped_matches += 1
#     print(f'dropped {dropped_matches} matches')
#     return data


# def normalize_data(input_data):
#     stacked_data = np.concatenate([x[0] for x in input_data])
#     print(type(stacked_data[0][0]))
#     print(stacked_data.shape)
#     mean = np.mean(stacked_data, axis=0)
#     print(mean.shape)
#     std = np.std(stacked_data, axis=0)

#     for i, x in enumerate(input_data):
#         normalized = (x[0] - mean) / std
#         input_data[i][0] = normalized
#     return input_data


# data_base_path= 'data/'
# def get_data(open_years, normalize=False):
#     data = []
#     print(open_years)
#     for open_year in open_years:
#         matches_path = os.path.join(data_base_path, f'{open_year}-matches.csv')
#         points_path = os.path.join(data_base_path, f'{open_year}-points.csv')
#         gollub_prematch__path = os.path.join(data_base_path, 'gollubdata', f'gollub-prematch-{open_year}.csv')
#         data += get_training_data_from_open(matches_path, points_path, gollub_prematch__path)
#     if normalize:
#         data = normalize_data(data)
#     return data



data_base_path= 'data/'
def get_final_data(open_years, get_match_info=False, normalize=False):
    data = []
    print(open_years)
    for open_year in open_years:
        parsed_csv_path = os.path.join(data_base_path, 'final_data', f'{open_year}-points-final.csv')
        gollub_prematch__path = os.path.join(data_base_path, 'gollubdata', f'gollub-prematch-{open_year}.csv')
        data += get_training_data_from_parsed_csvs(parsed_csv_path, gollub_prematch__path, get_match_info)
    if normalize:
        data = normalize_data(data)
    return data

def get_training_data_from_parsed_csvs(parsed_points_path, gollub_prematch_path, get_match_info=False):
    all_points = pd.read_csv(parsed_points_path) 

    good_matches = all_points.loc[all_points['winner'] != 0]['match_id'].unique()

    golub_probs = pd.read_csv(gollub_prematch_path) 

    # non_nan_matches = open_matches[open_matches['winner'].isin([1,2])]
    data = []
    dropped_matches = 0

    for match_id in good_matches:
        try:
            match_data = all_points.loc[all_points['match_id'] == match_id]
            prematch_probs = golub_probs.loc[golub_probs['match_id_x'] == match_id][prematch_fields].iloc[0].to_numpy(dtype=np.float)
            t_data, prematch_probs, label = get_parsed_match_data(match_id, match_data, prematch_probs)
            if get_match_info:
                p1 = match_data['player1'].iloc[0]
                p2 = match_data['player2'].iloc[0]
                winner = match_data['winner'].iloc[0]
                data.append([t_data, prematch_probs, label, f'{p1} vs {p2} winner was {winner}, {match_id}'])
            else:
                data.append([t_data, prematch_probs, label])
        except Exception as e:
            # logging.exception(e)
            # print(match_id)
            dropped_matches += 1
    print(f'dropped {dropped_matches} matches')
    return data


def get_parsed_match_data(match_id, point_data, prematch_probs):
    
    winner = point_data.iloc[0]['winner'] - 1
    prematch_probs[np.isnan(prematch_probs)]=0.5

    parsed_point_data = extract_numpy_from_parsed_match(point_data)
    num_points = parsed_point_data.shape[0]

    assert num_points > 0, "dropped match due to no points played"
    y_gt = np.full(shape=num_points, fill_value=winner, dtype='float')    
    return parsed_point_data, prematch_probs, y_gt

def extract_numpy_from_parsed_match(match_points):
    
    match_points_copy = match_points.copy()
    parsed_time = match_points_copy
    # parsed_time['ElapsedTime'] = parsed_time['ElapsedTime'].map(lambda x: parse_time(x))
    parsed_scores = parsed_time.replace('AD', 55)
    
    scores = parsed_scores[valid_fields].to_numpy(dtype=np.float)
    # scores = scores.fillna(0)
    assert np.sum(np.isnan(scores)) < 1, f"hit a nan {scores}"
    # scores = scores[~np.isnan(scores).any(axis=1)]
    return scores
