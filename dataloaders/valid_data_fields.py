valid_fields = {
# 'ElapsedTime',
# 'SetNo',
'P1GamesWon',
'P2GamesWon',
# 'SetWinner',
# 'GameNo',
# 'GameWinner',
# 'PointNumber',
# 'PointWinner',
# 'PointServer',
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
# 'Speed_MPH',
# 'P1BreakPointMissed',
# 'P2BreakPointMissed',
# 'ServeIndicator',
# 'P1TurningPoint', 
# 'P2TurningPoint'
}

prematch_fields = {
    'logit_elo_538_prob',
    'logit_elo_prob',
    'logit_elo_diff_prob',
    'logit_elo_diff_538_prob',
    'match_prob_kls',
    'match_prob_kls_JS',
    'match_prob_sf_kls',
    'match_prob_sf_kls_JS',
    'match_prob_adj_kls',
    'match_prob_adj_kls_JS',
    'elo_prob',
    'elo_prob_538',
    'sf_elo_prob',
    'sf_elo_prob_538'
}

#usually the better seed is player 1, so we randomly shuffle who is p1 vs p2 during training so the 
#net does not develop a bias towards seeding
shuffle_pairs = {
    ('P1GamesWon','P2GamesWon'),
    ('P1BreakPointMissed','P2BreakPointMissed'),
    ('P1Score', 'P2Score'),
    ('P1Momentum', 'P2Momentum'),
    ('P1PointsWon', 'P2PointsWon'),
    ('P1Ace', 'P2Ace'),
    ('P1Winner', 'P2Winner'),
    ('P1DoubleFault', 'P2DoubleFault'),
    ('P1UnfErr', 'P2UnfErr'),
    ('P1NetPoint', 'P2NetPoint'),
    ('P1NetPointWon', 'P2NetPointWon'),
    ('P1BreakPoint', 'P2BreakPoint'),
    ('P1BreakPointWon', 'P2BreakPointWon'),
    ('P1FirstSrvIn', 'P2FirstSrvIn'),
    ('P1FirstSrvWon', 'P2FirstSrvWon'),
    ('P1SecondSrvIn', 'P2SecondSrvIn'), 
    ('P1SecondSrvWon', 'P2SecondSrvWon'),
    ('P1ForcedError', 'P2ForcedError'),
    ('P1TurningPoint', 'P2TurningPoint')
}