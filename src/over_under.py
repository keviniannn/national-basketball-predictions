from data_generation import fetch_player_data
from over_under_model import generate_player_model
from predict_over_under import predict_over_under

season = '2024-25'
games = 10

player_name = 'julius randle'
stat_expr = 'PTS' # 'PTS+REB+AST'
line = 20.5

force_re = False

full_name = fetch_player_data(player_name=player_name, force_refresh=force_re)

generate_player_model(player_name=full_name, 
                      line=line, 
                      stat_expr=stat_expr,
                      games=games,
                      force_retrain=force_re)

predict_over_under(player_name=full_name,
    stat_expr=stat_expr,
    line=line,
    games=games)