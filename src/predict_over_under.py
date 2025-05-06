import pandas as pd
import joblib
import os

def predict_over_under(player_name: str, stat_expr: str, line: float, games: int = 10) -> float:
    """
    predicts the probability that a player will go over a stat line in their next game

    :param player_name: str, full player name like 'LeBron James'
    :param stat_expr: str, stat or combo like 'PTS', 'REB+AST', 'PTS+REB+AST'
    :param line: float, the over/under line to evaluate
    :param games: int, how many recent games to use for rolling averages
    :return: float, probability of going over the line
    """
    file_safe_name = player_name.replace(" ", "_")
    data_path = f"../data/{file_safe_name}_data.csv"
    model_path = f"../models/{file_safe_name}_{stat_expr.replace('+', '_')}_model.pkl"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing data file: {data_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")

    # load data
    player_data = pd.read_csv(data_path)
    player_data['GAME_DATE'] = pd.to_datetime(player_data['GAME_DATE'])
    player_data = player_data.sort_values('GAME_DATE')

    # get the most recent N games
    recent_games = player_data.tail(games)
    if len(recent_games) < games:
        raise ValueError(f"Not enough recent games for {player_name} (need at least {games})")
    
    # base features from last game
    last_game = recent_games.iloc[-1]
    base_stats = [
        'MIN', 'FGA', 'FG3A', 'FTA', 'REB', 'AST', 'TOV',
        'FGM', 'FG3M', 'FTM', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
        'OREB', 'DREB'
    ]

    avg_features = {stat: last_game[stat] for stat in base_stats}

    # rolling averages for each stat part
    for stat in list(avg_features.keys()):
        avg_features[f"{stat}_avg"] = recent_games[stat].mean()

    ast = recent_games['AST'].mean()
    tov = recent_games['TOV'].mean()
    avg_features['AST_TOV_ratio_avg'] = ast / (tov + 1e-5)

    # add rolling averages for components of stat_expr
    stat_parts = [s.strip() for s in stat_expr.split('+')]
    for stat in stat_parts:
        avg_features[f"{stat}_avg"] = recent_games[stat].mean()

    # load model
    model = joblib.load(model_path)

    # construct input dataframe with correct feature order
    features = list(model.feature_names_in_)
    input_df = pd.DataFrame([avg_features])[features]

    # make prediction
    probs = model.predict_proba(input_df)[0]

    # handle case where model only trained on one class
    if len(probs) == 1:
        only_class = model.classes_[0]
        prob_over = probs[0] if only_class == 1 else 1 - probs[0]
        print(f"model only trained on one class ({only_class}). using fallback prob: {prob_over:.2f}")
    else:
        prob_over = probs[1]
    label = "OVER" if prob_over > 0.5 else "UNDER"

    print(f"\npredicted: {label} (probability: {prob_over:.2f}) for {player_name} — {stat_expr} > {line}")
    return prob_over