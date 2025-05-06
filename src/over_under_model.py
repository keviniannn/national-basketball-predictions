import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def generate_player_model(player_name: str, line: float, stat_expr: str, games=5):
    """
    trains a binary over/under model for a given player and stat expression

    :param player_name: str, full player name like 'LeBron James'
    :param line: float, over/under line to model
    :param stat_expr: str, like 'PTS', 'REB', 'PTS+REB+AST'
    :param season: str, e.g. '2024-25'
    :param games: int, number of games to use for rolling averages
    """
    # load player data
    file_safe_name = player_name.replace(" ", "_")
    player_data = pd.read_csv(f'../data/{file_safe_name}_data.csv')

    # sort chronologically
    player_data['GAME_DATE'] = pd.to_datetime(player_data['GAME_DATE'])
    player_data = player_data.sort_values('GAME_DATE')

    # create target column
    player_data['_TARGET_COMBO'] = player_data.eval(stat_expr)
    player_data['OVER_TARGET'] = (player_data['_TARGET_COMBO'] > line).astype(int)
    player_data.drop(columns=['_TARGET_COMBO'], inplace=True)

    base_features = ['MIN', 'FGA', 'FG3A', 'FTA', 'REB', 'AST', 'TOV']

    # calculate rolling averages
    player_data['PTS_avg'] = player_data['PTS'].rolling(games).mean()
    player_data['MIN_avg'] = player_data['MIN'].rolling(games).mean()
    player_data['FGA_avg'] = player_data['FGA'].rolling(games).mean()
    player_data['FTA_avg'] = player_data['FTA'].rolling(games).mean()
    player_data['AST_avg'] = player_data['AST'].rolling(games).mean()
    player_data['REB_avg'] = player_data['REB'].rolling(games).mean()

    # extract rolling average for components in stat_expr
    stat_parts = [s.strip() for s in stat_expr.split('+')]
    for stat in stat_parts:
        col_name = f"{stat}_avg"
        if col_name not in player_data.columns:
            player_data[col_name] = player_data[stat].rolling(games).mean()

    # build final feature list
    rolling_features = [f"{stat}_avg" for stat in stat_parts]
    features = base_features + rolling_features

    # drop NaNs
    print("Total rows before rolling averages:", len(player_data))
    player_data = player_data.dropna(subset=features + ['OVER_TARGET'])
    print("Rows after dropping NaNs:", len(player_data))

    # split features and target
    X = player_data[features]
    y = player_data['OVER_TARGET']

    # split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # initialize and train random forest classifier
    model = RandomForestClassifier(class_weight='balanced',random_state=42)
    model.fit(X_train, y_train)

    # predict and evaluate model on test set
    y_pred = model.predict(X_test)
    print(f"\naccuracy for {player_name} ({stat_expr} > {line}): {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # save model
    model_path = f"../models/{file_safe_name}_{stat_expr.replace('+', '_')}_model.pkl"
    joblib.dump(model, model_path)
    print(f"model saved to: {model_path}")