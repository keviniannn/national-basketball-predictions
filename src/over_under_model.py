import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from sklearn.ensemble import RandomForestClassifier # old model
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def generate_player_model(player_name: str, line: float, stat_expr: str, games=10, force_retrain=False):
    """
    trains a binary over/under model for a given player and stat expression

    :param player_name: str, full player name like 'LeBron James'
    :param line: float, over/under line to model
    :param stat_expr: str, like 'PTS', 'REB', 'PTS+REB+AST'
    :param season: str, e.g. '2024-25'
    :param games: int, number of games to use for rolling averages
    """
    file_safe_name = player_name.replace(" ", "_")
    model_path = f"../models/{file_safe_name}_{stat_expr.replace('+', '_')}_model.pkl"

    # if model exists skip
    if os.path.exists(model_path) and not force_retrain:
        print(f"model already exists: {model_path}, skipping retrain.")
        return

    # load player data
    player_data = pd.read_csv(f'../data/{file_safe_name}_data.csv')

    # sort chronologically
    player_data['GAME_DATE'] = pd.to_datetime(player_data['GAME_DATE'])
    player_data = player_data.sort_values('GAME_DATE')

    # create target column
    player_data['_TARGET_COMBO'] = player_data.eval(stat_expr)
    player_data['OVER_TARGET'] = (player_data['_TARGET_COMBO'] > line).astype(int)
    player_data.drop(columns=['_TARGET_COMBO'], inplace=True)

    # set base features
    base_features = [
        'MIN', 'FGA', 'FG3A', 'FTA', 'REB', 'AST', 'TOV',
        'FGM', 'FG3M', 'FTM', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
        'OREB', 'DREB'
    ]

    # assist/turnover ratio
    player_data['AST_TOV_ratio'] = player_data['AST'] / (player_data['TOV'] + 1e-5)

    # compute rolling averages for selected features
    for feature in base_features + ['AST_TOV_ratio']:
        player_data[f'{feature}_avg'] = player_data[feature].rolling(games).mean()

    # extract rolling average for components in stat_expr
    stat_parts = [s.strip() for s in stat_expr.split('+')]
    for stat in stat_parts:
        col_name = f"{stat}_avg"
        if col_name not in player_data.columns:
            player_data[col_name] = player_data[stat].rolling(games).mean()

    # build final feature list
    rolling_features = [f"{stat}_avg" for stat in stat_parts]
    custom_rolling = [f"{feature}_avg" for feature in base_features + ['AST_TOV_ratio']]
    features = pd.Index(base_features + custom_rolling + rolling_features).unique().tolist()

    # drop NaNs
    # print("total rows before rolling averages:", len(player_data))
    player_data = player_data.dropna(subset=features + ['OVER_TARGET'])
    # print("rows after dropping NaNs:", len(player_data))

    # split features and target
    X = player_data[features].apply(pd.to_numeric, errors='coerce')
    X = X.dropna()
    y = player_data.loc[X.index, 'OVER_TARGET']

    # split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train = X_train.select_dtypes(include=['number']).astype('float32')
    X_test = X_test.select_dtypes(include=['number']).astype('float32')
    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]

    # initialize and train random forest classifier
    # model = RandomForestClassifier(class_weight='balanced',random_state=42)
    # model.fit(X_train, y_train)

    # calculate class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # grid search with cross-validation
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'scale_pos_weight': [scale_pos_weight]
    }

    model = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    model = grid.best_estimator_
    print("best parameters:", grid.best_params_)

    # predict and evaluate model on test set
    y_pred = model.predict(X_test)
    print(f"\naccuracy for {player_name} ({stat_expr} > {line}): {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # save model
    model_path = f"../models/{file_safe_name}_{stat_expr.replace('+', '_')}_model.pkl"
    joblib.dump(model, model_path)
    print(f"model saved to: {model_path}")