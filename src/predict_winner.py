import pandas as pd
import joblib

team_data = pd.read_csv('../data/team_data.csv')
model = joblib.load('../models/nba_prediction_model.pkl')

# function to predict winner between two teams
def predict_winner(team1, team2, team_stats):
    """
    predict the winner between two teams using the pre-trained model.
    
    :param team1: name of the first team (str)
    :param team2: name of the second team (str)
    :param team_stats: DataFrame containing stats for all teams
    
    :return: predicted winner (str)
    """
    # make sure team exists in dataset
    if team1 not in team_stats['TEAM_NAME'].values:
        raise ValueError(f"Team {team1} not found in the dataset.")
    if team2 not in team_stats['TEAM_NAME'].values:
        raise ValueError(f"Team {team2} not found in the dataset.")
    
    # stats for both teams
    team1_stats = team_stats.loc[team_stats['TEAM_NAME'] == team1].iloc[0]
    team2_stats = team_stats.loc[team_stats['TEAM_NAME'] == team2].iloc[0]
    
    # stat differences
    input_data = pd.DataFrame([{
        'PTS_diff': team1_stats['PTS'] - team2_stats['PTS'],
        'REB_diff': team1_stats['REB'] - team2_stats['REB'],
        'AST_diff': team1_stats['AST'] - team2_stats['AST'],
        'STL_diff': team1_stats['STL'] - team2_stats['STL'],
        'BLK_diff': team1_stats['BLK'] - team2_stats['BLK'],
        'TOV_diff': team1_stats['TOV'] - team2_stats['TOV']
    }])

    # predict outcome
    prediction = model.predict(input_data)[0]
    return team1 if prediction == 1 else team2

team1 = "Boston Celtics"
team2 = "Washington Wizards"
try:
    winner = predict_winner(team1, team2, team_data)
    print(f"the predicted winner is: {winner}")
except ValueError as e:
    print(e)