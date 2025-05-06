import pandas as pd
import joblib
from predict_winner import predict_winner

team_data = pd.read_csv('../data/team_data.csv')
model = joblib.load('../models/nba_prediction_model.pkl')

# first-round matches with full team names
first_round_matches = [
    ("New York Knicks", "Boston Celtics"),  # Match 1
    ("Cleveland Cavaliers", "Indiana Pacers"),   # Match 2
    ("Oklahoma City Thunder", "Denver Nuggets"),           # Match 3
    ("Minnesota Timberwolves", "Golden State Warriors")            # Match 4
]

# recursive function to simulate the bracket
def simulate_bracket(matches, team_stats, model, round_name="Quarterfinals"):
    """
    Simulate the entire bracket recursively and print results for each round.

    :param matches: List of tuples, where each tuple is a matchup (team1, team2).
    :param team_stats: DataFrame containing stats for all teams.
    :param model: Trained machine learning model.
    :param round_name: Name of the current round (str).
    :return: Winner of the tournament.
    """
    print(f"\n--- {round_name} ---")
    next_round = []
    
    # predict each match and print the results
    for team1, team2 in matches:
        winner = predict_winner(team1, team2, team_stats)
        print(f"{team1} vs. {team2} -> {winner} wins")
        next_round.append(winner)
    
    # if final round, return the champion
    if len(next_round) == 1:
        return next_round[0]
    
    # generate next round matches by pairing winners
    next_round_matches = [(next_round[i], next_round[i+1]) for i in range(0, len(next_round), 2)]
    
    # determine the next round's name
    next_round_name = {
        4: "Semifinals",
        2: "Final",
    }.get(len(next_round), f"Round of {len(next_round)}")
    
    # recursively simulate the next round
    return simulate_bracket(next_round_matches, team_stats, model, next_round_name)

# predict the NBA Cup winner and output each round
champion = simulate_bracket(first_round_matches, team_data, model)
print(f"\nThe predicted NBA Cup Champion is: {champion}")