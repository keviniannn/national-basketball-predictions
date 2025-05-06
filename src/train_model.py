import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# load data
team_data = pd.read_csv('../data/team_data.csv')
game_data = pd.read_csv('../data/game_data.csv')

# drop duplicates if any exist
game_data = game_data.drop_duplicates()
team_data = team_data.drop_duplicates()

# merge team stats with game data
game_data = game_data.merge(team_data, left_on='TEAM_ID_HOME', right_on='TEAM_ID', suffixes=('', '_HOME'))
game_data = game_data.merge(team_data, left_on='TEAM_ID_AWAY', right_on='TEAM_ID', suffixes=('', '_AWAY'))

# remove duplicate columns
if game_data.columns.duplicated().sum() > 0:
    game_data = game_data.loc[:, ~game_data.columns.duplicated()]

# stat differences
game_data['PTS_diff'] = game_data['PTS_HOME'] - game_data['PTS_AWAY']
game_data['REB_diff'] = game_data['REB_HOME'] - game_data['REB_AWAY']
game_data['AST_diff'] = game_data['AST_HOME'] - game_data['AST_AWAY']
game_data['STL_diff'] = game_data['STL_HOME'] - game_data['STL_AWAY']
game_data['BLK_diff'] = game_data['BLK_HOME'] - game_data['BLK_AWAY']
game_data['TOV_diff'] = game_data['TOV_HOME'] - game_data['TOV_AWAY']

# prepare features
X = game_data[['PTS_diff', 'REB_diff', 'AST_diff', 'STL_diff', 'BLK_diff', 'TOV_diff']]
y = game_data['HOME_WIN']

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# save model to file
joblib.dump(model, '../models/nba_prediction_model.pkl')
print("\nmodel exported to '../models/nba_prediction_model.pkl'")