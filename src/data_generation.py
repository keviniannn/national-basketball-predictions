import os
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, leaguedashteamstats, playergamelog
from nba_api.stats.static import players

def fetch_game_data(season):
    """
    Fetches NBA game data for a given season, processes it to merge home and away stats,
    and saves it as a CSV file.

    :param season: str, the NBA season in "YYYY-YY" format (e.g., "2024-25").
    """
    # fetch game data
    gamelog = leaguegamelog.LeagueGameLog(season=season)
    data = gamelog.get_data_frames()[0]

    # split data into home and away teams
    home_teams = data[data['MATCHUP'].str.contains("vs.")]
    away_teams = data[data['MATCHUP'].str.contains("@")]

    # merge data for the same game based on GAME_ID
    combined = pd.merge(home_teams, away_teams, on='GAME_ID', suffixes=('_HOME',    '_AWAY'))

    # select relevant columns for analysis
    # keep stats for home and away teams, and any additional metadata
    combined = combined[[
        'GAME_ID',
        'TEAM_ID_HOME', 'PTS_HOME', 'REB_HOME', 'AST_HOME', 'STL_HOME', 'BLK_HOME',     'TOV_HOME',
        'TEAM_ID_AWAY', 'PTS_AWAY', 'REB_AWAY', 'AST_AWAY', 'STL_AWAY', 'BLK_AWAY',     'TOV_AWAY',
        'WL_HOME'
    ]]

    # rename the column
    combined = combined.rename(columns={'WL_HOME': 'HOME_WIN'})

    combined['HOME_WIN'] = combined['HOME_WIN'].apply(lambda x: 1 if x == 'W' else  0)

    combined.to_csv("../data/game_data.csv", index=False)
    print('game_data.csv saved')

def fetch_team_data(season):
    """
    Fetches NBA team statistics for a given season and saves them as a CSV file.

    :param season: str, the NBA season in "YYYY-YY" format (e.g., "2024-25").
    """

    # fetch team stats from the current NBA season
    team_stats = leaguedashteamstats.LeagueDashTeamStats(season=season)
    data = team_stats.get_data_frames()[0]

    data.to_csv("../data/team_data.csv", index=False)
    print('team_data.csv saved')

def fetch_player_data(player_name: str, season='2024-25', force_refresh=False) -> str:
    """
    fetch individual nba player data for a given season and save to a csv

    :param season: str
    :param player: str, in the form of an id (Nikola Jokić = '203999')
    :return: player full name
    """
    file_safe_name = player_name.replace(" ", "_")
    file_path = f"../data/{file_safe_name}_data.csv"

    if os.path.exists(file_path) and not force_refresh:
        print(f"using cached csv: {file_path}")
        return file_safe_name
    
    result = players.find_players_by_full_name(player_name)

    player = result[0]
    player_id = player['id']
    full_name = player['full_name']
    file_safe_name = full_name.replace(" ", "_")

    player_stats = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    data = player_stats.get_data_frames()[0]

    data.to_csv(f"../data/{file_safe_name}_data.csv", index=False)
    print(f'{file_safe_name}_data.csv saved')
    
    return file_safe_name

if __name__ == "__main__":
    season = '2024-25'
    player_name = 'lebron'

    fetch_game_data(season=season)
    fetch_team_data(season=season)
    fetch_player_data(season=season, player_name=player_name)