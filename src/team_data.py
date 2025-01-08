from nba_api.stats.endpoints import leaguedashteamstats
import pandas as pd

# fetch team stats from the current NBA season
team_stats = leaguedashteamstats.LeagueDashTeamStats(season='2024-25')
data = team_stats.get_data_frames()[0]

data.to_csv("../data/team_data.csv", index=False)
print('csv saved')