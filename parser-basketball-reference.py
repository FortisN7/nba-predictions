### Most code from https://www.youtube.com/watch?v=o6Ih934hADU

# Import packages (Missing optional dependency 'lxml'.  Use pip or conda to install lxml. so python3 -m pip install --user lxml)
import os
import pandas as pd # python3 -m pip install --user pandas
from bs4 import BeautifulSoup
from io import StringIO # Fixes warning message

# Directory where box score HTML files are stored
SCORE_DIR = 'data/scores'

# Get list of all .html files in the scores directory
box_scores = os.listdir(SCORE_DIR)
box_scores = [os.path.join(SCORE_DIR, f) for f in box_scores if f.endswith('.html')]

# Function to parse HTML content into a BeautifulSoup object
def parse_html(box_score):
    with open(box_score, encoding='utf-8') as f:
        html = f.read()
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Remove header rows that mess with table parsing
    [s.decompose() for s in soup.select('tr.over_header')]
    [s.decompose() for s in soup.select('tr.thead')]

    return soup

# Read the final score from the line score table
def read_line_score(soup):
    line_score = pd.read_html(StringIO(str(soup)), attrs={'id': 'line_score'})[0]

    # Fix column names to make things easier
    cols = list(line_score.columns)
    cols[0] = 'team'
    cols[-1] = 'total'
    line_score.columns = cols

    # Only keep team name and total score
    line_score = line_score[['team', 'total']]
    return line_score

# Read stats for a given team and stat type (basic or advanced)
def read_stats(soup, team, stat):
    df = pd.read_html(StringIO(str(soup)), attrs={'id': f'box-{team}-game-{stat}'}, index_col=0)[0]

    # Try converting everything to numeric if possible (non-numeric entries become NaN)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

# Extract the season info from the navigation links at the bottom
def read_season_info(soup):
    nav = soup.select('#bottom_nav_container')[0]
    hrefs = [a['href'] for a in nav.find_all('a')]
    
    # Extract season string from second hyperlink
    season = os.path.basename(hrefs[1]).split('_')[0]
    return season

# Main function to loop through all games and extract data
def main():
    print('Starting parser-basketball-reference.py')

    base_cols = None
    games = []
    
    for box_score in box_scores:
        soup = parse_html(box_score)
        line_score = read_line_score(soup)
        teams = list(line_score['team'])  # Get the two teams that played

        summaries = []
        for team in teams:
            # Get basic and advanced stats
            basic = read_stats(soup, team, 'basic')
            advanced = read_stats(soup, team, 'advanced')

            # Get total stats (last row in both tables)
            totals = pd.concat([basic.iloc[-1, :], advanced.iloc[-1, :]])
            totals.index = totals.index.str.lower()

            # Get max individual player stats (all rows except the last)
            maxes = pd.concat([basic.iloc[:-1, :].max(), advanced.iloc[:-1, :].max()])
            maxes.index = maxes.index.str.lower() + '_max'

            # Combine totals and maxes into one summary row
            summary = pd.concat([totals, maxes])

            # Store base column names on first run (drop 'bpm' to avoid issues)
            if base_cols is None:
                base_cols = list(summary.index.drop_duplicates(keep='first'))
                base_cols = [b for b in base_cols if 'bpm' not in b]

            # Filter to base columns only
            summary = summary[base_cols]
            summaries.append(summary)

        # Combine both teams' summaries into one DataFrame
        summary = pd.concat(summaries, axis=1).T

        # Combine with line scores
        game = pd.concat([summary, line_score], axis=1)

        # Assign home (1) and away (0)
        game['home'] = [0, 1]

        # Create opponent columns by flipping the DataFrame
        game_opp = game.iloc[::-1].reset_index()
        game_opp.columns += '_opp'

        # Merge with opponent data side-by-side
        full_game = pd.concat([game, game_opp], axis=1)

        # Add season info
        full_game['season'] = read_season_info(soup)

        # Parse date from file name (e.g. '20230115_...' becomes datetime object)
        full_game['date'] = os.path.basename(box_score)[:8]
        full_game['date'] = pd.to_datetime(full_game['date'], format='%Y%m%d')

        # Add a boolean column for whether this team won
        full_game['won'] = full_game['total'] > full_game['total_opp']

        # Save game to list
        games.append(full_game)

        # Print progress every 100 games
        if len(games) % 100 == 0:
            print(f'{len(games)} / {len(box_scores)}')

    # Combine all game data into one big DataFrame and write to CSV
    games_df = pd.concat(games, ignore_index=True)
    games_df.to_csv('nba_games.csv')
    print('Output saved to nba_games.csv')

    print('Finished parser-basketball-reference.py')

# Run the script
if __name__ == '__main__':
    main() # Time Estimate: 3 hours