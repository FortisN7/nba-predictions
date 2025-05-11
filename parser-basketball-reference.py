import os
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO

SCORE_DIR = 'data/scores'
OUTPUT_CSV = 'nba_games2.csv'

def parse_html(box_score):
    with open(box_score, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    [s.decompose() for s in soup.select('tr.over_header')]
    [s.decompose() for s in soup.select('tr.thead')]
    return soup

def read_line_score(soup):
    df = pd.read_html(StringIO(str(soup)), attrs={'id': 'line_score'})[0]
    cols = list(df.columns)
    cols[0], cols[-1] = 'team', 'total'
    df.columns = cols
    return df[['team', 'total']]

def read_stats(soup, team, stat):
    df = pd.read_html(
        StringIO(str(soup)),
        attrs={'id': f'box-{team}-game-{stat}'},
        index_col=0
    )[0]
    return df.apply(pd.to_numeric, errors='coerce')

def read_season_info(soup):
    nav = soup.select('#bottom_nav_container')[0]
    hrefs = [a['href'] for a in nav.find_all('a')]
    return os.path.basename(hrefs[1]).split('_')[0]

def main():
    print('Starting parser-basketball-reference.py')

    all_files = sorted(f for f in os.listdir(SCORE_DIR) if f.endswith('.html'))
    all_paths = [os.path.join(SCORE_DIR, f) for f in all_files]

    existing_df = None
    master_cols = None

    if os.path.exists(OUTPUT_CSV):
        # read header to get exact column names (including duplicates)
        with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
            header = f.readline().rstrip('\n').split(',')
        master_cols = header[1:]  # drop the index column name

        # read without parsing dates: keep old 'date' strings intact
        existing_df = pd.read_csv(OUTPUT_CSV, index_col=0, dtype=str)

        processed = set(existing_df['source_file'])
        to_process = [p for p in all_paths if p not in processed]
        print(f"Found existing CSV. {len(processed)} processed; {len(to_process)} new.")
    else:
        to_process = all_paths
        print("No existing CSV found; parsing all files.")

    if not to_process:
        print("No new games to parse. Exiting.")
        return

    base_cols = None
    new_games = []
    for idx, box_score in enumerate(to_process, 1):
        soup = parse_html(box_score)
        line_score = read_line_score(soup)
        teams = list(line_score['team'])

        summaries = []
        for team in teams:
            basic    = read_stats(soup, team, 'basic')
            advanced = read_stats(soup, team, 'advanced')
            totals = pd.concat([basic.iloc[-1], advanced.iloc[-1]])
            totals.index = totals.index.str.lower()
            maxes = pd.concat([basic.iloc[:-1].max(), advanced.iloc[:-1].max()])
            maxes.index = maxes.index.str.lower() + '_max'

            if base_cols is None:
                cols = list(pd.concat([totals, maxes]).index.drop_duplicates())
                base_cols = [c for c in cols if 'bpm' not in c]

            summaries.append(pd.concat([totals, maxes])[base_cols])

        summary = pd.concat(summaries, axis=1).T
        game = pd.concat([summary, line_score], axis=1)
        game['home'] = [0, 1]
        opp = game.iloc[::-1].reset_index()
        opp.columns = [f"{c}_opp" for c in opp.columns]
        full = pd.concat([game, opp], axis=1)

        full['season']      = read_season_info(soup)
        full['date']        = pd.to_datetime(os.path.basename(box_score)[:8], format='%Y%m%d')
        full['won']         = full['total'] > full['total_opp']
        full['source_file'] = box_score

        new_games.append(full)
        if idx % 100 == 0 or idx == len(to_process):
            print(f"Parsed {idx}/{len(to_process)}")

    new_df = pd.concat(new_games, ignore_index=True)
    # convert new dates to 'YYYY-MM-DD' strings
    new_df['date'] = new_df['date'].dt.strftime('%Y-%m-%d')

    if existing_df is not None:
        # stack existing and new as raw numpy arrays to preserve duplicate columns
        existing_vals = existing_df.to_numpy()
        new_vals      = new_df.to_numpy()
        all_vals = np.vstack([existing_vals, new_vals])
        combined = pd.DataFrame(all_vals, columns=master_cols)
    else:
        combined = new_df

    combined.to_csv(OUTPUT_CSV)
    print(f"Appended {len(new_df)} new games. Saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
