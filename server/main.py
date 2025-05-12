"""
Pipeline: scrape, parse, model refresh, then predict and append new games to predictions.csv.
If predictions.csv missing: predict past 7 days.
"""
import os
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from scraper_basketball_reference import main as scrape_main
from parser_basketball_reference import main as parse_main
from model import main as model_main, clean_data, find_team_averages, add_shifted_cols
from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Paths
data_dir      = 'data'
csv_games     = os.path.join(data_dir, 'nba_games.csv')
csv_preds     = os.path.join(data_dir, 'predictions.csv')
model_path    = 'models/ridge_classifier.joblib'
features_path = 'models/feature_selector.joblib'

# Balldontlie API configuration
load_dotenv()
API_KEY   = os.getenv('API_KEY')
headers   = {'Authorization': f'Bearer {API_KEY}'}
games_api = "https://api.balldontlie.io/v1/games"

# Load persisted model & features
model    = joblib.load(model_path)
features = joblib.load(features_path)

def build_live_features(df_raw):
    # 1) history
    df_hist = pd.read_csv(csv_games, index_col=0)
    df_hist = clean_data(df_hist)

    # 2) raw stats (team and opponent)
    #    — now includes every non-_10 column, so opponent fields come through
    raw_cols = [
        c for c in features
        if c in df_hist.columns and not c.endswith('_10')
    ]
    raw_hist = df_hist[['team','date'] + raw_cols]
    raw_last = (
        raw_hist.sort_values(['team','date'])
                .groupby('team', as_index=False)
                .last()
    )

    # 3) rolling-10 for team only
    roll = (
        df_hist
        .groupby(['team','season'], group_keys=False)
        .apply(find_team_averages)
    )
    roll.columns = [f"{c}_10" for c in roll.columns]
    roll_hist = pd.concat([df_hist[['team','date']], roll], axis=1)
    roll_last = (
        roll_hist.sort_values(['team','date'])
                 .groupby('team', as_index=False)
                 .last()
                 .drop(columns=['date'])
    )

    # 4) next-game flags
    shifts = add_shifted_cols(df_hist, ['home','team_opp','date'])
    shifts_hist = pd.concat([df_hist[['team','date']], shifts], axis=1)
    shifts_last = (
        shifts_hist.sort_values(['team','date'])
                    .groupby('team', as_index=False)
                    .last()
                    .drop(columns=['date'])
    )

    # 5) merge team stats
    team_stats = (
        raw_last
        .merge(roll_last,   on='team', how='left')
        .merge(shifts_last, on='team', how='left')
    )

    # 6) prepare opponent rolling-10 (rename team→team_opp)
    opp_roll_last = roll_last.rename(
        columns={'team':'team_opp',
                 **{c:f"{c}_opp" for c in roll_last.columns if c!='team'}}
    )

    # 7) prepare opponent raw-last (also rename)
    opp_raw_last = raw_last.rename(
        columns={'team':'team_opp',
                 **{c:f"{c}_opp" for c in raw_last.columns if c not in ('team','date')}}
    )

    # 8) final merge
    live = (
        df_raw
        .merge(team_stats,   on='team',     how='left')
        .merge(opp_raw_last, on='team_opp', how='left')
        .merge(opp_roll_last,on='team_opp', how='left')
    )

    # 9) fill any holes
    return live.fillna(0.0)

def fetch_games_by_date(date_str):
    try:
        resp = requests.get(
            games_api,
            params={'dates[]': date_str, 'per_page': 100},
            headers=headers,
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json().get('data', [])
    except (requests.HTTPError, requests.RequestException, ValueError) as e:
        print(f"Error fetching games for {date_str}: {e}")
        return pd.DataFrame()

    rows = []
    for g in data:
        rows.append({
            'season':   g['season'],
            'date':     date_str,
            'home':     1,
            'team':     g['home_team']['abbreviation'],
            'team_opp': g['visitor_team']['abbreviation'],
        })
    return pd.DataFrame(rows)

def predict_and_append():
    today = datetime.now().date()
    if not os.path.exists(csv_preds):
        start = today - timedelta(days=29)
        end   = today
    else:
        preds     = pd.read_csv(csv_preds, parse_dates=['date'])
        last_pred = preds['date'].max().date()
        start     = last_pred + timedelta(days=1)
        end       = today
    if start > end:
        print('No new games to predict.')
        return

    all_out = []
    for n in range((end - start).days + 1):
        date_str = (start + timedelta(days=n)).strftime('%Y-%m-%d')
        df_new   = fetch_games_by_date(date_str)
        if df_new.empty:
            continue

        df_live = build_live_features(df_new)

        # ensure all trained features exist
        for feat in features:
            if feat not in df_live.columns:
                df_live[feat] = 0.0

        X = df_live[features]

        df_new['prediction'] = model.predict(X)
        all_out.append(df_new[['date','team','team_opp','prediction']])

    if not all_out:
        print('No games found in range.')
        return

    df_append = pd.concat(all_out, ignore_index=True)
    if os.path.exists(csv_preds):
        df_existing = pd.read_csv(csv_preds)
        df_out      = pd.concat([df_existing, df_append], ignore_index=True)
    else:
        df_out = df_append

    df_out.to_csv(csv_preds, index=False)
    print(f'Predicted and saved {len(df_append)} games from {start} to {end}.')

app = FastAPI()

# allow your React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or "*" for any
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def start_scheduler():
    run_pipeline()
    scheduler = AsyncIOScheduler()
    scheduler.add_job(run_pipeline, 'cron', hour=11, minute=0)
    scheduler.start()

@app.post("/run-pipeline")
def manual_trigger():
    run_pipeline()
    return {"status": "pipeline completed"}

@app.get("/predictions/last7")
def get_last7_predictions():
    # load and sort
    df = pd.read_csv(csv_preds, parse_dates=['date'])
    df = df.sort_values('date')

    # take last 7 unique dates
    dates = df['date'].dt.strftime('%Y-%m-%d').unique()[-7:]
    df7 = df[df['date'].dt.strftime('%Y-%m-%d').isin(dates)]

    # return as list of dicts
    return df7.to_dict(orient='records')


@app.get("/games/last7")
def get_last7_games():
    df = pd.read_csv(csv_games, parse_dates=['date'])
    # only past or today’s games
    today = pd.Timestamp.now().normalize()
    df = df[df['date'] <= today]
    df = df.sort_values('date')

    dates = df['date'].dt.strftime('%Y-%m-%d').unique()[-7:]
    df7 = df[df['date'].dt.strftime('%Y-%m-%d').isin(dates)]

    df7 = df7.replace({np.nan: None})

    return df7.to_dict(orient='records')

def run_pipeline():
    #scrape_main()
    #parse_main()
    #model_main()
    predict_and_append()

if __name__ == '__main__':
    run_pipeline()
