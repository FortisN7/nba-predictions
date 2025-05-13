"""
Pipeline: scrape, parse, model refresh, then predict and append new games to predictions.csv.
If predictions.csv missing: predict past 7 days.
"""
import os
import sys
import subprocess
import requests
import pandas as pd
import numpy as np
import joblib
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from scraper_basketball_reference import main as scrape_main  # still used if running stand‐alone
from parser_basketball_reference import main as parse_main
from model import main as model_main, clean_data, find_team_averages, add_shifted_cols

# On Windows, switch to selector event loop for any internal asyncio use
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Base paths & interpreter
BASE      = os.path.dirname(__file__)
PYTHON    = sys.executable

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR      = 'data'
CSV_GAMES     = os.path.join(DATA_DIR, 'nba_games.csv')
CSV_PREDS     = os.path.join(DATA_DIR, 'predictions.csv')
MODEL_PATH    = 'models/ridge_classifier.joblib'
FEATURES_PATH = 'models/feature_selector.joblib'

# ── Load persisted model & features ─────────────────────────────────────────────
model    = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

# ── Balldontlie API config ─────────────────────────────────────────────────────
load_dotenv()
API_KEY   = os.getenv('API_KEY')
HEADERS   = {'Authorization': f'Bearer {API_KEY}'}
GAMES_API = "https://api.balldontlie.io/v1/games"

# ── Feature-building for live games ─────────────────────────────────────────────
def build_live_features(df_raw):
    df_hist = pd.read_csv(CSV_GAMES, index_col=0)
    df_hist = clean_data(df_hist)

    raw_cols = [c for c in features if c in df_hist.columns and not c.endswith('_10')]
    raw_hist = df_hist[['team','date'] + raw_cols]
    raw_last = raw_hist.sort_values(['team','date']).groupby('team', as_index=False).last()

    roll = df_hist.groupby(['team','season'], group_keys=False).apply(find_team_averages)
    roll.columns = [f"{c}_10" for c in roll.columns]
    roll_hist = pd.concat([df_hist[['team','date']], roll], axis=1)
    roll_last = roll_hist.sort_values(['team','date']).groupby('team', as_index=False).last().drop(columns=['date'])

    shifts = add_shifted_cols(df_hist, ['home','team_opp','date'])
    shifts_hist = pd.concat([df_hist[['team','date']], shifts], axis=1)
    shifts_last = shifts_hist.sort_values(['team','date']).groupby('team', as_index=False).last().drop(columns=['date'])

    team_stats = (
        raw_last
        .merge(roll_last,   on='team', how='left')
        .merge(shifts_last, on='team', how='left')
    )

    opp_roll_last = roll_last.rename(
        columns={'team':'team_opp', **{c:f"{c}_opp" for c in roll_last.columns if c!='team'}}
    )
    opp_raw_last = raw_last.rename(
        columns={'team':'team_opp', **{c:f"{c}_opp" for c in raw_last.columns if c not in ('team','date')}}
    )

    return (
        df_raw
        .merge(team_stats,   on='team',     how='left')
        .merge(opp_raw_last, on='team_opp', how='left')
        .merge(opp_roll_last,on='team_opp', how='left')
    ).fillna(0.0)

# ── Fetch upcoming games via API ────────────────────────────────────────────────
def fetch_games_by_date(date_str):
    try:
        resp = requests.get(
            GAMES_API,
            params={'dates[]': date_str, 'per_page': 100},
            headers=HEADERS,
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json().get('data', [])
    except Exception as e:
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

# ── Predict & append new games ─────────────────────────────────────────────────
def predict_and_append():
    today = datetime.now().date()
    if not os.path.exists(CSV_PREDS):
        start, end = today - timedelta(days=29), today
    else:
        preds     = pd.read_csv(CSV_PREDS, parse_dates=['date'])
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
        for feat in features:
            if feat not in df_live.columns:
                df_live[feat] = 0.0

        df_new['prediction'] = model.predict(df_live[features])
        all_out.append(df_new[['date','team','team_opp','prediction']])

    if not all_out:
        print('No games found in range.')
        return

    df_append = pd.concat(all_out, ignore_index=True)
    if os.path.exists(CSV_PREDS):
        df_existing = pd.read_csv(CSV_PREDS)
        df_out      = pd.concat([df_existing, df_append], ignore_index=True)
    else:
        df_out = df_append

    df_out.to_csv(CSV_PREDS, index=False)
    print(f'Predicted and saved {len(df_append)} games from {start} to {end}.')

# ── FastAPI endpoints & scheduler ─────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def start_scheduler():
    # Run scraper in separate process, then in-process parse→model→predict
    await run_pipeline()

    scheduler = AsyncIOScheduler(timezone="America/New_York")
    scheduler.add_job(run_pipeline, CronTrigger(hour=11, minute=0))
    scheduler.start()

@app.post("/run-pipeline")
async def manual_trigger():
    await run_pipeline()
    return {"status": "pipeline completed"}

@app.get("/predictions/last7")
def get_last7_predictions():
    df = pd.read_csv(CSV_PREDS, parse_dates=['date']).sort_values('date')
    dates = df['date'].dt.strftime('%Y-%m-%d').unique()[-7:]
    return df[df['date'].dt.strftime('%Y-%m-%d').isin(dates)].to_dict(orient='records')

@app.get("/games/last7")
def get_last7_games():
    df = pd.read_csv(CSV_GAMES, parse_dates=['date'])
    today = pd.Timestamp.now().normalize()
    df = df[df['date'] <= today].sort_values('date')
    dates = df['date'].dt.strftime('%Y-%m-%d').unique()[-7:]
    return df[df['date'].dt.strftime('%Y-%m-%d').isin(dates)].replace({np.nan: None}).to_dict(orient='records')

# ── Async pipeline so scraper runs in its own process ─────────────────────────
async def run_pipeline():
    subprocess.run([PYTHON, 'scraper_basketball_reference.py'], cwd=BASE, check=True)
    parse_main()
    model_main()
    predict_and_append()

if __name__ == '__main__':
    asyncio.run(run_pipeline())
