import os
from datetime import datetime, time
import subprocess
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE,'data')
NBA  = os.path.join(DATA,'nba_games2.csv')
PRED = os.path.join(DATA,'predictions.csv')

app = FastAPI(title="NBA Predictions")

# serve /data/*
app.mount("/data", StaticFiles(directory=DATA), name="data")

def most_recent_date():
    if not os.path.exists(NBA):
        return None
    df = pd.read_csv(NBA, parse_dates=['date'])
    return df['date'].max().date()

def run_pipeline():
    print(datetime.now(), "→ pipeline start")
    steps = [
      ['python3','scraper-basketball-reference.py'],
      ['python3','parser-basketball-reference.py'],
      ['python3','model.py'],
      ['python3','predict.py'],
    ]
    for cmd in steps:
        subprocess.run(cmd, cwd=BASE, check=True)

@app.on_event("startup")
async def startup_event():
    # on‐launch freshness check
    today = datetime.now().date()
    now   = datetime.now().time()
    last  = most_recent_date()

    if last is None or last != today or (last == today and now > time(11,0)):
        run_pipeline()

    # schedule daily at 11:00 America/New_York
    sched = AsyncIOScheduler(timezone="America/New_York")
    trigger = CronTrigger(hour=11, minute=0)
    sched.add_job(run_pipeline, trigger)
    sched.start()

@app.get("/")
async def health():
    return {"status":"ok","desc":"NBA Predictions backend; CSVs at /data/"}
