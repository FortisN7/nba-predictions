import os
import pandas as pd
import joblib

# load parsed games
df = pd.read_csv('data/nba_games2.csv', parse_dates=['date'])
from model import clean_data, add_shifted_cols, find_team_averages
df = clean_data(df)
# rebuild features just like in model.py
full = find_team_averages(add_shifted_cols(df))
# load
clf = joblib.load(os.path.join('models','ridge_classifier.joblib'))
sel = joblib.load(os.path.join('models','feature_selector.joblib'))
X = full[sel.get_feature_names_out()]

preds = clf.predict(X)
out = full[['date','team','team_opp']].copy()
out['prediction'] = preds
out.to_csv('data/predictions.csv', index=False)
print("Wrote data/predictions.csv")
