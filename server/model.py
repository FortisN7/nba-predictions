import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Data cleaning
def clean_data(df):
    # sort & drop unused cols
    df = df.sort_values('date').reset_index(drop=True)
    df = df.drop(columns=['mp.1','mp_opp.1','index_opp'], errors=True)

    # vectorized next-game target
    df['target'] = df.groupby('team')['won'].shift(-1)
    df.loc[df['target'].isna(), 'target'] = 2
    df['target'] = df['target'].astype(int)

    # drop any columns still containing nulls
    null_counts = df.isnull().sum()
    to_keep = null_counts[null_counts == 0].index
    return df[to_keep].copy()

# backtest helper
def backtest(data, model, predictors, start=2, step=1):
    all_preds = []
    seasons = sorted(data['season'].unique())
    for i in range(start, len(seasons), step):
        train = data[data['season'] < seasons[i]]
        test  = data[data['season'] == seasons[i]]
        model.fit(train[predictors], train['target'])
        preds = model.predict(test[predictors])
        dfp  = pd.DataFrame({'actual': test['target'], 'prediction': preds}, index=test.index)
        all_preds.append(dfp)
    return pd.concat(all_preds)

# rolling averages on numeric columns
def find_team_averages(team):
    numeric_cols = team.select_dtypes(include=[np.number]).columns
    return team[numeric_cols].rolling(10).mean()

def add_shifted_cols(df, col_names):
    shifted = {}
    for c in col_names:
        shifted[f"{c}_next"] = df.groupby('team')[c].shift(-1)
    return pd.DataFrame(shifted, index=df.index)

# main model creation with only full backtest
def create_model(df):
    rr = RidgeClassifier(alpha=1)
    tscv = TimeSeriesSplit(n_splits=3)
    scaler = MinMaxScaler()

    # initial features (exclude metadata)
    drop0 = ['season','date','won','target','team','team_opp', 'source_file']
    features = [c for c in df.columns if c not in drop0]
    df[features] = scaler.fit_transform(df[features])

    # compute rolling averages and next-game columns
    roll = (
        df[features + ['won','team','season']]
        .groupby(['team','season'], group_keys=False)
        .apply(find_team_averages, include_groups=False)
        .dropna()
    )
    roll.columns = [f"{c}_10" for c in roll.columns]
    shifted = add_shifted_cols(df, ['home','team_opp','date'])
    df2 = pd.concat([df, roll, shifted], axis=1).dropna()

    # merge opponent stats
    opp_stats = df2[[c for c in df2.columns if c.endswith('_10')] + ['team_opp_next','date_next']]
    full = df2.merge(
        opp_stats,
        left_on=['team','date_next'],
        right_on=['team_opp_next','date_next'],
        suffixes=('','_opp')
    )

    # final feature selection on full set
    drop1 = [c for c in full.columns if full[c].dtype == 'object'] + drop0
    feats2 = [c for c in full.columns if c not in drop1]
    sfs2 = SequentialFeatureSelector(rr,
                                     n_features_to_select=30,
                                     direction='forward',
                                     cv=tscv,
                                     n_jobs=1)
    sfs2.fit(full[feats2], full['target'])
    sel2 = [feats2[i] for i, flag in enumerate(sfs2.get_support()) if flag]

    # full backtest
    preds1 = backtest(full, rr, sel2)
    print("Backtest accuracy:", accuracy_score(preds1['actual'], preds1['prediction']))

    # ——— NEW: persist model and feature list ———
    os.makedirs('models', exist_ok=True)
    joblib.dump(rr,  os.path.join('models','ridge_classifier.joblib'))
    joblib.dump(sel2, os.path.join('models','feature_selector.joblib'))
    print("Saved model → models/ridge_classifier.joblib")
    print("Saved features → models/feature_selector.joblib")

# entry point
def main():
    df = pd.read_csv('data/nba_games.csv', index_col=0)
    df = clean_data(df)
    create_model(df)

if __name__ == '__main__':
    main()

'''
$ python3 model.py
Backtest accuracy: 0.6364154528182394

New with data 2016-2025: Backtest accuracy: 0.6393188854489165
New with data 2023-2025: Backtest accuracy: 0.6599118942731278
'''
