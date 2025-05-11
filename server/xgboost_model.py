import time
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
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

# create next-game shift columns
def add_shifted_cols(df, col_names):
    shifted = {}
    for c in col_names:
        shifted[f"{c}_next"] = df.groupby('team')[c].shift(-1)
    return pd.DataFrame(shifted, index=df.index)

# main model creation with hyperparameter tuning and full backtest
def create_model(df):
    # prepare features
    drop0 = ['season','date','won','target','team','team_opp']
    features = [c for c in df.columns if c not in drop0]

    # scale features
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    # hyperparameter tuning for XGBoost (multi-class softmax)
    xgb = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        verbosity=0,
        random_state=42
    )
    tscv = TimeSeriesSplit(n_splits=3)
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    rnd_search = RandomizedSearchCV(
        xgb, param_dist,
        n_iter=20,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    rnd_search.fit(df[features], df['target'])
    best_xgb = rnd_search.best_estimator_
    print("Best hyperparameters:", rnd_search.best_params_)
    print("CV accuracy (tuning):", rnd_search.best_score_)

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

    # feature selection using tuned XGBoost
    drop1 = [c for c in full.columns if full[c].dtype == 'object'] + drop0
    feats2 = [c for c in full.columns if c not in drop1]
    sfs = SequentialFeatureSelector(
        best_xgb,
        n_features_to_select=30,
        direction='forward',
        cv=tscv,
        n_jobs=-1
    )
    sfs.fit(full[feats2], full['target'])
    sel_feats = [feats2[i] for i, flag in enumerate(sfs.get_support()) if flag]

    # full backtest
    preds = backtest(full, best_xgb, sel_feats)
    acc = accuracy_score(preds['actual'], preds['prediction'])
    print("Final backtest accuracy:", acc)
    return acc

# entry point
def main():
    df = pd.read_csv('nba_games.csv', index_col=0)
    df = clean_data(df)

    start = time.perf_counter()
    create_model(df)
    end = time.perf_counter()
    print(f"Total runtime: {(end - start)/60:.1f} minutes")

if __name__ == '__main__':
    main()

'''
$ python3 xgboost_model.py
Fitting 3 folds for each of 20 candidates, totalling 60 fits
Best hyperparameters: {'subsample': 0.6, 'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.05, 'colsample_bytree': 0.8}
CV accuracy (tuning): 0.5431563114659262
Final backtest accuracy: 0.628435718809373
Total runtime: 66.1 minutes
'''
# No good !