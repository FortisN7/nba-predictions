# nba-predictions
Project for CS485-006 that aims to create a prediction model for NBA games

# CS485006 Project Proposal - Nicholas Fortis

## Title: NBA Game Outcome Prediction Model

## Description
This project aims to predict NBA game outcomes using machine learning. The workflow includes web scraping NBA box scores, cleaning and structuring the data, performing feature selection, and training a RidgeClassifier model. The final model will be served via a FastAPI backend, and a React-based frontend will allow users to simulate their own games and view daily NBA game predictions. The goal is to create an end-to-end pipeline that processes NBA data, trains an ML model, and provides real-time predictions through an interactive web interface.

## Background & Motivation
I’ve always been into sports and sports betting, so this project is really exciting for me. I even talked to a recruiter from Caesars Sportsbook, so working in this field could be a real option for me. Predicting NBA games is a mix of data science, machine learning, and sports, which makes it a great fit for my interests.

There are a lot of different ways to predict game outcomes. One tutorial I found ([link](https://medium.com/@juliuscecilia33/predicting-nba-game-results-using-machine-learning-and-python-6be209d6d165)) uses a RandomForest model to predict NBA winners. Another study ([link](https://digitalcommons.bryant.edu/cgi/viewcontent.cgi?article=1000&context=honors_data_science)) looks at different models like logistic regression, XGB Classifier, and Gaussian Naïve Bayes to see which works best. I’ll be using ideas from both of these to build my own model.

For data, I’ll scrape NBA stats from [Basketball Reference](https://www.basketball-reference.com/), which has tons of historical game data. The goal is to train a solid prediction model and build a website where people can check daily NBA predictions and even create their own matchups to see who would win.

## Objectives
- Scrape NBA game historical data.
- Develop a machine learning model (RidgeClassifier) to predict NBA game outcomes using historical box scores.
- Build a FastAPI backend that serves predictions efficiently without retraining the model on every request.
- Develop a React frontend that allows users to view daily NBA predictions and create custom matchups for simulation.

## Methodology
- **Data Collection:** Scrape NBA box scores and standings from Basketball Reference using BeautifulSoup.
- **Data Processing:** Clean and structure the data into pandas DataFrames, selecting relevant features for prediction.
- **Model Training:** Use RidgeClassifier for initial predictions and experiment with other models for accuracy improvements.
- **Backend Development:** Load the trained model in a FastAPI backend, exposing an API for predictions.
- **Frontend Development:** Create a React-based UI that allows users to input team matchups, view predictions, and see daily NBA game predictions.

## Timeline

| Week         | Goal |
|-------------|------|
| March 3-10  | Set up project repo, scrape and store NBA box scores |
| March 11-17 | Clean and structure data, perform feature selection |
| March 18-24 | Train initial RidgeClassifier model and evaluate accuracy |
| March 25-31 | Experiment with other ML models and tune hyperparameters |
| April 1-7   | Set up FastAPI backend, load trained model, and test API |
| April 8-14  | Develop React frontend, integrate with FastAPI backend |
| April 15-21 | Add ability for users to simulate fake matchups |
| April 22-28 | Test full pipeline, debug errors, optimize performance |
| April 29-May 5 | Final testing, polish UI, and prepare presentation |

## Expected Outcomes
By the end of the project, I expect to have:
- A trained machine learning model capable of predicting NBA game outcomes.
- A functional FastAPI backend that serves predictions efficiently.
- A React-based frontend where users can view daily NBA game predictions and simulate matchups.

This project will provide a practical demonstration of using data science, machine learning, and full-stack development to solve a real-world problem in sports analytics.
