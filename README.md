# NBA Game Outcome Prediction Model

**Author:** Nicholas Fortis  
**Course:** CS485-006  
**Repository:** [github.com/FortisN7/nba-predictions](https://github.com/FortisN7/nba-predictions)

---

## ⚙️ Setup Instructions

### Prerequisites
- Install **Node.js**: [https://nodejs.org/](https://nodejs.org/)
- Install **Python 3**: [https://www.python.org/downloads/](https://www.python.org/downloads/)

### Running the Project (in 2 Git Bash instances)

#### 1️⃣ Backend Setup
```bash
cd server
# If first time running:
python3 -m venv venv
source ./venv/Scripts/activate  # Use 'source ./venv/bin/activate' on Mac/Linux
pip install -r requirements.txt
playwright install

# Create a .env file and add your balldontlie API key:
# .env
# API_KEY=your_api_key_here

npm run start
```

#### 2️⃣ Frontend Setup
```bash
cd client
npm install
npm start
```

---

## 📌 Project Overview

This project aims to build a machine learning pipeline that predicts the outcomes of NBA games. It involves web scraping historical box scores, cleaning and structuring the data, training a RidgeClassifier model, and deploying the final model using a FastAPI backend. A React-based frontend will display daily NBA predictions and evaluate model accuracy against actual results.

---

## 🎯 Objectives

1. Scrape historical NBA box score data from Basketball Reference.  
2. Build and evaluate a RidgeClassifier-based prediction model.  
3. Serve predictions through a FastAPI backend.  
4. Develop a React frontend to:
   - Display current NBA games and predicted winners.
   - Show outcomes of previous games and whether predictions were correct.

---

## 💡 Background & Motivation

As someone passionate about sports and betting, this project merges my interests in sports analytics and machine learning. After speaking with a recruiter from Caesars Sportsbook, I became even more motivated to explore this field professionally.

Resources that inspired this project:

- [Medium Tutorial using RandomForest](https://medium.com/@juliuscecilia33/predicting-nba-game-results-using-machine-learning-and-python-6be209d6d165)
- [Research Paper Comparing ML Models](https://digitalcommons.bryant.edu/cgi/viewcontent.cgi?article=1000&context=honors_data_science)
- [YouTube Scraping Tutorial](https://www.youtube.com/watch?v=o6Ih934hADU)

---

## 🔧 Methodology

### 1. Data Collection
- Scrape NBA box scores and standings using BeautifulSoup and Playwright.
- Source: [Basketball Reference](https://www.basketball-reference.com/)

### 2. Data Processing
- Clean HTML tables and transform data into pandas DataFrames.
- Perform feature engineering and selection.

### 3. Model Training
- Train a RidgeClassifier model initially.
- Compare with other models like Gaussian Naïve Bayes based on performance.

### 4. Backend Development
- Use FastAPI to expose prediction endpoints.
- Load trained model without retraining on every request.

### 5. Frontend Development
- Create a React UI to:
  - Display daily NBA games and predictions.
  - Show previous day’s outcomes and prediction accuracy.

---

## 🗓️ Timeline

| Week         | Goal                                                                 |
|--------------|----------------------------------------------------------------------|
| Mar 25–31     | Scrape data from Basketball Reference                                |
| Apr 1–7       | Train RidgeClassifier, evaluate performance, experiment with models  |
| Apr 8–14      | Develop FastAPI backend and React frontend                           |
| Apr 15–21     | Add prediction display for today’s and past games                    |
| Apr 22–28     | Debug, test end-to-end pipeline                                      |
| Apr 29–May 5  | Final testing, polish UI, prepare presentation                       |

---

## ✅ Progress Update (as of 04/09/2025)

- ✅ Scraped NBA HTML pages (2016–present) using `scraper-basketball-reference.py`
- ✅ Parsed data using `parser-basketball-reference.py`
- ✅ Final dataset stored here: [`nba_games.csv`](https://github.com/FortisN7/nba-predictions/blob/main/nba_games.csv)

Next Steps:
- Train RidgeClassifier model  
- Follow [RidgeClassifier Tutorial](https://www.youtube.com/watch?v=egTylm6C2is)  
- Compare performance with Gaussian Naïve Bayes (from [this paper](https://digitalcommons.bryant.edu/cgi/viewcontent.cgi?article=1000&context=honors_data_science))

**Note:**  
Due to time constraints, the scope has narrowed:
- ❌ Custom game simulations are removed  
- ✅ Focus is on daily game predictions and model performance  
- ✅ May test an additional model for comparison only

---

## 🎯 Expected Outcomes

By the end of this project, I expect to have:

- 📊 A trained NBA game prediction model  
- ⚙️ A FastAPI backend serving real-time predictions  
- 🌐 A React frontend that:
  - Displays today’s predictions
  - Shows previous game outcomes and model accuracy  

This project demonstrates how machine learning and full-stack development can be applied to solve real-world problems in sports analytics.
