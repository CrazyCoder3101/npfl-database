NPFL Match Outcome Prediction (Project "NaijaBallboy")
📌 Project Overview
This project aims to build a Machine Learning model tailored specifically for the Nigeria Professional Football League (NPFL). Unlike generic global models, this system is designed to account for the unique hyper-local variables of the Nigerian game, specifically the statistically significant Home Advantage.
Goal: To predict match outcomes (Home Win / Draw / Away Win) with higher accuracy than a random baseline, eventually incorporating variables like travel distance and form.
📅 Project Log & Status
Current Date: Saturday, Nov 29, 2025
Current Version: v0.1 (Baseline)
Accomplishments (Nov 28 - Nov 29)
1. Data Extraction:
   * Built a custom Python scraper (npfl_scraper.py) using requests and pandas.
   * Successfully scraped historical match results from Wikipedia for seasons: 2021/22, 2022/23 (Abridged), 2023/24, and 2024/25.
   * Total Dataset: ~1,320 Matches.
   * Note: Wikipedia scraping required a "User-Agent" header to bypass 403 Forbidden errors.
2. Data Cleaning & Audit:
   * Performed an audit on team names to resolve inconsistencies (e.g., merging "Enyimba" and "Enyimba Int'l").
   * Identified 26 unique teams across the 4-year period.
   * Statistical Reality Check: Confirmed a massive Home Advantage in the dataset (Home Win rate sits between 65-75%, significantly higher than European leagues).
3. Baseline Modeling:
   * Implemented a Logistic Regression model using scikit-learn.
   * Features: Home Team ID, Away Team ID.
   * Baseline Accuracy: 68.56%.
   * Finding: The model currently relies heavily on the "Home Win" bias. To improve this, we must introduce dynamic features (Form, Goals Scored).
📂 File Structure & Usage
1. npfl_scraper.py
Purpose: Scrapes raw match data from Wikipedia.
* Method: Scans for square results matrices (Teams x Teams) and "melts" them into a list of matches.
* Key Tech: Uses iloc (positional indexing) to handle abbreviated column headers.
* Output: npfl_historical_data.csv
2. data_cleaning.py
Purpose: Audits the CSV for errors.
* Function: Checks for duplicate team names and prints the win/draw/loss percentages to ensure the data aligns with reality.
3. first_model.py
Purpose: The training ground.
* Function: Loads data, encodes Team Names into Integers (0, 1, 2...), splits into Train/Test sets (80/20), and calculates the Model Accuracy Score.
4. power_rankings.py
Purpose: Visual Analysis.
* Function: Extracts the "Coefficients" from the Logistic Regression model to determine which teams are mathematically the strongest at home.
* Output: Generates npfl_rankings.png (Bar chart of team strength).
5. predict_matchday.py
Purpose: The User Interface.
* Function: Interactive console tool. Allows the user to input specific fixtures (e.g., "Remo vs Enyimba") and returns the probability of Home/Draw/Away. Includes a typo-fixer using difflib.
🛠️ Installation & Requirements
To run these scripts, the following Python libraries are required (installed via Miniconda):
pip install pandas requests lxml html5lib scikit-learn matplotlib seaborn

🚀 Next Steps (Phase 2)
1. Feature Engineering: Implement "Rolling Averages" (Form Guide) to track the last 5 games for each team.
2. Attack/Defense Metrics: Calculate average goals scored/conceded to predict Draws better.
3. Cold Start Handling: Create a generic profile for newly promoted teams (e.g., Ikorodu City) to prevent skewing stats due to small sample sizes.
