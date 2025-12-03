import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import difflib

# CONFIGURATION
TRAINING_FILE = 'npfl_training_data.csv'

def load_and_train():
    print("🧠 Loading Smart Data and Training Brain...")
    try:
        df = pd.read_csv(TRAINING_FILE)
    except FileNotFoundError:
        print(f"❌ Error: {TRAINING_FILE} not found. Run feature_engineering.py first!")
        return None, None, None

    # 1. PREPARE FEATURES
    # We are no longer using "Team Name" (ID). We are using "Team Strength".
    # Features: [Home_Attack, Home_Defense, Away_Attack, Away_Defense]
    feature_cols = ['Home_Attack', 'Home_Defense', 'Away_Attack', 'Away_Defense']
    
    X = df[feature_cols]
    y = df['Outcome']

    # 2. TRAIN MODEL (Using Random Forest for better complexity handling)
    # Random Forest is better at finding non-linear patterns than Logistic Regression
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 3. BUILD STATS LOOKUP TABLE
    # We need to know the stats for every team so we can predict future games.
    # Since the stats are constant in our current file, we just grab the first row for each team.
    
    team_stats = {}
    
    # Get Home Stats
    home_group = df.groupby('Home_Team')[['Home_Attack', 'Home_Defense']].first()
    for team, row in home_group.iterrows():
        if team not in team_stats: team_stats[team] = {}
        team_stats[team]['Home_Attack'] = row['Home_Attack']
        team_stats[team]['Home_Defense'] = row['Home_Defense']

    # Get Away Stats
    away_group = df.groupby('Away_Team')[['Away_Attack', 'Away_Defense']].first()
    for team, row in away_group.iterrows():
        if team not in team_stats: team_stats[team] = {}
        team_stats[team]['Away_Attack'] = row['Away_Attack']
        team_stats[team]['Away_Defense'] = row['Away_Defense']

    teams_list = list(team_stats.keys())
    return model, team_stats, teams_list

def get_closest_team(user_input, team_list):
    matches = difflib.get_close_matches(user_input, team_list, n=1, cutoff=0.4)
    return matches[0] if matches else None

def main():
    model, team_stats, team_list = load_and_train()
    if not model: return

    print("\n" + "="*50)
    print("🚀 NPFL ADVANCED PREDICTOR (STATS ENGINE) 🚀")
    print("="*50)
    print(f"Loaded Stats for {len(team_list)} teams.")
    print("Type 'done' to exit.\n")

    while True:
        print("-" * 30)
        h_input = input("🏠 Home Team: ").strip()
        if h_input.lower() == 'done': break
        
        home = get_closest_team(h_input, team_list)
        if not home: 
            print("❌ Team not found."); continue
        print(f"   Selected: {home}")

        a_input = input("✈️ Away Team: ").strip()
        away = get_closest_team(a_input, team_list)
        if not away: 
            print("❌ Team not found."); continue
        print(f"   Selected: {away}")

        # RETRIEVE STATS
        try:
            h_att = team_stats[home]['Home_Attack']
            h_def = team_stats[home]['Home_Defense']
            a_att = team_stats[away]['Away_Attack']
            a_def = team_stats[away]['Away_Defense']
            
            # Predict
            features = pd.DataFrame([[h_att, h_def, a_att, a_def]], 
                                  columns=['Home_Attack', 'Home_Defense', 'Away_Attack', 'Away_Defense'])
            
            probs = model.predict_proba(features)[0]
            classes = model.classes_
            result_probs = dict(zip(classes, probs))
            
            winner = max(result_probs, key=result_probs.get)
            conf = result_probs[winner]

            # CALCULATE EXPECTED GOALS (For "Why")
            exp_home_goals = (h_att + a_def) / 2
            exp_away_goals = (a_att + h_def) / 2
            
            print(f"\n📊 MATCH ANALYSIS:")
            print(f"   {home} Attack Rating: {h_att:.2f}")
            print(f"   {away} Defense Rating: {a_def:.2f}")
            print(f"   Expected Score: {home} {exp_home_goals:.1f} - {exp_away_goals:.1f} {away}")
            
            print(f"\n🔮 PREDICTION: {winner} ({conf:.1%})")
            
        except KeyError as e:
            print(f"❌ Error: Missing stats for one of these teams. (Maybe they haven't played enough games yet?)")

if __name__ == "__main__":
    main()