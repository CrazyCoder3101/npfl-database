import pandas as pd
import numpy as np

# CONFIGURATION
INPUT_FILE = 'npfl_historical_data.csv'
OUTPUT_FILE = 'npfl_training_data.csv'

def add_features():
    print(f"🔄 Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("❌ Error: Input file not found.")
        return

    print("🧠 Calculating Attack & Defense Ratings...")

    # --- STEP 1: CALCULATE GLOBAL AVERAGES (The "League Average") ---
    # We need a baseline. e.g., The average NPFL home team scores 1.5 goals.
    avg_home_goals_league = df['Home_Goals'].mean()
    avg_away_goals_league = df['Away_Goals'].mean()
    
    # --- STEP 2: CALCULATE TEAM AVERAGES ---
    # Group by Home Team to get their Home Stats
    home_stats = df.groupby('Home_Team')[['Home_Goals', 'Away_Goals']].mean()
    home_stats.columns = ['Home_Attack', 'Home_Defense'] # Rename for clarity

    # Group by Away Team to get their Away Stats
    away_stats = df.groupby('Away_Team')[['Home_Goals', 'Away_Goals']].mean()
    away_stats.columns = ['Away_Defense', 'Away_Attack'] # Note the flip: Home Goals = Away Defense weakness
    
    # --- STEP 3: MERGE STATS BACK INTO MATCHES ---
    # For every match, we look up the stats of the teams playing
    
    # Merge Home Stats
    df = df.merge(home_stats, left_on='Home_Team', right_index=True, how='left')
    
    # Merge Away Stats
    df = df.merge(away_stats, left_on='Away_Team', right_index=True, how='left')

    # --- STEP 4: CALCULATE RELATIVE STRENGTH ---
    # We create features that show the MISMATCH.
    # e.g., If Enyimba Home Attack (2.0) plays Pillars Away Defense (1.5)
    
    # Feature 1: Goal Expectancy (Home Team)
    # (Home Attack) vs (Away Defense)
    df['Home_Exp_Goals'] = (df['Home_Attack'] + df['Away_Defense']) / 2
    
    # Feature 2: Goal Expectancy (Away Team)
    # (Away Attack) vs (Home Defense)
    df['Away_Exp_Goals'] = (df['Away_Attack'] + df['Home_Defense']) / 2
    
    # Feature 3: The "Power Diff"
    # Positive number = Home Team is stronger. Negative = Away Team is stronger.
    df['Power_Diff'] = df['Home_Exp_Goals'] - df['Away_Exp_Goals']

    # Rounding for cleanliness
    cols_to_round = ['Home_Attack', 'Home_Defense', 'Away_Attack', 'Away_Defense', 
                     'Home_Exp_Goals', 'Away_Exp_Goals', 'Power_Diff']
    df[cols_to_round] = df[cols_to_round].round(2)

    # --- HANDLING NEW TEAMS (The "Cold Start" Fix) ---
    # If a team is new, they might have NaN (empty) stats. Fill with League Average.
    df.fillna(value={
        'Home_Attack': avg_home_goals_league,
        'Home_Defense': avg_away_goals_league, # Home Defense = resisting Away Goals
        'Away_Attack': avg_away_goals_league,
        'Away_Defense': avg_home_goals_league
    }, inplace=True)

    # 5. SAVE
    print(f"✅ Calculated features for {len(df)} matches.")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Saved smart data to {OUTPUT_FILE}")
    
    # 6. PREVIEW (The "Sanity Check")
    print("\n👀 Preview: Enyimba's Home Strength vs Opponent's Weakness")
    sample = df[df['Home_Team'].str.contains('Enyimba', na=False)].head(5)
    print(sample[['Home_Team', 'Away_Team', 'Outcome', 'Home_Attack', 'Away_Defense', 'Power_Diff']])

if __name__ == "__main__":
    add_features()