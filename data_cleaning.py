import pandas as pd

# --- CONFIGURATION ---
CSV_FILENAME = 'npfl_historical_data.csv'

def text_audit():
    print(f"Loading {CSV_FILENAME}...")
    try:
        df = pd.read_csv(CSV_FILENAME)
    except FileNotFoundError:
        print("❌ Error: CSV file not found.")
        return

    # --- CHECK 1: THE HOME ADVANTAGE MATH ---
    print("\n--- 📊 THE REALITY CHECK ---")
    total_games = len(df)
    
    # Count outcomes
    outcomes = df['Outcome'].value_counts()
    percentages = df['Outcome'].value_counts(normalize=True) * 100
    
    print(f"Total Matches Analyzed: {total_games}")
    print("\nOutcome Probabilities:")
    for result, pct in percentages.items():
        count = outcomes[result]
        print(f"  {result}: {pct:.2f}%  ({count} games)")

    # LOGIC CHECK
    home_win_pct = percentages.get('Home Win', 0)
    if home_win_pct > 65:
        print("\n✅ VERDICT: This looks like the NPFL. (Home Wins > 65%)")
    elif home_win_pct > 45:
        print("\n⚠️ VERDICT: Looks like the Premier League. (Home Wins 45-65%)")
    else:
        print("\n❌ VERDICT: Something is wrong. Home Advantage is too low.")

    # --- CHECK 2: THE NAME LIST ---
    print("\n--- 🔍 TEAM NAME AUDIT ---")
    all_teams = pd.concat([df['Home_Team'], df['Away_Team']]).unique()
    all_teams.sort() # Sorting helps spot "Akwa Utd" next to "Akwa United"
    
    print(f"Found {len(all_teams)} unique team names.")
    print("Scan this list for duplicates:")
    print("-" * 30)
    for team in all_teams:
        print(f"  {team}")
    print("-" * 30)

if __name__ == "__main__":
    text_audit()