import pandas as pd
import numpy as np
import re
import requests
from io import StringIO

# CONFIGURATION
MASTER_FILE = 'npfl_historical_data.csv'
NEW_SEASON_URL = "https://en.wikipedia.org/wiki/2025%E2%80%9326_Nigeria_Premier_Football_League"
NEW_SEASON_LABEL = "2025-26"

def scrape_new_season():
    print(f"🌍 Connecting to Wikipedia ({NEW_SEASON_LABEL})...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(NEW_SEASON_URL, headers=headers)
        response.raise_for_status()
        
        # Read tables
        tables = pd.read_html(StringIO(response.text))
        print(f"   Found {len(tables)} tables.")

        matches = []
        
        # Search for the Results Matrix
        for df in tables:
            # Matrix check: Square-ish shape > 8x8
            if df.shape[0] > 8 and df.shape[1] > 8:
                if abs(df.shape[0] - df.shape[1]) < 3:
                    
                    # Found the matrix
                    results_matrix = df.set_index(df.columns[0])
                    teams = results_matrix.index.tolist()
                    
                    # Iterate (Home vs Away)
                    rows_count = len(teams)
                    cols_count = len(results_matrix.columns)
                    n_teams = min(rows_count, cols_count)

                    for r in range(n_teams):
                        for c in range(n_teams):
                            if r == c: continue
                            
                            home_team = teams[r]
                            away_team = teams[c] # inferred from row order
                            
                            # Extract score
                            try:
                                score_cell = results_matrix.iloc[r, c]
                            except:
                                continue

                            if pd.isna(score_cell) or str(score_cell) in ['—', '-', 'nan', '.']:
                                continue

                            # Clean score
                            clean_score = re.sub(r'\[.*?\]', '', str(score_cell))
                            clean_score = re.sub(r'\(.*?\)', '', clean_score).strip()

                            splitter = None
                            if '–' in clean_score: splitter = '–'
                            elif '-' in clean_score: splitter = '-'
                            
                            if splitter:
                                try:
                                    parts = clean_score.split(splitter)
                                    if len(parts) == 2:
                                        h_goals = int(parts[0].strip())
                                        a_goals = int(parts[1].strip())
                                        
                                        matches.append({
                                            'Season': NEW_SEASON_LABEL,
                                            'Home_Team': home_team,
                                            'Away_Team': away_team,
                                            'Home_Goals': h_goals,
                                            'Away_Goals': a_goals,
                                            'Outcome': 'Home Win' if h_goals > a_goals else ('Away Win' if a_goals > h_goals else 'Draw')
                                        })
                                except:
                                    continue
        
        if not matches:
            print("❌ No matches found. Wikipedia table might be empty or formatted differently.")
            return None
            
        print(f"✅ Scraped {len(matches)} matches from {NEW_SEASON_LABEL}.")
        return pd.DataFrame(matches)

    except Exception as e:
        print(f"❌ Error scraping: {e}")
        return None

def update_master_file():
    # 1. Load Existing Data
    try:
        master_df = pd.read_csv(MASTER_FILE)
        print(f"📂 Loaded Master File: {len(master_df)} matches.")
    except FileNotFoundError:
        print("⚠️ Master file not found. Creating new one.")
        master_df = pd.DataFrame()

    # 2. Scrape New Data
    new_data = scrape_new_season()
    
    if new_data is not None and not new_data.empty:
        # 3. Combine
        combined_df = pd.concat([master_df, new_data])
        
        # 4. Remove Duplicates
        # We check duplicates based on Season, Home, Away to avoid double-counting
        before_dedup = len(combined_df)
        combined_df.drop_duplicates(subset=['Season', 'Home_Team', 'Away_Team'], keep='last', inplace=True)
        after_dedup = len(combined_df)
        
        added_count = after_dedup - len(master_df)
        
        # 5. Save
        combined_df.to_csv(MASTER_FILE, index=False)
        print(f"💾 Updated {MASTER_FILE}")
        print(f"   Total Matches: {after_dedup}")
        print(f"   New Matches Added: {added_count}")
    else:
        print("⚠️ No new data added.")

if __name__ == "__main__":
    update_master_file()