import pandas as pd
import numpy as np
import re
import requests
from io import StringIO

def scrape_npfl_season(url, season_label):
    """
    Scrapes the 'Results' table from a Wikipedia NPFL season page.
    Uses positional indexing (iloc) to handle abbreviated column headers.
    """
    print(f"--- Processing: {season_label} ---")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Read all tables
        tables = pd.read_html(StringIO(response.text))
        print(f"   Found {len(tables)} tables on page.")

        matches = []
        tables_processed = 0
        
        for df in tables:
            # 1. Filter: Must be roughly square (Teams x Teams)
            if df.shape[0] < 8 or df.shape[1] < 8:
                continue
            
            # Allow for slight difference in rows/cols (sometimes headers mess up count)
            if abs(df.shape[0] - df.shape[1]) > 3:
                continue

            # 2. Filter: "Dash Density" Check
            # Results tables are full of "1-0" or "–". Standings tables are just numbers.
            # We convert to string and count dashes.
            df_str = df.astype(str)
            dash_count = df_str.apply(lambda x: x.str.contains(r'[–-]', regex=True)).sum().sum()
            total_cells = df.size
            
            # If less than 5% of cells have dashes, it's probably not a results matrix
            if dash_count / total_cells < 0.05:
                continue

            # --- PROCESS CANDIDATE TABLE ---
            tables_processed += 1
            
            # Set first column as index (Home Teams)
            # We assume the First Column contains the Home Team names
            results_matrix = df.set_index(df.columns[0])
            
            # Get list of Home Teams (Rows)
            home_teams = results_matrix.index.tolist()
            
            # We verify if the number of columns (excluding index) matches the number of rows
            # This helps confirm we have the right matrix structure
            # Wikipedia Result matrices usually have len(columns) == len(rows)
            
            # Iterate by POSITION (i, j)
            # We assume Row[i] corresponds to Col[j] when i==j (Diagonal)
            # Therefore, the Away Team for Column j is the name of Home Team at Row j
            
            rows_count = len(home_teams)
            cols_count = len(results_matrix.columns)
            
            # Safety clamp: only iterate up to the smaller dimension
            n_teams = min(rows_count, cols_count)

            for r in range(n_teams):      # Row Index (Home)
                for c in range(n_teams):  # Col Index (Away)
                    if r == c:
                        continue # Skip Diagonal
                    
                    home_team_name = home_teams[r]
                    away_team_name = home_teams[c] # inferred from row order
                    
                    # Extract score using integer location
                    score_cell = results_matrix.iloc[r, c]

                    # Skip empty/invalid
                    if pd.isna(score_cell) or str(score_cell) in ['—', '-', 'nan', 'nan', '.']:
                        continue

                    # Clean Score
                    clean_score = re.sub(r'\[.*?\]', '', str(score_cell)) # Remove [a]
                    clean_score = re.sub(r'\(.*?\)', '', clean_score).strip() # Remove (citation)

                    splitter = None
                    if '–' in clean_score: splitter = '–'
                    elif '-' in clean_score: splitter = '-'
                    
                    if not splitter:
                        continue

                    try:
                        parts = clean_score.split(splitter)
                        if len(parts) == 2:
                            h_goals = int(parts[0].strip())
                            a_goals = int(parts[1].strip())
                            
                            matches.append({
                                'Season': season_label,
                                'Home_Team': home_team_name,
                                'Away_Team': away_team_name,
                                'Home_Goals': h_goals,
                                'Away_Goals': a_goals,
                                'Outcome': 'Home Win' if h_goals > a_goals else ('Away Win' if a_goals > h_goals else 'Draw')
                            })
                    except ValueError:
                        continue

        if tables_processed == 0:
            print(f"❌ No valid Results Matrix found for {season_label}.")
        else:
            print(f"✅ Extracted {len(matches)} matches from {season_label} ({tables_processed} tables).")
        
        return pd.DataFrame(matches)

    except Exception as e:
        print(f"❌ Critical Error scraping {season_label}: {e}")
        return pd.DataFrame()

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    seasons = [
        ("2024-25", "https://en.wikipedia.org/wiki/2024%E2%80%9325_Nigeria_Premier_Football_League"),
        ("2023-24", "https://en.wikipedia.org/wiki/2023%E2%80%9324_Nigeria_Premier_Football_League"),
        ("2022-23", "https://en.wikipedia.org/wiki/2022%E2%80%9323_Nigeria_Professional_Football_League"),
        ("2021-22", "https://en.wikipedia.org/wiki/2021%E2%80%9322_Nigeria_Professional_Football_League"),
    ]

    all_data = []

    for label, link in seasons:
        df = scrape_npfl_season(link, label)
        if not df.empty:
            all_data.append(df)

    if all_data:
        final_dataset = pd.concat(all_data, ignore_index=True)
        filename = "npfl_historical_data.csv"
        final_dataset.to_csv(filename, index=False)
        print(f"\n🎉 GRAND TOTAL: Saved {len(final_dataset)} matches to '{filename}'.")
        print(final_dataset['Season'].value_counts())
    else:
        print("\n⚠️ No data scraped.")