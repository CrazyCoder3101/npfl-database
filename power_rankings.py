import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# 1. RELOAD & PREP DATA
print("📊 Loading data...")
df = pd.read_csv('npfl_historical_data.csv')
df['Home_Team_Code'] = df['Home_Team'].astype('category').cat.codes
df['Away_Team_Code'] = df['Away_Team'].astype('category').cat.codes

# 2. TRAIN A BETTER MODEL (ONE-HOT ENCODING)
# To get a "Power Ranking," we need to give every team its own column
print("🧠 Training the Ranking Model...")
model_data = pd.get_dummies(df[['Home_Team', 'Away_Team']], prefix=['Home', 'Away'])
X = model_data
y = df['Outcome']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# 3. EXTRACT "STRENGTH" SCORES
# We look at who predicts a "Home Win" most strongly.
home_win_index = list(model.classes_).index('Home Win')
home_coeffs = model.coef_[home_win_index]

feature_names = X.columns
team_scores = []

for i, name in enumerate(feature_names):
    if "Home_" in name:
        clean_name = name.replace("Home_", "")
        score = home_coeffs[i]
        team_scores.append({'Team': clean_name, 'Score': score})

# 4. CREATE THE RANKING
ranking_df = pd.DataFrame(team_scores).sort_values(by='Score', ascending=False)

# 5. VISUALIZE AND SAVE
print("🎨 Generating Graph...")
plt.figure(figsize=(12, 10))
sns.barplot(data=ranking_df.head(15), x='Score', y='Team', palette='viridis')
plt.title('Top 15 Strongest Home Teams (NPFL Historical Data)', fontsize=15)
plt.xlabel('Strength Score (Coefficient)')
plt.axvline(0, color='k', linestyle='--') # Add a center line

# SAVE INSTEAD OF SHOW (Prevents Crashes)
filename = 'npfl_rankings.png'
plt.savefig(filename)
print(f"✅ Graph saved as '{filename}' in your folder.")

print("\n🏆 TOP 5 STRONGEST TEAMS (HOME):")
print(ranking_df.head(5)[['Team', 'Score']])

print("\n📉 BOTTOM 5 WEAKEST TEAMS (HOME):")
print(ranking_df.tail(5)[['Team', 'Score']])