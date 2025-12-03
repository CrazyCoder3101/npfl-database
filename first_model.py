import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. LOAD DATA
df = pd.read_csv('npfl_historical_data.csv')

# 2. PREPROCESS: TURN NAMES INTO NUMBERS
# We use .astype('category').cat.codes to assign a unique number to each team
# E.g., Abia Warriors = 0, Enyimba = 4, etc.
df['Home_Team_Code'] = df['Home_Team'].astype('category').cat.codes
df['Away_Team_Code'] = df['Away_Team'].astype('category').cat.codes

# Create a dictionary so we can look up names later
team_map = dict(enumerate(df['Home_Team'].astype('category').cat.categories))

# 3. DEFINE FEATURES (X) AND TARGET (y)
# X = The input (Who is playing?)
# y = The output (Did Home Win, Draw, or Away Win?)
X = df[['Home_Team_Code', 'Away_Team_Code']]
y = df['Outcome']

# 4. SPLIT DATA
# Train on 80% of matches, Test on 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. TRAIN THE MODEL
print("🤖 Training the model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. EVALUATE
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n" + "="*40)
print(f"🎯 MODEL ACCURACY: {accuracy:.2%}")
print("="*40)
print("\nWhat this means:")
print(f"- If you guessed blindly, you'd get ~33% right.")
print(f"- Your model is getting {accuracy:.0%} of matches right.")
print("="*40)

# 7. THE FUN PART: PREDICT A FAKE MATCH
# Let's pick two random teams from your map
team_a_id = 0  # Likely Abia Warriors (alphabetical)
team_b_id = 4  # Likely someone like Akwa or Bendel

team_a_name = team_map[team_a_id]
team_b_name = team_map[team_b_id]

print(f"\n🔮 PREDICTION TEST: {team_a_name} (Home) vs {team_b_name} (Away)")
# We have to reshape the input to look like a list of matches
match_input = pd.DataFrame([[team_a_id, team_b_id]], columns=['Home_Team_Code', 'Away_Team_Code'])
pred = model.predict(match_input)
probs = model.predict_proba(match_input)

print(f"   Model Predicts: {pred[0]}")
print(f"   Confidence: {max(probs[0]):.2%} sure.")