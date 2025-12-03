import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the SMART data (with Attack/Defense ratings)
df = pd.read_csv('npfl_training_data.csv')

# Features: Strength Ratings
X = df[['Home_Attack', 'Home_Defense', 'Away_Attack', 'Away_Defense']]
y = df['Outcome']

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
# limiting max_depth prevents the model from "memorizing" too much detail
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# --- THE OVERFITTING CHECK ---
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

print(f"📘 TRAINING Accuracy (Memorization): {train_acc:.2%}")
print(f"🔮 TESTING Accuracy (Real World):    {test_acc:.2%}")
print("-" * 40)

gap = train_acc - test_acc
if gap > 0.10:
    print(f"⚠️ DANGER: Overfitting detected! (Gap: {gap:.1%})")
    print("   The model is memorizing the data. We need to simplify it.")
elif gap < 0.05:
    print(f"✅ EXCELLENT: The model is robust. (Gap: {gap:.1%})")
    print("   It performs just as well on new data as old data.")
else:
    print(f"ℹ️ ACCEPTABLE: Slight gap, but normal. (Gap: {gap:.1%})")

print("-" * 40)
print(f"📊 FINAL MODEL ACCURACY: {test_acc:.2%}")