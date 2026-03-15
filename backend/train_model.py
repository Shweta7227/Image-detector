import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_brain(csv_file="training_data.csv"):
    df = pd.read_csv(csv_file)
    X = df.drop("label", axis=1) # The 8 scores
    y = df["label"]              # 0 for Real, 1 for Fake

    model = RandomForestClassifier(n_estimators=200, max_depth=10)
    model.fit(X, y)
    
    joblib.dump(model, 'deepfake_brain.pkl')
    print("🧠 Brain trained and saved as deepfake_brain.pkl")

if __name__ == "__main__":
    train_brain()