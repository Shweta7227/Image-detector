import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

def train_brain(csv_file="training_data.csv"):
    df = pd.read_csv(csv_file)
    X = df.drop("label", axis=1) 
    y = df["label"]              

    # The Pipeline ensures data is ALWAYS scaled before reaching the AI
    model = Pipeline([
        ('scaler', StandardScaler()), 
        ('classifier', RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=2))
    ])
    
    model.fit(X, y)
    
    joblib.dump(model, 'deepfake_brain.pkl')
    print("🧠 Brain trained with Scaling and saved as deepfake_brain.pkl")

if __name__ == "__main__":
    train_brain()