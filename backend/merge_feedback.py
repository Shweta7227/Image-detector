import pandas as pd
import json
import os

def merge_corrections():
    csv_path = "training_data.csv"
    feedback_path = "feedback.json"

    if not os.path.exists(feedback_path) or os.stat(feedback_path).st_size <= 2:
        print("No new feedback to merge.")
        return

    df = pd.read_csv(csv_path)

    with open(feedback_path, "r") as f:
        feedbacks = json.load(f)

    new_rows = []
    for entry in feedbacks:
        if "actual_label" in entry:
            label = 0 if entry["actual_label"] == "Real" else 1
            row = entry["scores"]
            row["label"] = label
            
            # OVERSAMPLING: Add this correction 50 times so the AI 
            # actually pays attention to it!
            for _ in range(50):
                new_rows.append(row.copy())

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Put the new high-priority data at the top
        updated_df = pd.concat([new_df, df], ignore_index=True)
        updated_df.to_csv(csv_path, index=False)
        
        # Clear feedback.json
        with open(feedback_path, "w") as f:
            json.dump([], f)
            
        print(f"✅ Merged {len(feedbacks)} feedback entries (multiplied to {len(new_rows)} rows for weight).")

if __name__ == "__main__":
    merge_corrections()