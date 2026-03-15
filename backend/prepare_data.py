import os
import pandas as pd
import cv2
from detector import DeepfakeDetector

def generate_csv(real_folder, fake_folder, output_file="training_data.csv"):
    detector = DeepfakeDetector()
    dataset = []

# Process Real Images (Label 0)
    print("Processing Real Images...")
    real_files = os.listdir(real_folder)
    for i, filename in enumerate(real_files):
        img = cv2.imread(os.path.join(real_folder, filename))
        if img is not None:
            features = detector.extract_all_features(img)
            features['label'] = 0 
            dataset.append(features)
        
        # This will now correctly print every 10 images
        if i % 10 == 0:
            print(f" > Real Image {i}/{len(real_files)} done...")

    # Process Fake Images (Label 1)
    print("\nProcessing Fake Images...")
    fake_files = os.listdir(fake_folder)
    for i, filename in enumerate(fake_files):
        img = cv2.imread(os.path.join(fake_folder, filename))
        if img is not None:
            features = detector.extract_all_features(img)
            features['label'] = 1
            dataset.append(features)
        
        if i % 50 == 0: # Prints every 50 for the 10,000 images
            print(f" > Fake Image {i}/{len(fake_files)} done...")

    # Convert to DataFrame and Save
    df = pd.DataFrame(dataset)
    df.to_csv(output_file, index=False)
    print(f"Done! {output_file} created with {len(df)} rows.")

if __name__ == "__main__":
    generate_csv("data/train/Real", "data/train/Fake")