import os
import shutil

# This script lives in /backend
def move_fake_data(download_path):
    target_path = "data/train/fake"
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # We only move .jpg files that have a matching .json 
    # as per the dataset README 
    files = os.listdir(download_path)
    for f in files:
        if f.endswith(".jpg"):
            json_file = f.replace(".jpg", ".json")
            if json_file in files:
                shutil.copy(os.path.join(download_path, f), os.path.join(target_path, f))
    print("Fake images organized.")

if __name__ == "__main__":
    # Point this to where you unzipped the 10,000 photos
    move_fake_data("data/raw_downloads/generated.photos")