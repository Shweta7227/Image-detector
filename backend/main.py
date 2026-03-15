from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from detector import DeepfakeDetector
import json
import os
from datetime import datetime
import time
import pandas as pd  # NEW


app = FastAPI(title="Deepfake Detector Backend")

# Allow frontend (Vite runs on http://localhost:5173 by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = DeepfakeDetector()

FEEDBACK_FILE = "feedback.json"

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    start_time = time.time()  # optional: for timing

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_cv is None:
            raise ValueError("Could not decode image")

        # Resize if image is too large (this is now in the right place!)
        if max(img_cv.shape[:2]) > 1024:
            scale = 1024 / max(img_cv.shape[:2])
            img_cv = cv2.resize(img_cv, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Run detection
        result = detector.predict_from_array(img_cv)

        # Add simple explanation based on top scores
        sorted_scores = sorted(result["scores"].items(), key=lambda x: x[1], reverse=True)
        top_reasons = [f"{prop}: {score:.0f}%" for prop, score in sorted_scores[:3]]
        result["explanation"] = "Main signals: " + ", ".join(top_reasons)

        # Optional: add processing time to response
        result["processing_time_seconds"] = round(time.time() - start_time, 2)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


# @app.post("/feedback")
# async def save_feedback(data: dict):
#     entry = {
#         "timestamp": datetime.now().isoformat(),
#         **data
#     }

#     if os.path.exists(FEEDBACK_FILE):
#         with open(FEEDBACK_FILE, "r") as f:
#             feedback_list = json.load(f)
#     else:
#         feedback_list = []

#     feedback_list.append(entry)

#     with open(FEEDBACK_FILE, "w") as f:
#         json.dump(feedback_list, f, indent=2)

#     return {"status": "feedback saved"}
@app.post("/feedback")
async def save_feedback(data: dict):
    file_path = "feedback.json"
    
    # 1. Load existing feedback safely
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        feedbacks = []
    else:
        with open(file_path, "r") as f:
            feedbacks = json.load(f)

    # 2. Add the new correction
    feedbacks.append(data)

    # 3. Save back to file
    with open(file_path, "w") as f:
        json.dump(feedbacks, f, indent=4)
        
    return {"status": "Feedback saved. This will be used in the next training cycle."}


@app.get("/health")
def health_check():
    return {"status": "ok"}
# in trail 4
@app.get("/export_training_data")
def export_data():
    if not os.path.exists(FEEDBACK_FILE):
        return {"error": "No feedback yet"}
    
    with open(FEEDBACK_FILE, "r") as f:
        feedback = json.load(f)
    
    rows = []
    for entry in feedback:
        # entry["user_correct_label"] would come from your frontend button
        # entry["scores"] is what the detector originally saw
        row = entry["scores"]
        row["label"] = 1 if entry["user_label"] == "Fake" else 0
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv("training_data.csv", index=False)
    return {"status": "training_data.csv updated!"}