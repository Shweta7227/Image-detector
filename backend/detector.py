import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import easyocr
from spellchecker import SpellChecker
import joblib
import pandas as pd
import os


class DeepfakeDetector:
    def __init__(self, model_path='deepfake_brain.pkl'):
        # 1. Initialize Face Landmarker
        face_base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        face_options = vision.FaceLandmarkerOptions(
            base_options=face_base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

        # 2. Initialize Hand Landmarker
        hand_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        # 3. Initialize OCR and Spellchecker
        self.reader = easyocr.Reader(['en', 'hi'], gpu=False)
        self.spell = SpellChecker()

        # 4. Load the Machine Learning Model
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print("✅ ML Model loaded successfully!")
        else:
            self.model = None
            print("⚠️ No ML model found. Using manual weights fallback.")

    # --- FEATURE EXTRACTION FUNCTIONS ---

    def check_frequency_artifacts(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        high_freq_power = np.sum(magnitude[magnitude > np.percentile(magnitude, 95)])
        score = min(100, high_freq_power / (np.mean(magnitude) * 40))
        return round(score, 1)

    def check_texture_noise(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        score = max(0, min(100, (8000 - variance) / 60)) if variance < 8000 else 0
        return round(score, 1)

    def check_lighting_shadows(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
        
        mean_val = np.mean(sobel)
        std_val = np.std(sobel)
        
        # FIX: Ensure the score never goes below 0 or above 100
        # This prevents the "negative millions" error you saw
        inconsistency = std_val / (mean_val + 1e-6) 
        final_score = np.clip(inconsistency * 10, 0, 100) 
        return float(final_score)

    def check_facial_landmarks(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        results = self.face_landmarker.detect(mp_image)

        if not results.face_landmarks:
            return 25.0

        try:
            landmarks = results.face_landmarks[0]
            eye_dist = abs(landmarks[33].x - landmarks[263].x)
            mouth_width = abs(landmarks[61].x - landmarks[291].x)
            ratio = mouth_width / (eye_dist + 1e-5)
            anomaly = 1 if 0.8 < ratio < 1.3 else 0
            score = 60 if anomaly == 0 else 20
        except:
            score = 25.0
        return round(score, 1)

    def check_hands_fingers(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        results = self.hand_landmarker.detect(mp_image)

        if not results.hand_landmarks:
            return 15.0

        total_anomaly = 0
        for hand_landmarks_list in results.hand_landmarks:
            finger_count = self._count_fingers_new(hand_landmarks_list)
            if finger_count != 5:
                total_anomaly += 40
            total_anomaly += 30
        return min(100, total_anomaly)

    def _count_fingers_new(self, landmarks):
        tips = [4, 8, 12, 16, 20]
        count = 0
        if landmarks[4].x < landmarks[3].x: count += 1
        for i in range(1, 5):
            if landmarks[tips[i]].y < landmarks[tips[i] - 1].y:
                count += 1
        return count

    def check_text_errors(self, img):
        results = self.reader.readtext(img, detail=1)
        if not results: return 15.0
        words = " ".join([res[1] for res in results]).split()
        if not words: return 15.0
        misspelled = self.spell.unknown(words)
        error_rate = len(misspelled) / len(words) * 100
        total_score = min(95, error_rate * 3 + 25)
        return round(total_score, 1)

    def check_morphological_view(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(gray, kernel, iterations=1)
        eroded = cv2.erode(gray, kernel, iterations=1)
        morph_diff = cv2.absdiff(dilated, eroded)
        score = min(90, (np.mean(morph_diff) / 255 * 100) * 2.5)
        return round(score, 1)

    def check_extra_properties(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        skin_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return 40.0 if skin_var < 800 else 15.0

    def extract_all_features(self, img_cv):
        """Used for both training (CSV) and prediction."""
        return {
            "freq_artifacts": self.check_frequency_artifacts(img_cv),
            "texture_noise": self.check_texture_noise(img_cv),
            "lighting_shadows": self.check_lighting_shadows(img_cv),
            "facial_landmarks": self.check_facial_landmarks(img_cv),
            "hand_fingers": self.check_hands_fingers(img_cv),
            "text_errors": self.check_text_errors(img_cv),
            "morphological": self.check_morphological_view(img_cv),
            "extra_props": self.check_extra_properties(img_cv)
        }

    def predict_from_array(self, img_cv):
        features = self.extract_all_features(img_cv)
        
        if self.model:
            df = pd.DataFrame([features])
            prob = self.model.predict_proba(df)[0][1] * 100
            final_score = round(prob, 1)
        else:
            weights = [0.20, 0.15, 0.12, 0.15, 0.20, 0.08, 0.07, 0.03]
            final_score = sum(s * w for s, w in zip(features.values(), weights))

        if final_score >= 70: verdict = "HIGH probability DEEPFAKE"
        elif final_score >= 50: verdict = "SUSPICIOUS - Likely Deepfake"
        else: verdict = "Likely REAL image"

        return {
            "verdict": verdict,
            "final_score": final_score,
            "scores": features,
            "explanation": "Analyzing texture, symmetry, and artifacts..."
        }
# import cv2
# import numpy as np
# import mediapipe as mp                  # ← ADD THIS LINE
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# # from mediapipe.framework.formats import landmark_pb2
# from PIL import Image
# import easyocr
# from spellchecker import SpellChecker
# class DeepfakeDetector:
#     def __init__(self):
#         # self.mp_face = mp.solutions.face_mesh
#         # self.mp_hands = mp.solutions.hands
#         # self.face_mesh = self.mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
#         # self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
#         # Face Landmarker
#         face_base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
#         face_options = vision.FaceLandmarkerOptions(
#             base_options=face_base_options,
#             running_mode=vision.RunningMode.IMAGE,
#             num_faces=1,
#             min_face_detection_confidence=0.5,
#             min_face_presence_confidence=0.5,
#             min_tracking_confidence=0.5,
#             output_face_blendshapes=False,   # we don't need blendshapes for now
#             output_facial_transformation_matrixes=False
#         )
#         self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

#         # Hand Landmarker
#         hand_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
#         hand_options = vision.HandLandmarkerOptions(
#             base_options=hand_base_options,
#             running_mode=vision.RunningMode.IMAGE,
#             num_hands=2,
#             min_hand_detection_confidence=0.5,
#             min_hand_presence_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
#         self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
                
#         self.reader = easyocr.Reader(['en'], gpu=False)
#         self.spell = SpellChecker()
        
#         print("✅ DeepfakeDetector initialized - checks all requested properties!")

#     def load_image(self, image_path):
#         img = cv2.imread(image_path)
#         if img is None:
#             raise ValueError(f"Cannot read image: {image_path}")
#         return img

#     def check_frequency_artifacts(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         f = np.fft.fft2(gray)
#         fshift = np.fft.fftshift(f)
#         magnitude = 20 * np.log(np.abs(fshift) + 1)
#         high_freq_power = np.sum(magnitude[magnitude > np.percentile(magnitude, 95)])
#         score = min(100, high_freq_power / (np.mean(magnitude) * 40))
#         return round(score, 1)

#     def check_texture_noise(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#         variance = laplacian.var()
#         score = max(0, min(100, (8000 - variance) / 60)) if variance < 8000 else 0
#         return round(score, 1)

#     def check_lighting_shadows(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
#         inconsistency = np.std(sobel) / (np.mean(sobel) + 1e-5)
#         score = min(100, inconsistency * 25)
#         return round(score, 1)

#     # def check_facial_landmarks(self, img):
#         # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # results = self.face_mesh.process(rgb)
#         # if not results.multi_face_landmarks:
#         #     return 25.0
#         # landmarks = results.multi_face_landmarks[0].landmark
#         # left_eye = landmarks[33]
#         # right_eye = landmarks[263]
#         # mouth_width = abs(landmarks[61].x - landmarks[291].x)
#         # eye_dist = abs(left_eye.x - right_eye.x)
#         # ratio = mouth_width / (eye_dist + 1e-5)
#         # anomaly = 1 if 0.8 < ratio < 1.3 else 0
#         # score = 60 if anomaly == 0 else 20
#         # return round(score, 1)
#     def check_facial_landmarks(self, img):
#     # Convert OpenCV BGR to RGB and to MediaPipe Image
#         rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

#         # Run detection
#         results = self.face_landmarker.detect(mp_image)

#         if not results.face_landmarks:
#             return 25.0  # neutral if no face

#         # Take first face (we set num_faces=1)
#         landmarks = results.face_landmarks[0]  # list of NormalizedLandmark

#         # Your original check: eye-mouth ratio (adapt to new landmark indices)
#         # MediaPipe Face Landmarker uses 478 points, indices similar but verify
#         # Common: left eye outer ~33, right eye outer ~263, mouth center bottom ~13, etc.
#         # For simplicity, use approximate indices (test and adjust if needed)
#         try:
#             left_eye = landmarks[33]   # left eye outer corner
#             right_eye = landmarks[263] # right eye outer corner
#             mouth_left = landmarks[61]
#             mouth_right = landmarks[291]

#             eye_dist = abs(left_eye.x - right_eye.x)
#             mouth_width = abs(mouth_left.x - mouth_right.x)
#             ratio = mouth_width / (eye_dist + 1e-5)

#             anomaly = 1 if 0.8 < ratio < 1.3 else 0
#             score = 60 if anomaly == 0 else 20
#         except IndexError:
#             score = 25.0  # fallback if indices wrong

#         return round(score, 1)

#     # def check_hands_fingers(self, img):
#     #     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     #     results = self.hands.process(rgb)
#     #     if not results.multi_hand_landmarks:
#     #         return 15.0
        
#     #     total_anomaly = 0
#     #     for hand_landmarks in results.multi_hand_landmarks:
#     #         finger_count = self._count_fingers(hand_landmarks)
#     #         if finger_count != 5:
#     #             total_anomaly += 40
#     #         total_anomaly += 30   # placeholder structure penalty
#     #     return min(100, total_anomaly)
#     def check_hands_fingers(self, img):
#         rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

#         results = self.hand_landmarker.detect(mp_image)

#         if not results.hand_landmarks:
#             return 15.0

#         total_anomaly = 0
#         for hand_landmarks_list in results.hand_landmarks:
#             # hand_landmarks_list is list of 21 NormalizedLandmark per hand
#             finger_count = self._count_fingers_new(hand_landmarks_list)
#             if finger_count != 5:
#                 total_anomaly += 40

#             # Keep simple structure penalty (you can improve later)
#             total_anomaly += 30

#         return min(100, total_anomaly)

#     def _count_fingers_new(self, landmarks):
#         """Adapted finger count for new format (21 landmarks per hand)"""
#         # Landmark indices (standard MediaPipe hand model):
#         # 0: wrist, 1-4: thumb, 5-8: index, 9-12: middle, 13-16: ring, 17-20: pinky
#         # Tip indices: thumb=4, index=8, middle=12, ring=16, pinky=20
#         tips = [4, 8, 12, 16, 20]
#         pip_mcp = [2, 5, 9, 13, 17]  # approximate for raised check

#         count = 0

#         # Thumb (special horizontal check)
#         if landmarks[4].x < landmarks[3].x:  # tip left of IP joint (adjust for orientation)
#             count += 1

#         # Other fingers: tip above PIP joint
#         for i in range(1, 5):  # index to pinky
#             tip_y = landmarks[tips[i]].y
#             pip_y = landmarks[tips[i] - 1].y   # PIP joint
#             if tip_y < pip_y:  # tip higher than PIP (raised)
#                 count += 1

#         return count                    

#     # def _count_fingers(self, landmarks):
#         tips = [4, 8, 12, 16, 20]
#         count = 0
#         for i in range(1, 5):
#             if landmarks.landmark[tips[i]].y < landmarks.landmark[tips[i] - 2].y:
#                 count += 1
#         thumb_tip = landmarks.landmark[4]
#         thumb_ip = landmarks.landmark[3]
#         if abs(thumb_tip.x - thumb_ip.x) > abs(thumb_tip.y - thumb_ip.y):
#             count += 1
#         return count

#     def check_text_errors(self, img):
#         results = self.reader.readtext(img, detail=1)
#         if not results:
#             return 15.0
#         full_text = " ".join([res[1] for res in results])
#         words = full_text.split()
#         if not words:
#             return 15.0
        
#         misspelled = self.spell.unknown(words)
#         spelling_error_rate = len(misspelled) / len(words) * 100
        
#         shape_anomaly = 0
#         for (_, _, conf) in results:
#             if conf < 0.6:
#                 shape_anomaly += 20
#         total_score = min(95, spelling_error_rate * 3 + shape_anomaly + 25)
#         return round(total_score, 1)

#     def check_morphological_view(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         kernel = np.ones((5, 5), np.uint8)
#         dilated = cv2.dilate(gray, kernel, iterations=1)
#         eroded = cv2.erode(gray, kernel, iterations=1)
#         morph_diff = cv2.absdiff(dilated, eroded)
#         anomaly = np.mean(morph_diff) / 255 * 100
#         score = min(90, anomaly * 2.5)
#         return round(score, 1)

#     def check_extra_properties(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         skin_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#         extra_score = 40 if skin_var < 800 else 15
#         return round(extra_score, 1)

#     def predict_from_array(self, img_cv):
#         """Run detection on OpenCV image array (no path needed)"""
#         scores = {
#             "Frequency Artifacts (Google SynthID)": self.check_frequency_artifacts(img_cv),
#             "Texture/Noise (Meta DFDC)": self.check_texture_noise(img_cv),
#             "Lighting/Shadows (Microsoft)": self.check_lighting_shadows(img_cv),
#             "Facial Landmarks (Intel)": self.check_facial_landmarks(img_cv),
#             "Hands/Fingers Structure & Count": self.check_hands_fingers(img_cv),
#             "Text Spelling + Letter Shapes": self.check_text_errors(img_cv),
#             "Morphological View": self.check_morphological_view(img_cv),
#             "Extra (Skin/Eyes)": self.check_extra_properties(img_cv)
#         }

#         weights = [0.20, 0.15, 0.12, 0.15, 0.20, 0.08, 0.07, 0.03]
#         final_score = sum(s * w for s, w in zip(scores.values(), weights))

#         if final_score >= 70:
#             verdict = "HIGH probability DEEPFAKE"
#         elif final_score >= 50:
#             verdict = "SUSPICIOUS - Likely Deepfake"
#         else:
#             verdict = "Likely REAL image"

#         # Simple explanation from top 3 scores
#         sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
#         top_reasons = [f"{prop}: {score:.0f}%" for prop, score in sorted_scores]
#         explanation = "Main signals: " + ", ".join(top_reasons)

#         return {
#             "verdict": verdict,
#             "final_score": round(final_score, 1),
#             "scores": scores,
#             "explanation": explanation
#         }
# import cv2
# import numpy as np
# import mediapipe as mp                  # ← ADD THIS LINE
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# # from mediapipe.framework.formats import landmark_pb2
# from PIL import Image
# import easyocr
# from spellchecker import SpellChecker
# class DeepfakeDetector:
#     def __init__(self):
#         # self.mp_face = mp.solutions.face_mesh
#         # self.mp_hands = mp.solutions.hands
#         # self.face_mesh = self.mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
#         # self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
#         # Face Landmarker
#         face_base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
#         face_options = vision.FaceLandmarkerOptions(
#             base_options=face_base_options,
#             running_mode=vision.RunningMode.IMAGE,
#             num_faces=1,
#             min_face_detection_confidence=0.5,
#             min_face_presence_confidence=0.5,
#             min_tracking_confidence=0.5,
#             output_face_blendshapes=False,   # we don't need blendshapes for now
#             output_facial_transformation_matrixes=False
#         )
#         self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

#         # Hand Landmarker
#         hand_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
#         hand_options = vision.HandLandmarkerOptions(
#             base_options=hand_base_options,
#             running_mode=vision.RunningMode.IMAGE,
#             num_hands=2,
#             min_hand_detection_confidence=0.5,
#             min_hand_presence_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
#         self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
                
#         self.reader = easyocr.Reader(['en'], gpu=False)
#         self.spell = SpellChecker()
        
#         print("✅ DeepfakeDetector initialized - checks all requested properties!")

#     def load_image(self, image_path):
#         img = cv2.imread(image_path)
#         if img is None:
#             raise ValueError(f"Cannot read image: {image_path}")
#         return img

#     def check_frequency_artifacts(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         f = np.fft.fft2(gray)
#         fshift = np.fft.fftshift(f)
#         magnitude = 20 * np.log(np.abs(fshift) + 1)
#         high_freq_power = np.sum(magnitude[magnitude > np.percentile(magnitude, 95)])
#         score = min(100, high_freq_power / (np.mean(magnitude) * 40))
#         return round(score, 1)

#     def check_texture_noise(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#         variance = laplacian.var()
#         score = max(0, min(100, (8000 - variance) / 60)) if variance < 8000 else 0
#         return round(score, 1)

#     def check_lighting_shadows(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
#         inconsistency = np.std(sobel) / (np.mean(sobel) + 1e-5)
#         score = min(100, inconsistency * 25)
#         return round(score, 1)

#     # def check_facial_landmarks(self, img):
#         # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # results = self.face_mesh.process(rgb)
#         # if not results.multi_face_landmarks:
#         #     return 25.0
#         # landmarks = results.multi_face_landmarks[0].landmark
#         # left_eye = landmarks[33]
#         # right_eye = landmarks[263]
#         # mouth_width = abs(landmarks[61].x - landmarks[291].x)
#         # eye_dist = abs(left_eye.x - right_eye.x)
#         # ratio = mouth_width / (eye_dist + 1e-5)
#         # anomaly = 1 if 0.8 < ratio < 1.3 else 0
#         # score = 60 if anomaly == 0 else 20
#         # return round(score, 1)
#     def check_facial_landmarks(self, img):
#     # Convert OpenCV BGR to RGB and to MediaPipe Image
#         rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

#         # Run detection
#         results = self.face_landmarker.detect(mp_image)

#         if not results.face_landmarks:
#             return 25.0  # neutral if no face

#         # Take first face (we set num_faces=1)
#         landmarks = results.face_landmarks[0]  # list of NormalizedLandmark

#         # Your original check: eye-mouth ratio (adapt to new landmark indices)
#         # MediaPipe Face Landmarker uses 478 points, indices similar but verify
#         # Common: left eye outer ~33, right eye outer ~263, mouth center bottom ~13, etc.
#         # For simplicity, use approximate indices (test and adjust if needed)
#         try:
#             left_eye = landmarks[33]   # left eye outer corner
#             right_eye = landmarks[263] # right eye outer corner
#             mouth_left = landmarks[61]
#             mouth_right = landmarks[291]

#             eye_dist = abs(left_eye.x - right_eye.x)
#             mouth_width = abs(mouth_left.x - mouth_right.x)
#             ratio = mouth_width / (eye_dist + 1e-5)

#             anomaly = 1 if 0.8 < ratio < 1.3 else 0
#             score = 60 if anomaly == 0 else 20
#         except IndexError:
#             score = 25.0  # fallback if indices wrong

#         return round(score, 1)

#     # def check_hands_fingers(self, img):
#     #     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     #     results = self.hands.process(rgb)
#     #     if not results.multi_hand_landmarks:
#     #         return 15.0
        
#     #     total_anomaly = 0
#     #     for hand_landmarks in results.multi_hand_landmarks:
#     #         finger_count = self._count_fingers(hand_landmarks)
#     #         if finger_count != 5:
#     #             total_anomaly += 40
#     #         total_anomaly += 30   # placeholder structure penalty
#     #     return min(100, total_anomaly)
#     def check_hands_fingers(self, img):
#         rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

#         results = self.hand_landmarker.detect(mp_image)

#         if not results.hand_landmarks:
#             return 15.0

#         total_anomaly = 0
#         for hand_landmarks_list in results.hand_landmarks:
#             # hand_landmarks_list is list of 21 NormalizedLandmark per hand
#             finger_count = self._count_fingers_new(hand_landmarks_list)
#             if finger_count != 5:
#                 total_anomaly += 40

#             # Keep simple structure penalty (you can improve later)
#             total_anomaly += 30

#         return min(100, total_anomaly)

#     def _count_fingers_new(self, landmarks):
#         """Adapted finger count for new format (21 landmarks per hand)"""
#         # Landmark indices (standard MediaPipe hand model):
#         # 0: wrist, 1-4: thumb, 5-8: index, 9-12: middle, 13-16: ring, 17-20: pinky
#         # Tip indices: thumb=4, index=8, middle=12, ring=16, pinky=20
#         tips = [4, 8, 12, 16, 20]
#         pip_mcp = [2, 5, 9, 13, 17]  # approximate for raised check

#         count = 0

#         # Thumb (special horizontal check)
#         if landmarks[4].x < landmarks[3].x:  # tip left of IP joint (adjust for orientation)
#             count += 1

#         # Other fingers: tip above PIP joint
#         for i in range(1, 5):  # index to pinky
#             tip_y = landmarks[tips[i]].y
#             pip_y = landmarks[tips[i] - 1].y   # PIP joint
#             if tip_y < pip_y:  # tip higher than PIP (raised)
#                 count += 1

#         return count                    

#     # def _count_fingers(self, landmarks):
#         tips = [4, 8, 12, 16, 20]
#         count = 0
#         for i in range(1, 5):
#             if landmarks.landmark[tips[i]].y < landmarks.landmark[tips[i] - 2].y:
#                 count += 1
#         thumb_tip = landmarks.landmark[4]
#         thumb_ip = landmarks.landmark[3]
#         if abs(thumb_tip.x - thumb_ip.x) > abs(thumb_tip.y - thumb_ip.y):
#             count += 1
#         return count

#     def check_text_errors(self, img):
#         results = self.reader.readtext(img, detail=1)
#         if not results:
#             return 15.0
#         full_text = " ".join([res[1] for res in results])
#         words = full_text.split()
#         if not words:
#             return 15.0
        
#         misspelled = self.spell.unknown(words)
#         spelling_error_rate = len(misspelled) / len(words) * 100
        
#         shape_anomaly = 0
#         for (_, _, conf) in results:
#             if conf < 0.6:
#                 shape_anomaly += 20
#         total_score = min(95, spelling_error_rate * 3 + shape_anomaly + 25)
#         return round(total_score, 1)

#     def check_morphological_view(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         kernel = np.ones((5, 5), np.uint8)
#         dilated = cv2.dilate(gray, kernel, iterations=1)
#         eroded = cv2.erode(gray, kernel, iterations=1)
#         morph_diff = cv2.absdiff(dilated, eroded)
#         anomaly = np.mean(morph_diff) / 255 * 100
#         score = min(90, anomaly * 2.5)
#         return round(score, 1)

#     def check_extra_properties(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         skin_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#         extra_score = 40 if skin_var < 800 else 15
#         return round(extra_score, 1)

#     def predict_from_array(self, img_cv):
#         """Run detection on OpenCV image array (no path needed)"""
#         scores = {
#             "Frequency Artifacts (Google SynthID)": self.check_frequency_artifacts(img_cv),
#             "Texture/Noise (Meta DFDC)": self.check_texture_noise(img_cv),
#             "Lighting/Shadows (Microsoft)": self.check_lighting_shadows(img_cv),
#             "Facial Landmarks (Intel)": self.check_facial_landmarks(img_cv),
#             "Hands/Fingers Structure & Count": self.check_hands_fingers(img_cv),
#             "Text Spelling + Letter Shapes": self.check_text_errors(img_cv),
#             "Morphological View": self.check_morphological_view(img_cv),
#             "Extra (Skin/Eyes)": self.check_extra_properties(img_cv)
#         }

#         weights = [0.20, 0.15, 0.12, 0.15, 0.20, 0.08, 0.07, 0.03]
#         final_score = sum(s * w for s, w in zip(scores.values(), weights))

#         if final_score >= 70:
#             verdict = "HIGH probability DEEPFAKE"
#         elif final_score >= 50:
#             verdict = "SUSPICIOUS - Likely Deepfake"
#         else:
#             verdict = "Likely REAL image"

#         # Simple explanation from top 3 scores
#         sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
#         top_reasons = [f"{prop}: {score:.0f}%" for prop, score in sorted_scores]
#         explanation = "Main signals: " + ", ".join(top_reasons)

#         return {
#             "verdict": verdict,
#             "final_score": round(final_score, 1),
#             "scores": scores,
#             "explanation": explanation
#         }